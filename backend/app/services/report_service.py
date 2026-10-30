import os
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.evidence import Evidence, AuditLog
from app.models.analysis import (
    MediaAnalysis,
    FaceDetection,
    DeepfakeAnalysis,
    ElaAnalysis,
    FftAnalysis
)
from app.models.report import ForensicReport
from app.services.hashing_service import HashingService
from app.services.report_pdf_service import ReportPdfService

class ReportService:
    @staticmethod
    def select_run(query, model):
        """Prefer latest completed work; retain newer attempt state explicitly."""
        ordering = (model.started_at.desc(), model.id.desc())
        latest = query.order_by(*ordering).first()
        completed = query.filter(model.status == "COMPLETED").order_by(*ordering).first()
        selected = completed or latest
        latest_attempt = None if latest is None else {
            "run_id": latest.id, "status": latest.status,
            "started_at": latest.started_at.isoformat(),
            "error_message": latest.error_message,
        }
        return selected, latest_attempt

    @staticmethod
    def verify_original(db: Session, evidence: Evidence) -> str:
        path = Path(evidence.storage_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Original evidence file is missing.")
        current_hash = HashingService.calculate_sha256(path)
        if current_hash != evidence.sha256_hash:
            db.add(AuditLog(evidence_id=evidence.id, action="INTEGRITY_VERIFICATION_FAILED",
                            details="Original evidence hash mismatch during report generation."))
            db.commit()
            raise HTTPException(status_code=409, detail="Original evidence integrity mismatch. Report aborted.")
        return datetime.utcnow().isoformat() + "Z"

    @staticmethod
    def generate_report(db: Session, evidence_id: int) -> ForensicReport:
        # 1. Verify evidence exists
        evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
        if not evidence:
            raise HTTPException(status_code=404, detail=f"Evidence record with ID {evidence_id} not found.")

        integrity_checked_at = ReportService.verify_original(db, evidence)

        # Ensure reports directory exists
        settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate unique report UUID and file path
        report_uuid = str(uuid.uuid4())
        pdf_filename = f"report_{evidence_id}_{report_uuid}.pdf"
        pdf_path = settings.REPORTS_DIR / pdf_filename
        snapshot_path = pdf_path.with_suffix('.json')

        # Initialize ForensicReport database record
        db_report = ForensicReport(
            evidence_id=evidence_id,
            report_uuid=report_uuid,
            status="GENERATING",
            generated_at=datetime.utcnow(),
            file_path=str(pdf_path),
            file_hash=None,
            error_message=None
        )
        db.add(db_report)

        # Log started audit event
        db_audit_start = AuditLog(
            evidence_id=evidence_id,
            action="REPORT_GENERATION_STARTED",
            timestamp=datetime.utcnow(),
            details=f"Forensic PDF report generation initialized. UUID: {report_uuid}"
        )
        db.add(db_audit_start)
        db.commit()
        db.refresh(db_report)

        try:
            # 2. Gather preprocessing & face detection data
            preprocessing_run, pre_latest = ReportService.select_run(db.query(MediaAnalysis).filter(
                MediaAnalysis.evidence_id == evidence_id,
                MediaAnalysis.analysis_type == "PREPROCESSING_FACE_DETECTION"
            ), MediaAnalysis)

            pre_dict = {
                "latest_attempt": pre_latest,
                "run_id": None,
                "status": "NOT_RUN",
                "face_detector": "Haar Cascade (Frontal Face Default)",
                "detected_faces": 0,
                "faces": []
            }
            integrity_status = "VERIFIED"

            if preprocessing_run:
                pre_dict["run_id"] = preprocessing_run.id
                pre_dict["status"] = preprocessing_run.status
                pre_dict["error_message"] = preprocessing_run.error_message
                pre_dict["integrity_verified_at_preprocessing"] = preprocessing_run.integrity_verified

                # Fetch detected faces
                faces = db.query(FaceDetection).filter(FaceDetection.analysis_id == preprocessing_run.id).all()
                pre_dict["detected_faces"] = len(faces)
                for f in faces:
                    pre_dict["faces"].append({
                        "face_index": f.face_index,
                        "frame_index": f.frame_index,
                        "timestamp_seconds": f.timestamp_seconds,
                        "x": f.x,
                        "y": f.y,
                        "width": f.width,
                        "height": f.height,
                        "crop_path": f.crop_path
                    })

                # Check for video metadata payload in error_message
                if preprocessing_run.status == "COMPLETED" and preprocessing_run.error_message:
                    try:
                        meta_dict = json.loads(preprocessing_run.error_message)
                        if "frame_count" in meta_dict:
                            pre_dict["video_metadata"] = meta_dict
                    except Exception:
                        pass

            # 3. Gather deepfake analysis data
            deepfake_run, df_latest = ReportService.select_run(db.query(DeepfakeAnalysis).filter(
                DeepfakeAnalysis.evidence_id == evidence_id,
                DeepfakeAnalysis.preprocessing_analysis_id == (preprocessing_run.id if preprocessing_run else -1)
            ), DeepfakeAnalysis)

            df_dict = {
                "latest_attempt": df_latest,
                "run_id": None,
                "status": "NOT_RUN",
                "model_name": settings.DEEPFAKE_MODEL_NAME,
                "model_version": settings.DEEPFAKE_MODEL_VERSION,
                "model_status": "NOT_RUN",
                "overall_result": "N/A",
                "overall_confidence": "N/A"
            }

            if deepfake_run:
                df_dict["run_id"] = deepfake_run.id
                df_dict["stored_status"] = deepfake_run.status
                df_dict["stored_overall_result"] = deepfake_run.overall_label
                df_dict["preprocessing_run_id"] = deepfake_run.preprocessing_analysis_id
                df_dict["error_message"] = deepfake_run.error_message
                df_dict["status"] = deepfake_run.status
                df_dict["model_name"] = deepfake_run.model_name
                df_dict["model_version"] = deepfake_run.model_version
                df_dict["overall_result"] = deepfake_run.overall_label
                df_dict["overall_confidence"] = f"{deepfake_run.overall_confidence * 100:.2f}%" if deepfake_run.overall_confidence is not None else "N/A"
                # Check model configuration status
                if deepfake_run.status in ("QUEUED", "PROCESSING"):
                    df_dict.update(model_status="NOT_RUN", overall_result="NOT_COMPLETED", overall_confidence="N/A")
                elif deepfake_run.status != "COMPLETED" or deepfake_run.overall_label == "MODEL_ERROR":
                    df_dict["model_status"] = "MODEL_ERROR"
                    df_dict["overall_result"] = "MODEL_ERROR"
                    df_dict["overall_confidence"] = "N/A"
                elif deepfake_run.overall_label == "NO_FACE_DETECTED":
                    df_dict["model_status"] = "NOT_RUN"
                elif deepfake_run.model_version == "NOT_CONFIGURED" or deepfake_run.overall_label == "MODEL_NOT_CONFIGURED":
                    df_dict.update(model_status="MODEL_NOT_CONFIGURED", overall_result="MODEL_NOT_CONFIGURED",
                                   overall_confidence="N/A")
                else:
                    df_dict["model_status"] = "CONFIGURED"
                # Old persisted rows may contain REAL despite failed child predictions.
                if any(p.model_status != "CONFIGURED" or p.label not in ("REAL", "FAKE")
                       for p in deepfake_run.predictions):
                    if deepfake_run.overall_label != "MODEL_NOT_CONFIGURED":
                        df_dict.update(model_status="MODEL_ERROR", overall_result="MODEL_ERROR",
                                       overall_confidence="N/A", status="FAILED")

            # 4. Gather ELA data
            ela_run, ela_latest = ReportService.select_run(db.query(ElaAnalysis).filter(
                ElaAnalysis.evidence_id == evidence_id
            ), ElaAnalysis)

            ela_dict = {
                "latest_attempt": ela_latest,
                "run_id": None,
                "status": "NOT_RUN",
                "jpeg_quality": 95,
                "mean_error": "N/A",
                "max_error": "N/A",
                "error_variance": "N/A",
                "visualization_path": None
            }

            if ela_run:
                ela_dict["run_id"] = ela_run.id
                ela_dict["error_message"] = ela_run.error_message
                ela_dict["status"] = ela_run.status
                ela_dict["jpeg_quality"] = ela_run.jpeg_quality
                ela_dict["mean_error"] = ela_run.mean_error if ela_run.mean_error is not None else "N/A"
                ela_dict["max_error"] = ela_run.max_error if ela_run.max_error is not None else "N/A"
                ela_dict["error_variance"] = ela_run.error_variance if ela_run.error_variance is not None else "N/A"
                ela_dict["visualization_path"] = ela_run.visualization_path

            # 5. Gather FFT data
            fft_run, fft_latest = ReportService.select_run(db.query(FftAnalysis).filter(
                FftAnalysis.evidence_id == evidence_id
            ), FftAnalysis)

            fft_dict = {
                "latest_attempt": fft_latest,
                "run_id": None,
                "status": "NOT_RUN",
                "high_freq_radius_percent": 15.0,
                "average_magnitude": "N/A",
                "high_frequency_ratio": "N/A",
                "spectral_entropy": "N/A",
                "visualization_path": None
            }

            if fft_run:
                fft_dict["run_id"] = fft_run.id
                fft_dict["error_message"] = fft_run.error_message
                fft_dict["status"] = fft_run.status
                fft_dict["high_freq_radius_percent"] = fft_run.high_freq_radius_percent
                fft_dict["average_magnitude"] = fft_run.average_magnitude if fft_run.average_magnitude is not None else "N/A"
                fft_dict["high_frequency_ratio"] = fft_run.high_frequency_ratio if fft_run.high_frequency_ratio is not None else "N/A"
                fft_dict["spectral_entropy"] = fft_run.spectral_entropy if fft_run.spectral_entropy is not None else "N/A"
                fft_dict["visualization_path"] = fft_run.visualization_path

            # 6. Gather Audit Logs data
            for selected, item in ((preprocessing_run, pre_dict), (deepfake_run, df_dict),
                                   (ela_run, ela_dict), (fft_run, fft_dict)):
                if selected is not None:
                    item['started_at'] = selected.started_at.isoformat()
                    item['completed_at'] = selected.completed_at.isoformat() if selected.completed_at else None
            audit_logs = db.query(AuditLog).filter(
                AuditLog.evidence_id == evidence_id
            ).order_by(AuditLog.timestamp.asc()).all()

            audit_list = []
            for log in audit_logs:
                audit_list.append({
                    "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "action": log.action,
                    "details": log.details or ""
                })

            # Assemble the complete report data dictionary (provisional pass)
            report_data = {
                "report_uuid": report_uuid,
                "evidence_id": evidence_id,
                "generated_at": db_report.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "evidence_integrity_status": integrity_status,
                "integrity_checked_at": integrity_checked_at,
                "snapshot_schema_version": 1,
                "evidence": {
                    "original_filename": evidence.original_filename,
                    "media_category": evidence.media_category,
                    "mime_type": evidence.mime_type,
                    "file_size": evidence.file_size,
                    "uploaded_at": evidence.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "sha256_hash": evidence.sha256_hash
                },
                "preprocessing": pre_dict,
                "deepfake": df_dict,
                "ela": ela_dict,
                "fft": fft_dict,
                "audit_logs": audit_list
            }

            # Freeze selected run IDs and values once. No latest-run queries occur
            # during rendering. The PDF carries the digest of this JSON snapshot.
            snapshot_bytes = json.dumps(report_data, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')
            snapshot_path.write_bytes(snapshot_bytes)
            report_data['snapshot_sha256'] = hashlib.sha256(snapshot_bytes).hexdigest()
            ReportPdfService.generate_pdf(report_data, str(pdf_path))
            ReportService.verify_original(db, evidence)

            # Calculate SHA-256 hash of the final PDF file
            pdf_hash = HashingService.calculate_sha256(pdf_path)

            # Update ForensicReport record status and hash
            db_report.status = "COMPLETED"
            db_report.file_hash = pdf_hash
            db.add(AuditLog(evidence_id=evidence_id, action="REPORT_GENERATION_COMPLETED",
                            details=f"Report UUID: {report_uuid}; snapshot SHA-256: {report_data['snapshot_sha256']}"))
            db.add(db_report)
            db.commit()
            db.refresh(db_report)
            return db_report

        except Exception as e:
            db.rollback()
            pdf_path.unlink(missing_ok=True)
            snapshot_path.unlink(missing_ok=True)
            # Mark the report record as FAILED and store the exception message
            failed_report = db.query(ForensicReport).filter(ForensicReport.id == db_report.id).first()
            if failed_report:
                failed_report.status = "FAILED"
                failed_report.error_message = str(e)[:500]
                db.add(failed_report)

            db_audit_fail = AuditLog(
                evidence_id=evidence_id,
                action="REPORT_GENERATION_FAILED",
                timestamp=datetime.utcnow(),
                details=f"Report generation error encountered: {str(e)}"
            )
            db.add(db_audit_fail)
            db.commit()
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(
                status_code=500,
                detail=f"Forensic report generation failed: {str(e)}"
            )

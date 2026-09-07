import os
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
    def generate_report(db: Session, evidence_id: int) -> ForensicReport:
        # 1. Verify evidence exists
        evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
        if not evidence:
            raise HTTPException(status_code=404, detail=f"Evidence record with ID {evidence_id} not found.")

        # Ensure reports directory exists
        settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate unique report UUID and file path
        report_uuid = str(uuid.uuid4())
        pdf_filename = f"report_{evidence_id}_{report_uuid}.pdf"
        pdf_path = settings.REPORTS_DIR / pdf_filename

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
            preprocessing_run = db.query(MediaAnalysis).filter(
                MediaAnalysis.evidence_id == evidence_id,
                MediaAnalysis.analysis_type == "PREPROCESSING_FACE_DETECTION"
            ).order_by(MediaAnalysis.started_at.desc()).first()

            pre_dict = {
                "status": "NOT_RUN",
                "face_detector": "Haar Cascade (Frontal Face Default)",
                "detected_faces": 0,
                "faces": []
            }
            integrity_status = "UNVERIFIED"

            if preprocessing_run:
                pre_dict["status"] = preprocessing_run.status
                if preprocessing_run.integrity_verified:
                    integrity_status = "VERIFIED"
                else:
                    integrity_status = "FAILED"

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
            deepfake_run = db.query(DeepfakeAnalysis).filter(
                DeepfakeAnalysis.evidence_id == evidence_id
            ).order_by(DeepfakeAnalysis.started_at.desc()).first()

            df_dict = {
                "status": "NOT_RUN",
                "model_name": settings.DEEPFAKE_MODEL_NAME,
                "model_version": settings.DEEPFAKE_MODEL_VERSION,
                "model_status": "NOT_RUN",
                "overall_result": "N/A",
                "overall_confidence": "N/A"
            }

            if deepfake_run:
                df_dict["status"] = deepfake_run.status
                df_dict["model_name"] = deepfake_run.model_name
                df_dict["model_version"] = deepfake_run.model_version
                df_dict["overall_result"] = deepfake_run.overall_label
                df_dict["overall_confidence"] = f"{deepfake_run.overall_confidence * 100:.2f}%" if deepfake_run.overall_confidence is not None else "N/A"
                # Check model configuration status
                if deepfake_run.model_version == "NOT_CONFIGURED" or deepfake_run.overall_label == "MODEL_NOT_CONFIGURED":
                    df_dict["model_status"] = "MODEL_NOT_CONFIGURED"
                else:
                    df_dict["model_status"] = "CONFIGURED"

            # 4. Gather ELA data
            ela_run = db.query(ElaAnalysis).filter(
                ElaAnalysis.evidence_id == evidence_id
            ).order_by(ElaAnalysis.started_at.desc()).first()

            ela_dict = {
                "status": "NOT_RUN",
                "jpeg_quality": 95,
                "mean_error": "N/A",
                "max_error": "N/A",
                "error_variance": "N/A",
                "visualization_path": None
            }

            if ela_run:
                ela_dict["status"] = ela_run.status
                ela_dict["jpeg_quality"] = ela_run.jpeg_quality
                ela_dict["mean_error"] = ela_run.mean_error if ela_run.mean_error is not None else "N/A"
                ela_dict["max_error"] = ela_run.max_error if ela_run.max_error is not None else "N/A"
                ela_dict["error_variance"] = ela_run.error_variance if ela_run.error_variance is not None else "N/A"
                ela_dict["visualization_path"] = ela_run.visualization_path

            # 5. Gather FFT data
            fft_run = db.query(FftAnalysis).filter(
                FftAnalysis.evidence_id == evidence_id
            ).order_by(FftAnalysis.started_at.desc()).first()

            fft_dict = {
                "status": "NOT_RUN",
                "high_freq_radius_percent": 15.0,
                "average_magnitude": "N/A",
                "high_frequency_ratio": "N/A",
                "spectral_entropy": "N/A",
                "visualization_path": None
            }

            if fft_run:
                fft_dict["status"] = fft_run.status
                fft_dict["high_freq_radius_percent"] = fft_run.high_freq_radius_percent
                fft_dict["average_magnitude"] = fft_run.average_magnitude if fft_run.average_magnitude is not None else "N/A"
                fft_dict["high_frequency_ratio"] = fft_run.high_frequency_ratio if fft_run.high_frequency_ratio is not None else "N/A"
                fft_dict["spectral_entropy"] = fft_run.spectral_entropy if fft_run.spectral_entropy is not None else "N/A"
                fft_dict["visualization_path"] = fft_run.visualization_path

            # 6. Gather Audit Logs data
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

            # First-pass PDF generation (ensures PDF can be successfully created before committing completion event)
            ReportPdfService.generate_pdf(report_data, str(pdf_path))

            # Commit the REPORT_GENERATION_COMPLETED audit event to the database
            completion_time = datetime.utcnow()
            completion_details = f"Forensic PDF report successfully generated. UUID: {report_uuid}"
            db_audit_done = AuditLog(
                evidence_id=evidence_id,
                action="REPORT_GENERATION_COMPLETED",
                timestamp=completion_time,
                details=completion_details
            )
            db.add(db_audit_done)
            db.commit()

            # Retrieve the updated list of audit logs (which now includes the real REPORT_GENERATION_COMPLETED event)
            updated_audit_logs = db.query(AuditLog).filter(
                AuditLog.evidence_id == evidence_id
            ).order_by(AuditLog.timestamp.asc()).all()

            updated_audit_list = []
            for log in updated_audit_logs:
                updated_audit_list.append({
                    "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "action": log.action,
                    "details": log.details or ""
                })

            # Update the report data with the actual custody log records from database
            report_data["audit_logs"] = updated_audit_list

            # Second-pass PDF generation (includes the logged completion event)
            ReportPdfService.generate_pdf(report_data, str(pdf_path))

            # Calculate SHA-256 hash of the final PDF file
            pdf_hash = HashingService.calculate_sha256(pdf_path)

            # Update ForensicReport record status and hash
            db_report.status = "COMPLETED"
            db_report.file_hash = pdf_hash
            db.add(db_report)
            db.commit()
            db.refresh(db_report)
            return db_report

        except Exception as e:
            db.rollback()
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

            raise HTTPException(
                status_code=500,
                detail=f"Forensic report generation failed: {str(e)}"
            )

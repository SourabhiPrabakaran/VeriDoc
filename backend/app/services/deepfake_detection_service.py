import os
import math
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.evidence import Evidence, AuditLog
from app.models.analysis import MediaAnalysis, FaceDetection, DeepfakeAnalysis, DeepfakePrediction
from app.services.hashing_service import HashingService
from app.services.deepfake_model_interface import DeepfakeModelInterface

class DeepfakeDetectionService:
    @staticmethod
    def run_deepfake_analysis(db: Session, evidence_id: int) -> DeepfakeAnalysis:
        # 1. Confirm that the evidence exists
        evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
        if not evidence:
            raise HTTPException(status_code=404, detail="Evidence item not found.")

        # 2. Confirm stored evidence file exists on disk
        file_path = Path(evidence.storage_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Stored evidence file not found on storage vault.")

        # 3. Recalculate SHA-256 hash for integrity checking
        try:
            current_hash = HashingService.calculate_sha256(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to calculate file hash: {str(e)}")

        # 4. Abort with HTTP 409 Conflict if integrity fails
        if current_hash != evidence.sha256_hash:
            db_audit = AuditLog(
                evidence_id=evidence_id,
                action="INTEGRITY_VERIFICATION_FAILED",
                timestamp=datetime.utcnow(),
                details=f"SHA-256 mismatch before deepfake run. Expected: {evidence.sha256_hash}, Found: {current_hash}"
            )
            db.add(db_audit)
            db.commit()
            raise HTTPException(
                status_code=409,
                detail="SHA-256 cryptographic integrity mismatch. Analysis aborted."
            )

        # 5. Verify that Phase 3 preprocessing has completed successfully
        preprocessing_run = db.query(MediaAnalysis).filter(
            MediaAnalysis.evidence_id == evidence_id,
            MediaAnalysis.analysis_type == "PREPROCESSING_FACE_DETECTION"
        ).order_by(MediaAnalysis.started_at.desc()).first()

        if not preprocessing_run or preprocessing_run.status != "COMPLETED":
            db_audit = AuditLog(
                evidence_id=evidence_id,
                action="DEEPFAKE_ANALYSIS_FAILED",
                timestamp=datetime.utcnow(),
                details="Attempted deepfake analysis before completing media preprocessing."
            )
            db.add(db_audit)
            db.commit()
            raise HTTPException(
                status_code=400,
                detail="Preprocessing analysis is not completed. Run preprocessing first."
            )

        # 6. Initialize DeepfakeAnalysis record with status PROCESSING
        db_analysis = DeepfakeAnalysis(
            evidence_id=evidence_id,
            preprocessing_analysis_id=preprocessing_run.id,
            status="PROCESSING",
            model_name=settings.DEEPFAKE_MODEL_NAME,
            model_version=settings.DEEPFAKE_MODEL_VERSION,
            started_at=datetime.utcnow(),
            overall_label="MODEL_NOT_CONFIGURED",
            overall_confidence=None
        )
        db.add(db_analysis)

        # Log started audit event
        db_audit_start = AuditLog(
            evidence_id=evidence_id,
            action="DEEPFAKE_ANALYSIS_STARTED",
            timestamp=datetime.utcnow(),
            details=f"Deepfake analysis initialized using model {settings.DEEPFAKE_MODEL_NAME} ({settings.DEEPFAKE_MODEL_VERSION})."
        )
        db.add(db_audit_start)
        db.commit()
        db.refresh(db_analysis)

        try:
            # 7. Retrieve detected face crops
            faces = db.query(FaceDetection).filter(FaceDetection.analysis_id == preprocessing_run.id).all()
            
            # 8. Aggregation rules & prediction persistence
            if len(faces) == 0:
                overall_label = "NO_FACE_DETECTED"
                overall_confidence = None
            else:
                # Check if model path is configured
                if settings.DEEPFAKE_MODEL_PATH is None or settings.DEEPFAKE_MODEL_VERSION == "NOT_CONFIGURED":
                    overall_label = "MODEL_NOT_CONFIGURED"
                    overall_confidence = None

                    # Log MODEL_NOT_CONFIGURED audit event
                    db_audit_model = AuditLog(
                        evidence_id=evidence_id,
                        action="MODEL_NOT_CONFIGURED",
                        timestamp=datetime.utcnow(),
                        details=f"Inference bypassed: model {settings.DEEPFAKE_MODEL_NAME} weights are not configured."
                    )
                    db.add(db_audit_model)

                    # Persist unconfigured predictions for every face crop
                    for face in faces:
                        pred_dict = {"label": "MODEL_NOT_CONFIGURED", "confidence": None,
                                     "model_status": "MODEL_NOT_CONFIGURED"}
                        db_pred = DeepfakePrediction(
                            analysis_id=db_analysis.id,
                            face_detection_id=face.id,
                            label=pred_dict["label"],
                            confidence=pred_dict["confidence"],
                            model_status=pred_dict["model_status"],
                            created_at=datetime.utcnow()
                        )
                        db.add(db_pred)
                else:
                    # Model is configured. Attempt loading and prediction.
                    model = DeepfakeModelInterface.load_model()
                    
                    if model is None:
                        status_label = DeepfakeModelInterface._model_status
                        overall_label = status_label
                        overall_confidence = None
                        
                        # Log audit event for failure or unconfigured state
                        if status_label == "MODEL_ERROR":
                            db_audit_model = AuditLog(
                                evidence_id=evidence_id,
                                action="MODEL_ERROR",
                                timestamp=datetime.utcnow(),
                                details=f"Inference bypassed: model {settings.DEEPFAKE_MODEL_NAME} failed to load."
                            )
                        else:
                            db_audit_model = AuditLog(
                                evidence_id=evidence_id,
                                action="MODEL_NOT_CONFIGURED",
                                timestamp=datetime.utcnow(),
                                details=f"Inference bypassed: model {settings.DEEPFAKE_MODEL_NAME} weights are not configured."
                            )
                        db.add(db_audit_model)
                        
                        # Persist fallback predictions
                        for face in faces:
                            db_pred = DeepfakePrediction(
                                analysis_id=db_analysis.id,
                                face_detection_id=face.id,
                                label=status_label,
                                confidence=None,
                                model_status=status_label,
                                created_at=datetime.utcnow()
                            )
                            db.add(db_pred)
                    else:
                        # Log model loading success
                        db_audit_model = AuditLog(
                            evidence_id=evidence_id,
                            action="MODEL_CONFIGURED",
                            timestamp=datetime.utcnow(),
                            details=f"Deepfake model weights verified. Running inference using model {settings.DEEPFAKE_MODEL_NAME}."
                        )
                        db.add(db_audit_model)
                        
                        preds = []
                        valid_predictions = []
                        failed_predictions = 0
                        
                        for face in faces:
                            # Construct physical absolute crop path
                            crop_filename = face.crop_path
                            crop_physical_path = settings.FACE_CROPS_DIR / crop_filename
                            
                            pred_dict = DeepfakeModelInterface.predict(str(crop_physical_path))
                            confidence = pred_dict.get("confidence")
                            valid = (
                                pred_dict.get("model_status") == "CONFIGURED"
                                and pred_dict.get("label") in ("REAL", "FAKE")
                                and isinstance(confidence, (int, float))
                                and not isinstance(confidence, bool)
                                and math.isfinite(confidence)
                                and 0.0 <= confidence <= 1.0
                            )
                            if not valid:
                                failed_predictions += 1
                                pred_dict = {"label": "MODEL_ERROR", "confidence": None,
                                             "model_status": "MODEL_ERROR"}
                            else:
                                valid_predictions.append(pred_dict)
                            
                            db_pred = DeepfakePrediction(
                                analysis_id=db_analysis.id,
                                face_detection_id=face.id,
                                label=pred_dict["label"],
                                confidence=pred_dict["confidence"],
                                model_status=pred_dict["model_status"],
                                created_at=datetime.utcnow()
                            )
                            db.add(db_pred)
                            
                            if pred_dict["confidence"] is not None:
                                preds.append(pred_dict["confidence"])
                                
                        if failed_predictions:
                            overall_label = "MODEL_ERROR"
                            overall_confidence = None
                            db_analysis.error_message = (
                                f"{failed_predictions} of {len(faces)} face predictions failed. "
                                "No overall classification was produced."
                            )
                        else:
                            fake_count = sum(1 for p in valid_predictions if p.get("label") == "FAKE")
                            total_valid = len(valid_predictions)
                            fake_ratio = fake_count / max(1, total_valid)

                            # Distinguish single IMAGE vs multi-frame VIDEO aggregation
                            if evidence.media_category == "VIDEO" and total_valid > 1:
                                agg_method = getattr(settings, "DEEPFAKE_VIDEO_AGGREGATION_METHOD", "fraction_threshold")
                                ratio_threshold = getattr(settings, "DEEPFAKE_VIDEO_FAKE_RATIO_THRESHOLD", 0.25)

                                if agg_method == "fraction_threshold":
                                    overall_label = "FAKE" if fake_ratio >= ratio_threshold else "REAL"
                                elif agg_method == "mean_probability":
                                    mean_p = sum(p.get("raw_probability", 0.5) for p in valid_predictions) / total_valid
                                    overall_label = "REAL" if mean_p >= 0.5 else "FAKE"
                                else:  # 'any_fake'
                                    overall_label = "FAKE" if fake_count > 0 else "REAL"

                                db_analysis.error_message = (
                                    f"Video aggregation ({agg_method}): {fake_count}/{total_valid} frames "
                                    f"({fake_ratio * 100:.1f}%) classified as FAKE "
                                    f"(threshold: {ratio_threshold * 100:.1f}%)."
                                )
                            else:
                                # Standard image decision rule
                                overall_label = "FAKE" if fake_count > 0 else "REAL"

                            overall_confidence = sum(preds) / len(preds) if preds else None

            # 9. Update analysis status to COMPLETED and commit
            db_analysis.status = "FAILED" if overall_label == "MODEL_ERROR" else "COMPLETED"
            db_analysis.overall_label = overall_label
            db_analysis.overall_confidence = overall_confidence
            db_analysis.completed_at = datetime.utcnow()

            # Log completion audit event
            db_audit_done = AuditLog(
                evidence_id=evidence_id,
                action="DEEPFAKE_ANALYSIS_FAILED" if db_analysis.status == "FAILED" else "DEEPFAKE_ANALYSIS_COMPLETED",
                timestamp=datetime.utcnow(),
                details=f"Deepfake analysis pipeline completed. Overall result: {overall_label}."
            )
            db.add(db_audit_done)
            db.commit()
            
            db.refresh(db_analysis)
            return db_analysis

        except Exception as e:
            db.rollback()
            
            # Recover analysis record and write FAILED state
            failed_run = db.query(DeepfakeAnalysis).filter(DeepfakeAnalysis.id == db_analysis.id).first()
            if failed_run:
                failed_run.status = "FAILED"
                failed_run.overall_label = "MODEL_ERROR"
                failed_run.overall_confidence = None
                failed_run.completed_at = datetime.utcnow()
                failed_run.error_message = str(e)[:500]
                db.add(failed_run)

            # Log failure audit event
            db_audit_fail = AuditLog(
                evidence_id=evidence_id,
                action="DEEPFAKE_ANALYSIS_FAILED",
                timestamp=datetime.utcnow(),
                details=f"Deepfake analysis failed: {str(e)}"
            )
            db.add(db_audit_fail)
            db.commit()
            
            raise HTTPException(status_code=500, detail=f"Deepfake analysis pipeline execution failed: {str(e)}")

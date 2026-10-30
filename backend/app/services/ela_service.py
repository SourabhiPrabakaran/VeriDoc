import os
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.evidence import Evidence, AuditLog
from app.models.analysis import ElaAnalysis
from app.services.hashing_service import HashingService

class ElaService:
    @staticmethod
    def run_ela(db: Session, evidence_id: int) -> ElaAnalysis:
        # 1. Confirm that the evidence exists
        evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
        if not evidence:
            raise HTTPException(status_code=404, detail="Evidence item not found.")

        # 2. ELA is currently IMAGE-ONLY
        if evidence.media_category != "IMAGE":
            raise HTTPException(
                status_code=400,
                detail="Error Level Analysis (ELA) is only supported for image evidence."
            )

        # 3. Confirm stored file exists on disk
        file_path = Path(evidence.storage_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Stored evidence file not found on storage vault.")

        # 4. SHA-256 integrity verification immediately before analysis
        try:
            current_hash = HashingService.calculate_sha256(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to calculate file hash: {str(e)}")

        if current_hash != evidence.sha256_hash:
            db_audit = AuditLog(
                evidence_id=evidence_id,
                action="INTEGRITY_VERIFICATION_FAILED",
                timestamp=datetime.utcnow(),
                details=f"SHA-256 mismatch before ELA run. Expected: {evidence.sha256_hash}, Found: {current_hash}"
            )
            db.add(db_audit)
            db.commit()
            raise HTTPException(
                status_code=409,
                detail="SHA-256 cryptographic integrity mismatch. ELA aborted."
            )

        # 5. Initialize ElaAnalysis database entry
        db_analysis = ElaAnalysis(
            evidence_id=evidence_id,
            status="PROCESSING",
            started_at=datetime.utcnow(),
            jpeg_quality=95
        )
        db.add(db_analysis)

        # Log started audit event
        db_audit_start = AuditLog(
            evidence_id=evidence_id,
            action="ELA_ANALYSIS_STARTED",
            timestamp=datetime.utcnow(),
            details="Error Level Analysis (ELA) initialized at JPEG quality 95."
        )
        db.add(db_audit_start)
        db.commit()
        db.refresh(db_analysis)

        temp_path = None
        visualization_path = None
        try:
            import cv2
            import numpy as np

            settings.ELA_DIR.mkdir(parents=True, exist_ok=True)
            settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)

            # Load the original image
            orig = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if orig is None:
                raise ValueError("OpenCV failed to decode original image file.")

            if orig.dtype != np.uint8:
                raise ValueError("ELA currently supports only 8-bit image samples.")
            # Compare in one explicit color space for grayscale, BGR and BGRA.
            if len(orig.shape) == 2:
                orig_bgr = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
            elif len(orig.shape) == 3 and orig.shape[2] == 4:
                # Convert to BGR to drop alpha channel safely for JPEG compression
                orig_bgr = cv2.cvtColor(orig, cv2.COLOR_BGRA2BGR)
            else:
                orig_bgr = orig

            # Create temporary JPEG file path
            temp_filename = f"temp_ela_{db_analysis.id}_{uuid.uuid4().hex}.jpg"
            temp_path = settings.TEMP_DIR / temp_filename

            # Save image at defined JPEG quality
            if not cv2.imwrite(str(temp_path), orig_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError("Failed to write temporary ELA JPEG.")

            # Read back compressed image
            compressed = cv2.imread(str(temp_path))
            if compressed is None:
                if temp_path.exists():
                    temp_path.unlink()
                raise ValueError("OpenCV failed to decode compressed temporary JPEG file.")

            # Ensure spatial shapes match (resizing if OpenCV added padding, though standard imwrite maintains shape)
            if orig_bgr.shape != compressed.shape:
                compressed = cv2.resize(compressed, (orig_bgr.shape[1], orig_bgr.shape[0]))

            # Compute absolute error level difference
            diff = cv2.absdiff(orig_bgr, compressed)

            # Compute raw difference measurements
            mean_error = float(np.mean(diff))
            max_error = float(np.max(diff))
            error_variance = float(np.var(diff))

            # Scale error difference for visualization (scale factor of 15)
            # Clip values to range [0, 255] and cast to uint8
            scaled_diff = cv2.multiply(diff, 15)
            scaled_diff = np.clip(scaled_diff, 0, 255).astype(np.uint8)

            # Save the ELA visualization artifact
            visualization_filename = f"ela_{db_analysis.id}_{uuid.uuid4().hex}.jpg"
            visualization_path = settings.ELA_DIR / visualization_filename
            if not cv2.imwrite(str(visualization_path), scaled_diff):
                raise ValueError("Failed to write ELA visualization.")

            # Clean up temporary JPEG resave file
            if temp_path.exists():
                temp_path.unlink()

            # Update database record
            db_analysis.status = "COMPLETED"
            db_analysis.completed_at = datetime.utcnow()
            db_analysis.visualization_path = visualization_filename
            db_analysis.mean_error = round(mean_error, 4)
            db_analysis.max_error = round(max_error, 4)
            db_analysis.error_variance = round(error_variance, 4)

            # Log completion audit event
            db_audit_done = AuditLog(
                evidence_id=evidence_id,
                action="ELA_ANALYSIS_COMPLETED",
                timestamp=datetime.utcnow(),
                details=f"ELA completed. Mean error: {db_analysis.mean_error}, Var: {db_analysis.error_variance}."
            )
            db.add(db_audit_done)
            db.commit()
            db.refresh(db_analysis)

            return db_analysis

        except Exception as e:
            db.rollback()
            if visualization_path is not None:
                visualization_path.unlink(missing_ok=True)

            # Recover analysis record and update status to FAILED
            failed_run = db.query(ElaAnalysis).filter(ElaAnalysis.id == db_analysis.id).first()
            if failed_run:
                failed_run.status = "FAILED"
                failed_run.completed_at = datetime.utcnow()
                failed_run.error_message = str(e)[:500]
                db.add(failed_run)

            # Log failure audit log
            db_audit_fail = AuditLog(
                evidence_id=evidence_id,
                action="ELA_ANALYSIS_FAILED",
                timestamp=datetime.utcnow(),
                details=f"Error Level Analysis failed: {str(e)}"
            )
            db.add(db_audit_fail)
            db.commit()

            raise HTTPException(status_code=500, detail=f"ELA analysis pipeline execution failed: {str(e)}")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

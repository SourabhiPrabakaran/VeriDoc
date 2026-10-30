import os
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.evidence import Evidence, AuditLog
from app.models.analysis import FftAnalysis
from app.services.hashing_service import HashingService

class FftService:
    @staticmethod
    def run_fft(db: Session, evidence_id: int) -> FftAnalysis:
        # 1. Confirm that the evidence exists
        evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
        if not evidence:
            raise HTTPException(status_code=404, detail="Evidence item not found.")

        # 2. FFT is currently IMAGE-ONLY
        if evidence.media_category != "IMAGE":
            raise HTTPException(
                status_code=400,
                detail="Fast Fourier Transform (FFT) analysis is only supported for image evidence."
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
                details=f"SHA-256 mismatch before FFT run. Expected: {evidence.sha256_hash}, Found: {current_hash}"
            )
            db.add(db_audit)
            db.commit()
            raise HTTPException(
                status_code=409,
                detail="SHA-256 cryptographic integrity mismatch. FFT aborted."
            )

        # 5. Initialize FftAnalysis database entry
        db_analysis = FftAnalysis(
            evidence_id=evidence_id,
            status="PROCESSING",
            started_at=datetime.utcnow(),
            high_freq_radius_percent=15.0
        )
        db.add(db_analysis)

        # Log started audit event
        db_audit_start = AuditLog(
            evidence_id=evidence_id,
            action="FFT_ANALYSIS_STARTED",
            timestamp=datetime.utcnow(),
            details="2D Fast Fourier Transform (FFT) initialized. High-frequency threshold set to 15.0% radius."
        )
        db.add(db_audit_start)
        db.commit()
        db.refresh(db_analysis)

        try:
            import cv2

            settings.FFT_DIR.mkdir(parents=True, exist_ok=True)

            # Load the original image in grayscale as FFT represents coordinate frequency components
            img_gray = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise ValueError("OpenCV failed to decode original image file in grayscale.")

            # Compute centered 2D Fast Fourier Transform
            fft = np.fft.fft2(img_gray)
            fft_shift = np.fft.fftshift(fft)

            # Compute raw magnitude (absolute value of complex Fourier output)
            magnitude = np.abs(fft_shift)

            # Calculate raw average magnitude
            average_magnitude = float(np.mean(magnitude))

            # Compute logarithmic magnitude spectrum for visualization: 20 * log(abs(shifted) + 1)
            magnitude_spectrum = 20 * np.log(magnitude + 1.0)

            # Normalize magnitude spectrum to range [0, 255] for standard display
            mag_min = np.min(magnitude_spectrum)
            mag_max = np.max(magnitude_spectrum)
            if mag_max > mag_min:
                normalized = ((magnitude_spectrum - mag_min) / (mag_max - mag_min) * 255.0).astype(np.uint8)
            else:
                normalized = np.zeros_like(img_gray, dtype=np.uint8)

            # Define high-frequency mask: outer region beyond 15% radius
            h, w = img_gray.shape[:2]
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # 15% of the image diagonal
            diagonal = np.sqrt(h**2 + w**2)
            threshold_r = 0.15 * diagonal

            high_freq_mask = dist_from_center > threshold_r

            # Compute high-frequency energy ratio
            high_freq_energy = np.sum(magnitude[high_freq_mask])
            total_energy = np.sum(magnitude)
            high_frequency_ratio = float(high_freq_energy / total_energy) if total_energy > 0 else 0.0

            # Compute Spectral Entropy: normalized power spectral density entropy
            power = magnitude ** 2
            total_power = np.sum(power)
            if total_power > 0:
                psd = power / total_power
                spectral_entropy = float(-np.sum(psd * np.log2(psd + 1e-12)))
            else:
                spectral_entropy = 0.0

            # Save the FFT magnitude spectrum visualization artifact
            visualization_filename = f"fft_{db_analysis.id}_{uuid.uuid4().hex}.png"
            visualization_path = settings.FFT_DIR / visualization_filename
            cv2.imwrite(str(visualization_path), normalized)

            # Update database record
            db_analysis.status = "COMPLETED"
            db_analysis.completed_at = datetime.utcnow()
            db_analysis.visualization_path = visualization_filename
            db_analysis.average_magnitude = round(average_magnitude, 4)
            db_analysis.high_frequency_ratio = round(high_frequency_ratio, 4)
            db_analysis.spectral_entropy = round(spectral_entropy, 4)

            # Log completion audit event
            db_audit_done = AuditLog(
                evidence_id=evidence_id,
                action="FFT_ANALYSIS_COMPLETED",
                timestamp=datetime.utcnow(),
                details=f"FFT completed. Average magnitude: {db_analysis.average_magnitude}, High-freq ratio: {db_analysis.high_frequency_ratio}."
            )
            db.add(db_audit_done)
            db.commit()
            db.refresh(db_analysis)

            return db_analysis

        except Exception as e:
            db.rollback()

            # Recover analysis record and update status to FAILED
            failed_run = db.query(FftAnalysis).filter(FftAnalysis.id == db_analysis.id).first()
            if failed_run:
                failed_run.status = "FAILED"
                failed_run.completed_at = datetime.utcnow()
                failed_run.error_message = str(e)[:500]
                db.add(failed_run)

            # Log failure audit log
            db_audit_fail = AuditLog(
                evidence_id=evidence_id,
                action="FFT_ANALYSIS_FAILED",
                timestamp=datetime.utcnow(),
                details=f"Fast Fourier Transform failed: {str(e)}"
            )
            db.add(db_audit_fail)
            db.commit()

            raise HTTPException(status_code=500, detail=f"FFT analysis pipeline execution failed: {str(e)}")

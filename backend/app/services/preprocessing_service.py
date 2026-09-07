import os
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.evidence import Evidence, AuditLog
from app.models.analysis import MediaAnalysis, FaceDetection
from app.services.hashing_service import HashingService
from app.services.face_detection_service import FaceDetectionService

class PreprocessingService:
    @staticmethod
    def run_preprocessing(db: Session, evidence_id: int) -> MediaAnalysis:
        # 1. Verify evidence exists
        evidence = db.query(Evidence).filter(Evidence.id == evidence_id).first()
        if not evidence:
            raise HTTPException(status_code=404, detail=f"Evidence record with ID {evidence_id} not found.")

        # 2. Verify stored file exists
        file_path = Path(evidence.storage_path)
        if not file_path.exists():
            # Log failure audit log
            db_audit = AuditLog(
                evidence_id=evidence_id,
                action="MEDIA_PREPROCESSING_FAILED",
                timestamp=datetime.utcnow(),
                details=f"Stored evidence file missing at: {evidence.storage_path}"
            )
            db.add(db_audit)
            db.commit()
            raise HTTPException(status_code=404, detail="Stored evidence file not found on disk.")

        # 3. Recalculate SHA-256 for integrity verification
        try:
            current_hash = HashingService.calculate_sha256(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to calculate SHA-256 hash: {str(e)}")

        # 4. Compare hashes
        if current_hash != evidence.sha256_hash:
            # Audit log integrity failure
            db_audit = AuditLog(
                evidence_id=evidence_id,
                action="INTEGRITY_VERIFICATION_FAILED",
                timestamp=datetime.utcnow(),
                details=f"Cryptographic hash mismatch. Expected: {evidence.sha256_hash}, Found: {current_hash}"
            )
            db.add(db_audit)
            
            # Create failed analysis record
            db_analysis = MediaAnalysis(
                evidence_id=evidence_id,
                analysis_type="PREPROCESSING_FACE_DETECTION",
                status="FAILED",
                integrity_verified=False,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                error_message="SHA-256 cryptographic integrity verification failed. Media altered or corrupted."
            )
            db.add(db_analysis)
            db.commit()
            
            raise HTTPException(
                status_code=409,
                detail="SHA-256 cryptographic integrity mismatch. Processing aborted."
            )

        # 5. Initialize processing analysis record
        db_analysis = MediaAnalysis(
            evidence_id=evidence_id,
            analysis_type="PREPROCESSING_FACE_DETECTION",
            status="PROCESSING",
            integrity_verified=True,
            started_at=datetime.utcnow()
        )
        db.add(db_analysis)
        
        # Log starting audit log
        db_audit = AuditLog(
            evidence_id=evidence_id,
            action="PREPROCESSING_STARTED",
            timestamp=datetime.utcnow(),
            details="SHA-256 verified successfully. Initializing preprocessing pipeline."
        )
        db.add(db_audit)
        db.commit()
        db.refresh(db_analysis)

        # Run pipeline
        try:
            import cv2
            import numpy as np

            detector = FaceDetectionService()
            settings.FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)

            face_count = 0
            video_meta_payload = None

            if evidence.media_category == "IMAGE":
                # IMAGE PREPROCESSING
                img = cv2.imread(str(file_path))
                if img is None:
                    raise ValueError("OpenCV failed to decode the registered image file.")

                height, width = img.shape[:2]
                
                # Convert color space for grayscale Haar Cascade face detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Run face detection
                faces = detector.detect_faces(gray)
                
                # Crop and record face regions
                for i, face_coords in enumerate(faces):
                    x, y, w, h = face_coords["x"], face_coords["y"], face_coords["width"], face_coords["height"]
                    face_crop = img[y:y+h, x:x+w]
                    
                    if face_crop.size == 0:
                        continue
                    
                    # Generate safe unique filename
                    crop_filename = f"crop_{db_analysis.id}_face_{i}_{uuid.uuid4().hex}.jpg"
                    crop_path = settings.FACE_CROPS_DIR / crop_filename
                    
                    cv2.imwrite(str(crop_path), face_crop)
                    
                    db_face = FaceDetection(
                        analysis_id=db_analysis.id,
                        frame_index=None,
                        timestamp_seconds=None,
                        face_index=i,
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        crop_path=crop_filename # Store relative filename reference
                    )
                    db.add(db_face)
                    face_count += 1

            elif evidence.media_category == "VIDEO":
                # VIDEO PREPROCESSING (Frame Sampling + Detection)
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    raise ValueError("OpenCV failed to open the registered video file stream.")

                # Read parameters
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps <= 0 or frame_count <= 0:
                    cap.release()
                    raise ValueError("Invalid FPS or frame count read from video stream metadata.")

                duration = frame_count / fps
                
                # Frame sampling intervals calculation
                sample_interval_frames = int(round(fps * settings.VIDEO_SAMPLE_INTERVAL_SECONDS))
                if sample_interval_frames <= 0:
                    sample_interval_frames = 1

                sampled_frames_count = 0
                current_frame_idx = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Check if current frame index is inside sampling slot
                    if current_frame_idx % sample_interval_frames == 0:
                        sampled_frames_count += 1
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = detector.detect_faces(gray)
                        
                        timestamp = float(current_frame_idx) / fps
                        
                        for i, face_coords in enumerate(faces):
                            x, y, w, h = face_coords["x"], face_coords["y"], face_coords["width"], face_coords["height"]
                            face_crop = frame[y:y+h, x:x+w]
                            
                            if face_crop.size == 0:
                                continue
                            
                            crop_filename = f"crop_{db_analysis.id}_frame_{current_frame_idx}_face_{i}_{uuid.uuid4().hex}.jpg"
                            crop_path = settings.FACE_CROPS_DIR / crop_filename
                            
                            cv2.imwrite(str(crop_path), face_crop)
                            
                            db_face = FaceDetection(
                                analysis_id=db_analysis.id,
                                frame_index=current_frame_idx,
                                timestamp_seconds=round(timestamp, 2),
                                face_index=i,
                                x=x,
                                y=y,
                                width=w,
                                height=h,
                                crop_path=crop_filename
                            )
                            db.add(db_face)
                            face_count += 1

                    current_frame_idx += 1
                
                cap.release()

                video_meta_payload = {
                    "width": width,
                    "height": height,
                    "fps": round(fps, 2),
                    "frame_count": frame_count,
                    "duration_seconds": round(duration, 2),
                    "sampled_frames": sampled_frames_count
                }

            # 9. Commit transaction and log audit trail
            db_analysis.status = "COMPLETED"
            db_analysis.completed_at = datetime.utcnow()
            
            # Repurpose error_message to persist JSON serialized video metadata if it exists
            if video_meta_payload:
                db_analysis.error_message = json.dumps(video_meta_payload)
            else:
                db_analysis.error_message = None

            # Add completion audit events
            db_audit_faces = AuditLog(
                evidence_id=evidence_id,
                action="FACE_DETECTION_COMPLETED",
                timestamp=datetime.utcnow(),
                details=f"Face detection search completed. Found {face_count} face crops."
            )
            db.add(db_audit_faces)

            db_audit_done = AuditLog(
                evidence_id=evidence_id,
                action="PREPROCESSING_COMPLETED",
                timestamp=datetime.utcnow(),
                details="Media stream preprocessing completed successfully."
            )
            db.add(db_audit_done)
            db.commit()
            
            db.refresh(db_analysis)
            return db_analysis

        except Exception as e:
            # Handle failures, rollback DB, update analysis row to FAILED
            db.rollback()
            
            # Re-fetch/re-open session specifically to write FAILED state
            failed_analysis = db.query(MediaAnalysis).filter(MediaAnalysis.id == db_analysis.id).first()
            if failed_analysis:
                failed_analysis.status = "FAILED"
                failed_analysis.completed_at = datetime.utcnow()
                failed_analysis.error_message = str(e)[:500] # Limit size of error log
                db.add(failed_analysis)
                
            db_audit_fail = AuditLog(
                evidence_id=evidence_id,
                action="MEDIA_PREPROCESSING_FAILED",
                timestamp=datetime.utcnow(),
                details=f"Preprocessing error encountered: {str(e)}"
            )
            db.add(db_audit_fail)
            db.commit()
            
            raise HTTPException(
                status_code=500,
                detail=f"Preprocessing failed during execution: {str(e)}"
            )

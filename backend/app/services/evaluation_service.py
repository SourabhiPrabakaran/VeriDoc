import os
import re
import cv2
import hashlib
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session

from app.models.evaluation import EvaluationRun, EvaluationPrediction
from app.core.config import settings

class EvaluationService:
    @staticmethod
    def create_run(db: Session, dataset_name: str, dataset_split: str, checkpoint_sha256: str) -> EvaluationRun:
        run = EvaluationRun(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            model_name=settings.DEEPFAKE_MODEL_NAME,
            model_version=settings.DEEPFAKE_MODEL_VERSION,
            checkpoint_sha256=checkpoint_sha256,
            started_at=datetime.utcnow(),
            status="PROCESSING"
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run

    @staticmethod
    def add_prediction(
        db: Session,
        run_id: int,
        dataset_name: str,
        source_video_id: str,
        frame_number: int,
        ground_truth: str,
        predicted_label: str,
        probability: float,
        confidence: float,
        inference_time_ms: float,
        crop_path: str
    ) -> EvaluationPrediction:
        pred = EvaluationPrediction(
            evaluation_run_id=run_id,
            dataset_name=dataset_name,
            source_video_id=source_video_id,
            frame_number=frame_number,
            ground_truth=ground_truth,
            predicted_label=predicted_label,
            probability=probability,
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            crop_path=crop_path
        )
        db.add(pred)
        db.flush() # flush to get auto-incremented ID, but commit later for efficiency
        return pred

    @staticmethod
    def complete_run(
        db: Session,
        run_id: int,
        total_samples: int,
        real_samples: int,
        fake_samples: int,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        specificity: float,
        roc_auc: float = None,
        status: str = "COMPLETED"
    ) -> EvaluationRun:
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if not run:
            raise ValueError(f"EvaluationRun with ID {run_id} not found.")

        run.completed_at = datetime.utcnow()
        run.total_samples = total_samples
        run.real_samples = real_samples
        run.fake_samples = fake_samples
        run.accuracy = accuracy
        run.precision = precision
        run.recall = recall
        run.f1_score = f1_score
        run.specificity = specificity
        run.roc_auc = roc_auc
        run.status = status
        
        db.commit()
        db.refresh(run)
        return run

    @staticmethod
    def parse_filename_provenance(filename: str) -> tuple[str | None, int | None]:
        """
        Attempts to extract source_video_id and frame_number from standard benchmark naming formats.
        Returns:
            (source_video_id, frame_number) or (None, None) if format is unrecognized.
        """
        # Formats like: 123_456.jpg (where 123 is video_id, 456 is frame)
        m1 = re.match(r"^(\d+)_(\d+)\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
        if m1:
            return m1.group(1), int(m1.group(2))

        # Formats like: video_123_frame_456.png
        m2 = re.match(r"^video_(\w+)_frame_(\d+)\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
        if m2:
            return m2.group(1), int(m2.group(2))

        # Crop filenames like: crop_11_face_0_a444d4.jpg
        # Let's map "11" as the source ID, face_index not frame
        m3 = re.match(r"^crop_(\d+)_face_(\d+)_\w+\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
        if m3:
            return m3.group(1), None

        # Return None, None to prevent fabricating metadata
        return None, None

    @classmethod
    def scan_and_validate_dataset(cls, dataset_dir: Path) -> dict:
        """
        Scans directories, validates formats, and reads images to identify corruptions.
        Returns a dict containing valid files with their parsed provenance.
        """
        real_dir = dataset_dir / "real"
        fake_dir = dataset_dir / "fake"
        
        results = {
            "real": [],
            "fake": [],
            "corrupted": [],
            "invalid_format": []
        }
        
        valid_extensions = {".jpg", ".jpeg", ".png"}
        
        for category, directory in [("REAL", real_dir), ("FAKE", fake_dir)]:
            if not directory.exists():
                continue
                
            for root, _, files in os.walk(directory):
                for f in files:
                    file_path = Path(root) / f
                    ext = file_path.suffix.lower()
                    
                    if ext not in valid_extensions:
                        results["invalid_format"].append(str(file_path))
                        continue
                        
                    # Check image readability
                    try:
                        img = cv2.imread(str(file_path))
                        if img is None:
                            raise ValueError("OpenCV returned None")
                            
                        # Valid image, parse provenance
                        video_id, frame_num = cls.parse_filename_provenance(f)
                        
                        entry = {
                            "path": str(file_path),
                            "filename": f,
                            "video_id": video_id,
                            "frame_number": frame_num
                        }
                        
                        if category == "REAL":
                            results["real"].append(entry)
                        else:
                            results["fake"].append(entry)
                            
                    except Exception as e:
                        results["corrupted"].append({
                            "path": str(file_path),
                            "error": str(e)
                        })
                        
        return results

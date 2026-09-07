import os
import sys
import unittest
import tempfile
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend to path
sys.path.append(str(Path(__file__).resolve().parent))

from app.core.config import settings
from app.core.database import Base
from app.models.evaluation import EvaluationRun, EvaluationPrediction
from app.services.deepfake_model_interface import DeepfakeModelInterface
from app.services.evaluation_service import EvaluationService
from scripts.prepare_evaluation_dataset import audit_dataset
from scripts.run_benchmark_evaluation import run_benchmark

# Setup in-memory sqlite engine for test isolation
test_engine = create_engine("sqlite:///:memory:")
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

class TestPhase8Infrastructure(unittest.TestCase):
    def setUp(self):
        # Create temp directory for evaluation data
        self.test_dir = tempfile.mkdtemp()
        self.dataset_dir = Path(self.test_dir) / "evaluation_dataset"
        self.results_dir = Path(self.test_dir) / "evaluation_results"
        
        self.real_dir = self.dataset_dir / "real"
        self.fake_dir = self.dataset_dir / "fake"
        
        self.real_dir.mkdir(parents=True, exist_ok=True)
        self.fake_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup model settings
        self.orig_path = settings.DEEPFAKE_MODEL_PATH
        self.orig_version = settings.DEEPFAKE_MODEL_VERSION
        
        # Setup in-memory DB tables
        Base.metadata.create_all(bind=test_engine)
        self.db = TestingSessionLocal()
        
    def tearDown(self):
        # Close DB
        self.db.close()
        Base.metadata.drop_all(bind=test_engine)
        
        # Remove temp directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
        # Restore settings
        settings.DEEPFAKE_MODEL_PATH = self.orig_path
        settings.DEEPFAKE_MODEL_VERSION = self.orig_version
        
    def write_dummy_image(self, path):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(path), img)
        
    def test_table_creation(self):
        # Verify both table names exist in the Base metadata
        self.assertIn("evaluation_run", Base.metadata.tables)
        self.assertIn("evaluation_prediction", Base.metadata.tables)
        
    def test_evaluation_run_creation(self):
        run = EvaluationService.create_run(
            db=self.db,
            dataset_name="Celeb-DF",
            dataset_split="test",
            checkpoint_sha256="abc123hash"
        )
        self.assertIsNotNone(run.id)
        self.assertEqual(run.dataset_name, "Celeb-DF")
        self.assertEqual(run.status, "PROCESSING")
        
    def test_evaluation_prediction_creation(self):
        run = EvaluationService.create_run(
            db=self.db,
            dataset_name="Celeb-DF",
            dataset_split="test",
            checkpoint_sha256="abc123hash"
        )
        pred = EvaluationService.add_prediction(
            db=self.db,
            run_id=run.id,
            dataset_name="Celeb-DF",
            source_video_id="video_01",
            frame_number=45,
            ground_truth="REAL",
            predicted_label="REAL",
            probability=0.85,
            confidence=0.7,
            inference_time_ms=25.0,
            crop_path="/tmp/crop.jpg"
        )
        self.db.commit()
        
        self.assertIsNotNone(pred.id)
        self.assertEqual(pred.evaluation_run_id, run.id)
        self.assertEqual(pred.source_video_id, "video_01")
        self.assertEqual(pred.frame_number, 45)
        self.assertEqual(pred.ground_truth, "REAL")
        
    def test_foreign_key_relationship(self):
        # Test cascade deletion
        run = EvaluationService.create_run(
            db=self.db,
            dataset_name="Celeb-DF",
            dataset_split="test",
            checkpoint_sha256="abc123hash"
        )
        EvaluationService.add_prediction(
            db=self.db,
            run_id=run.id,
            dataset_name="Celeb-DF",
            source_video_id="video_01",
            frame_number=10,
            ground_truth="REAL",
            predicted_label="REAL",
            probability=0.9,
            confidence=0.8,
            inference_time_ms=12.0,
            crop_path="crop.jpg"
        )
        self.db.commit()
        
        # Verify prediction exists
        preds = self.db.query(EvaluationPrediction).filter(EvaluationPrediction.evaluation_run_id == run.id).all()
        self.assertEqual(len(preds), 1)
        
        # Delete run and verify prediction is cascade-deleted
        self.db.delete(run)
        self.db.commit()
        
        preds_after = self.db.query(EvaluationPrediction).filter(EvaluationPrediction.evaluation_run_id == run.id).all()
        self.assertEqual(len(preds_after), 0)
        
    def test_duplicate_evaluation_run(self):
        # We can create multiple runs with the same checkpoint/dataset names without database restrictions
        run1 = EvaluationService.create_run(self.db, "Celeb-DF", "test", "hash1")
        run2 = EvaluationService.create_run(self.db, "Celeb-DF", "test", "hash1")
        self.db.commit()
        self.assertNotEqual(run1.id, run2.id)
        
    def test_failed_evaluation_run(self):
        # Test that benchmark failure updates status to FAILED
        self.write_dummy_image(self.real_dir / "real.jpg")
        
        # Mock predict to raise an exception
        with patch.object(DeepfakeModelInterface, 'predict', side_effect=Exception("Inference crash")):
            with self.assertRaises(Exception):
                run_benchmark(self.dataset_dir, self.results_dir, db=self.db)
                
            # Verify the run status is updated to FAILED in database
            run = self.db.query(EvaluationRun).first()
            self.assertIsNotNone(run)
            self.assertEqual(run.status, "FAILED")
            
    def test_database_rollback_behavior(self):
        # Test that if run benchmark fails during execution, predictions are rolled back
        self.write_dummy_image(self.real_dir / "real.jpg")
        
        # Setup run_benchmark with an exception mid-way
        with patch.object(DeepfakeModelInterface, 'predict', side_effect=Exception("Rollback crash")):
            with self.assertRaises(Exception):
                run_benchmark(self.dataset_dir, self.results_dir, db=self.db)
                
        # Assert that no predictions are persisted in the database for the failed run
        predictions = self.db.query(EvaluationPrediction).all()
        self.assertEqual(len(predictions), 0)
        
    def test_empty_dataset(self):
        # Run audit
        report = audit_dataset(self.dataset_dir)
        self.assertEqual(report["status"], "EMPTY")
        
        # Run benchmark
        res = run_benchmark(self.dataset_dir, self.results_dir, db=self.db)
        self.assertEqual(res["evaluation_status"], "EMPTY")
        
    def test_only_real_samples(self):
        for i in range(5):
            self.write_dummy_image(self.real_dir / f"real_{i}.jpg")
            
        report = audit_dataset(self.dataset_dir)
        self.assertEqual(report["real_count"], 5)
        self.assertEqual(report["status"], "PRELIMINARY")
        
        with patch.object(DeepfakeModelInterface, 'predict', return_value={
            "label": "REAL",
            "confidence": 0.6,
            "model_status": "CONFIGURED",
            "model_name": "MesoInception-4"
        }):
            res = run_benchmark(self.dataset_dir, self.results_dir, db=self.db)
            self.assertEqual(res["evaluation_status"], "PRELIMINARY")
            metrics = res["metrics"]
            self.assertEqual(metrics["real_count"], 5)
            self.assertEqual(metrics["correct_predictions"], 5)
            
    def test_only_fake_samples(self):
        for i in range(5):
            self.write_dummy_image(self.fake_dir / f"fake_{i}.png")
            
        report = audit_dataset(self.dataset_dir)
        self.assertEqual(report["fake_count"], 5)
        
        with patch.object(DeepfakeModelInterface, 'predict', return_value={
            "label": "FAKE",
            "confidence": 0.8,
            "model_status": "CONFIGURED",
            "model_name": "MesoInception-4"
        }):
            res = run_benchmark(self.dataset_dir, self.results_dir, db=self.db)
            self.assertEqual(res["evaluation_status"], "PRELIMINARY")
            metrics = res["metrics"]
            self.assertEqual(metrics["fake_count"], 5)
            
    def test_balanced_real_fake_dataset(self):
        for i in range(3):
            # Filenames like 001_010.jpg to test provenance parsing
            self.write_dummy_image(self.real_dir / f"{i:03d}_010.jpg")
            self.write_dummy_image(self.fake_dir / f"{i:03d}_020.jpg")
            
        predict_side_effects = [
            {"label": "REAL", "confidence": 0.8, "model_status": "CONFIGURED", "model_name": "MesoInception-4"},
            {"label": "FAKE", "confidence": 0.4, "model_status": "CONFIGURED", "model_name": "MesoInception-4"},
            {"label": "REAL", "confidence": 0.6, "model_status": "CONFIGURED", "model_name": "MesoInception-4"},
            {"label": "FAKE", "confidence": 0.8, "model_status": "CONFIGURED", "model_name": "MesoInception-4"},
            {"label": "FAKE", "confidence": 0.6, "model_status": "CONFIGURED", "model_name": "MesoInception-4"},
            {"label": "REAL", "confidence": 0.2, "model_status": "CONFIGURED", "model_name": "MesoInception-4"}
        ]
        
        with patch.object(DeepfakeModelInterface, 'predict', side_effect=predict_side_effects):
            res = run_benchmark(self.dataset_dir, self.results_dir, db=self.db)
            metrics = res["metrics"]
            
            self.assertAlmostEqual(metrics["accuracy"], 4.0 / 6.0)
            self.assertEqual(metrics["confusion_matrix"]["tp"], 2)
            self.assertEqual(metrics["confusion_matrix"]["tn"], 2)
            
            # Check provenance parsing in database
            predictions = self.db.query(EvaluationPrediction).order_by(EvaluationPrediction.id.asc()).all()
            self.assertEqual(len(predictions), 6)
            
            # Predictions 0 to 2 are REAL files (filename like 000_010.jpg)
            # video_id="000", frame_number=10
            self.assertEqual(predictions[0].source_video_id, "000")
            self.assertEqual(predictions[0].frame_number, 10)
            self.assertEqual(predictions[0].ground_truth, "REAL")
            
    def test_corrupted_image(self):
        corrupted_path = self.real_dir / "corrupt.jpg"
        with open(corrupted_path, "w") as f:
            f.write("this is not an image file")
            
        report = audit_dataset(self.dataset_dir)
        self.assertEqual(len(report["corrupted_files"]), 1)
        
    def test_invalid_image_format(self):
        invalid_path = self.real_dir / "test.txt"
        with open(invalid_path, "w") as f:
            f.write("text content")
            
        report = audit_dataset(self.dataset_dir)
        self.assertEqual(len(report["invalid_formats"]), 1)
        
    def test_model_loading_failure(self):
        self.write_dummy_image(self.real_dir / "real.jpg")
        
        settings.DEEPFAKE_MODEL_PATH = "nonexistent_model.pth"
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"
        
        res = run_benchmark(self.dataset_dir, self.results_dir, db=self.db)
        self.assertEqual(res["evaluation_status"], "MODEL_LOAD_FAILED")
        
    def test_deterministic_result_formatting(self):
        self.write_dummy_image(self.real_dir / "real.jpg")
        self.write_dummy_image(self.fake_dir / "fake.jpg")
        
        predict_side_effects = [
            {"label": "REAL", "confidence": 0.8, "model_status": "CONFIGURED", "model_name": "MesoInception-4"},
            {"label": "FAKE", "confidence": 0.8, "model_status": "CONFIGURED", "model_name": "MesoInception-4"}
        ]
        
        with patch.object(DeepfakeModelInterface, 'predict', side_effect=predict_side_effects):
            run_benchmark(self.dataset_dir, self.results_dir, db=self.db)
            
            # Check results file
            results_file = self.results_dir / "benchmark_results.json"
            self.assertTrue(results_file.exists())
            
            with open(results_file, "r") as f:
                data = json.load(f)
                
            self.assertIn("evaluation_status", data)
            self.assertIn("metrics", data)
            self.assertIn("raw_predictions", data)

if __name__ == "__main__":
    unittest.main()

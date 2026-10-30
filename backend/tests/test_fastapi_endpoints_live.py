"""
Rigorous verification of the Deep Learning inference lifecycle, Grad-CAM generation,
and dashboard reporting through the application's service layer and route handlers.
"""
import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import Base
from app.models.evidence import Evidence, AuditLog
from app.models.analysis import MediaAnalysis, FaceDetection, DeepfakeAnalysis, DeepfakePrediction, ElaAnalysis, FftAnalysis
from app.services.deepfake_model_interface import DeepfakeModelInterface, MesoInception4
from app.services.deepfake_detection_service import DeepfakeDetectionService
from app.services.gradcam_service import GradCAMService
from app.api.routes.health import get_dashboard_stats
from app.api.routes.analysis import generate_face_gradcam, get_gradcam_visualization


class TestFastAPIEndpointsLive(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Save settings
        self.saved_settings = {
            "BACKEND_DIR": settings.BACKEND_DIR,
            "DEEPFAKE_MODEL_PATH": settings.DEEPFAKE_MODEL_PATH,
            "DEEPFAKE_MODEL_VERSION": settings.DEEPFAKE_MODEL_VERSION,
            "DEEPFAKE_DEVICE": settings.DEEPFAKE_DEVICE,
        }

        # Point backend dir to isolated temporary directory (all storage paths derive from this)
        settings.BACKEND_DIR = self.root
        settings.DEEPFAKE_DEVICE = "cpu"

        # In-memory database
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()

        # Reset model interface
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        for k, v in self.saved_settings.items():
            setattr(settings, k, v)
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"
        self.temp_dir.cleanup()

    def create_fixture_evidence_and_face(self):
        """Creates an evidence record with an associated completed media analysis and face crop."""
        # 1. Create crop file
        crop_filename = "crop_1_face_0_test.jpg"
        crop_path = settings.FACE_CROPS_DIR / crop_filename
        face_img = np.zeros((256, 256, 3), dtype=np.uint8)
        face_img[:] = (180, 150, 130)
        cv2.circle(face_img, (100, 100), 20, (50, 50, 50), -1)
        cv2.circle(face_img, (156, 100), 20, (50, 50, 50), -1)
        cv2.imwrite(str(crop_path), face_img)

        # 2. Ingest Evidence
        orig_path = self.root / "test_orig.jpg"
        cv2.imwrite(str(orig_path), face_img)
        file_bytes = orig_path.read_bytes()
        sha256 = hashlib.sha256(file_bytes).hexdigest()

        evidence = Evidence(
            original_filename="test_orig.jpg",
            stored_filename="test_orig.jpg",
            media_category="IMAGE",
            mime_type="image/jpeg",
            file_size=len(file_bytes),
            file_extension=".jpg",
            sha256_hash=sha256,
            storage_path=str(orig_path),
            status="REGISTERED"
        )
        self.db.add(evidence)
        self.db.commit()
        self.db.refresh(evidence)

        # 3. Create MediaAnalysis with status COMPLETED
        media_analysis = MediaAnalysis(
            evidence_id=evidence.id,
            status="COMPLETED",
            integrity_verified=True
        )
        self.db.add(media_analysis)
        self.db.commit()
        self.db.refresh(media_analysis)

        # 4. Create FaceDetection
        face = FaceDetection(
            analysis_id=media_analysis.id,
            face_index=0,
            x=20,
            y=20,
            width=200,
            height=200,
            crop_path=crop_filename
        )
        self.db.add(face)
        self.db.commit()
        self.db.refresh(face)

        return evidence, media_analysis, face

    def test_unconfigured_model_behavior(self):
        """Verify that an unconfigured model returns MODEL_NOT_CONFIGURED and does not fabricate labels."""
        evidence, media_analysis, face = self.create_fixture_evidence_and_face()

        settings.DEEPFAKE_MODEL_PATH = None
        settings.DEEPFAKE_MODEL_VERSION = "NOT_CONFIGURED"
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"

        result = DeepfakeDetectionService.run_deepfake_analysis(self.db, evidence.id)
        self.assertEqual(result.model_version, "NOT_CONFIGURED")
        self.assertEqual(result.overall_label, "MODEL_NOT_CONFIGURED")

    def test_configured_model_inference_and_gradcam_flow(self):
        """
        Verify live checkpoint loading, batched inference, prediction persistence,
        and Grad-CAM heatmap visualization generation.
        """
        evidence, media_analysis, face = self.create_fixture_evidence_and_face()

        # Locate trained checkpoint
        backend_dir = Path(__file__).resolve().parent.parent
        checkpoint_path = backend_dir / "training" / "weights" / "meso_inception_best.pth"
        self.assertTrue(checkpoint_path.exists(), f"Checkpoint must exist at {checkpoint_path}")

        # Configure model
        settings.DEEPFAKE_MODEL_PATH = str(checkpoint_path.resolve())
        settings.DEEPFAKE_MODEL_VERSION = "meso-trained-v1.0"
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"

        # 1. Run live deepfake analysis
        result = DeepfakeDetectionService.run_deepfake_analysis(self.db, evidence.id)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.model_version, "meso-trained-v1.0")
        self.assertIn(result.overall_label, ["REAL", "FAKE"])
        self.assertTrue(0.0 <= result.overall_confidence <= 1.0)
        self.assertEqual(len(result.predictions), 1)
        self.assertEqual(result.predictions[0].face_detection_id, face.id)

        # Verify DB records created
        db_analysis = self.db.query(DeepfakeAnalysis).filter(DeepfakeAnalysis.evidence_id == evidence.id).first()
        self.assertIsNotNone(db_analysis)
        self.assertEqual(db_analysis.status, "COMPLETED")
        self.assertEqual(db_analysis.overall_label, result.overall_label)

        db_preds = self.db.query(DeepfakePrediction).filter(DeepfakePrediction.analysis_id == db_analysis.id).all()
        self.assertEqual(len(db_preds), 1)
        self.assertTrue(0.0 <= db_preds[0].confidence <= 1.0)

        # 2. Run Grad-CAM route handler
        gradcam_res = generate_face_gradcam(evidence.id, face.id, db=self.db)
        self.assertEqual(gradcam_res["status"], "COMPLETED")
        self.assertIn("gradcam_filename", gradcam_res)
        self.assertEqual(gradcam_res["layer_inspected"], "conv4")

        # Verify Grad-CAM overlay file exists on disk
        cam_file = settings.GRADCAM_DIR / gradcam_res["gradcam_filename"]
        self.assertTrue(cam_file.exists())
        self.assertGreater(cam_file.stat().st_size, 0)

        # 3. Verify get_gradcam_visualization route
        file_resp = get_gradcam_visualization(gradcam_res["gradcam_filename"])
        self.assertEqual(file_resp.media_type, "image/jpeg")

    def test_dashboard_stats_endpoint(self):
        """Verify that get_dashboard_stats accurately reflects database entities."""
        evidence, media_analysis, face = self.create_fixture_evidence_and_face()

        # Ingest one completed deepfake analysis
        df_analysis = DeepfakeAnalysis(
            evidence_id=evidence.id,
            preprocessing_analysis_id=media_analysis.id,
            status="COMPLETED",
            overall_label="FAKE",
            overall_confidence=0.92,
            model_name="MesoInception-4",
            model_version="1.0"
        )
        self.db.add(df_analysis)
        self.db.commit()

        stats = get_dashboard_stats(db=self.db)
        self.assertEqual(stats["total_evidence"], 1)
        self.assertEqual(stats["analyses_completed"], 1)
        self.assertEqual(stats["high_risk_findings"], 1)
        self.assertEqual(stats["integrity_warnings"], 0)


if __name__ == "__main__":
    unittest.main()

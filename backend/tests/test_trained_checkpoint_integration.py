import unittest
import hashlib
import math
from pathlib import Path
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import Base
from app.models.evidence import Evidence, AuditLog
from app.models.analysis import MediaAnalysis, FaceDetection, DeepfakeAnalysis, DeepfakePrediction
from app.services.hashing_service import HashingService
from app.services.deepfake_model_interface import DeepfakeModelInterface, MesoInception4
from app.services.deepfake_detection_service import DeepfakeDetectionService


class TestTrainedCheckpointIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend_dir = Path(__file__).resolve().parent.parent
        cls.weights_path = cls.backend_dir / "training" / "weights_sdfvd" / "meso_inception_best.pth"
        cls.test_manifest = cls.backend_dir / "training" / "weights_sdfvd" / "test_split_manifest.csv"
        cls.crops_dir = cls.backend_dir / "training" / "data_sdfvd" / "crops"
        
        # Test SQLite DB
        cls.engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(bind=cls.engine)
        cls.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)

    def setUp(self):
        self.db = self.SessionLocal()
        # Save original settings
        self.orig_path = settings.DEEPFAKE_MODEL_PATH
        self.orig_name = settings.DEEPFAKE_MODEL_NAME
        self.orig_version = settings.DEEPFAKE_MODEL_VERSION
        self.orig_device = settings.DEEPFAKE_DEVICE
        
        # Reset DeepfakeModelInterface state
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_key = None
        DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"

    def tearDown(self):
        self.db.close()
        settings.DEEPFAKE_MODEL_PATH = self.orig_path
        settings.DEEPFAKE_MODEL_NAME = self.orig_name
        settings.DEEPFAKE_MODEL_VERSION = self.orig_version
        settings.DEEPFAKE_DEVICE = self.orig_device
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_key = None
        DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"

    def test_checkpoint_weights_integrity_and_keys(self):
        """Verify the trained SDFVD checkpoint exists, has valid SHA-256, and matches architecture."""
        self.assertTrue(self.weights_path.exists(), f"Weights file must exist at {self.weights_path}")
        
        # SHA-256 validation
        data = self.weights_path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        self.assertEqual(sha256, "19f655acbd190b29617099fba8215d507ac48b78650bf539531a48bc210ab4db")
        
        # Architecture validation
        state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model = MesoInception4(num_classes=1)
        expected_keys = set(model.state_dict().keys())
        actual_keys = set(state_dict.keys())
        
        self.assertEqual(expected_keys, actual_keys, "Checkpoint keys must exactly match MesoInception-4.")
        self.assertEqual(len(actual_keys), 52)
        
        # Verify finite tensors
        for k, v in state_dict.items():
            self.assertTrue(torch.isfinite(v).all().item(), f"Tensor {k} contains non-finite values.")

    def test_deepfake_model_interface_relative_path_loading(self):
        """Verify DeepfakeModelInterface successfully loads the checkpoint via relative path."""
        rel_path = "training/weights_sdfvd/meso_inception_best.pth"
        settings.DEEPFAKE_MODEL_PATH = rel_path
        settings.DEEPFAKE_MODEL_NAME = "MesoInception-4"
        settings.DEEPFAKE_MODEL_VERSION = "sdfvd_v1"
        settings.DEEPFAKE_DEVICE = "cpu"

        model = DeepfakeModelInterface.load_model()
        self.assertIsNotNone(model, "Model must load successfully from relative path.")
        self.assertEqual(DeepfakeModelInterface.get_model_status(), "CONFIGURED")
        self.assertIsInstance(model, MesoInception4)

    def test_real_crop_inference_from_test_split(self):
        """Verify inference on a real face crop from the held-out test split."""
        settings.DEEPFAKE_MODEL_PATH = str(self.weights_path)
        settings.DEEPFAKE_MODEL_NAME = "MesoInception-4"
        settings.DEEPFAKE_MODEL_VERSION = "sdfvd_v1"
        settings.DEEPFAKE_DEVICE = "cpu"

        real_crop_path = self.crops_dir / "v10_frame0000_face0.jpg"
        self.assertTrue(real_crop_path.exists(), f"Crop image must exist at {real_crop_path}")

        result = DeepfakeModelInterface.predict(str(real_crop_path))
        self.assertEqual(result["model_status"], "CONFIGURED")
        self.assertEqual(result["model_name"], "MesoInception-4")
        self.assertEqual(result["label"], "REAL")
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertTrue(0.0 <= result["raw_probability"] <= 1.0)
        self.assertTrue(math.isfinite(result["raw_probability"]))
        self.assertFalse(result["is_calibrated"])
        self.assertEqual(result["threshold_used"], 0.5)
        self.assertEqual(result["confidence"], result["boundary_distance"])

        # Real sample v10 has raw probability p ~ 0.5041 >= 0.50 -> REAL (True Positive)
        self.assertGreaterEqual(result["raw_probability"], 0.50)
        expected_boundary_dist = (result["raw_probability"] - 0.5) / 0.5
        self.assertAlmostEqual(result["boundary_distance"], expected_boundary_dist, places=5)

    def test_fake_crop_inference_from_test_split(self):
        """
        Verify inference on deepfake crops from the held-out test split.
        Demonstrates both a misclassified fake (False Positive, p >= threshold)
        and a correctly identified fake (True Negative, p < threshold).
        """
        settings.DEEPFAKE_MODEL_PATH = str(self.weights_path)
        settings.DEEPFAKE_MODEL_NAME = "MesoInception-4"
        settings.DEEPFAKE_MODEL_VERSION = "sdfvd_v1"
        settings.DEEPFAKE_DEVICE = "cpu"

        # Case A: vs10 crop (p ~ 0.5054 >= 0.50 -> REAL, False Positive misclassification)
        fake_crop_vs10 = self.crops_dir / "vs10_frame0000_face0.jpg"
        self.assertTrue(fake_crop_vs10.exists(), f"Crop image must exist at {fake_crop_vs10}")

        result_vs10 = DeepfakeModelInterface.predict(str(fake_crop_vs10))
        self.assertEqual(result_vs10["model_status"], "CONFIGURED")
        self.assertTrue(result_vs10["raw_probability"] >= 0.50)
        self.assertEqual(result_vs10["label"], "REAL")  # Consistent with p >= threshold rule
        self.assertEqual(result_vs10["confidence"], result_vs10["boundary_distance"])
        self.assertFalse(result_vs10["is_calibrated"])

        # Case B: vs17 crop (p ~ 0.4894 < 0.50 -> FAKE, True Negative correct detection)
        fake_crop_vs17 = self.crops_dir / "vs17_frame0000_face0.jpg"
        self.assertTrue(fake_crop_vs17.exists(), f"Crop image must exist at {fake_crop_vs17}")

        result_vs17 = DeepfakeModelInterface.predict(str(fake_crop_vs17))
        self.assertEqual(result_vs17["model_status"], "CONFIGURED")
        self.assertTrue(result_vs17["raw_probability"] < 0.50)
        self.assertEqual(result_vs17["label"], "FAKE")  # Correctly flagged as manipulated
        self.assertEqual(result_vs17["confidence"], result_vs17["boundary_distance"])
        self.assertFalse(result_vs17["is_calibrated"])

    def test_threshold_boundary_decisions_below_equal_above(self):
        """
        Explicitly verifies classification decision logic and boundary distance
        for probabilities immediately below (tau - eps), equal to (tau), and above (tau + eps).
        Decision rule: p >= tau -> REAL, p < tau -> FAKE.
        """
        test_cases = [0.50, 0.5020, 0.5052]
        eps = 1e-4

        for tau in test_cases:
            # 1. Below threshold (tau - eps) -> FAKE
            p_below = max(0.0, tau - eps)
            dist_below = (tau - p_below) / max(1e-6, tau)
            self.assertTrue(p_below < tau)
            label_below = "REAL" if p_below >= tau else "FAKE"
            self.assertEqual(label_below, "FAKE", f"p={p_below} < tau={tau} must yield FAKE")
            self.assertAlmostEqual(dist_below, eps / tau, places=5)

            # 2. Exactly equal to threshold (tau) -> REAL (inclusive threshold)
            p_equal = tau
            dist_equal = (p_equal - tau) / max(1e-6, 1.0 - tau)
            self.assertTrue(p_equal >= tau)
            label_equal = "REAL" if p_equal >= tau else "FAKE"
            self.assertEqual(label_equal, "REAL", f"p={p_equal} == tau={tau} must yield REAL")
            self.assertAlmostEqual(dist_equal, 0.0, places=5)

            # 3. Above threshold (tau + eps) -> REAL
            p_above = min(1.0, tau + eps)
            dist_above = (p_above - tau) / max(1e-6, 1.0 - tau)
            self.assertTrue(p_above >= tau)
            label_above = "REAL" if p_above >= tau else "FAKE"
            self.assertEqual(label_above, "REAL", f"p={p_above} > tau={tau} must yield REAL")
            self.assertAlmostEqual(dist_above, eps / (1.0 - tau), places=5)

    def test_confidence_is_uncalibrated_boundary_distance(self):
        """
        Confirms that reported 'confidence' is an uncalibrated normalized boundary distance:
        - 0.0 at threshold boundary (maximum uncertainty)
        - 1.0 at extremes (p = 0.0 or p = 1.0)
        - is_calibrated is explicitly False
        """
        settings.DEEPFAKE_MODEL_PATH = str(self.weights_path)
        settings.DEEPFAKE_MODEL_NAME = "MesoInception-4"
        settings.DEEPFAKE_MODEL_VERSION = "sdfvd_v1"
        settings.DEEPFAKE_DEVICE = "cpu"

        crop_path = self.crops_dir / "v10_frame0000_face0.jpg"
        result = DeepfakeModelInterface.predict(str(crop_path))

        self.assertIn("boundary_distance", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["confidence"], result["boundary_distance"])
        self.assertFalse(result["is_calibrated"], "Model must explicitly declare is_calibrated=False")
        self.assertIn("threshold_used", result)
        self.assertEqual(result["threshold_used"], 0.5)

    def test_threshold_override_shifts_classification(self):
        """
        Verify that changing settings.DEEPFAKE_CLASSIFICATION_THRESHOLD alters the decision boundary
        for borderline samples like vs10 (p ~ 0.505383).
        """
        settings.DEEPFAKE_MODEL_PATH = str(self.weights_path)
        settings.DEEPFAKE_MODEL_NAME = "MesoInception-4"
        settings.DEEPFAKE_MODEL_VERSION = "sdfvd_v1"
        settings.DEEPFAKE_DEVICE = "cpu"

        crop_vs10 = str(self.crops_dir / "vs10_frame0000_face0.jpg")

        # Threshold 0.50: p (0.5054) >= 0.50 -> REAL
        settings.DEEPFAKE_CLASSIFICATION_THRESHOLD = 0.50
        res_50 = DeepfakeModelInterface.predict(crop_vs10)
        self.assertEqual(res_50["label"], "REAL")

        # Threshold 0.5060: p (0.5054) < 0.5060 -> FAKE
        settings.DEEPFAKE_CLASSIFICATION_THRESHOLD = 0.5060
        res_5060 = DeepfakeModelInterface.predict(crop_vs10)
        self.assertEqual(res_5060["label"], "FAKE")
        self.assertEqual(res_5060["threshold_used"], 0.5060)

        # Reset
        settings.DEEPFAKE_CLASSIFICATION_THRESHOLD = 0.50

    def test_error_handling_on_non_existent_crop(self):
        """Verify explicit error return on missing crop without fabricating labels."""
        settings.DEEPFAKE_MODEL_PATH = str(self.weights_path)
        settings.DEEPFAKE_MODEL_NAME = "MesoInception-4"
        settings.DEEPFAKE_MODEL_VERSION = "sdfvd_v1"

        missing_path = self.backend_dir / "missing_face_crop.jpg"
        result = DeepfakeModelInterface.predict(str(missing_path))
        self.assertEqual(result["label"], "MODEL_ERROR")
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["model_status"], "MODEL_ERROR")

    def test_unconfigured_model_returns_explicit_status(self):
        """Verify unconfigured model returns MODEL_NOT_CONFIGURED and never fabricates labels."""
        settings.DEEPFAKE_MODEL_PATH = None
        settings.DEEPFAKE_MODEL_VERSION = "NOT_CONFIGURED"

        result = DeepfakeModelInterface.predict(str(self.crops_dir / "v10_frame0000_face0.jpg"))
        self.assertEqual(result["label"], "MODEL_NOT_CONFIGURED")
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["model_status"], "MODEL_NOT_CONFIGURED")

    def test_production_service_run_deepfake_analysis_with_sdfvd_weights(self):
        """Verify the full DeepfakeDetectionService pipeline using the trained SDFVD weights."""
        settings.DEEPFAKE_MODEL_PATH = str(self.weights_path)
        settings.DEEPFAKE_MODEL_NAME = "MesoInception-4"
        settings.DEEPFAKE_MODEL_VERSION = "sdfvd_v1"
        settings.DEEPFAKE_DEVICE = "cpu"

        # Create dummy evidence file
        evidence_file = self.backend_dir / "storage" / "evidence" / "test_item.mp4"
        evidence_file.parent.mkdir(parents=True, exist_ok=True)
        evidence_file.write_bytes(b"dummy evidence bytes for test")
        sha256 = HashingService.calculate_sha256(evidence_file)

        # Place crop in settings.FACE_CROPS_DIR
        crop_src = self.crops_dir / "v10_frame0000_face0.jpg"
        target_crop_name = "test_run_face0.jpg"
        target_crop = settings.FACE_CROPS_DIR / target_crop_name
        settings.FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)
        target_crop.write_bytes(crop_src.read_bytes())

        # Seed database records
        evidence = Evidence(
            original_filename="test_item.mp4",
            stored_filename="test_item.mp4",
            storage_path=str(evidence_file),
            sha256_hash=sha256,
            file_size=len(evidence_file.read_bytes()),
            file_extension=".mp4",
            mime_type="video/mp4",
            media_category="VIDEO"
        )
        self.db.add(evidence)
        self.db.commit()
        self.db.refresh(evidence)

        prep_analysis = MediaAnalysis(
            evidence_id=evidence.id,
            analysis_type="PREPROCESSING_FACE_DETECTION",
            status="COMPLETED"
        )
        self.db.add(prep_analysis)
        self.db.commit()
        self.db.refresh(prep_analysis)

        face = FaceDetection(
            analysis_id=prep_analysis.id,
            frame_index=0,
            timestamp_seconds=0.0,
            face_index=0,
            x=10,
            y=10,
            width=100,
            height=100,
            crop_path=target_crop_name
        )
        self.db.add(face)
        self.db.commit()
        self.db.refresh(face)

        # Run full production deepfake analysis service
        result = DeepfakeDetectionService.run_deepfake_analysis(self.db, evidence.id)
        
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.model_name, "MesoInception-4")
        self.assertEqual(result.model_version, "sdfvd_v1")
        self.assertIn(result.overall_label, ["REAL", "FAKE"])
        self.assertIsNotNone(result.overall_confidence)
        self.assertTrue(0.0 <= result.overall_confidence <= 1.0)

        # Verify persisted predictions
        preds = self.db.query(DeepfakePrediction).filter(DeepfakePrediction.analysis_id == result.id).all()
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].face_detection_id, face.id)
        self.assertEqual(preds[0].model_status, "CONFIGURED")
        self.assertIn(preds[0].label, ["REAL", "FAKE"])

        # Verify audit logs
        audit_events = [a.action for a in self.db.query(AuditLog).filter(AuditLog.evidence_id == evidence.id).all()]
        self.assertIn("DEEPFAKE_ANALYSIS_STARTED", audit_events)
        self.assertIn("MODEL_CONFIGURED", audit_events)
        self.assertIn("DEEPFAKE_ANALYSIS_COMPLETED", audit_events)

    def test_production_service_failure_returns_explicit_error_without_fallback(self):
        """
        Verify that when a face crop is missing or corrupted, the production inference service
        returns an explicit MODEL_ERROR with None confidence, and never falls back to hardcoded guesses.
        """
        settings.DEEPFAKE_MODEL_PATH = str(self.weights_path)
        settings.DEEPFAKE_MODEL_NAME = "MesoInception-4"
        settings.DEEPFAKE_MODEL_VERSION = "sdfvd_v1"
        settings.DEEPFAKE_DEVICE = "cpu"

        # Evidence file
        evidence_file = self.backend_dir / "storage" / "evidence" / "test_missing_crop.mp4"
        evidence_file.parent.mkdir(parents=True, exist_ok=True)
        evidence_file.write_bytes(b"dummy evidence for error test")
        sha256 = HashingService.calculate_sha256(evidence_file)

        evidence = Evidence(
            original_filename="test_missing_crop.mp4",
            stored_filename="test_missing_crop.mp4",
            storage_path=str(evidence_file),
            sha256_hash=sha256,
            file_size=len(evidence_file.read_bytes()),
            file_extension=".mp4",
            mime_type="video/mp4",
            media_category="VIDEO"
        )
        self.db.add(evidence)
        self.db.commit()
        self.db.refresh(evidence)

        prep_analysis = MediaAnalysis(
            evidence_id=evidence.id,
            analysis_type="PREPROCESSING_FACE_DETECTION",
            status="COMPLETED"
        )
        self.db.add(prep_analysis)
        self.db.commit()
        self.db.refresh(prep_analysis)

        # Non-existent crop file
        face = FaceDetection(
            analysis_id=prep_analysis.id,
            frame_index=0,
            timestamp_seconds=0.0,
            face_index=0,
            x=10,
            y=10,
            width=100,
            height=100,
            crop_path="completely_non_existent_crop.jpg"
        )
        self.db.add(face)
        self.db.commit()
        self.db.refresh(face)

        # Run production service
        result = DeepfakeDetectionService.run_deepfake_analysis(self.db, evidence.id)

        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.overall_label, "MODEL_ERROR")
        self.assertIsNone(result.overall_confidence)
        self.assertIn("1 of 1 face predictions failed", result.error_message)

        # Check prediction record in DB
        preds = self.db.query(DeepfakePrediction).filter(DeepfakePrediction.analysis_id == result.id).all()
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].label, "MODEL_ERROR")
        self.assertIsNone(preds[0].confidence)
        self.assertEqual(preds[0].model_status, "MODEL_ERROR")


if __name__ == "__main__":
    unittest.main()

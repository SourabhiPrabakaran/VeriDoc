import os
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch

from app.core.config import settings
from app.services.deepfake_model_interface import MesoInception4, DeepfakeModelInterface
from app.services.gradcam_service import GradCAMService
from training.dataset import (
    DeepfakeDataset,
    create_grouped_splits,
    generate_synthetic_verification_dataset,
    LABEL_REAL,
    LABEL_FAKE
)
from training.evaluate import (
    compute_roc_curve,
    compute_equal_error_rate,
    compute_expected_calibration_error
)
from training.dual_stream_model import DualStreamForensicNet


class TestDeepLearningPipeline(unittest.TestCase):
    def test_model_forward_logits_and_probabilities(self):
        """Verify MesoInception-4 returns both unscaled logits and sigmoid probabilities."""
        model = MesoInception4(num_classes=1)
        model.eval()
        dummy_input = torch.randn(2, 3, 256, 256)

        # 1. Forward logits
        logits = model(dummy_input, return_logits=True)
        self.assertEqual(logits.shape, (2, 1))

        # 2. Forward probabilities
        probs = model(dummy_input, return_logits=False)
        self.assertEqual(probs.shape, (2, 1))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

        # 3. Explicit logits match
        direct_logits = model.forward_logits(dummy_input)
        self.assertTrue(torch.allclose(logits, direct_logits))

        # 4. Features shape (16, 8, 8)
        feats = model.forward_features(dummy_input)
        self.assertEqual(feats.shape, (2, 16, 8, 8))

    def test_dataset_grouped_split_leakage_prevention(self):
        """Verify that frames from the same video ID are strictly never split across train/val/test."""
        # Create 10 fake videos with 5 frames each, and 10 real videos with 5 frames each
        samples = []
        for v in range(10):
            for f in range(5):
                samples.append((f"/path/real_vid{v}_f{f}.jpg", LABEL_REAL, f"real_vid{v}"))
                samples.append((f"/path/fake_vid{v}_f{f}.jpg", LABEL_FAKE, f"fake_vid{v}"))

        train_s, val_s, test_s = create_grouped_splits(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

        train_vids = {s[2] for s in train_s}
        val_vids = {s[2] for s in val_s}
        test_vids = {s[2] for s in test_s}

        # Check total count
        self.assertEqual(len(train_s) + len(val_s) + len(test_s), len(samples))

        # Check strict disjointness (no video ID leakage between any splits)
        self.assertEqual(len(train_vids.intersection(val_vids)), 0, "Data leakage between train and val!")
        self.assertEqual(len(train_vids.intersection(test_vids)), 0, "Data leakage between train and test!")
        self.assertEqual(len(val_vids.intersection(test_vids)), 0, "Data leakage between val and test!")

    def test_evaluation_metric_calculations(self):
        """Verify ROC-AUC, EER, and calibration error calculations with known mathematical ground truths."""
        # 1. Perfect classifier: scores for positive class strictly higher
        y_true_perfect = np.array([0, 0, 0, 1, 1, 1])
        y_score_perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        _, _, _, auc_perf = compute_roc_curve(y_true_perfect, y_score_perfect)
        self.assertAlmostEqual(auc_perf, 1.0, places=4)

        # 2. Inverse classifier: scores strictly inverted
        y_score_inverse = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        _, _, _, auc_inv = compute_roc_curve(y_true_perfect, y_score_inverse)
        self.assertAlmostEqual(auc_inv, 0.0, places=4)

        # 3. Equal Error Rate (EER)
        fpr, tpr, _, _ = compute_roc_curve(y_true_perfect, y_score_perfect)
        eer, _ = compute_equal_error_rate(fpr, tpr)
        self.assertAlmostEqual(eer, 0.0, places=4)

        # 4. Expected Calibration Error (ECE) for well-calibrated vs miscalibrated
        ece = compute_expected_calibration_error(y_true_perfect, y_score_perfect)
        self.assertTrue(0.0 <= ece <= 1.0)

    def test_gradcam_service_unconfigured_and_configured(self):
        """Test Grad-CAM behavior when model is unconfigured and when loaded."""
        # Unconfigured behavior
        original_path = settings.DEEPFAKE_MODEL_PATH
        original_version = settings.DEEPFAKE_MODEL_VERSION
        settings.DEEPFAKE_MODEL_PATH = None
        settings.DEEPFAKE_MODEL_VERSION = "NOT_CONFIGURED"
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"

        res = GradCAMService.generate_heatmap("nonexistent.jpg")
        self.assertEqual(res["status"], "MODEL_NOT_CONFIGURED")

        # Configured behavior with mock weights
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            weight_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
            crop_path = img_f.name

        try:
            model = MesoInception4(num_classes=1)
            torch.save(model.state_dict(), weight_path)

            dummy_crop = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            cv2.imwrite(crop_path, dummy_crop)

            settings.DEEPFAKE_MODEL_PATH = weight_path
            settings.DEEPFAKE_MODEL_VERSION = "test-1.0"
            DeepfakeModelInterface._model = None

            cam_res = GradCAMService.generate_heatmap(crop_path, target_class_is_real=False)
            self.assertEqual(cam_res["status"], "COMPLETED")
            self.assertIsNotNone(cam_res["gradcam_filename"])
            self.assertEqual(cam_res["layer_inspected"], "conv4")

            # Check that generated visualization exists on disk
            overlay_file = settings.GRADCAM_DIR / cam_res["gradcam_filename"]
            self.assertTrue(overlay_file.exists())
            overlay_file.unlink(missing_ok=True)
        finally:
            settings.DEEPFAKE_MODEL_PATH = original_path
            settings.DEEPFAKE_MODEL_VERSION = original_version
            DeepfakeModelInterface._model = None
            DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"
            if os.path.exists(weight_path):
                os.unlink(weight_path)
            if os.path.exists(crop_path):
                os.unlink(crop_path)

    def test_dual_stream_network(self):
        """Verify DualStreamForensicNet combining spatial and frequency streams."""
        net = DualStreamForensicNet(num_classes=1)
        net.eval()
        dummy_rgb = torch.randn(2, 3, 256, 256)

        # Verify FFT magnitude transformation
        fft_mag = DualStreamForensicNet.compute_fft_magnitude(dummy_rgb)
        self.assertEqual(fft_mag.shape, (2, 1, 256, 256))

        # Verify dual stream forward pass
        out = net(dummy_rgb)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_dual_stream_batch_invariance_regression(self):
        """
        Regression test: Verify that DualStreamForensicNet produces consistent intermediate
        frequency representations and final predictions regardless of batch size (B = 1, 2, 8, 16),
        using a numerical tolerance rather than requiring bit-for-bit identity.
        """
        net = DualStreamForensicNet(num_classes=1)
        net.eval()

        torch.manual_seed(42)
        images = torch.rand(16, 3, 256, 256)
        atol = 1e-5

        # 1. Baseline: Compute individual single-crop outputs (B = 1)
        single_ffts = []
        single_preds = []
        with torch.no_grad():
            for i in range(16):
                sample = images[i:i + 1]
                fft_mag = DualStreamForensicNet.compute_fft_magnitude(sample)
                pred = net(sample).item()
                single_ffts.append(fft_mag)
                single_preds.append(pred)

        # 2. Evaluate batch sizes 1, 2, 8, and 16
        for bs in [1, 2, 8, 16]:
            with torch.no_grad():
                for start_idx in range(0, 16, bs):
                    batch_samples = images[start_idx:start_idx + bs]

                    # Test intermediate frequency representation
                    batch_fft = DualStreamForensicNet.compute_fft_magnitude(batch_samples)
                    self.assertEqual(batch_fft.shape, (bs, 1, 256, 256))

                    # Test final model prediction
                    batch_preds = net(batch_samples).squeeze(-1)
                    if bs == 1:
                        batch_preds = batch_preds.unsqueeze(0)

                    for j in range(bs):
                        global_idx = start_idx + j

                        # Frequency representation tolerance check
                        fft_diff = torch.max(torch.abs(batch_fft[j:j + 1] - single_ffts[global_idx])).item()
                        self.assertLess(
                            fft_diff, atol,
                            f"FFT magnitude batch variance at B={bs}, sample={global_idx}: diff={fft_diff:.7f} >= {atol}"
                        )

                        # Final prediction tolerance check
                        pred_diff = abs(batch_preds[j].item() - single_preds[global_idx])
                        self.assertLess(
                            pred_diff, atol,
                            f"Prediction batch variance at B={bs}, sample={global_idx}: diff={pred_diff:.7f} >= {atol}"
                        )


if __name__ == "__main__":
    unittest.main()

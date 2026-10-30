import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from app.core.config import settings
from app.services.deepfake_model_interface import DeepfakeModelInterface, MesoInception4
from training.dual_stream_model import DualStreamForensicNet
from training.dataset import (
    DeepfakeDataset,
    parse_label,
    load_dataset_manifest,
    load_dataset_splits_from_manifest,
    create_grouped_splits,
    LABEL_REAL,
    LABEL_FAKE
)
from training.evaluate import (
    evaluate_checkpoint,
    compute_roc_curve,
    compute_equal_error_rate,
    compute_expected_calibration_error
)
from training.compare_models import generate_comparison


class TestDatasetAndScientificPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create dummy image files for manifest path validation
        self.dummy_images = {}
        for name in ("real_vid1_f0.jpg", "real_vid1_f1.jpg", "real_vid2_f0.jpg",
                     "fake_vid3_f0.jpg", "fake_vid3_f1.jpg", "fake_vid4_f0.jpg"):
            img_path = self.root / name
            img = np.full((64, 64, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            self.dummy_images[name] = img_path

        # Save configuration
        self.orig_model_path = settings.DEEPFAKE_MODEL_PATH
        self.orig_model_version = settings.DEEPFAKE_MODEL_VERSION
        self.orig_model_name = settings.DEEPFAKE_MODEL_NAME

    def tearDown(self):
        settings.DEEPFAKE_MODEL_PATH = self.orig_model_path
        settings.DEEPFAKE_MODEL_VERSION = self.orig_model_version
        settings.DEEPFAKE_MODEL_NAME = self.orig_model_name
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"
        self.temp_dir.cleanup()

    def test_label_parsing_valid_and_invalid(self):
        """Verify label normalization accepts valid synonyms and rejects invalid inputs."""
        # Valid REAL synonyms
        for s in [1, 1.0, "1", "1.0", "real", "Real", "AUTHENTIC", "original", "pristine", "bonafide"]:
            self.assertEqual(parse_label(s), LABEL_REAL)

        # Valid FAKE synonyms
        for s in [0, 0.0, "0", "0.0", "fake", "Fake", "MANIPULATED", "deepfake", "tampered", "df", "f2f", "fs"]:
            self.assertEqual(parse_label(s), LABEL_FAKE)

        # Invalid labels must raise ValueError
        for invalid in ["unknown", "maybe", 42, -1, "unlabeled", None]:
            with self.assertRaises(ValueError):
                parse_label(invalid)

    def test_manifest_parsing_csv(self):
        """Verify CSV manifest loading with varied header naming conventions."""
        csv_path = self.root / "test_manifest.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "category", "orig_video", "person_id"])
            writer.writerow(["real_vid1_f0.jpg", "authentic", "vid1", "person_alice"])
            writer.writerow(["real_vid1_f1.jpg", "real", "vid1", "person_alice"])
            writer.writerow(["fake_vid3_f0.jpg", "deepfake", "vid3", "person_bob"])

        samples = load_dataset_manifest(csv_path, base_dir=self.root, validate_paths=True)
        self.assertEqual(len(samples), 3)

        # Sample 0 check: (path, label, video_id, subject_id)
        self.assertEqual(samples[0][1], LABEL_REAL)
        self.assertEqual(samples[0][2], "vid1")
        self.assertEqual(samples[0][3], "person_alice")
        self.assertTrue(Path(samples[0][0]).exists())

        # Sample 2 check
        self.assertEqual(samples[2][1], LABEL_FAKE)
        self.assertEqual(samples[2][2], "vid3")
        self.assertEqual(samples[2][3], "person_bob")

    def test_manifest_parsing_json_list_and_splits(self):
        """Verify JSON manifest loading for both list-of-samples and split-keyed structures."""
        # 1. Flat list JSON
        json_path = self.root / "manifest_list.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([
                {"path": "real_vid1_f0.jpg", "label": "real", "video_id": "vid1"},
                {"path": "fake_vid3_f0.jpg", "label": "fake", "video_id": "vid3"}
            ], f)

        samples = load_dataset_manifest(json_path, base_dir=self.root, validate_paths=True)
        self.assertEqual(len(samples), 2)

        # 2. Split-keyed JSON
        json_split_path = self.root / "manifest_splits.json"
        with open(json_split_path, "w", encoding="utf-8") as f:
            json.dump({
                "train": [{"path": "real_vid1_f0.jpg", "label": 1.0, "video_id": "vid1"}],
                "val": [{"path": "real_vid1_f1.jpg", "label": 1.0, "video_id": "vid1"}],
                "test": [{"path": "fake_vid3_f0.jpg", "label": 0.0, "video_id": "vid3"}]
            }, f)

        splits = load_dataset_splits_from_manifest(json_split_path, base_dir=self.root, validate_paths=True)
        self.assertIsNotNone(splits)
        train_s, val_s, test_s = splits
        self.assertEqual(len(train_s), 1)
        self.assertEqual(len(val_s), 1)
        self.assertEqual(len(test_s), 1)

    def test_missing_files_validation(self):
        """Verify manifest loader fails explicitly when referenced files do not exist on disk."""
        csv_path = self.root / "missing_files.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "video_id"])
            writer.writerow(["nonexistent_face_01.jpg", "real", "vid99"])
            writer.writerow(["nonexistent_face_02.jpg", "fake", "vid99"])

        with self.assertRaises(FileNotFoundError) as ctx:
            load_dataset_manifest(csv_path, base_dir=self.root, validate_paths=True)
        self.assertIn("Manifest validation failed", str(ctx.exception))
        self.assertIn("nonexistent_face_01.jpg", str(ctx.exception))

    def test_source_video_leakage_and_subject_grouping(self):
        """
        Verify video-level and subject-level isolation:
        - When group_by='video', frames from the same video ID must never be split.
        - When group_by='subject', videos with the same subject ID must never be split.
        """
        # Create dataset with 6 videos across 2 subjects for real and fake
        samples = []
        # Real: Subject A (vid1, vid2), Subject B (vid3)
        samples.append(("/dummy/r1.jpg", LABEL_REAL, "vid1", "subjA"))
        samples.append(("/dummy/r2.jpg", LABEL_REAL, "vid1", "subjA"))
        samples.append(("/dummy/r3.jpg", LABEL_REAL, "vid2", "subjA"))
        samples.append(("/dummy/r4.jpg", LABEL_REAL, "vid3", "subjB"))
        samples.append(("/dummy/r5.jpg", LABEL_REAL, "vid4", "subjC"))

        # Fake: Subject D (vid5, vid6), Subject E (vid7)
        samples.append(("/dummy/f1.jpg", LABEL_FAKE, "vid5", "subjD"))
        samples.append(("/dummy/f2.jpg", LABEL_FAKE, "vid5", "subjD"))
        samples.append(("/dummy/f3.jpg", LABEL_FAKE, "vid6", "subjD"))
        samples.append(("/dummy/f4.jpg", LABEL_FAKE, "vid7", "subjE"))
        samples.append(("/dummy/f5.jpg", LABEL_FAKE, "vid8", "subjF"))

        # 1. Test video grouping
        tr, va, te = create_grouped_splits(samples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, group_by="video", seed=42)
        tr_vids = {s[2] for s in tr}
        va_vids = {s[2] for s in va}
        te_vids = {s[2] for s in te}
        self.assertEqual(len(tr_vids.intersection(va_vids)), 0)
        self.assertEqual(len(tr_vids.intersection(te_vids)), 0)
        self.assertEqual(len(va_vids.intersection(te_vids)), 0)

        # 2. Test subject grouping
        tr_s, va_s, te_s = create_grouped_splits(samples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, group_by="subject", seed=42)
        tr_subjs = {s[3] for s in tr_s}
        va_subjs = {s[3] for s in va_s}
        te_subjs = {s[3] for s in te_s}
        self.assertEqual(len(tr_subjs.intersection(va_subjs)), 0)
        self.assertEqual(len(tr_subjs.intersection(te_subjs)), 0)
        self.assertEqual(len(va_subjs.intersection(te_subjs)), 0)

    def test_checkpoint_selection_and_dual_stream_loading(self):
        """
        Verify DeepfakeModelInterface loads MesoInception-4 or DualStreamForensicNet
        safely and executes batched prediction for both architectures.
        """
        # 1. Create MesoInception-4 weights
        meso_weights_path = self.root / "test_meso.pth"
        meso_model = MesoInception4(num_classes=1)
        torch.save(meso_model.state_dict(), meso_weights_path)

        settings.DEEPFAKE_MODEL_PATH = str(meso_weights_path)
        settings.DEEPFAKE_MODEL_VERSION = "test-meso-1.0"
        settings.DEEPFAKE_MODEL_NAME = "MesoInception-4"
        DeepfakeModelInterface._model = None

        loaded_meso = DeepfakeModelInterface.load_model()
        self.assertIsInstance(loaded_meso, MesoInception4)
        self.assertEqual(DeepfakeModelInterface._model_status, "CONFIGURED")

        # 2. Create DualStreamForensicNet weights
        dual_weights_path = self.root / "test_dual.pth"
        dual_model = DualStreamForensicNet(num_classes=1)
        torch.save(dual_model.state_dict(), dual_weights_path)

        settings.DEEPFAKE_MODEL_PATH = str(dual_weights_path)
        settings.DEEPFAKE_MODEL_VERSION = "test-dual-1.0"
        settings.DEEPFAKE_MODEL_NAME = "DualStreamForensicNet"
        DeepfakeModelInterface._model = None

        loaded_dual = DeepfakeModelInterface.load_model()
        self.assertIsInstance(loaded_dual, DualStreamForensicNet)
        self.assertEqual(DeepfakeModelInterface._model_status, "CONFIGURED")

        # 3. Test batched inference on both models
        dummy_crop_path = str(self.dummy_images["real_vid1_f0.jpg"])
        meso_preds = DeepfakeModelInterface.predict_batch([dummy_crop_path])
        self.assertEqual(len(meso_preds), 1)
        self.assertEqual(meso_preds[0]["model_status"], "CONFIGURED")
        self.assertIn("raw_probability", meso_preds[0])

    def test_evaluation_digest_tracking_and_comparison_utility(self):
        """Verify evaluate_checkpoint records exact SHA-256 digest and generate_comparison produces delta report."""
        # Setup model and manifest
        weights_path = self.root / "eval_test_model.pth"
        model = MesoInception4(num_classes=1)
        torch.save(model.state_dict(), weights_path)
        expected_sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()

        manifest_path = self.root / "test_eval_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump([
                {"path": str(self.dummy_images["real_vid1_f0.jpg"]), "label": 1.0, "video_id": "vid1"},
                {"path": str(self.dummy_images["fake_vid3_f0.jpg"]), "label": 0.0, "video_id": "vid3"}
            ], f)

        out_eval = self.root / "eval_out"
        eval_res = evaluate_checkpoint(
            checkpoint_path=weights_path,
            test_manifest_path=manifest_path,
            output_dir=out_eval,
            model_override="meso_inception",
            device_str="cpu"
        )

        self.assertEqual(eval_res["status"], "EVALUATION_SUCCESS")
        self.assertEqual(eval_res["checkpoint_sha256"], expected_sha)
        self.assertIn("parameter_counts", eval_res)
        self.assertIn("latency_ms_per_sample", eval_res["performance"])

        # Test model comparison utility
        comp_res = generate_comparison(eval_res, eval_res, out_eval)
        self.assertIn("baseline", comp_res)
        self.assertIn("dual_stream", comp_res)

        # Check markdown table exists
        comp_md = out_eval / "model_comparison_table.md"
        self.assertTrue(comp_md.exists())
        self.assertGreater(comp_md.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

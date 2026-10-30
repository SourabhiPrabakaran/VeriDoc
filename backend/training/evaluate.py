import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.deepfake_model_interface import MesoInception4
from training.dual_stream_model import DualStreamForensicNet
from training.dataset import (
    DeepfakeDataset,
    load_dataset_manifest,
    LABEL_REAL,
    LABEL_FAKE,
    LABEL_TO_CLASS
)


def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Computes false positive rates, true positive rates, thresholds, and AUC.
    y_true: binary labels (0 or 1).
    y_score: predicted probability of class 1.
    """
    desc_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_indices]
    y_score_sorted = y_score[desc_indices]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_indices]
    fps = np.cumsum(y_true_sorted == 0)[threshold_indices]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    tpr = tps / max(1, n_pos)
    fpr = fps / max(1, n_neg)

    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[y_score_sorted[0] + 1.0, y_score_sorted[threshold_indices]]

    # Trapezoidal rule for AUC
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, thresholds, auc


def compute_equal_error_rate(fpr: np.ndarray, tpr: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) where FPR == FNR (1 - TPR).
    Returns (eer, eer_threshold_estimate).
    """
    fnr = 1.0 - tpr
    diffs = np.abs(fpr - fnr)
    min_idx = np.argmin(diffs)
    eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    return eer, min_idx


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5).astype(int))
            avg_confidence_in_bin = np.mean(np.maximum(y_prob[in_bin], 1.0 - y_prob[in_bin]))
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def evaluate_checkpoint(
    checkpoint_path: Path,
    test_manifest_path: Path,
    output_dir: Path,
    base_dir: Optional[Path] = None,
    val_manifest_path: Optional[Path] = None,
    model_override: str = "auto",
    device_str: str = "auto",
    smoke_test_override: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Performs full scientific evaluation of a trained model against the held-out test manifest.
    Computes both crop-level and video-level metrics (Accuracy, ROC-AUC, EER, ECE, Confusion Matrix).
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    test_manifest_path = Path(test_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        return {
            "status": "NOT_EVALUATED",
            "blocker": f"Trained checkpoint not found at: {checkpoint_path}",
            "remedy": "Run training with 'python backend/training/train.py' first."
        }

    if not test_manifest_path.exists():
        return {
            "status": "NOT_EVALUATED",
            "blocker": f"Test manifest not found at: {test_manifest_path}",
            "remedy": "Generate test split or specify a valid test manifest."
        }

    # 1. Compute Checkpoint Cryptographic Hash
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()

    # 2. Load manifest samples (supports CSV and JSON)
    try:
        test_samples = load_dataset_manifest(test_manifest_path, base_dir=base_dir, validate_paths=True)
    except Exception as e:
        return {
            "status": "NOT_EVALUATED",
            "blocker": f"Failed to load test manifest ({test_manifest_path}): {e}",
            "remedy": "Ensure manifest file contains valid columns and existing image paths."
        }

    if not test_samples:
        return {
            "status": "NOT_EVALUATED",
            "blocker": "Test manifest contains 0 samples.",
            "remedy": "Ensure test dataset contains valid real and fake media samples."
        }

    # 2b. Load validation samples for leakage-safe threshold calibration (prevents test set peeking)
    val_samples = []
    val_source = "NONE"
    if val_manifest_path and Path(val_manifest_path).exists():
        try:
            val_samples = load_dataset_manifest(Path(val_manifest_path).resolve(), base_dir=base_dir, validate_paths=True)
            val_source = str(val_manifest_path)
        except Exception as e:
            print(f"[EVALUATION] Error loading val_manifest_path ({val_manifest_path}): {e}")
    if not val_samples:
        for cand in [test_manifest_path.parent / "val_split_manifest.json", test_manifest_path.parent / "val_split_manifest.csv"]:
            if cand.exists():
                try:
                    val_samples = load_dataset_manifest(cand, base_dir=base_dir, validate_paths=True)
                    val_source = cand.name
                    break
                except Exception as e:
                    print(f"[EVALUATION] Error loading candidate val manifest ({cand}): {e}")

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # 3. Load Checkpoint and Detect Architecture
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = raw_checkpoint
    if isinstance(raw_checkpoint, dict):
        if "model_state_dict" in raw_checkpoint:
            state_dict = raw_checkpoint["model_state_dict"]
        elif "state_dict" in raw_checkpoint:
            state_dict = raw_checkpoint["state_dict"]
        elif "model" in raw_checkpoint:
            state_dict = raw_checkpoint["model"]

    # Determine architecture
    if model_override in ("dual_stream", "dualstream", "dualstreamforensicnet"):
        model = DualStreamForensicNet(num_classes=1)
        model_name = "DualStreamForensicNet"
    elif model_override in ("meso_inception", "mesoinception", "meso"):
        model = MesoInception4(num_classes=1)
        model_name = "MesoInception-4"
    else:
        # Auto-detect from state_dict keys
        if isinstance(state_dict, dict) and any(k.startswith("spatial_stream.") for k in state_dict.keys()):
            model = DualStreamForensicNet(num_classes=1)
            model_name = "DualStreamForensicNet"
        else:
            model = MesoInception4(num_classes=1)
            model_name = "MesoInception-4"

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 4. Inference on Test Set with Latency Benchmarking
    test_dataset = DeepfakeDataset(test_samples, transform=DeepfakeDataset.default_eval_transforms())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_probs = []
    all_targets = []
    inference_start = time.time()

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            probs = model(inputs).squeeze(-1).cpu().numpy()
            targets = labels.squeeze(-1).cpu().numpy()

            all_probs.extend(np.atleast_1d(probs))
            all_targets.extend(np.atleast_1d(targets))

    total_inference_time = time.time() - inference_start
    latency_ms_per_sample = (total_inference_time / max(1, len(test_samples))) * 1000.0

    # 4b. Calibrate Threshold on Validation Set (Strictly Zero Test Peeking)
    val_calibrated_threshold = 0.5
    val_video_threshold = 0.5
    calibration_source = "DEFAULT_0.50 (No validation set provided)"
    if val_samples:
        val_dataset = DeepfakeDataset(val_samples, transform=DeepfakeDataset.default_eval_transforms())
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        v_probs, v_targets = [], []
        with torch.inference_mode():
            for v_in, v_lbl in val_loader:
                v_in = v_in.to(device)
                v_probs.extend(np.atleast_1d(model(v_in).squeeze(-1).cpu().numpy()))
                v_targets.extend(np.atleast_1d(v_lbl.squeeze(-1).cpu().numpy()))
        y_v_t = np.array(v_targets, dtype=np.float32)
        y_v_p = np.array(v_probs, dtype=np.float32)
        fpr_v, tpr_v, th_v, _ = compute_roc_curve(y_v_t, y_v_p)
        j_v = tpr_v - fpr_v
        best_v_idx = np.argmax(j_v)
        v_cand = float(th_v[best_v_idx]) if best_v_idx < len(th_v) else 0.5
        if np.isfinite(v_cand) and 0.0 < v_cand < 1.0:
            val_calibrated_threshold = v_cand
            calibration_source = f"VALIDATION_DATASET ({len(val_samples)} crops from {val_source})"

        # Video-level validation calibration
        v_records: Dict[str, Dict[str, Any]] = {}
        for (s_path, s_label, s_vid, *rest), prob in zip(val_samples, v_probs):
            if s_vid not in v_records:
                v_records[s_vid] = {"target": s_label, "probs": []}
            v_records[s_vid]["probs"].append(prob)
        y_v_vid_t = np.array([v_records[v]["target"] for v in sorted(v_records.keys())], dtype=np.float32)
        y_v_vid_p = np.array([float(np.mean(v_records[v]["probs"])) for v in sorted(v_records.keys())], dtype=np.float32)
        fpr_v_vid, tpr_v_vid, th_v_vid, _ = compute_roc_curve(y_v_vid_t, y_v_vid_p)
        j_v_vid = tpr_v_vid - fpr_v_vid
        best_v_vid_idx = np.argmax(j_v_vid)
        v_vid_cand = float(th_v_vid[best_v_vid_idx]) if best_v_vid_idx < len(th_v_vid) else 0.5
        if np.isfinite(v_vid_cand) and 0.0 < v_vid_cand < 1.0:
            val_video_threshold = v_vid_cand

    y_true = np.array(all_targets, dtype=np.float32)
    y_prob = np.array(all_probs, dtype=np.float32)

    # 5. Crop-Level Metrics
    fpr, tpr, thresholds, auc_roc = compute_roc_curve(y_true, y_prob)
    eer, eer_idx = compute_equal_error_rate(fpr, tpr)
    ece = compute_expected_calibration_error(y_true, y_prob)

    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5
    if not np.isfinite(optimal_threshold) or optimal_threshold <= 0.0 or optimal_threshold >= 1.0:
        optimal_threshold = 0.5

    def get_metrics_at_threshold(scores: np.ndarray, targets: np.ndarray, th: float):
        pred = (scores >= th).astype(int)
        y_int = targets.astype(int)
        tp = int(np.sum((pred == 1) & (y_int == 1)))
        tn = int(np.sum((pred == 0) & (y_int == 0)))
        fp = int(np.sum((pred == 1) & (y_int == 0)))
        fn = int(np.sum((pred == 0) & (y_int == 1)))

        acc = float((tp + tn) / max(1, len(targets)))
        prec = float(tp / max(1, tp + fp))
        rec = float(tp / max(1, tp + fn))
        spec = float(tn / max(1, tn + fp))
        fpr_val = float(fp / max(1, fp + tn))
        fnr_val = float(fn / max(1, fn + tp))
        f1 = float(2 * prec * rec / max(1e-7, prec + rec))
        return {
            "threshold": float(th),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(spec),
            "false_positive_rate": float(fpr_val),
            "false_negative_rate": float(fnr_val),
            "f1_score": float(f1),
            "confusion_matrix": {
                "true_positives_real": tp,
                "true_negatives_fake": tn,
                "false_positives_fake_classified_real": fp,
                "false_negatives_real_classified_fake": fn
            }
        }

    crop_standard_eval = get_metrics_at_threshold(y_prob, y_true, 0.5)
    crop_val_calibrated_eval = get_metrics_at_threshold(y_prob, y_true, val_calibrated_threshold)
    crop_oracle_eval = get_metrics_at_threshold(y_prob, y_true, optimal_threshold)

    # 6. Video-Level Metrics (Mean Probability Aggregation)
    video_records: Dict[str, Dict[str, Any]] = {}
    for (s_path, s_label, s_vid, *rest), prob in zip(test_samples, all_probs):
        if s_vid not in video_records:
            video_records[s_vid] = {"target": s_label, "probs": []}
        video_records[s_vid]["probs"].append(prob)

    vids_sorted = sorted(list(video_records.keys()))
    y_vid_true = np.array([video_records[v]["target"] for v in vids_sorted], dtype=np.float32)
    y_vid_prob = np.array([float(np.mean(video_records[v]["probs"])) for v in vids_sorted], dtype=np.float32)

    fpr_vid, tpr_vid, th_vid, auc_vid = compute_roc_curve(y_vid_true, y_vid_prob)
    eer_vid, _ = compute_equal_error_rate(fpr_vid, tpr_vid)
    j_vid = tpr_vid - fpr_vid
    opt_vid_idx = np.argmax(j_vid)
    opt_th_vid = float(th_vid[opt_vid_idx]) if opt_vid_idx < len(th_vid) else 0.5
    if not np.isfinite(opt_th_vid) or opt_th_vid <= 0.0 or opt_th_vid >= 1.0:
        opt_th_vid = 0.5

    video_standard_eval = get_metrics_at_threshold(y_vid_prob, y_vid_true, 0.5)
    video_val_calibrated_eval = get_metrics_at_threshold(y_vid_prob, y_vid_true, val_video_threshold)
    video_oracle_eval = get_metrics_at_threshold(y_vid_prob, y_vid_true, opt_th_vid)

    # 7. Experiment Classification
    if smoke_test_override is not None:
        is_smoke = smoke_test_override
    else:
        is_smoke = (
            "synthetic" in str(checkpoint_path).lower()
            or "synthetic" in str(test_manifest_path).lower()
            or len(y_true) < 50
        )

    experiment_type = "SMOKE_TEST_SYNTHETIC" if is_smoke else "BENCHMARK_EVALUATION"

    results = {
        "status": "EVALUATION_SUCCESS",
        "experiment_type": experiment_type,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": model_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "test_manifest_path": str(test_manifest_path),
        "total_test_samples": len(y_true),
        "real_samples": int(np.sum(y_true == 1)),
        "fake_samples": int(np.sum(y_true == 0)),
        "total_test_videos": len(vids_sorted),
        "real_videos": int(np.sum(y_vid_true == 1)),
        "fake_videos": int(np.sum(y_vid_true == 0)),
        "parameter_counts": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        },
        "performance": {
            "latency_ms_per_sample": round(latency_ms_per_sample, 2),
            "device": str(device)
        },
        "crop_level": {
            "roc_auc": float(auc_roc),
            "equal_error_rate": float(eer),
            "expected_calibration_error": float(ece),
            "calibration_source": calibration_source,
            "optimal_threshold_youden": float(val_calibrated_threshold),
            "validation_calibrated_threshold": float(val_calibrated_threshold),
            "oracle_test_youden_threshold_uncalibrated": float(optimal_threshold),
            "metrics_at_threshold_0_5": crop_standard_eval,
            "metrics_at_optimal_threshold": crop_val_calibrated_eval,
            "metrics_at_oracle_test_threshold": crop_oracle_eval
        },
        "video_level": {
            "roc_auc": float(auc_vid),
            "equal_error_rate": float(eer_vid),
            "calibration_source": calibration_source,
            "optimal_threshold_youden": float(val_video_threshold),
            "validation_calibrated_threshold": float(val_video_threshold),
            "oracle_test_youden_threshold_uncalibrated": float(opt_th_vid),
            "metrics_at_threshold_0_5": video_standard_eval,
            "metrics_at_optimal_threshold": video_val_calibrated_eval,
            "metrics_at_oracle_test_threshold": video_oracle_eval
        },
        # Top-level backward compatibility aliases
        "roc_auc": float(auc_roc),
        "equal_error_rate": float(eer),
        "expected_calibration_error": float(ece),
        "optimal_threshold_youden": float(val_calibrated_threshold),
        "metrics_at_threshold_0_5": crop_standard_eval,
        "metrics_at_optimal_threshold": crop_val_calibrated_eval
    }

    # Save JSON summary
    summary_filename = f"evaluation_summary_{model_name.lower().replace('-', '_')}.json"
    summary_path = output_dir / summary_filename
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown report
    if is_smoke:
        banner = """
> [!CAUTION]
> **SMOKE TEST PIPELINE VERIFICATION ONLY — NOT A REAL BENCHMARK**
> This evaluation was run on a synthetic verification dataset or low-sample fixture to verify code paths, split grouping, and pipeline execution.
> An Expected Calibration Error (ECE) of ~0.0036 is a mathematical consequence of unconfident predictions (~0.5036) on balanced synthetic noise, NOT a trained deepfake detector.
> Do NOT cite these metrics as real deepfake detection performance. Genuine benchmark numbers require training and testing on datasets such as SDFVD, FaceForensics++, or Celeb-DF.
"""
    else:
        banner = """
> [!NOTE]
> **REAL DATASET BENCHMARK EVALUATION (PILOT EXPERIMENT)**
> This evaluation was executed against an independent held-out test split of real facial media.
> All metrics reflect genuine experimental evaluation on the documented benchmark partition.
> Note: SDFVD is a small pilot dataset (106 videos); results demonstrate pipeline feasibility but may not generalize to unseen generative models.
"""

    report_filename = f"evaluation_report_{model_name.lower().replace('-', '_')}.md"
    report_path = output_dir / report_filename
    report_md = f"""# VeriDoc Deepfake Detection Model — Evaluation Report
{banner}
**Evaluation Timestamp:** {results['timestamp']}  
**Experiment Type:** `{experiment_type}`  
**Model Architecture:** {model_name} (PyTorch)  
**Checkpoint Path:** `{checkpoint_path.name}`  
**Checkpoint SHA-256:** `{checkpoint_sha256}`  
**Test Manifest:** `{test_manifest_path.name}`  
**Test Set Size (Crops):** {results['total_test_samples']} facial crops ({results['real_samples']} REAL, {results['fake_samples']} FAKE)  
**Test Set Size (Videos):** {results['total_test_videos']} videos ({results['real_videos']} REAL, {results['fake_videos']} FAKE)  
**Inference Latency:** {results['performance']['latency_ms_per_sample']} ms/crop ({device})  
**Parameters:** {total_params:,} total  

---

## 1. Crop-Level (Frame-Level) Performance Metrics

| Metric | Threshold = 0.50 | Val-Calibrated Threshold ({val_calibrated_threshold:.4f}) |
|---|---|---|
| **Accuracy** | {crop_standard_eval['accuracy'] * 100:.2f}% | {crop_val_calibrated_eval['accuracy'] * 100:.2f}% |
| **ROC-AUC** | **{auc_roc:.4f}** | **{auc_roc:.4f}** |
| **Equal Error Rate (EER)** | **{eer * 100:.2f}%** | **{eer * 100:.2f}%** |
| **F1 Score** | {crop_standard_eval['f1_score']:.4f} | {crop_val_calibrated_eval['f1_score']:.4f} |
| **Precision (Authentic)** | {crop_standard_eval['precision']:.4f} | {crop_val_calibrated_eval['precision']:.4f} |
| **Recall / Sensitivity** | {crop_standard_eval['recall']:.4f} | {crop_val_calibrated_eval['recall']:.4f} |
| **Specificity (True Negative)** | {crop_standard_eval['specificity']:.4f} | {crop_val_calibrated_eval['specificity']:.4f} |
| **False Positive Rate (FPR)** | {crop_standard_eval['false_positive_rate'] * 100:.2f}% | {crop_val_calibrated_eval['false_positive_rate'] * 100:.2f}% |
| **False Negative Rate (FNR)** | {crop_standard_eval['false_negative_rate'] * 100:.2f}% | {crop_val_calibrated_eval['false_negative_rate'] * 100:.2f}% |
| **Expected Calibration Error** | {ece:.4f} | {ece:.4f} |

---

## 2. Video-Level (Aggregated) Performance Metrics

| Metric | Threshold = 0.50 | Val-Calibrated Threshold ({val_video_threshold:.4f}) |
|---|---|---|
| **Video Accuracy** | {video_standard_eval['accuracy'] * 100:.2f}% | {video_val_calibrated_eval['accuracy'] * 100:.2f}% |
| **Video ROC-AUC** | **{auc_vid:.4f}** | **{auc_vid:.4f}** |
| **Video EER** | **{eer_vid * 100:.2f}%** | **{eer_vid * 100:.2f}%** |
| **Video F1 Score** | {video_standard_eval['f1_score']:.4f} | {video_val_calibrated_eval['f1_score']:.4f} |
| **Video Precision** | {video_standard_eval['precision']:.4f} | {video_val_calibrated_eval['precision']:.4f} |
| **Video Recall** | {video_standard_eval['recall']:.4f} | {video_val_calibrated_eval['recall']:.4f} |

---

## 3. Confusion Matrix (Crop-Level, Threshold = 0.50)

| Ground Truth \\ Prediction | Classified FAKE | Classified REAL |
|---|---|---|
| **Actual FAKE** (Manipulated) | **{crop_standard_eval['confusion_matrix']['true_negatives_fake']} (TN)** | {crop_standard_eval['confusion_matrix']['false_positives_fake_classified_real']} (FP) |
| **Actual REAL** (Authentic) | {crop_standard_eval['confusion_matrix']['false_negatives_real_classified_fake']} (FN) | **{crop_standard_eval['confusion_matrix']['true_positives_real']} (TP)** |

---

## 4. Scientific Notes & Limitations

- **Label Semantics:** Target class 1 denotes authentic media (REAL); target class 0 denotes manipulated media (FAKE).
- **Video Independence:** The test split was generated by grouping frames strictly by source video ID and pair identity to prevent leakage.
- **Aggregation:** Video-level probabilities represent mean confidence across sampled facial crops.
- **Pilot Dataset Notice:** With {results['total_test_videos']} test videos, discrete step increments are {1.0/max(1, results['real_videos']):.3f}. Larger multi-method benchmarks are recommended for generalizability.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    with open(output_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[EVALUATION] Summary saved to: {summary_path}")
    print(f"[EVALUATION] Markdown report saved to: {report_path}")
    return results


def resolve_io_path(p_str: str, default_subpath: str) -> Path:
    """Safely resolves path regardless of whether invoked from root or backend/."""
    if not p_str:
        return (backend_dir / default_subpath).resolve()
    p = Path(p_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    try:
        rel = p.relative_to("backend")
        return (backend_dir / rel).resolve()
    except ValueError:
        pass
    if (backend_dir / p).exists():
        return (backend_dir / p).resolve()
    return (backend_dir / p).resolve()


def main():
    parser = argparse.ArgumentParser(description="VeriDoc Deepfake Model Scientific Evaluation")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint weights (.pth or .pt)")
    parser.add_argument("--manifest", type=str, default="", help="Path to test split manifest (.json or .csv)")
    parser.add_argument("--val-manifest", type=str, default="", help="Path to validation split manifest for leakage-free threshold calibration")
    parser.add_argument("--base-dir", type=str, default="", help="Base directory for relative paths in manifest")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "meso_inception", "dual_stream"],
                        help="Model architecture ('auto', 'meso_inception', 'dual_stream')")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for evaluation artifacts")
    parser.add_argument("--smoke-test", action="store_true", help="Explicitly label as smoke test verification run")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', or 'auto')")

    args = parser.parse_args()

    checkpoint_path = resolve_io_path(args.checkpoint, "training/weights/meso_inception_best.pth")
    manifest_path = resolve_io_path(args.manifest, "training/weights/test_split_manifest.json")
    val_manifest_path = resolve_io_path(args.val_manifest, "") if args.val_manifest else None
    output_dir = resolve_io_path(args.output_dir, "training/eval_results")
    base_dir = resolve_io_path(args.base_dir, "") if args.base_dir else None

    smoke_override = True if args.smoke_test else None

    results = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        test_manifest_path=manifest_path,
        val_manifest_path=val_manifest_path,
        output_dir=output_dir,
        base_dir=base_dir,
        model_override=args.model,
        device_str=args.device,
        smoke_test_override=smoke_override
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

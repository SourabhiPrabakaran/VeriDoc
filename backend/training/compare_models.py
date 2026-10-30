import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from training.evaluate import evaluate_checkpoint, resolve_io_path


def format_delta(val1: float, val2: float, higher_is_better: bool = True, is_pct: bool = False) -> str:
    """Formats the delta between Model 2 (Dual-Stream) and Model 1 (Baseline)."""
    diff = val2 - val1
    sign = "+" if diff > 0 else ""
    suffix = "%" if is_pct else ""
    fmt_diff = f"{sign}{diff * 100:.2f}%" if is_pct else f"{sign}{diff:.4f}"
    return fmt_diff


def generate_comparison(
    baseline_summary: Dict[str, Any],
    dualstream_summary: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Generates a structured comparison between baseline MesoInception-4 and DualStreamForensicNet.
    Outputs comparison summary JSON and formatted Markdown report.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    b_std = baseline_summary.get("metrics_at_threshold_0_5", {})
    d_std = dualstream_summary.get("metrics_at_threshold_0_5", {})

    b_opt = baseline_summary.get("metrics_at_optimal_threshold", {})
    d_opt = dualstream_summary.get("metrics_at_optimal_threshold", {})

    b_params = baseline_summary.get("parameter_counts", {})
    d_params = dualstream_summary.get("parameter_counts", {})

    b_perf = baseline_summary.get("performance", {})
    d_perf = dualstream_summary.get("performance", {})

    is_smoke = (
        baseline_summary.get("experiment_type") == "SMOKE_TEST_SYNTHETIC"
        or dualstream_summary.get("experiment_type") == "SMOKE_TEST_SYNTHETIC"
    )

    comparison_data = {
        "experiment_type": "SMOKE_TEST_SYNTHETIC" if is_smoke else "BENCHMARK_COMPARISON",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_manifest": baseline_summary.get("test_manifest_path"),
        "total_test_samples": baseline_summary.get("total_test_samples"),
        "baseline": {
            "model_name": baseline_summary.get("model_architecture", "MesoInception-4"),
            "checkpoint_sha256": baseline_summary.get("checkpoint_sha256"),
            "parameters": b_params.get("total_parameters"),
            "latency_ms": b_perf.get("latency_ms_per_sample"),
            "accuracy_0_5": b_std.get("accuracy"),
            "accuracy_optimal": b_opt.get("accuracy"),
            "optimal_threshold": baseline_summary.get("optimal_threshold_youden"),
            "roc_auc": baseline_summary.get("roc_auc"),
            "eer": baseline_summary.get("equal_error_rate"),
            "ece": baseline_summary.get("expected_calibration_error"),
            "f1_score": b_std.get("f1_score"),
            "precision": b_std.get("precision"),
            "recall": b_std.get("recall"),
            "specificity": b_std.get("specificity")
        },
        "dual_stream": {
            "model_name": dualstream_summary.get("model_architecture", "DualStreamForensicNet"),
            "checkpoint_sha256": dualstream_summary.get("checkpoint_sha256"),
            "parameters": d_params.get("total_parameters"),
            "latency_ms": d_perf.get("latency_ms_per_sample"),
            "accuracy_0_5": d_std.get("accuracy"),
            "accuracy_optimal": d_opt.get("accuracy"),
            "optimal_threshold": dualstream_summary.get("optimal_threshold_youden"),
            "roc_auc": dualstream_summary.get("roc_auc"),
            "eer": dualstream_summary.get("equal_error_rate"),
            "ece": dualstream_summary.get("expected_calibration_error"),
            "f1_score": d_std.get("f1_score"),
            "precision": d_std.get("precision"),
            "recall": d_std.get("recall"),
            "specificity": d_std.get("specificity")
        }
    }

    # Save JSON summary
    json_path = output_dir / "model_comparison_table.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    # Build Markdown table
    b = comparison_data["baseline"]
    d = comparison_data["dual_stream"]

    if is_smoke:
        banner = """
> [!CAUTION]
> **SMOKE TEST PIPELINE VERIFICATION ONLY — NOT A REAL BENCHMARK**
> This comparison was generated from synthetic test fixtures to verify multi-model evaluation pipelines.
> Do NOT interpret these metrics as experimental evidence of real deepfake detection performance.
"""
    else:
        banner = """
> [!NOTE]
> **CONTROLLED BENCHMARK COMPARISON**
> Both models were evaluated on the identical held-out test split under identical preprocessing conditions.
"""

    md_report = f"""# VeriDoc Deep Learning Model Comparison
{banner}
**Comparison Timestamp:** {comparison_data['timestamp']}  
**Test Manifest:** `{Path(str(comparison_data['test_manifest'])).name}` ({comparison_data['total_test_samples']} facial crops)  

---

## 1. Architectural and Complexity Comparison

| Dimension | Baseline ({b['model_name']}) | Proposed ({d['model_name']}) | Delta (Dual-Stream vs Baseline) |
|---|---|---|---|
| **Input Streams** | 1 (Spatial RGB) | 2 (Spatial RGB + 2D FFT Spectrum) | +1 Spectral Stream |
| **Total Parameters** | {b['parameters']:,} | {d['parameters']:,} | {format_delta(b['parameters'], d['parameters'])} |
| **Inference Latency** | {b['latency_ms']} ms/crop | {d['latency_ms']} ms/crop | {format_delta(b['latency_ms'], d['latency_ms'])} ms |
| **Checkpoint SHA-256** | `{b['checkpoint_sha256'][:16]}...` | `{d['checkpoint_sha256'][:16]}...` | N/A |

---

## 2. Forensic Classification Performance

| Forensic Metric | Baseline ({b['model_name']}) | Proposed ({d['model_name']}) | Absolute Delta ($\Delta$) |
|---|---|---|---|
| **Accuracy (th = 0.50)** | {b['accuracy_0_5'] * 100:.2f}% | {d['accuracy_0_5'] * 100:.2f}% | **{format_delta(b['accuracy_0_5'], d['accuracy_0_5'], is_pct=True)}** |
| **Accuracy (Val-Calibrated th)** | {b['accuracy_optimal'] * 100:.2f}% | {d['accuracy_optimal'] * 100:.2f}% | **{format_delta(b['accuracy_optimal'], d['accuracy_optimal'], is_pct=True)}** |
| **ROC-AUC** | **{b['roc_auc']:.4f}** | **{d['roc_auc']:.4f}** | **{format_delta(b['roc_auc'], d['roc_auc'])}** |
| **Equal Error Rate (EER)** | **{b['eer'] * 100:.2f}%** | **{d['eer'] * 100:.2f}%** | **{format_delta(b['eer'], d['eer'], higher_is_better=False, is_pct=True)}** |
| **Calibration Error (ECE)** | {b['ece']:.4f} | {d['ece']:.4f} | {format_delta(b['ece'], d['ece'], higher_is_better=False)} |
| **F1 Score** | {b['f1_score']:.4f} | {d['f1_score']:.4f} | {format_delta(b['f1_score'], d['f1_score'])} |
| **Precision (Authentic)** | {b['precision']:.4f} | {d['precision']:.4f} | {format_delta(b['precision'], d['precision'])} |
| **Recall / Sensitivity** | {b['recall']:.4f} | {d['recall']:.4f} | {format_delta(b['recall'], d['recall'])} |
| **Specificity (True Neg)** | {b['specificity']:.4f} | {d['specificity']:.4f} | {format_delta(b['specificity'], d['specificity'])} |
| **Val-Calibrated Threshold** | {b['optimal_threshold']:.4f} | {d['optimal_threshold']:.4f} | N/A |

---

## 3. Scientific Discussion and Forensic Insights

1. **Multimodal Complementarity:** MesoInception-4 relies purely on spatial pixel domain correlations, making it susceptible to subtle blending boundaries. DualStreamForensicNet captures 2D Fourier magnitude spectrum anomalies, providing complementary frequency-domain signals.
2. **Computational Overhead:** The frequency branch and cross-modal fusion add {d['parameters'] - b['parameters']:,} parameters and ~{d['latency_ms'] - b['latency_ms']:.2f} ms latency per crop.
3. **Threshold Calibration:** Neither model should be deployed in forensic casework using a naive $0.50$ cutoff without empirical threshold calibration on the validation split.
"""

    md_path = output_dir / "model_comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"[COMPARISON] Comparison summary saved to: {json_path}")
    print(f"[COMPARISON] Markdown report saved to: {md_path}")
    print(md_report)
    return comparison_data


def main():
    parser = argparse.ArgumentParser(description="VeriDoc Multi-Model Comparison Utility")
    parser.add_argument("--baseline-checkpoint", type=str, default="", help="Path to baseline MesoInception-4 checkpoint")
    parser.add_argument("--dualstream-checkpoint", type=str, default="", help="Path to DualStreamForensicNet checkpoint")
    parser.add_argument("--baseline-eval", type=str, default="", help="Path to precomputed baseline evaluation summary JSON")
    parser.add_argument("--dualstream-eval", type=str, default="", help="Path to precomputed dual-stream evaluation summary JSON")
    parser.add_argument("--manifest", type=str, default="", help="Test split manifest for live evaluation")
    parser.add_argument("--base-dir", type=str, default="", help="Base directory for relative paths in manifest")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for comparison outputs")
    parser.add_argument("--smoke-test", action="store_true", help="Explicitly label as smoke test verification run")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', or 'auto')")

    args = parser.parse_args()

    out_dir = resolve_io_path(args.output_dir, "training/eval_results")
    manifest_p = resolve_io_path(args.manifest, "training/weights/test_split_manifest.json")
    base_dir_p = resolve_io_path(args.base_dir, "") if args.base_dir else None

    # Load or compute baseline summary
    if args.baseline_eval and Path(args.baseline_eval).exists():
        with open(args.baseline_eval, "r", encoding="utf-8") as f:
            baseline_summary = json.load(f)
    else:
        b_chk = resolve_io_path(args.baseline_checkpoint, "training/weights/meso_inception_best.pth")
        print(f"[COMPARISON] Evaluating baseline checkpoint: {b_chk}")
        baseline_summary = evaluate_checkpoint(
            checkpoint_path=b_chk,
            test_manifest_path=manifest_p,
            output_dir=out_dir,
            base_dir=base_dir_p,
            model_override="meso_inception",
            device_str=args.device,
            smoke_test_override=True if args.smoke_test else None
        )

    # Load or compute dual-stream summary
    if args.dualstream_eval and Path(args.dualstream_eval).exists():
        with open(args.dualstream_eval, "r", encoding="utf-8") as f:
            dualstream_summary = json.load(f)
    else:
        d_chk = resolve_io_path(args.dualstream_checkpoint, "training/weights/dual_stream_best.pth")
        print(f"[COMPARISON] Evaluating dual-stream checkpoint: {d_chk}")
        dualstream_summary = evaluate_checkpoint(
            checkpoint_path=d_chk,
            test_manifest_path=manifest_p,
            output_dir=out_dir,
            base_dir=base_dir_p,
            model_override="dual_stream",
            device_str=args.device,
            smoke_test_override=True if args.smoke_test else None
        )

    generate_comparison(baseline_summary, dualstream_summary, out_dir)


if __name__ == "__main__":
    main()

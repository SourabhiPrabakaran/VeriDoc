import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add backend directory to path so app modules can be imported
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.deepfake_model_interface import MesoInception4
from training.dual_stream_model import DualStreamForensicNet
from training.dataset import (
    DeepfakeDataset,
    scan_directory_dataset,
    load_dataset_manifest,
    load_dataset_splits_from_manifest,
    create_grouped_splits,
    generate_synthetic_verification_dataset,
    LABEL_REAL,
    LABEL_FAKE
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes basic classification metrics for validation tracking."""
    y_pred = (y_prob >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    tp = np.sum((y_pred == 1) & (y_true_int == 1))
    tn = np.sum((y_pred == 0) & (y_true_int == 0))
    fp = np.sum((y_pred == 1) & (y_true_int == 0))
    fn = np.sum((y_pred == 0) & (y_true_int == 1))

    accuracy = float((tp + tn) / max(1, len(y_true)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2 * precision * recall / max(1e-7, precision + recall))

    # Fast AUC calculation via Wilcoxon-Mann-Whitney U-statistic
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        ranks = np.arange(len(y_prob), 0, -1)
        sum_pos_ranks = np.sum(ranks[y_true_sorted == 1])
        auc = float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs, return_logits=True)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_targets = []
    all_probs = []

    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs, return_logits=True)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_batches += 1

            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            targets = labels.squeeze(-1).cpu().numpy()

            all_probs.extend(np.atleast_1d(probs))
            all_targets.extend(np.atleast_1d(targets))

    avg_loss = total_loss / max(1, total_batches)
    metrics = compute_binary_metrics(np.array(all_targets), np.array(all_probs))
    return avg_loss, metrics


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


def run_training(
    data_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
    model_name: str = "meso_inception",
    group_by: str = "video",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    dropout_rate: float = 0.5,
    selection_metric: str = "val_loss",
    patience: int = 4,
    seed: int = 42,
    device_str: str = "auto",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    smoke_test: bool = False
) -> Dict[str, Any]:
    set_seed(seed)
    out_dir = Path(output_dir or (backend_dir / "training" / "weights"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[TRAINING] Target execution device: {device}")
    print(f"[TRAINING] Model Architecture: {model_name}")

    # 1. Dataset Loading and Splitting
    predefined_splits = None
    if manifest_path and manifest_path.exists():
        print(f"[TRAINING] Loading dataset manifest from: {manifest_path}")
        predefined_splits = load_dataset_splits_from_manifest(manifest_path, base_dir=base_dir, validate_paths=True)
        if predefined_splits:
            train_samples, val_samples, test_samples = predefined_splits
            all_samples = train_samples + val_samples + test_samples
            print(f"[TRAINING] Using pre-defined splits from manifest -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        else:
            all_samples = load_dataset_manifest(manifest_path, base_dir=base_dir, validate_paths=True)
            print(f"[TRAINING] Loaded {len(all_samples)} total samples from manifest. Generating grouped splits (group_by={group_by})...")
            train_samples, val_samples, test_samples = create_grouped_splits(
                all_samples, train_ratio=(1.0 - val_ratio - test_ratio), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed, group_by=group_by
            )
    elif data_dir and Path(data_dir).exists():
        print(f"[TRAINING] Scanning dataset directory: {data_dir}")
        all_samples = scan_directory_dataset(Path(data_dir))
        if not all_samples:
            raise ValueError(f"No valid image samples found under {data_dir}. Expected 'real/' and 'fake/' folders.")
        print(f"[TRAINING] Scanned {len(all_samples)} samples. Generating grouped splits (group_by={group_by})...")
        train_samples, val_samples, test_samples = create_grouped_splits(
            all_samples, train_ratio=(1.0 - val_ratio - test_ratio), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed, group_by=group_by
        )
    else:
        raise ValueError("Either --manifest or a valid --data-dir must be provided for training.")

    real_count = sum(1 for s in all_samples if s[1] == LABEL_REAL)
    fake_count = sum(1 for s in all_samples if s[1] == LABEL_FAKE)
    print(f"[TRAINING] Total samples: {len(all_samples)} (REAL: {real_count}, FAKE: {fake_count})")
    print(f"[TRAINING] Splits -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Save test split manifest for independent evaluation
    test_manifest_json = out_dir / "test_split_manifest.json"
    with open(test_manifest_json, "w", encoding="utf-8") as f:
        json.dump([
            {
                "path": s[0],
                "label": s[1],
                "video_id": s[2],
                "subject_id": s[3] if len(s) > 3 else None
            } for s in test_samples
        ], f, indent=2)

    test_manifest_csv = out_dir / "test_split_manifest.csv"
    with open(test_manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "video_id", "subject_id"])
        for s in test_samples:
            writer.writerow([s[0], s[1], s[2], s[3] if len(s) > 3 else ""])

    # Save validation split manifest for threshold calibration without test set peeking
    val_manifest_json = out_dir / "val_split_manifest.json"
    with open(val_manifest_json, "w", encoding="utf-8") as f:
        json.dump([
            {
                "path": s[0],
                "label": s[1],
                "video_id": s[2],
                "subject_id": s[3] if len(s) > 3 else None
            } for s in val_samples
        ], f, indent=2)

    val_manifest_csv = out_dir / "val_split_manifest.csv"
    with open(val_manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "video_id", "subject_id"])
        for s in val_samples:
            writer.writerow([s[0], s[1], s[2], s[3] if len(s) > 3 else ""])

    train_dataset = DeepfakeDataset(train_samples, transform=DeepfakeDataset.default_train_transforms())
    val_dataset = DeepfakeDataset(val_samples, transform=DeepfakeDataset.default_eval_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model Selection
    is_dual = model_name.lower() in ("dual_stream", "dualstream", "dualstreamforensicnet")
    if is_dual:
        model = DualStreamForensicNet(num_classes=1, dropout_rate=dropout_rate).to(device)
        model_canonical_name = "DualStreamForensicNet"
        weights_filename = "dual_stream_best.pth"
        checkpoint_filename = "dual_stream_checkpoint_best.pt"
    else:
        model = MesoInception4(num_classes=1, dropout_rate=dropout_rate).to(device)
        model_canonical_name = "MesoInception-4"
        weights_filename = "meso_inception_best.pth"
        checkpoint_filename = "training_checkpoint_best.pt"

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1
    best_val_auc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
        "val_f1": [],
        "learning_rate": [],
        "epoch_duration_seconds": []
    }

    # 3. Check for Checkpoint Resumption
    if resume_from and Path(resume_from).exists():
        print(f"[TRAINING] Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_auc = ckpt.get("val_auc", -1.0)
            best_val_loss = ckpt.get("val_loss", float("inf"))
            best_epoch = ckpt.get("epoch", 0)
            print(f"[TRAINING] Successfully resumed from epoch {start_epoch - 1} (Best Val AUC: {best_val_auc:.4f})")

    start_time = time.time()
    print(f"[TRAINING] Beginning training (epochs {start_epoch} to {epochs}) for {model_canonical_name}...")

    best_weights_path = out_dir / weights_filename
    full_checkpoint_path = out_dir / checkpoint_filename
    latest_checkpoint_path = out_dir / f"{model_name}_checkpoint_latest.pt"

    stopping_reason = f"Completed all {epochs} epochs."
    for epoch in range(start_epoch, epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])
        history["learning_rate"].append(current_lr)
        history["epoch_duration_seconds"].append(epoch_duration)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Save latest checkpoint after every epoch (for crash recovery / Colab timeout resilience)
        torch.save({
            "epoch": epoch,
            "model_name": model_canonical_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_auc": val_metrics["auc"],
            "seed": seed,
            "class_distribution": {"real": real_count, "fake": fake_count},
            "split_counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
            "is_smoke_test": smoke_test
        }, latest_checkpoint_path)

        if selection_metric == "val_loss":
            is_best = val_loss < best_val_loss or (
                np.isclose(val_loss, best_val_loss) and val_metrics["auc"] > best_val_auc
            )
        else:
            is_best = val_metrics["auc"] > best_val_auc or (
                np.isclose(val_metrics["auc"], best_val_auc) and val_loss < best_val_loss
            )

        if is_best:
            best_val_auc = val_metrics["auc"]
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # 1. Clean weights-only state dict
            torch.save(model.state_dict(), best_weights_path)
            weights_sha256 = hashlib.sha256(best_weights_path.read_bytes()).hexdigest()

            # 2. Comprehensive training state checkpoint
            torch.save({
                "epoch": epoch,
                "model_name": model_canonical_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_auc": val_metrics["auc"],
                "seed": seed,
                "weights_sha256": weights_sha256,
                "class_distribution": {"real": real_count, "fake": fake_count},
                "split_counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
                "is_smoke_test": smoke_test
            }, full_checkpoint_path)
            print(f"  --> Saved new best {model_canonical_name} checkpoint at epoch {epoch} (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopping_reason = f"Early stopping triggered after {patience} epochs without validation improvement."
                print(f"[TRAINING] {stopping_reason}")
                break

    total_training_time = time.time() - start_time
    print(f"[TRAINING] Training completed in {total_training_time:.1f}s. Best epoch: {best_epoch} (AUC: {best_val_auc:.4f})")

    final_sha256 = hashlib.sha256(best_weights_path.read_bytes()).hexdigest() if best_weights_path.exists() else None

    # Save training history
    history_filename = f"training_history_{model_name}.json"
    history_path = out_dir / history_filename
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_canonical_name,
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "best_val_loss": best_val_loss,
            "total_epochs_run": len(history["train_loss"]),
            "training_time_seconds": total_training_time,
            "stopping_reason": stopping_reason,
            "weights_path": str(best_weights_path.resolve()),
            "weights_sha256": final_sha256,
            "test_manifest_path": str(test_manifest_json.resolve()),
            "is_smoke_test": smoke_test,
            "history": history
        }, f, indent=2)

    return {
        "model_name": model_canonical_name,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_weights_path": str(best_weights_path.resolve()),
        "weights_sha256": final_sha256,
        "test_manifest_path": str(test_manifest_json.resolve())
    }


def main():
    parser = argparse.ArgumentParser(description="VeriDoc Deepfake Model Training Pipeline")
    parser.add_argument("--manifest", type=str, default="", help="Path to CSV or JSON manifest file")
    parser.add_argument("--data-dir", type=str, default="", help="Path to dataset folder containing 'real/' and 'fake/'")
    parser.add_argument("--base-dir", type=str, default="", help="Base directory for relative paths in manifest")
    parser.add_argument("--resume-from", type=str, default="", help="Path to checkpoint file (.pt) to resume training from")
    parser.add_argument("--model", type=str, default="meso_inception", choices=["meso_inception", "dual_stream"],
                        help="Model architecture ('meso_inception' or 'dual_stream')")
    parser.add_argument("--group-by", type=str, default="video", choices=["video", "subject"],
                        help="Splitting grouping key ('video' or 'subject')")
    parser.add_argument("--use-synthetic", action="store_true", help="Generate and train on synthetic dataset for verification")
    parser.add_argument("--smoke-test", action="store_true", help="Explicitly label as smoke test verification run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for checkpoint outputs")
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="Dropout rate in classification head")
    parser.add_argument("--selection-metric", type=str, default="val_loss", choices=["val_loss", "val_auc"],
                        help="Metric used to select best checkpoint ('val_loss' or 'val_auc')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', or 'auto')")

    args = parser.parse_args()

    out_path = resolve_io_path(args.output_dir, "training/weights")
    manifest_p = resolve_io_path(args.manifest, "") if args.manifest else None
    base_dir_p = resolve_io_path(args.base_dir, "") if args.base_dir else None
    resume_p = resolve_io_path(args.resume_from, "") if args.resume_from else None

    is_smoke = args.smoke_test or args.use_synthetic

    if args.use_synthetic or (not manifest_p and not args.data_dir):
        synth_dir = (backend_dir / "training" / "data_synthetic").resolve()
        print(f"[SETUP] Generating synthetic verification dataset in {synth_dir}...")
        generate_synthetic_verification_dataset(synth_dir, num_videos_per_class=6, frames_per_video=8, seed=args.seed)
        data_path = synth_dir
        is_smoke = True
    else:
        data_path = resolve_io_path(args.data_dir, "") if args.data_dir else None

    run_training(
        data_dir=data_path,
        manifest_path=manifest_p,
        base_dir=base_dir_p,
        output_dir=out_path,
        resume_from=resume_p,
        model_name=args.model,
        group_by=args.group_by,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        selection_metric=args.selection_metric,
        patience=args.patience,
        seed=args.seed,
        device_str=args.device,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        smoke_test=is_smoke
    )


if __name__ == "__main__":
    main()

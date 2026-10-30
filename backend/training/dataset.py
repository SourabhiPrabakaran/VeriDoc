import csv
import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

# Class label constants (Canonical Deep Learning definitions)
# Label 1.0 = REAL (Authentic facial media)
# Label 0.0 = FAKE (Deepfake / Manipulated facial media)
LABEL_REAL = 1.0
LABEL_FAKE = 0.0
CLASS_TO_LABEL = {"real": LABEL_REAL, "fake": LABEL_FAKE}
LABEL_TO_CLASS = {LABEL_REAL: "real", LABEL_FAKE: "fake"}

# Accepted string synonyms for dataset labels (case-insensitive)
REAL_LABEL_SYNONYMS = {
    "1", "1.0", "real", "authentic", "original", "pristine", "bonafide", "true"
}
FAKE_LABEL_SYNONYMS = {
    "0", "0.0", "fake", "manipulated", "deepfake", "tampered", "synthesized",
    "df", "f2f", "fs", "nt", "false"
}


def parse_label(val: Any) -> float:
    """
    Normalizes a dataset label value into either 1.0 (REAL) or 0.0 (FAKE).
    Raises ValueError with a clear diagnostic message if the label is unrecognized.
    """
    if isinstance(val, (int, float)):
        if np.isclose(val, 1.0):
            return LABEL_REAL
        if np.isclose(val, 0.0):
            return LABEL_FAKE
        raise ValueError(
            f"Invalid numeric dataset label: {val}. Expected 1.0 (REAL) or 0.0 (FAKE)."
        )

    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in REAL_LABEL_SYNONYMS:
            return LABEL_REAL
        if cleaned in FAKE_LABEL_SYNONYMS:
            return LABEL_FAKE
        raise ValueError(
            f"Invalid string dataset label: '{val}'. Expected one of {REAL_LABEL_SYNONYMS} "
            f"for REAL or {FAKE_LABEL_SYNONYMS} for FAKE."
        )

    raise ValueError(f"Unsupported label type {type(val)} for value: {val}")


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for Deepfake Detection (FaceForensics++, Celeb-DF, DFDC, etc.).
    Loads facial crops, applies normalization and augmentation, and returns (tensor, label).
    Supports tuples of (path, label, video_id) or (path, label, video_id, subject_id).
    """
    def __init__(
        self,
        samples: List[Tuple],
        transform: Optional[Callable] = None,
        image_size: int = 256
    ):
        self.samples = samples
        self.image_size = image_size
        self.transform = transform or self.default_eval_transforms(image_size)

    @staticmethod
    def default_train_transforms(image_size: int = 256) -> T.Compose:
        """Data augmentation pipeline for training."""
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),  # Converts PIL image [0, 255] to Tensor [0.0, 1.0]
        ])

    @staticmethod
    def default_eval_transforms(image_size: int = 256) -> T.Compose:
        """Deterministic preprocessing for validation and test evaluation."""
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_path = sample[0]
        label = sample[1]
        path_obj = Path(img_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset sample not found: {img_path}")

        try:
            with Image.open(str(path_obj)) as pil_img:
                image_rgb = pil_img.convert("RGB")
                tensor = self.transform(image_rgb)
        except Exception as e:
            raise IOError(f"Failed to read image at {img_path}: {e}")

        label_tensor = torch.tensor([float(label)], dtype=torch.float32)
        return tensor, label_tensor


def _extract_column(row: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
    """Finds the first matching key in a dictionary ignoring case and whitespace."""
    row_lower = {k.strip().lower(): v for k, v in row.items()}
    for c in candidates:
        if c.lower() in row_lower:
            val = row_lower[c.lower()]
            if val is not None and str(val).strip() != "":
                return val
    return None


def parse_manifest_sample(
    raw_item: Dict[str, Any],
    base_dir: Path
) -> Tuple[str, float, str, Optional[str], Optional[str]]:
    """
    Parses a single manifest record from CSV or JSON.
    Returns: (resolved_path, float_label, video_id, subject_id, split)
    """
    # 1. Path column candidates
    path_val = _extract_column(raw_item, ["path", "image_path", "filepath", "file_path", "filename", "file", "image"])
    if not path_val:
        raise ValueError(f"Manifest row is missing an image path column. Keys found: {list(raw_item.keys())}")

    path_obj = Path(str(path_val).strip())
    resolved_path = (base_dir / path_obj).resolve() if not path_obj.is_absolute() else path_obj.resolve()

    # 2. Label column candidates
    label_val = _extract_column(raw_item, ["label", "target", "class", "is_real", "ground_truth", "category"])
    if label_val is None:
        raise ValueError(f"Manifest row is missing a label column for path '{path_val}'. Keys found: {list(raw_item.keys())}")
    float_label = parse_label(label_val)

    # 3. Video ID candidates (defaults to stem prefix if not provided)
    vid_val = _extract_column(raw_item, ["video_id", "video", "source_video", "orig_video", "vid", "video_name"])
    if vid_val:
        video_id = str(vid_val).strip()
    else:
        stem = resolved_path.stem
        video_id = stem.split("_frame")[0] if "_frame" in stem else stem.split("_")[0]

    # 4. Optional Subject / Identity ID
    subj_val = _extract_column(raw_item, ["subject_id", "subject", "identity", "person_id", "actor_id", "id"])
    subject_id = str(subj_val).strip() if subj_val else None

    # 5. Optional Split partition (train, val, test)
    split_val = _extract_column(raw_item, ["split", "subset", "partition", "set"])
    split_name = str(split_val).strip().lower() if split_val else None

    return (str(resolved_path), float_label, video_id, subject_id, split_name)


def load_dataset_manifest(
    manifest_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    validate_paths: bool = True
) -> List[Tuple[str, float, str, Optional[str]]]:
    """
    Loads samples from a CSV or JSON manifest file.
    Returns: List of (image_path, float_label, video_id, subject_id) tuples.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    root_dir = Path(base_dir).resolve() if base_dir else manifest_path.parent
    raw_records: List[Dict[str, Any]] = []

    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            raw_records = [row for row in reader if any(v.strip() for v in row.values())]
    elif suffix in (".json", ".jsonl"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, list):
                raw_records = content
            elif isinstance(content, dict):
                if "samples" in content:
                    raw_records = content["samples"]
                elif "data" in content:
                    raw_records = content["data"]
                else:
                    # Flatten split keys if present {"train": [...], "val": [...]}
                    for k, v in content.items():
                        if isinstance(v, list):
                            for item in v:
                                if isinstance(item, dict):
                                    item.setdefault("split", k)
                                    raw_records.append(item)
    else:
        raise ValueError(f"Unsupported manifest format '{suffix}'. Expected .csv or .json.")

    if not raw_records:
        raise ValueError(f"Manifest file '{manifest_path}' contains 0 records.")

    samples = []
    missing_paths = []

    for idx, item in enumerate(raw_records):
        try:
            p, l, vid, sid, _ = parse_manifest_sample(item, root_dir)
        except Exception as e:
            raise ValueError(f"Error parsing manifest row {idx}: {e}")

        if validate_paths and not Path(p).exists():
            missing_paths.append(p)

        samples.append((p, l, vid, sid))

    if missing_paths:
        first_few = missing_paths[:5]
        raise FileNotFoundError(
            f"Manifest validation failed: {len(missing_paths)} image files do not exist on disk. "
            f"First missing paths: {first_few}"
        )

    return samples


def load_dataset_splits_from_manifest(
    manifest_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
    validate_paths: bool = True
) -> Optional[Tuple[List, List, List]]:
    """
    Checks whether a manifest contains predefined splits ('train', 'val', 'test').
    If yes, returns (train_samples, val_samples, test_samples).
    If no pre-defined splits are present, returns None.
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    root_dir = Path(base_dir).resolve() if base_dir else manifest_path.parent
    raw_records: List[Dict[str, Any]] = []

    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            raw_records = [row for row in reader if any(v.strip() for v in row.values())]
    elif suffix == ".json":
        with open(manifest_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, dict) and any(k in content for k in ("train", "val", "validation", "test")):
                train_raw = content.get("train", [])
                val_raw = content.get("val", content.get("validation", []))
                test_raw = content.get("test", [])

                def parse_split(records):
                    res = []
                    for r in records:
                        p, l, vid, sid, _ = parse_manifest_sample(r, root_dir)
                        if validate_paths and not Path(p).exists():
                            raise FileNotFoundError(f"File not found: {p}")
                        res.append((p, l, vid, sid))
                    return res

                return parse_split(train_raw), parse_split(val_raw), parse_split(test_raw)
            elif isinstance(content, list):
                raw_records = content
            elif isinstance(content, dict) and "samples" in content:
                raw_records = content["samples"]

    train_samples, val_samples, test_samples = [], [], []
    has_split_column = False

    for item in raw_records:
        p, l, vid, sid, split_name = parse_manifest_sample(item, root_dir)
        if validate_paths and not Path(p).exists():
            raise FileNotFoundError(f"File not found: {p}")

        if split_name:
            has_split_column = True
            if split_name in ("train", "training"):
                train_samples.append((p, l, vid, sid))
            elif split_name in ("val", "validation", "dev"):
                val_samples.append((p, l, vid, sid))
            elif split_name in ("test", "testing", "eval"):
                test_samples.append((p, l, vid, sid))

    if has_split_column:
        return train_samples, val_samples, test_samples

    return None


def scan_directory_dataset(data_dir: Path) -> List[Tuple[str, float, str, Optional[str]]]:
    """
    Scans a directory expecting subdirectories named 'real' and 'fake'.
    Returns list of (file_path, label, video_id, subject_id).
    """
    samples = []
    data_dir = Path(data_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for class_name, label in CLASS_TO_LABEL.items():
        class_folder = data_dir / class_name
        if class_folder.exists() and class_folder.is_dir():
            for file_path in class_folder.rglob("*"):
                if file_path.suffix.lower() in valid_exts and file_path.is_file():
                    stem = file_path.stem
                    video_id = stem.split("_frame")[0] if "_frame" in stem else stem.split("_")[0]
                    samples.append((str(file_path.resolve()), label, video_id, None))

    return samples


def create_grouped_splits(
    samples: List[Tuple],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_by: str = "video"
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Splits samples into train, validation, and test sets grouped by video_id or subject_id.

    GUARANTEE:
    All frames originating from the same group key are strictly kept in the same split.
    This prevents data leakage across training, validation, and test sets.

    LIMITATION NOTE ON IDENTITY-LEVEL SEPARATION:
    If group_by="video" or if metadata does not specify subject/actor identities, grouping
    is performed strictly by video_id. While this mathematically prevents temporal and frame-level
    leakage from the same video clip, cross-video identity leakage is still possible if the same
    actor appears in multiple independent source videos without explicit subject_id annotations.
    When subject_id is available, set group_by="subject" to enforce identity-level separation.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Ratios must sum to 1.0.")

    rng = random.Random(seed)

    # Group by key and label
    group_to_samples: Dict[str, List[Tuple]] = {}
    group_labels: Dict[str, float] = {}

    for sample in samples:
        path = sample[0]
        label = sample[1]
        vid = sample[2]
        sid = sample[3] if len(sample) > 3 else None

        if group_by == "subject" and sid:
            group_key = f"subj_{sid}"
        else:
            group_key = f"vid_{vid}"

        group_to_samples.setdefault(group_key, []).append(sample)
        group_labels[group_key] = label

    train_samples: List[Tuple] = []
    val_samples: List[Tuple] = []
    test_samples: List[Tuple] = []

    for target_label in (LABEL_REAL, LABEL_FAKE):
        class_groups = [g for g, l in group_labels.items() if l == target_label]
        rng.shuffle(class_groups)

        n_total = len(class_groups)
        if n_total >= 3:
            n_val = max(1, int(round(n_total * val_ratio)))
            n_test = max(1, int(round(n_total * test_ratio)))
            n_train = max(1, n_total - n_val - n_test)
            if n_train + n_val + n_test > n_total:
                n_train = max(1, n_total - 2)
                n_val = 1
                n_test = 1
        else:
            n_train = n_total
            n_val = 0
            n_test = 0

        train_keys = set(class_groups[:n_train])
        val_keys = set(class_groups[n_train:n_train + n_val])
        test_keys = set(class_groups[n_train + n_val:n_train + n_val + n_test])

        for k in train_keys:
            train_samples.extend(group_to_samples[k])
        for k in val_keys:
            val_samples.extend(group_to_samples[k])
        for k in test_keys:
            test_samples.extend(group_to_samples[k])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    return train_samples, val_samples, test_samples


def generate_synthetic_verification_dataset(
    output_dir: Path,
    num_videos_per_class: int = 6,
    frames_per_video: int = 8,
    seed: int = 42
) -> Path:
    """
    Generates a reproducible, self-contained synthetic image dataset for smoke-testing
    the training, validation, and evaluation pipeline without requiring external downloads.
    """
    rng = random.Random(seed)
    output_dir = Path(output_dir)

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for v in range(num_videos_per_class):
            video_id = f"{class_name}_vid{v:03d}"
            base_color = (
                rng.randint(120, 200),
                rng.randint(100, 180),
                rng.randint(80, 160)
            ) if label == LABEL_REAL else (
                rng.randint(60, 140),
                rng.randint(120, 200),
                rng.randint(140, 220)
            )

            for f in range(frames_per_video):
                filename = f"{video_id}_frame{f:03d}.jpg"
                file_path = class_dir / filename

                img = np.zeros((256, 256, 3), dtype=np.uint8)
                img[:] = base_color

                center = (128 + rng.randint(-5, 5), 128 + rng.randint(-5, 5))
                cv2.ellipse(img, center, (70, 95), 0, 0, 360, (210, 180, 160), -1)
                cv2.circle(img, (center[0] - 25, center[1] - 20), 8, (40, 40, 40), -1)
                cv2.circle(img, (center[0] + 25, center[1] - 20), 8, (40, 40, 40), -1)
                cv2.ellipse(img, (center[0], center[1] + 35), (20, 8), 0, 0, 180, (50, 50, 180), -1)

                if label == LABEL_FAKE:
                    cv2.rectangle(img, (center[0] - 35, center[1] - 30),
                                  (center[0] + 35, center[1] + 45), (180, 120, 90), 1)

                cv2.imwrite(str(file_path), img)

    return output_dir

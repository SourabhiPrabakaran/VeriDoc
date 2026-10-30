import argparse
import csv
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from training.dataset import LABEL_REAL, LABEL_FAKE, parse_label


def validate_zip(zip_path: Path) -> Tuple[bool, str, List[str], List[str]]:
    """
    Validates the integrity and contents of the SDFVD ZIP archive.
    Returns: (is_valid, message, real_video_entries, fake_video_entries)
    """
    if not zip_path.exists():
        return False, f"ZIP file not found at: {zip_path}", [], []

    if not zipfile.is_zipfile(zip_path):
        return False, f"File is not a valid ZIP archive: {zip_path}", [], []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            corrupt = zf.testzip()
            if corrupt is not None:
                return False, f"Corrupted file found in ZIP archive: {corrupt}", [], []

            namelist = zf.namelist()
            valid_exts = (".mp4", ".avi", ".mov", ".mkv")
            real_vids = [
                n for n in namelist
                if "videos_real" in n.lower() and n.lower().endswith(valid_exts) and not n.endswith("/")
            ]
            fake_vids = [
                n for n in namelist
                if "videos_fake" in n.lower() and n.lower().endswith(valid_exts) and not n.endswith("/")
            ]

            if not real_vids and not fake_vids:
                return False, "Archive does not contain expected 'videos_real' or 'videos_fake' directories.", [], []

            return True, f"Valid archive with {len(real_vids)} real and {len(fake_vids)} fake videos.", real_vids, fake_vids
    except Exception as e:
        return False, f"Failed to inspect ZIP archive: {e}", [], []


def extract_videos_safely(zip_path: Path, extract_dir: Path) -> Dict[str, Path]:
    """
    Safely extracts real and fake videos from SDFVD ZIP archive into extract_dir.
    Returns mapping of video_id -> extracted local Path.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_map = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name_lower = member.filename.lower()
            if not any(name_lower.endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv")):
                continue

            # Safe extraction path (avoid directory traversal)
            filename = Path(member.filename).name
            if "videos_real" in name_lower:
                sub_folder = extract_dir / "videos_real"
            elif "videos_fake" in name_lower:
                sub_folder = extract_dir / "videos_fake"
            else:
                sub_folder = extract_dir / "other"

            sub_folder.mkdir(parents=True, exist_ok=True)
            target_path = sub_folder / filename

            if not target_path.exists() or target_path.stat().st_size != member.file_size:
                with zf.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())

            video_id = target_path.stem  # e.g. 'v1', 'vs1'
            extracted_map[video_id] = target_path

    return extracted_map


def derive_pair_id(video_id: str) -> str:
    """
    Derives subject/pair ID from SDFVD video filename.
    Real video 'v12' and fake video 'vs12' both map to subject pair 'pair_12'.
    """
    vid_clean = video_id.lower().strip()
    if vid_clean.startswith("vs"):
        num_part = vid_clean[2:]
    elif vid_clean.startswith("v"):
        num_part = vid_clean[1:]
    else:
        num_part = vid_clean
    return f"pair_{num_part}"


def create_or_load_splits(
    video_records: List[Dict[str, Any]],
    split_file_path: Optional[Path] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_by_pair: bool = True
) -> Dict[str, str]:
    """
    Creates or reuses video-level and pair-level splits to guarantee zero data leakage.
    Returns mapping: video_id -> split ('train', 'val', 'test').
    """
    # 1. Try reusing existing valid split file if provided
    if split_file_path and split_file_path.exists():
        print(f"[PREPARATION] Reusing existing video split file from: {split_file_path}")
        existing_splits = {}
        with open(split_file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v_id = row.get("video_id") or row.get("video")
                s_name = row.get("split") or row.get("subset")
                if v_id and s_name:
                    existing_splits[v_id.strip()] = s_name.strip().lower()

        # Check coverage
        missing = [v["video_id"] for v in video_records if v["video_id"] not in existing_splits]
        if not missing:
            print(f"[PREPARATION] Successfully mapped all {len(video_records)} videos from existing split file.")
            return existing_splits
        else:
            print(f"[PREPARATION] Existing split file missing {len(missing)} videos. Recomputing deterministic split.")

    # 2. Compute deterministic, leakage-free pair-grouped split
    rng = random.Random(seed)

    if group_by_pair:
        # Group video records by pair_id
        pairs_dict: Dict[str, List[Dict[str, Any]]] = {}
        for rec in video_records:
            pair_id = rec["subject_id"]
            pairs_dict.setdefault(pair_id, []).append(rec)

        pair_keys = sorted(list(pairs_dict.keys()))
        rng.shuffle(pair_keys)

        n_pairs = len(pair_keys)
        n_val = max(1, int(round(n_pairs * val_ratio)))
        n_test = max(1, int(round(n_pairs * test_ratio)))
        n_train = max(1, n_pairs - n_val - n_test)
        if n_train + n_val + n_test > n_pairs:
            n_train = max(1, n_pairs - 2)
            n_val = 1
            n_test = 1

        train_pairs = set(pair_keys[:n_train])
        val_pairs = set(pair_keys[n_train:n_train + n_val])
        test_pairs = set(pair_keys[n_train + n_val:n_train + n_val + n_test])

        video_split_map = {}
        for p in train_pairs:
            for rec in pairs_dict[p]:
                video_split_map[rec["video_id"]] = "train"
        for p in val_pairs:
            for rec in pairs_dict[p]:
                video_split_map[rec["video_id"]] = "val"
        for p in test_pairs:
            for rec in pairs_dict[p]:
                video_split_map[rec["video_id"]] = "test"

    else:
        # Standard video-level split separated by class
        video_split_map = {}
        for label_name in ("real", "fake"):
            class_vids = [rec["video_id"] for rec in video_records if rec["label"] == label_name]
            class_vids.sort()
            rng.shuffle(class_vids)

            n_v = len(class_vids)
            n_val = max(1, int(round(n_v * val_ratio)))
            n_test = max(1, int(round(n_v * test_ratio)))
            n_train = max(1, n_v - n_val - n_test)

            for vid in class_vids[:n_train]:
                video_split_map[vid] = "train"
            for vid in class_vids[n_train:n_train + n_val]:
                video_split_map[vid] = "val"
            for vid in class_vids[n_train + n_val:n_train + n_val + n_test]:
                video_split_map[vid] = "test"

    return video_split_map


def extract_face_crops_from_video(
    video_path: Path,
    face_cascade: cv2.CascadeClassifier,
    output_crops_dir: Path,
    frames_per_video: int = 5,
    crop_size: int = 256,
    min_face_size: int = 40,
    margin_ratio: float = 0.15
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Samples representative frames from a video, detects faces, extracts 256x256 crops,
    and returns: (successful_crops, exclusions).
    """
    successful_crops = []
    exclusions = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        exclusions.append({
            "video_path": str(video_path),
            "reason": "UNREADABLE_VIDEO",
            "detail": "OpenCV failed to open video file."
        })
        return successful_crops, exclusions

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        exclusions.append({
            "video_path": str(video_path),
            "reason": "EMPTY_VIDEO",
            "detail": f"Video contains 0 frames (reported: {total_frames})."
        })
        return successful_crops, exclusions

    # Select representative evenly spaced frame indices
    if total_frames <= frames_per_video:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int).tolist()

    video_id = video_path.stem

    for f_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()

        if not ret or frame_bgr is None:
            exclusions.append({
                "video_id": video_id,
                "frame_index": f_idx,
                "reason": "FRAME_READ_FAILED",
                "detail": f"OpenCV read returned False at frame {f_idx}."
            })
            continue

        h, w, _ = frame_bgr.shape
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Primary Haar cascade detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size)
        )

        # 2. Fallback slightly more permissive search if no face found
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(max(24, min_face_size - 10), max(24, min_face_size - 10))
            )

        if len(faces) == 0:
            exclusions.append({
                "video_id": video_id,
                "frame_index": f_idx,
                "reason": "NO_FACE_DETECTED",
                "detail": f"Haar cascade detected 0 faces at frame {f_idx}."
            })
            continue

        # Choose the largest face bounding box
        largest_face = max(faces, key=lambda b: b[2] * b[3])
        fx, fy, fw, fh = largest_face

        # Add 15% bounding box padding
        pad_x = int(fw * margin_ratio)
        pad_y = int(fh * margin_ratio)

        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_y)

        face_crop = frame_bgr[y1:y2, x1:x2]

        if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            exclusions.append({
                "video_id": video_id,
                "frame_index": f_idx,
                "reason": "DEGENERATE_CROP_BOUNDS",
                "detail": f"Crop dimensions invalid: {face_crop.shape}."
            })
            continue

        # Check contrast / variance to exclude blank or corrupt crops
        crop_var = float(np.var(face_crop))
        if crop_var < 5.0:
            exclusions.append({
                "video_id": video_id,
                "frame_index": f_idx,
                "reason": "LOW_CONTRAST_OR_BLANK",
                "detail": f"Crop pixel variance too low: {crop_var:.2f}."
            })
            continue

        # Resize to target resolution (256x256)
        interp = cv2.INTER_AREA if (face_crop.shape[0] > crop_size or face_crop.shape[1] > crop_size) else cv2.INTER_LINEAR
        resized_crop = cv2.resize(face_crop, (crop_size, crop_size), interpolation=interp)

        # Save crop
        crop_filename = f"{video_id}_frame{f_idx:04d}_face0.jpg"
        crop_path = output_crops_dir / crop_filename
        cv2.imwrite(str(crop_path), resized_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        successful_crops.append({
            "crop_path": str(crop_path.resolve()),
            "crop_relative": f"crops/{crop_filename}",
            "video_id": video_id,
            "frame_index": f_idx,
            "box": [int(fx), int(fy), int(fw), int(fh)],
            "variance": round(crop_var, 2)
        })

    cap.release()
    return successful_crops, exclusions


def run_preparation(
    zip_path: Path,
    output_root: Path,
    split_file: Optional[Path] = None,
    frames_per_video: int = 5,
    crop_size: int = 256,
    seed: int = 42,
    group_by_pair: bool = True
) -> Dict[str, Any]:
    """
    Executes full end-to-end dataset preparation workflow:
    1. Validate ZIP
    2. Extract raw videos
    3. Generate/load deterministic splits
    4. Extract face crops
    5. Generate clean CSV manifest
    6. Output preparation summary
    """
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    raw_videos_dir = output_root / "raw_videos"
    crops_dir = output_root / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VERIDOC FORENSICS — SDFVD DATASET PREPARATION PIPELINE")
    print("=" * 70)

    # 1. Validate ZIP
    is_valid, msg, real_entries, fake_entries = validate_zip(zip_path)
    print(f"[PREPARATION] ZIP Validation: {msg}")
    if not is_valid:
        raise ValueError(f"ZIP validation failed: {msg}")

    # 2. Extract Videos
    print(f"[PREPARATION] Extracting raw video assets to: {raw_videos_dir}")
    extracted_map = extract_videos_safely(zip_path, raw_videos_dir)
    print(f"[PREPARATION] Total videos extracted: {len(extracted_map)}")

    # 3. Build Video Records with Labels and Pair IDs
    video_records = []
    for vid, p in extracted_map.items():
        is_real = "videos_real" in str(p.parent).lower() or vid.lower().startswith("v") and not vid.lower().startswith("vs")
        label = "real" if is_real else "fake"
        subject_id = derive_pair_id(vid)
        video_records.append({
            "video_id": vid,
            "path": p,
            "label": label,
            "subject_id": subject_id
        })

    # 4. Generate or Load Split Mapping
    split_save_path = output_root / "SDFVD_video_splits.csv"
    actual_split_file = split_file if (split_file and split_file.exists()) else (split_save_path if split_save_path.exists() else None)
    video_splits = create_or_load_splits(
        video_records,
        split_file_path=actual_split_file,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
        group_by_pair=group_by_pair
    )

    # Save split mapping CSV
    with open(split_save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "label", "subject_id", "split"])
        for rec in video_records:
            writer.writerow([rec["video_id"], rec["label"], rec["subject_id"], video_splits.get(rec["video_id"], "train")])
    print(f"[PREPARATION] Video splits saved to: {split_save_path}")

    # 5. Load Face Detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load OpenCV Haar Cascade from {cascade_path}")

    # 6. Extract Face Crops
    print(f"[PREPARATION] Extracting {frames_per_video} face crops per video (resolution: {crop_size}x{crop_size})...")
    all_manifest_rows = []
    all_exclusions = []
    videos_with_crops = set()

    for idx, rec in enumerate(video_records, 1):
        vid = rec["video_id"]
        v_path = rec["path"]
        label = rec["label"]
        subject_id = rec["subject_id"]
        split = video_splits.get(vid, "train")

        crops, excl = extract_face_crops_from_video(
            video_path=v_path,
            face_cascade=face_cascade,
            output_crops_dir=crops_dir,
            frames_per_video=frames_per_video,
            crop_size=crop_size
        )

        for c in crops:
            all_manifest_rows.append({
                "path": c["crop_path"],
                "relative_path": c["crop_relative"],
                "label": label,
                "video_id": vid,
                "subject_id": subject_id,
                "split": split,
                "frame_index": c["frame_index"],
                "box": c["box"]
            })
            videos_with_crops.add(vid)

        all_exclusions.extend(excl)

        if idx % 20 == 0 or idx == len(video_records):
            print(f"  --> Processed {idx}/{len(video_records)} videos... Total crops so far: {len(all_manifest_rows)}")

    # 7. Write Clean CSV Manifest
    manifest_path = output_root / "SDFVD_clean_manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "video_id", "subject_id", "split"])
        for r in all_manifest_rows:
            writer.writerow([r["path"], r["label"], r["video_id"], r["subject_id"], r["split"]])
    print(f"[PREPARATION] Clean CSV manifest saved to: {manifest_path}")

    # Also save relative path manifest for portability
    rel_manifest_path = output_root / "SDFVD_clean_manifest_relative.csv"
    with open(rel_manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "video_id", "subject_id", "split"])
        for r in all_manifest_rows:
            writer.writerow([r["relative_path"], r["label"], r["video_id"], r["subject_id"], r["split"]])

    # 8. Compile Comprehensive Summary
    real_crops = sum(1 for r in all_manifest_rows if r["label"] == "real")
    fake_crops = sum(1 for r in all_manifest_rows if r["label"] == "fake")

    train_crops = sum(1 for r in all_manifest_rows if r["split"] == "train")
    val_crops = sum(1 for r in all_manifest_rows if r["split"] == "val")
    test_crops = sum(1 for r in all_manifest_rows if r["split"] == "test")

    train_vids = sum(1 for v in video_records if video_splits.get(v["video_id"]) == "train")
    val_vids = sum(1 for v in video_records if video_splits.get(v["video_id"]) == "val")
    test_vids = sum(1 for v in video_records if video_splits.get(v["video_id"]) == "test")

    summary = {
        "status": "PREPARATION_COMPLETED",
        "zip_source": str(zip_path.resolve()),
        "output_directory": str(output_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "relative_manifest_path": str(rel_manifest_path.resolve()),
        "split_file_path": str(split_save_path.resolve()),
        "video_statistics": {
            "total_videos_in_zip": len(video_records),
            "videos_yielding_crops": len(videos_with_crops),
            "videos_with_zero_crops": len(video_records) - len(videos_with_crops),
            "real_videos": sum(1 for v in video_records if v["label"] == "real"),
            "fake_videos": sum(1 for v in video_records if v["label"] == "fake"),
            "video_split_distribution": {
                "train_videos": train_vids,
                "val_videos": val_vids,
                "test_videos": test_vids
            }
        },
        "crop_statistics": {
            "total_face_crops_extracted": len(all_manifest_rows),
            "real_face_crops": real_crops,
            "fake_face_crops": fake_crops,
            "crop_split_distribution": {
                "train_crops": train_crops,
                "val_crops": val_crops,
                "test_crops": test_crops
            }
        },
        "exclusions_count": len(all_exclusions),
        "crop_resolution": [crop_size, crop_size]
    }

    # Save summary JSON
    summary_path = output_root / "preparation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save exclusions log
    exclusions_path = output_root / "excluded_samples.json"
    with open(exclusions_path, "w", encoding="utf-8") as f:
        json.dump(all_exclusions, f, indent=2)

    print("\n" + "=" * 70)
    print("SDFVD DATASET PREPARATION SUMMARY")
    print("=" * 70)
    print(f"Total Videos Processed:    {len(video_records)} ({summary['video_statistics']['real_videos']} Real, {summary['video_statistics']['fake_videos']} Fake)")
    print(f"Videos with Face Crops:    {len(videos_with_crops)}")
    print(f"Total Face Crops:          {len(all_manifest_rows)} ({real_crops} Real, {fake_crops} Fake)")
    print(f"Train / Val / Test Crops:  {train_crops} / {val_crops} / {test_crops}")
    print(f"Train / Val / Test Videos: {train_vids} / {val_vids} / {test_vids}")
    print(f"Excluded Frames/Reasons:   {len(all_exclusions)} (logged to excluded_samples.json)")
    print(f"Clean Manifest CSV:        {manifest_path}")
    print("=" * 70 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="VeriDoc SDFVD Dataset Preparation and Face Extraction Pipeline")
    parser.add_argument("--zip-path", type=str, default="", help="Path to SDFVD.zip archive")
    parser.add_argument("--output-dir", type=str, default="", help="Output root directory for dataset artifacts")
    parser.add_argument("--split-file", type=str, default="", help="Optional existing video split CSV file to reuse")
    parser.add_argument("--frames-per-video", type=int, default=5, help="Number of representative frames to sample per video")
    parser.add_argument("--crop-size", type=int, default=256, help="Target face crop resolution (square 256x256)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed for splitting")
    parser.add_argument("--no-pair-grouping", action="store_true", help="Disable pair-level subject grouping")

    args = parser.parse_args()

    default_zip = (backend_dir / "training" / "SDFVD.zip").resolve()
    zip_path = Path(args.zip_path).resolve() if args.zip_path else default_zip

    default_out = (backend_dir / "training" / "data_sdfvd").resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_out

    split_f = Path(args.split_file).resolve() if args.split_file else None

    run_preparation(
        zip_path=zip_path,
        output_root=output_dir,
        split_file=split_f,
        frames_per_video=args.frames_per_video,
        crop_size=args.crop_size,
        seed=args.seed,
        group_by_pair=not args.no_pair_grouping
    )


if __name__ == "__main__":
    main()

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import json

def audit_dataset(dataset_dir):
    print("=== STARTING BENCHMARK DATASET AUDIT ===")
    dataset_path = Path(dataset_dir)
    real_dir = dataset_path / "real"
    fake_dir = dataset_path / "fake"
    
    # Ensure directories exist
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "status": "INCOMPLETE",
        "real_count": 0,
        "fake_count": 0,
        "corrupted_files": [],
        "invalid_formats": [],
        "valid_real_files": [],
        "valid_fake_files": [],
        "image_dimension_summary": {},
        "face_crop_heuristics": {
            "likely_face_crops": 0,
            "likely_full_scenes": 0
        }
    }
    
    valid_extensions = {".jpg", ".jpeg", ".png"}
    
    for category, directory in [("REAL", real_dir), ("FAKE", fake_dir)]:
        # Walk directories recursively
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()
                
                # Format validation
                if ext not in valid_extensions:
                    report["invalid_formats"].append(str(file_path))
                    continue
                    
                # Read and open validation
                try:
                    img = cv2.imread(str(file_path))
                    if img is None:
                        raise ValueError("OpenCV returned None")
                        
                    h, w, c = img.shape
                    dim_str = f"{w}x{h}"
                    
                    if category == "REAL":
                        report["real_count"] += 1
                        report["valid_real_files"].append(str(file_path))
                    else:
                        report["fake_count"] += 1
                        report["valid_fake_files"].append(str(file_path))
                        
                    # Track dimension frequencies
                    report["image_dimension_summary"][dim_str] = report["image_dimension_summary"].get(dim_str, 0) + 1
                    
                    # Heuristics for face crops (e.g. aspect ratio close to 1:1 and resolution under 600x600)
                    aspect_ratio = w / h
                    if 0.8 <= aspect_ratio <= 1.25 and w <= 600 and h <= 600:
                        report["face_crop_heuristics"]["likely_face_crops"] += 1
                    else:
                        report["face_crop_heuristics"]["likely_full_scenes"] += 1
                        
                except Exception as e:
                    report["corrupted_files"].append({
                        "file": str(file_path),
                        "error": str(e)
                    })
                    
    # Update Status
    if report["real_count"] >= 50 and report["fake_count"] >= 50:
        report["status"] = "READY"
    elif report["real_count"] > 0 or report["fake_count"] > 0:
        report["status"] = "PRELIMINARY"
    else:
        report["status"] = "EMPTY"
        
    # Print summary
    print(f"Status: {report['status']}")
    print(f"REAL Images: {report['real_count']}")
    print(f"FAKE Images: {report['fake_count']}")
    print(f"Invalid Format Files: {len(report['invalid_formats'])}")
    print(f"Corrupted Files: {len(report['corrupted_files'])}")
    print(f"Likely Face Crops: {report['face_crop_heuristics']['likely_face_crops']}")
    print(f"Likely Full Scenes: {report['face_crop_heuristics']['likely_full_scenes']}")
    
    return report

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent / "data" / "evaluation_dataset"
    audit_dataset(base_dir)

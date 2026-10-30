"""
VeriDoc Forensics - Training & Evaluation Pipeline
"""
from .dataset import (
    DeepfakeDataset,
    scan_directory_dataset,
    create_grouped_splits,
    LABEL_REAL,
    LABEL_FAKE
)

__all__ = [
    "DeepfakeDataset",
    "scan_directory_dataset",
    "create_grouped_splits",
    "LABEL_REAL",
    "LABEL_FAKE"
]

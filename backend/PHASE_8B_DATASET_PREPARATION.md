# Phase 8B — MesoInception-4 Evaluation Dataset Preparation

This document outlines the folder structure, expected file formats, validation rules, benchmark runner methodology, and commands to execute the evaluation benchmark.

---

## 1. Folder Structure
The benchmark runner expects the evaluation dataset to be placed under:
```text
backend/data/evaluation_dataset/
    real/
        image_01.jpg
        image_02.png
        ...
    fake/
        image_01.jpg
        image_02.png
        ...
```

*   **`real/`:** Contains authentic face images.
*   **`fake/`:** Contains manipulated (deepfaked) face images.

---

## 2. Expected Image Formats & Rules
*   **Extensions:** `.jpg`, `.jpeg`, or `.png`.
*   **Dimensions:** Preferably pre-cropped $256 \times 256$ pixels face regions.
*   **Content:** Only valid face images. Multi-face images or non-face scenes should be avoided or preprocessed to extract individual face crops first.
*   **No Modifications:** The script audits the directory recursively and validates readability without modifying or overwriting any input files.

---

## 3. Benchmark Methodology & Metrics
The evaluation script [`run_benchmark_evaluation.py`](file:///c:/Users/soura/OneDrive/Pictures/VeriDoc/backend/scripts/run_benchmark_evaluation.py) runs the identical production preprocessing and inference pipelines. 

### Classification Metrics (Positive Class = FAKE)
*   **Accuracy:** Overall fraction of correct predictions.
*   **Precision:** True Fakes / (True Fakes + False Fakes).
*   **Recall:** True Fakes / (True Fakes + Missed Fakes).
*   **F1-Score:** Harmonic mean of Precision and Recall.
*   **Specificity:** True Reals / (True Reals + False Fakes).
*   **FPR (False Positive Rate):** False Fakes / (True Reals + False Fakes).
*   **FNR (False Negative Rate):** Missed Fakes / (True Fakes + Missed Fakes).

### Output Files
*   **`backend/data/evaluation_results/benchmark_results.json`:** Raw list of file-level predictions and inference times.
*   **`backend/data/evaluation_results/benchmark_report.json`:** Structured summary table of metrics, confusion matrix, and average CPU latency.

---

## 4. Execution Command
To run the dataset preparation audit and benchmark evaluation, run:

```powershell
# 1. Audit and validate evaluation images folder
venv\Scripts\python scripts\prepare_evaluation_dataset.py

# 2. Run model inference benchmark and compute stats
venv\Scripts\python scripts\run_benchmark_evaluation.py
```

---

## 5. Limitations Notice
*   **Preliminary Evaluation:** If either folder contains fewer than **50 samples**, the benchmark status is flagged as `PRELIMINARY`.
*   **Out-of-Distribution Warning:** General-world accuracy claims must not be asserted from small, non-representative test benchmarks.

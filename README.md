# VeriDoc Forensics — AI-Assisted Deepfake Detection & Digital Evidence Integrity

VeriDoc Forensics is an end-to-end digital forensics platform designed to verify digital video and image authenticity. Developed as a machine learning and forensic engineering system, it bridges deep convolutional neural networks, frequency-domain forensic signals, and cryptographic chain-of-custody tracking into an interactive verification workstation.

---

## 1. Project Overview

Digital visual evidence presented in judicial proceedings, regulatory reviews, or investigative journalism faces an unprecedented challenge from generative artificial intelligence. Consumer-grade tools can generate convincing facial reenactments and face swaps with minimal visual flaws. 

VeriDoc Forensics addresses this challenge through a multi-layered verification strategy:
- **Cryptographic Preservation:** Immediate streaming SHA-256 digests establish an immutable chain of custody.
- **Deep Learning Detection:** Inspects micro-structural facial blending boundaries and high-frequency spectral patterns.
- **Signal-Level Diagnostics:** Independent Error Level Analysis (ELA) and 2D Fourier power spectra.
- **Explainability:** On-demand Gradient-weighted Class Activation Mapping (Grad-CAM) visualizations.
- **Tamper-Evident Reporting:** Automated PDF forensic reports with embedded cryptographic signatures.

---

## 2. Problem Statement

Modern facial manipulation pipelines (e.g., DeepFaceLab, FaceForensics++, SimSwap) introduce distinct forensic challenges:
1. **Spatial Boundary Smoothing:** Generative blending often conceals boundary edges in the spatial domain, causing standard convolutional networks to struggle or overfit on compression noise.
2. **Spectral Grid Artifacts:** Convolutional upsampling layers (transposed convolutions, nearest-neighbor interpolation) create periodic lattice spikes in the frequency domain that are invisible to the naked eye.
3. **Data Leakage Risks:** Datasets that pair authentic footage with reenacted fakes of the same person often suffer from identity and background leakage unless strictly grouped at the subject-pair level.
4. **Evidentiary Continuity:** Model predictions are legally meaningless if evidence integrity cannot be proven from ingestion through analysis.

---

## 3. Key Features

- **Chain-of-Custody Hashing:** Streaming SHA-256 digest computation upon evidence upload, with automated pre- and post-analysis tamper checks against an SQLite audit ledger.
- **Dual-Model Deep Learning Pipeline:**
  - **MesoInception-4 (Production Baseline):** 28,539 parameters; utilizes dilated Inception modules to detect micro-structural facial blending seams.
  - **DualStreamForensicNet (Experimental Research Architecture):** 775,932 parameters; multimodal late fusion of spatial convolutions with a 2D Fourier log-magnitude spectral CNN.
- **Descriptive Forensic Signals:**
  - **Error Level Analysis (ELA):** Recompresses images at 90% JPEG quality to surface compression rate discrepancies.
  - **2D FFT Radial Spectrum:** Computes radial frequency energy distributions and high-frequency ratios.
- **Explainable AI (Grad-CAM):** Interactive, on-demand activation heatmaps on target convolutional layers (`conv4`) to illustrate facial regions influencing model decisions.
- **Forensic PDF Export:** Clean, tamper-evident judicial summary reports generated via ReportLab with embedded SHA-256 verification hashes.

---

## 4. System Architecture

```text
               USER EVIDENCE UPLOAD (Image / Video)
                               │
               ┌───────────────▼───────────────┐
               │ Cryptographic Ingest Hashing  ├──► SHA-256 Audit Record (SQLite)
               └───────────────┬───────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
[Error Level Analysis]                       [Media Preprocessing]
(JPEG Recompression Map)                     (Face Localization & Uniform Sampling)
       │                                               │
       ▼                                               ▼
[2D FFT Spectral Analysis]                   [Facial Crop Normalization]
(High-Frequency Ratio)                       (256x256, RGB, [0, 1] Tensor)
       │                                               │
       │                               ┌───────────────┴───────────────┐
       │                               ▼                               ▼
       │                     [Spatial MesoInception-4]       [Dual-Stream Fusion]
       │                     (Conv4 Feature Map)             (Spatial + 2D FFT)
       │                               │                               │
       │                               ▼                               ▼
       │                      [Grad-CAM Heatmap]              [Model Prediction]
       │                      (On-Demand Overlay)                      │
       └───────────────────────────────┬───────────────────────────────┘
                                       ▼
                         [Video Aggregation Strategy]
                    (Fraction Threshold / Mean Probability)
                                       │
                                       ▼
                        [Tamper-Evident PDF Report]
                        (Embedded SHA-256 & Diagnostics)
```

---

## 5. Technology Stack

- **Backend:** Python 3.11, FastAPI 0.115, Uvicorn, SQLAlchemy 2.0, SQLite, Pydantic 2.10
- **Machine Learning & Vision:** PyTorch 2.5, Torchvision, OpenCV (`opencv-python` 4.10), NumPy
- **Forensic PDF Generation:** ReportLab 4.2
- **Frontend:** React 19, Vite 8, TailwindCSS 3.4, Lucide React
- **Testing & Verification:** Python `unittest`, Node.js test runner (`node:test`), Oxlint

---

## 6. Repository Structure

```text
VeriDoc/
├── backend/
│   ├── app/
│   │   ├── api/routes/          # REST API endpoints (analysis, evidence, reports, health)
│   │   ├── core/                # Configuration and database engine
│   │   ├── models/              # SQLAlchemy database models
│   │   ├── schemas/             # Pydantic validation schemas
│   │   ├── services/            # Forensic services (ELA, FFT, hashing, Grad-CAM, inference)
│   │   └── main.py              # FastAPI application entry point
│   ├── tests/                   # Automated backend test suite (65 tests)
│   ├── training/                # Training, evaluation, and dataset preparation
│   │   ├── dataset.py           # PyTorch Dataset and leakage-free pair splitting
│   │   ├── dual_stream_model.py # DualStreamForensicNet implementation
│   │   ├── evaluate.py          # Metric evaluation engine (ROC-AUC, EER, ECE)
│   │   ├── prepare_sdfvd.py     # Dataset extraction and manifest generation
│   │   ├── compare_models.py    # Side-by-side benchmark comparison script
│   │   └── train.py             # Model training engine with AdamW & Cosine Annealing
│   ├── requirements.txt         # Backend Python dependencies
│   └── .env.example             # Backend configuration template
├── notebooks/                   # Jupyter / Google Colab GPU training notebooks
├── public/                      # Static web assets
├── src/                         # React frontend source code
│   ├── components/              # Layout and UI components
│   ├── pages/                   # Application pages (Dashboard, Evidence, Analysis, Integrity, Reports)
│   └── services/api.js          # REST API client
├── .env.example                 # Root environment template
├── .gitignore                   # Comprehensive git exclusions
├── MODEL_CARD.md                # Formal model specifications, metrics, and limitations
├── package.json                 # Frontend dependencies and scripts
└── vite.config.js               # Vite configuration
```

---

## 7. Dataset Acquisition & Preparation

VeriDoc Forensics is benchmarked on the **Small-scale Deepfake Forgery Video Dataset (SDFVD)**.

### 7.1 Dataset Overview & Attribution
- **Dataset Title:** *SDFVD: Small-scale Deepfake Forgery Video Dataset*
- **Authors:** Shilpa Kaman and Dr. Aziz Makandar
- **Persistent DOI:** [10.17632/bcmkfgct2s.1](https://doi.org/10.17632/bcmkfgct2s.1) (Mendeley Data, 2024)
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Total Videos:** 106 MP4 video files (53 authentic videos from Pexels, 53 AI face-swap reenactments).

> [!NOTE]
> The dataset archive is **not included in this repository** due to size constraints. Users must download the dataset independently from Mendeley Data.

### 7.2 Download and Extraction Instructions
1. Download `SDFVD.zip` from the official Mendeley Data page: [https://doi.org/10.17632/bcmkfgct2s.1](https://doi.org/10.17632/bcmkfgct2s.1).
2. Place the downloaded `SDFVD.zip` inside the `backend/training/` directory:
   ```text
   backend/training/SDFVD.zip
   ```
3. Run the automated, leakage-safe dataset preparation script:
   ```powershell
   cd backend
   .\venv\Scripts\python.exe training\prepare_sdfvd.py `
       --zip-path "training/SDFVD.zip" `
       --output-dir "training/data_sdfvd" `
       --frames-per-video 5 `
       --crop-size 256 `
       --seed 42
   ```

### 7.3 Expected Output
The preparation script extracts 5 uniformly spaced frames per video, runs OpenCV Haar face detection, expands bounding boxes by 15%, resizes to $256 \times 256$ RGB JPEGs, and performs pair-grouped splitting:
```text
backend/training/data_sdfvd/
├── crops/                         # 513 facial crop JPEGs (257 Real, 256 Fake)
├── raw_videos/                    # Extracted 106 MP4 videos
├── SDFVD_clean_manifest.csv       # Master dataset manifest with pair IDs
├── SDFVD_video_splits.csv         # Video-level split assignments
├── preparation_summary.json       # Yield and split statistics
└── excluded_samples.json          # 17 candidate frames where 0 faces were detected
```

---

## 8. Model Checkpoint Setup

### 8.1 Model Roles and Current Status

| Model | Role in Project | Current Deployment Status | Weight Origin |
|---|---|---|---|
| **MesoInception-4** | Spatial baseline model | **Production-Configured** (Active in `backend/.env`) | Trained from scratch on SDFVD |
| **DualStreamForensicNet** | Spatial-spectral multi-stream model | **Experimental Research Model** (Evaluation artifact) | Trained from scratch on SDFVD |
| *Historical Buggy DualStream* | Historical experiment | **Deprecated** (Batch-roll bug) | Not recommended for inference |

> [!IMPORTANT]
> **Trained model weights (`.pth`, `.pt`) are NOT committed to GitHub.**  
> Users must either:
> 1. Supply their own trained checkpoint.
> 2. Train the models from scratch using the reproducible training instructions in [Section 12](#12-model-training--evaluation).

### 8.2 Placing and Configuring the Model Checkpoint
1. Copy or place your trained MesoInception-4 checkpoint at:
   ```text
   backend/training/weights/meso_inception_best.pth
   ```
2. Configure `backend/.env` (copied from `backend/.env.example`):
   ```env
   DEEPFAKE_MODEL_PATH=training/weights/meso_inception_best.pth
   DEEPFAKE_MODEL_NAME=MesoInception-4
   DEEPFAKE_MODEL_VERSION=sdfvd_v1
   DEEPFAKE_DEVICE=cpu
   DEEPFAKE_CLASSIFICATION_THRESHOLD=0.50
   DEEPFAKE_VIDEO_AGGREGATION_METHOD=fraction_threshold
   DEEPFAKE_VIDEO_FAKE_RATIO_THRESHOLD=0.25
   DEEPFAKE_BATCH_SIZE=16
   VIDEO_SAMPLE_INTERVAL_SECONDS=1.0
   ```

### 8.3 Unconfigured Model Behavior
If no checkpoint is placed or `DEEPFAKE_MODEL_PATH` is left blank, the application gracefully operates in **`MODEL_NOT_CONFIGURED`** mode:
- The backend remains fully operational.
- Evidence hashing, ELA, and 2D FFT spectral graphs execute normally.
- Deepfake predictions transparently display `UNCONFIGURED` in the web interface rather than generating false predictions or throwing errors.

---

## 9. Backend Installation

### Prerequisites
- Python 3.8 – 3.11 (Python 3.11 recommended; Python >= 3.13 is unsupported due to torch<2.5.0 wheels)
- Windows PowerShell, macOS zsh, or Linux bash

### Setup Steps
```powershell
# 1. Navigate to backend directory
cd backend

# 2. Create Python virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
# source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Initialize environment configuration
Copy-Item .env.example .env    # On Linux/macOS: cp .env.example .env
```

---

## 10. Frontend Installation

### Prerequisites
- Node.js 18+ (Node.js 20 LTS recommended)
- npm 9+

### Setup Steps
```powershell
# From the project root directory:
npm install
```

---

## 11. Running the Application

### Option A: Manual Launch (Two Terminals)

**Terminal 1 — Backend (FastAPI / Uvicorn):**
```powershell
cd backend
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend (Vite / React):**
```powershell
# From project root:
npm run dev
```

Open your browser at:
- **Web Dashboard:** [http://localhost:5173](http://localhost:5173)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Health Check:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Option B: 1-Click Launch via VS Code
Pre-configured tasks are provided in `.vscode/tasks.json`:
1. Press `Ctrl + Shift + P` (or `F1`).
2. Type **`Tasks: Run Task`** and press Enter.
3. Select **`Start Full-Stack (Backend + Frontend)`**.

---

## 12. Model Training & Evaluation

All training scripts support reproducible training from scratch using PyTorch default weight initialization (no external pretrained weights required).

### 12.1 Quick Smoke Test (Synthetic Dataset)
To verify your training pipeline without downloading SDFVD:
```powershell
cd backend
.\venv\Scripts\python.exe training\train.py --use-synthetic --epochs 3 --output-dir "training/weights_smoke"
```

### 12.2 Training MesoInception-4 from Scratch
```powershell
cd backend
.\venv\Scripts\python.exe training\train.py `
    --manifest "training/data_sdfvd/SDFVD_clean_manifest.csv" `
    --model "meso_inception" `
    --epochs 15 `
    --batch-size 16 `
    --lr 0.0003 `
    --dropout-rate 0.30 `
    --patience 6 `
    --output-dir "training/weights_meso"
```

### 12.3 Training DualStreamForensicNet from Scratch
Trains the multi-stream architecture with the corrected 2D FFT spatial shift:
```powershell
cd backend
.\venv\Scripts\python.exe training\train.py `
    --manifest "training/data_sdfvd/SDFVD_clean_manifest.csv" `
    --model "dual_stream" `
    --epochs 20 `
    --batch-size 16 `
    --lr 0.0003 `
    --dropout-rate 0.30 `
    --patience 6 `
    --output-dir "training/weights_dualstream"
```

### 12.4 Model Evaluation on Held-Out Test Manifest
```powershell
cd backend
.\venv\Scripts\python.exe training\evaluate.py `
    --checkpoint "training/weights_meso/meso_inception_best.pth" `
    --manifest "training/weights_meso/test_split_manifest.json" `
    --val-manifest "training/weights_meso/val_split_manifest.json" `
    --output-dir "training/eval_results"
```

### 12.5 Comparative Model Evaluation
```powershell
cd backend
.\venv\Scripts\python.exe training\compare_models.py `
    --baseline-checkpoint "training/weights_meso/meso_inception_best.pth" `
    --dualstream-checkpoint "training/weights_dualstream/dual_stream_best.pth" `
    --manifest "training/weights_meso/test_split_manifest.json" `
    --output-dir "training/comparison_results"
```

---

## 13. Testing Instructions

### Backend Automated Test Suite (65 Tests)
The backend test suite covers cryptographic hashing, tensor operations, FFT batch invariance, ELA differential signals, and live REST endpoints:

```powershell
cd backend
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

### Frontend Code Quality & Automated Tests
```powershell
# From project root:
npm run lint                                # Fast code linting via Oxlint
node --test src/pages/Integrity.test.mjs    # Frontend integrity & hash tests (Node.js test runner)
npm run build                               # Production bundle build
```

---

## 14. Model Performance & Scientific Limitations

Detailed performance statistics, confusion matrices, and risk assessments are documented in [`MODEL_CARD.md`](MODEL_CARD.md).

### 14.1 Held-Out Benchmark Performance ($N=16$ Test Videos / 80 Crops)

| Metric | Baseline MesoInception-4 (Production) | Retrained DualStreamForensicNet (Research) |
|---|:---:|:---:|
| **Crop ROC-AUC** | 0.4931 | **0.7731** |
| **Crop Equal Error Rate (EER)** | 52.50% | **30.00%** |
| **Crop Accuracy ($\tau=0.50$)** | 47.50% (38/80) | **73.75% (59/80)** |
| **Video ROC-AUC** | 0.4844 | **0.7969** |
| **Video Equal Error Rate (EER)** | 50.00% | **25.00%** |
| **Video Accuracy ($\tau=0.50$)** | 50.00% (8/16) | **75.00% (12/16)** |
| **Fake Detection Specificity** | 37.50% (3/8 caught) | **87.50% (7/8 caught)** |
| **Fake Specificity at $\tau_{\text{val}}$** | 50.00% (4/8 caught) | **100.00% (8/8 caught)** |

### 14.2 Essential Scientific Disclosures
1. **Sample Size Quantization:** The test split contains 16 videos (8 Real, 8 Fake). A single video changes reported accuracy by **$6.25\%$** ($1/16$). DualStream (12/16) leads MesoInception (11/16) by **exactly one video**, which is not statistically significant ($p > 0.05$).
2. **Wilson Score Uncertainty:** Observed 75% video accuracy on 16 samples has a 95% Wilson confidence interval of **$[50.50\%, 89.82\%]$**.
3. **Forensic Triangulation:** ELA, FFT spectral charts, and Grad-CAM overlays are supporting investigative aids, not conclusive proof of digital forgery.
4. **Integrity vs. Authenticity:** SHA-256 confirms that a file has not been modified since upload; it does **not** prove that the media was genuine when captured.

---

## 15. Research References

1. **Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018).** *MesoNet: a Compact Facial Video Forgery Detection Network.* IEEE International Workshop on Information Forensics and Security (WIFS). [DOI: 10.1109/WIFS.2018.8630761](https://doi.org/10.1109/WIFS.2018.8630761).
2. **Frank, J., Banzhaf, T., Holz, K., et al. (2020).** *Leveraging Frequency Analysis for Deep Fake Image Recognition.* International Conference on Machine Learning (ICML), PMLR 119:3247–3258.
3. **Selvaraju, R. R., Cogswell, M., Das, A., et al. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* IEEE International Conference on Computer Vision (ICCV), pp. 618–626.
4. **Kaman, S., & Makandar, A. (2024).** *SDFVD: Small-scale Deepfake Forgery Video Dataset.* Mendeley Data, [DOI: 10.17632/bcmkfgct2s.1](https://doi.org/10.17632/bcmkfgct2s.1).
5. **Krawetz, N. (2007).** *A Picture's Worth... Digital Image Analysis and Error Level Analysis.* Hacker Factor Solutions.

---

## 16. Future Work

- **Cross-Dataset Evaluation:** Benchmark checkpoints against FaceForensics++ (Raw, c23, c40) and Celeb-DF v2.
- **Advanced Face Alignment:** Upgrade from Haar cascades to 5-point landmark alignment (RetinaFace or MTCNN).
- **Ablation Studies:** Quantify the independent contribution of the spatial stream vs. the frequency stream across varying social media compression profiles.

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.

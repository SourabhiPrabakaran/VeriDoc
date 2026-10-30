# Model Card: VeriDoc Forensics Deepfake Detection Models

**Model Card Version:** 1.0.0  
**Date:** October 30, 2026  
**License:** MIT License  
**Framework:** PyTorch 2.5+  
**Target Repository:** VeriDoc Forensics Workstation  

---

## 1. Model Overview

VeriDoc Forensics incorporates two convolutional neural network models for binary facial manipulation detection (Authentic vs. Deepfake):

| Model Name | Role in Project | Current Deployment Status | Trainable Parameters | Input Modality |
|---|---|---|:---:|---|
| **MesoInception-4** | Spatial baseline model | **Production-Configured** (Active in `backend/.env`) | 28,539 | Normalized RGB facial crop |
| **DualStreamForensicNet** | Spatial-spectral multi-stream model | **Experimental Research Model** (Evaluation artifact) | 775,932 | RGB crop + 2D FFT Log-Magnitude |

> [!IMPORTANT]
> **Repository Weights Disclosure:**  
> In accordance with standard open-source research practices, **serialized model weights (`.pth`, `.pt`) are NOT tracked or committed to GitHub** due to file size constraints. Users must either:
> 1. Obtain serialized checkpoints from the project maintainer.
> 2. Reproduce the weights from scratch using the included reproducible training pipeline (`backend/training/train.py`).
>
> When no weights are configured in `backend/.env`, the application gracefully enters an unconfigured state, reporting `MODEL_NOT_CONFIGURED` without failing or generating misleading synthetic predictions.

---

## 2. Model Lineage & Theoretical Foundations

### 2.1 MesoInception-4 (Production Baseline)
- **Primary Citation:** Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). *MesoNet: a Compact Facial Video Forgery Detection Network*. IEEE International Workshop on Information Forensics and Security (WIFS).
- **Architecture Principle:** Traditional deep vision backbones (e.g., VGG, ResNet) overfit on facial images by memorizing high-level facial semantics. MesoInception-4 uses lightweight dilated Inception modules to detect micro-structural facial blending seams with very low parameter count (28,539 parameters).

### 2.2 DualStreamForensicNet (Experimental Research Architecture)
- **Primary Citations:** 
  - Spatial Stream: *Afchar et al. (2018)*.
  - Spectral Stream: Frank, J., Banzhaf, T., Holz, K., et al. (2020). *Leveraging Frequency Analysis for Deep Fake Image Recognition*. ICML.
- **Architecture Principle:** Deep generative synthesis methods (GANs, autoencoders, diffusion models) rely on transposed convolutions or interpolation-based upsampling. These leave periodic lattice artifacts and unnatural high-frequency energy distributions in the **2D Fourier transform (FFT)** domain. DualStreamForensicNet fuses spatial pixel convolutions with a 2D Fourier log-magnitude convolutional stream via late multimodal concatenation.

---

## 3. Architectural Specifications

### 3.1 Input Tensor Format
- **Shape:** `(B, 3, 256, 256)`
- **Color Space:** RGB
- **Data Type:** `torch.float32`
- **Value Range:** Normalized to `[0.0, 1.0]`

### 3.2 MesoInception-4 Topology
1. **Inception Block 1:** 4 parallel convolutional branches ($1\times 1$, $3\times 3$, dilated $3\times 3$ with $d=2$, $3\times 3$ maxpool) concatenating to 11 channels $\to$ BatchNorm $\to$ MaxPool($2\times 2$).
2. **Inception Block 2:** 4 parallel branches concatenating to 12 channels $\to$ BatchNorm $\to$ MaxPool($2\times 2$).
3. **Convolutional Block 3:** Conv2d($12 \to 16, k=5, p=2$) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool($2\times 2$).
4. **Convolutional Block 4:** Conv2d($16 \to 16, k=5, p=2$) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool($4\times 4$). Output feature map: `(B, 16, 8, 8)` (1,024 features).
5. **Dense Classification Head:** Flatten $\to$ Dropout($p=0.50$) $\to$ Linear($1024 \to 16$) $\to$ LeakyReLU(0.1) $\to$ Dropout($p=0.50$) $\to$ Linear($16 \to 1$).
6. **Trainable Parameters:** **28,539**.

### 3.3 DualStreamForensicNet Topology
1. **Stream 1 (Spatial):** MesoInception-4 convolutional backbone (`forward_features`) extracting a 1,024-dimensional feature embedding.
2. **2D FFT Spectral Preprocessor:**
   - Luminance conversion: $Y = 0.299R + 0.587G + 0.114B$.
   - 2D Fast Fourier Transform: `torch.fft.fft2(gray)`.
   - Dimensionally constrained spatial shift: `torch.fft.fftshift(fft, dim=(-2, -1))` to center the zero-frequency DC component without rolling the batch dimension.
   - Normalized log-magnitude: $\ln(|F_{\text{shifted}}| + 10^{-6})$, scaled to `[0.0, 1.0]`.
3. **Stream 2 (Spectral CNN — `FftFeatureExtractor`):**
   - Conv1: $1 \to 16$ ($k=5, s=2, p=2$) $\to$ BN $\to$ ReLU $\to$ MaxPool($2\times 2$).
   - Conv2: $16 \to 32$ ($k=3, p=1$) $\to$ BN $\to$ ReLU $\to$ MaxPool($2\times 2$).
   - Conv3: $32 \to 64$ ($k=3, p=1$) $\to$ BN $\to$ ReLU $\to$ AdaptiveAvgPool2d(($4, 4$)).
   - Dense: Flatten $\to$ Linear($1024 \to 512$) $\to$ LeakyReLU(0.1) $\to$ Dropout($p=0.50$).
   - Output: 512-dimensional spectral representation (548,576 parameters).
4. **Multimodal Late Fusion Head:**
   - Concatenation: $\text{dim} = 1024 + 512 = 1,536$.
   - Linear($1536 \to 128$) $\to$ LeakyReLU(0.1) $\to$ Dropout($p=0.30$) $\to$ Linear($128 \to 16$) $\to$ LeakyReLU(0.1) $\to$ Linear($16 \to 1$).
   - Parameters: 198,817.
5. **Total Trainable Parameters:** **775,932**.

---

## 4. The FFT Bug & Architectural Correction

During independent validation, an architectural defect was identified in early DualStream implementations:
- **Defect:** Calling `torch.fft.fftshift(fft)` without specifying dimensions causes PyTorch to circularly shift along **all 4 tensor dimensions**, including Dimension 0 (the mini-batch dimension). For batch size $B=16$, Dimension 0 was rolled by $\lfloor 16/2 \rfloor = 8$ positions.
- **Consequence:** Sample $i$'s spatial features were concatenated with sample $(i + 8) \pmod{16}$'s frequency features, training the model on misaligned cross-sample representations.
- **Correction:** Constraining the shift strictly to spatial dimensions in `dual_stream_model.py`:
  ```python
  fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
  ```
- **Regression Verification:** Batch invariance across batch sizes $B \in \{1, 2, 8, 16\}$ is strictly enforced by `test_dual_stream_batch_invariance_regression` in `backend/tests/test_deep_learning_pipeline.py` ($\text{atol} = 10^{-5}$, measured diff $< 10^{-7}$).
- **Checkpoints Note:** Earlier DualStream checkpoints trained with the unconstrained shift are invalid for the corrected architecture. The corrected model was retrained from scratch.

---

## 5. Training Dataset & Provenance

- **Dataset:** Small-scale Deepfake Forgery Video Dataset (SDFVD).
- **Attribution:** Shilpa Kaman and Dr. Aziz Makandar, Mendeley Data, 2024.
- **Persistent Identifier:** [DOI 10.17632/bcmkfgct2s.1](https://doi.org/10.17632/bcmkfgct2s.1)
- **Composition:** 106 videos (53 Real from Pexels, 53 Manipulated via face-swapping algorithms).
- **Extracted Crops:** 513 facial crops ($256 \times 256$ RGB, 257 Real, 256 Fake).
- **Pair-Grouped Splitting (Leakage-Safe):**
  Real video `v{i}` and fake video `vs{i}` contain the same human subject and background environment. Splitting was strictly performed at the subject pair level (`seed=42`):
  - **Train Split (70%):** 37 pairs = 74 videos $\to$ 355 crops (178 Real, 177 Fake).
  - **Validation Split (15%):** 8 pairs = 16 videos $\to$ 78 crops (39 Real, 39 Fake).
  - **Held-Out Test Split (15%):** 8 pairs = 16 videos $\to$ 80 crops (40 Real, 40 Fake).
- **External Pretraining:** **None.** All checkpoints were trained from scratch using PyTorch default random weight initialization. No external transfer learning weights were used.

---

## 6. Output Interpretation & Label Convention

### 6.1 Binary Label Mapping
- **Label 1.0:** `REAL` (Authentic facial media)
- **Label 0.0:** `FAKE` (Manipulated / Deepfake media)

### 6.2 Decision Thresholds
The neural network outputs a continuous authenticity probability $p \in [0.0, 1.0]$ via a Sigmoid activation:
- Standard default threshold: $\tau = 0.50$.
  - If $p \ge 0.50 \implies \textbf{REAL}$.
  - If $p < 0.50 \implies \textbf{FAKE}$.
- Validation-calibrated threshold: Derived from the validation partition using Youden's J statistic ($\tau_{\text{val}} = 0.7679$ for retrained DualStream). Calibrating on validation data shifts the model into a conservative high-specificity triage mode (100% fake video detection rate, but increased false positives on authentic clips).

### 6.3 Video Aggregation Strategy
For a video $V$ with $K$ uniformly sampled crops:
1. **Fraction Threshold (Default):** Flags video as `FAKE` if the proportion of manipulated crops exceeds `DEEPFAKE_VIDEO_FAKE_RATIO_THRESHOLD` (default: 25%).
2. **Mean Probability:** Averages authenticity scores across all crops: $\bar{p} = \frac{1}{K}\sum_{k=1}^K p_k$.

---

## 7. Quantitative Evaluation Results

Evaluated on the held-out SDFVD test partition (16 videos, 80 facial crops, zero identity overlap):

| Metric | Baseline MesoInception-4 (Active `.env`) | Extended MesoInception-4 (Research) | Retrained DualStreamForensicNet (Corrected FFT) |
|---|:---:|:---:|:---:|
| **Crop ROC-AUC** | 0.4931 | 0.7694 | **0.7731** |
| **Crop EER** | 52.50% | 30.00% | **30.00%** |
| **Crop Accuracy ($\tau=0.50$)** | 47.50% (38/80) | 71.25% (57/80) | **73.75% (59/80)** |
| **Video ROC-AUC** | 0.4844 | 0.7500 | **0.7969** |
| **Video EER** | 50.00% | 25.00% | **25.00%** |
| **Video Accuracy ($\tau=0.50$)** | 50.00% (8/16) | 68.75% (11/16) | **75.00% (12/16)** |
| **Fake Detection Specificity** | 37.50% (3/8 caught) | 75.00% (6/8 caught) | **87.50% (7/8 caught)** |
| **Fake Specificity at $\tau_{\text{val}}$** | 50.00% (4/8 caught) | 87.50% (7/8 caught) | **100.00% (8/8 caught)** |

---

## 8. Limitations & Threats to Validity

1. **Small Test Set Sample Size ($N=16$ Videos):**
   The test partition comprises exactly 16 videos (8 Real, 8 Fake). Each individual video classification changes the reported accuracy by **$6.25\%$** ($1/16$). The performance delta between DualStream (12/16) and Extended MesoInception (11/16) is exactly **one video**, which is not statistically significant ($p > 0.05$).
2. **Wilson Score Uncertainty:**
   An observed accuracy of $75.0\%$ on 16 samples carries a wide 95% Wilson score confidence interval of **$[50.50\%, 89.82\%]$**. These metrics represent pilot proof-of-concept indicators rather than certified production benchmarks.
3. **Cross-Dataset Generalization:**
   Models trained on SDFVD have not been validated against diverse commercial social media codecs (e.g., TikTok, WhatsApp re-encoding) or large-scale benchmarks like FaceForensics++ (c23/c40) and Celeb-DF v2.
4. **Haar Cascade Sensitivity:**
   OpenCV Haar cascade face detection can fail on extreme head poses, heavy motion blur, or partial occlusions (failed on 17 candidate frames during dataset preparation).

---

## 9. Intended and Prohibited Interpretations

### Intended Use
- **Investigative Decision Support:** Designed as a complementary analytical tool for forensic practitioners, researchers, and journalism evaluators.
- **Multimodal Triangulation:** Best used in conjunction with Error Level Analysis (ELA), 2D Fourier power spectra, and cryptographic SHA-256 custody tracking.

### Prohibited Interpretations
- **NO Legal Admissibility Claim:** VeriDoc outputs do not constitute legally certified evidence or self-authenticating proof under Federal Rules of Evidence Rule 902.
- **NO Conclusive Proof of Manipulation:** Neural network predictions, Grad-CAM overlays, and ELA heatmaps are diagnostic indicators, not conclusive proof of fraud.
- **SHA-256 Distinction:** SHA-256 verifies that a digital file has not been altered since ingestion; it does **not** verify that the video was authentic when recorded.

---

## 10. References

1. Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). *MesoNet: a Compact Facial Video Forgery Detection Network*. IEEE WIFS.
2. Frank, J., Banzhaf, T., Holz, K., et al. (2020). *Leveraging Frequency Analysis for Deep Fake Image Recognition*. ICML.
3. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE ICCV.
4. Kaman, S., & Makandar, A. (2024). *SDFVD: Small-scale Deepfake Forgery Video Dataset*. Mendeley Data, DOI: 10.17632/bcmkfgct2s.1.

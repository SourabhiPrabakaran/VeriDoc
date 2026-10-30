import torch
import torch.nn as nn
import numpy as np

from app.services.deepfake_model_interface import MesoInception4


class FftFeatureExtractor(nn.Module):
    """
    Convolutional frequency stream for 2D FFT magnitude spectra.
    Extracts frequency-domain forensic features (high-frequency energy anomalies,
    checkerboard artifacts, grid interpolation peaks).
    """
    def __init__(self, in_channels: int = 1):
        super(FftFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        return self.fc(x)


class DualStreamForensicNet(nn.Module):
    """
    Dual-Stream Neural Network for Deepfake Forensics.
    Fuses spatial RGB representations (Stream 1) with frequency FFT spectra (Stream 2).
    
    Academic Rationale:
    Generative manipulation techniques (GANs, Diffusion models, FaceSwap) leave subtle
    spectral footprints in high-frequency Fourier components that spatial CNNs often smooth out.
    Combining both streams via cross-modal feature fusion enhances forgery localization.
    """
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.5):
        super(DualStreamForensicNet, self).__init__()
        # Spatial stream: MesoInception-4 convolutional backbone
        self.spatial_stream = MesoInception4(num_classes=num_classes, dropout_rate=dropout_rate)

        # Frequency stream: FFT spectral feature extractor
        self.frequency_stream = FftFeatureExtractor(in_channels=1)

        # Spatial feature dimension: 16 * 8 * 8 = 1024
        # Frequency feature dimension: 512
        # Fused dimension: 1024 + 512 = 1536
        self.fusion_head = nn.Sequential(
            nn.Linear(1024 + 512, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(16, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def compute_fft_magnitude(rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes 2D Fast Fourier Transform magnitude spectrum from RGB tensor batch.
        Args:
            rgb_tensor: (B, 3, H, W) normalized tensor in range [0, 1].
        Returns:
            (B, 1, H, W) log-magnitude spectrum tensor normalized to [0, 1].
        """
        # Convert RGB to grayscale luminosity: 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * rgb_tensor[:, 0:1, :, :] + 0.587 * rgb_tensor[:, 1:2, :, :] + 0.114 * rgb_tensor[:, 2:3, :, :]
        # 2D FFT
        fft = torch.fft.fft2(gray)
        # Shift zero-frequency component to center along spatial dimensions (H, W) only
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        # Log-scaled magnitude spectrum
        magnitude = torch.log(torch.abs(fft_shifted) + 1e-6)
        # Normalize to [0, 1] per batch item
        b, c, h, w = magnitude.shape
        min_vals = magnitude.view(b, -1).min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        max_vals = magnitude.view(b, -1).max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        normalized = (magnitude - min_vals) / (max_vals - min_vals + 1e-6)
        return normalized

    def forward_logits(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through both streams and fusion head, returning unscaled logits."""
        # Stream 1: Spatial features
        spatial_feat_map = self.spatial_stream.forward_features(rgb_tensor)
        spatial_features = torch.flatten(spatial_feat_map, start_dim=1)  # (B, 1024)

        # Stream 2: Frequency features
        fft_tensor = self.compute_fft_magnitude(rgb_tensor)
        frequency_features = self.frequency_stream(fft_tensor)           # (B, 512)

        # Feature Fusion
        fused = torch.cat([spatial_features, frequency_features], dim=1)  # (B, 1536)
        logits = self.fusion_head(fused)
        return logits

    def forward(self, rgb_tensor: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass.
        If return_logits=True, returns raw logits suitable for BCEWithLogitsLoss.
        Otherwise applies sigmoid activation for probability output.
        """
        logits = self.forward_logits(rgb_tensor)
        if return_logits:
            return logits
        return self.sigmoid(logits)

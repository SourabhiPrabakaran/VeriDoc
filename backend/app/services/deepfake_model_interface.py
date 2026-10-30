import os
import hashlib
import io
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from app.core.config import settings

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5, out_7x7):
        super(InceptionBlock, self).__init__()
        
        # Branch 1: Conv 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: Conv 1x1 -> Conv 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: Conv 1x1 -> Conv 3x3 dilated (dilation=2)
        # Note: to preserve spatial resolution with dilation=2, kernel=3, padding must be 2.
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5, out_5x5, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: MaxPool 3x3 -> Conv 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_7x7, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class MesoInception4(nn.Module):
    def __init__(self, num_classes=1, dropout_rate: float = 0.5):
        super(MesoInception4, self).__init__()
        
        # Inception 1
        # Parameters match Keras: a=1, b=4, c=4, d=2
        # Concats to 1 + 4 + 4 + 2 = 11 channels
        self.inception1 = InceptionBlock(3, 1, 4, 4, 2)
        self.bn1 = nn.BatchNorm2d(11)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Inception 2
        # Parameters match Keras: a=2, b=4, c=4, d=2
        # Concats to 2 + 4 + 4 + 2 = 12 channels
        self.inception2 = InceptionBlock(11, 2, 4, 4, 2)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv 3
        # Keras: Conv2D(16, (5, 5), padding='same', activation='relu')
        self.conv3 = nn.Conv2d(12, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv 4
        # Keras: Conv2D(16, (5, 5), padding='same', activation='relu')
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        
        # Fully Connected Block
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(16 * 8 * 8, 16),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)
        )
        
        # Sigmoid activation for binary classification probability
        self.sigmoid = nn.Sigmoid()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional & pooling feature extractor."""
        x = self.inception1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = self.inception2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and linear head, returning raw unscaled logits."""
        features = self.forward_features(x)
        logits = self.fc(features)
        return logits

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass.
        If return_logits is True, returns raw logits suitable for BCEWithLogitsLoss.
        Otherwise applies sigmoid activation for probability output.
        """
        logits = self.forward_logits(x)
        if return_logits:
            return logits
        return self.sigmoid(logits)


class DeepfakeModelInterface:
    _model = None
    _model_status = "MODEL_NOT_CONFIGURED"
    _model_key = None

    @classmethod
    def get_model_status(cls) -> str:
        return cls._model_status

    @classmethod
    def load_model(cls):
        """
        Loads the PyTorch MesoInception-4 model state_dict from settings.DEEPFAKE_MODEL_PATH.
        """
        model_path = settings.DEEPFAKE_MODEL_PATH
        if model_path is None or settings.DEEPFAKE_MODEL_VERSION == "NOT_CONFIGURED":
            cls._model = None
            cls._model_key = None
            cls._model_status = "MODEL_NOT_CONFIGURED"
            return None

        path = Path(model_path)
        if not path.is_absolute():
            if path.exists() and path.is_file():
                path = path.resolve()
            elif (settings.BACKEND_DIR / path).exists() and (settings.BACKEND_DIR / path).is_file():
                path = (settings.BACKEND_DIR / path).resolve()
            elif (settings.BACKEND_DIR.parent / path).exists() and (settings.BACKEND_DIR.parent / path).is_file():
                path = (settings.BACKEND_DIR.parent / path).resolve()

        if not path.exists() or not path.is_file():
            cls._model = None
            cls._model_key = None
            cls._model_status = "MODEL_NOT_CONFIGURED"
            return None

        try:
            # Hash and load the same bytes; never reuse weights after configuration
            # or checkpoint contents change, even when the filename is unchanged.
            checkpoint_bytes = path.read_bytes()
            device = settings.DEEPFAKE_DEVICE or "cpu"
            model_key = (str(path.resolve()), hashlib.sha256(checkpoint_bytes).hexdigest(),
                         settings.DEEPFAKE_MODEL_VERSION, device)
            if cls._model is not None and cls._model_key == model_key:
                return cls._model
            # Load weights safely on device
            checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
            
            # Validate state_dict compatibility
            if isinstance(checkpoint, dict):
                # Unpack if state_dict is wrapped
                if "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    checkpoint = checkpoint["model"]

            # Select architecture: DualStreamForensicNet vs MesoInception4
            is_dual_stream = (
                settings.DEEPFAKE_MODEL_NAME.lower() in ("dualstreamforensicnet", "dual_stream", "dualstream")
                or (isinstance(checkpoint, dict) and any(k.startswith("spatial_stream.") for k in checkpoint.keys()))
            )
            if is_dual_stream:
                from training.dual_stream_model import DualStreamForensicNet
                model = DualStreamForensicNet(num_classes=1)
            else:
                model = MesoInception4(num_classes=1)
            
            expected = model.state_dict()
            if not isinstance(checkpoint, dict) or set(checkpoint) != set(expected):
                raise ValueError("Checkpoint must contain every model key and no unexpected keys.")
            for name, reference in expected.items():
                value = checkpoint[name]
                if not isinstance(value, torch.Tensor) or value.shape != reference.shape:
                    raise ValueError(f"Incompatible tensor shape/type for {name}.")
                if value.dtype != reference.dtype:
                    raise ValueError(f"Incompatible tensor dtype for {name}.")
                if not torch.isfinite(value).all().item():
                    raise ValueError(f"Non-finite checkpoint tensor: {name}.")
            model.load_state_dict(checkpoint, strict=True)
            model.to(device)
            model.eval()
            
            cls._model = model
            cls._model_key = model_key
            cls._model_status = "CONFIGURED"
            return model
            
        except Exception as e:
            cls._model = None
            cls._model_key = None
            cls._model_status = "MODEL_ERROR"
            print(f"Error loading deepfake model checkpoint from {model_path}: {e}")
            return None

    @classmethod
    def predict(cls, face_crop_path: str) -> dict:
        """
        Performs inference on a face crop file.
        Arguments:
            face_crop_path: Absolute physical path to the face crop image, or None (to trigger fallback).
        Returns:
            A structured dict with fields: label, confidence, model_status, model_name
        """
        # If face_crop_path is None or model isn't loaded/configured, trigger fallback
        if face_crop_path is None:
            return {
                "label": cls._model_status,
                "confidence": None,
                "model_status": cls._model_status,
                "model_name": settings.DEEPFAKE_MODEL_NAME
            }

        model = cls.load_model()
        if model is None:
            return {
                "label": cls._model_status,
                "confidence": None,
                "model_status": cls._model_status,
                "model_name": settings.DEEPFAKE_MODEL_NAME
            }

        try:
            # 1. Load crop from physical path
            img = cv2.imread(face_crop_path)
            if img is None:
                raise ValueError(f"OpenCV failed to load face crop image at {face_crop_path}")
                
            # 2. Resize to exactly 256x256
            img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # 3. Convert BGR -> RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # 4. Convert to float32 and normalize using /255.0
            img_normalized = img_rgb.astype(np.float32) / 255.0
            
            # 5. Transpose HWC -> CHW
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            
            # 6. Add batch dimension and convert to PyTorch Tensor
            tensor = torch.from_numpy(img_transposed).unsqueeze(0)
            
            # Move to device
            device = settings.DEEPFAKE_DEVICE or "cpu"
            tensor = tensor.to(device)
            
            # Run inference
            with torch.inference_mode():
                output = model(tensor)
                p = output.item()
                
            if not np.isfinite(p) or not (0.0 <= p <= 1.0):
                raise ValueError(f"Inference output probability {p} is out of bounds or not finite.")
                
            # Threshold and Decision Logic
            threshold = float(getattr(settings, "DEEPFAKE_CLASSIFICATION_THRESHOLD", 0.5))

            # Label Mapping
            # Output represents probability of AUTHENTICITY (REAL).
            # Output >= threshold is classified as REAL, output < threshold is classified as FAKE.
            if p >= threshold:
                label = "REAL"
                boundary_dist = (p - threshold) / max(1e-6, 1.0 - threshold)
            else:
                label = "FAKE"
                boundary_dist = (threshold - p) / max(1e-6, threshold)

            boundary_dist = float(np.clip(boundary_dist, 0.0, 1.0))

            return {
                "label": label,
                "confidence": boundary_dist,  # Boundary distance from threshold (0.0 at threshold, 1.0 at extremes)
                "boundary_distance": boundary_dist,
                "raw_probability": p,
                "threshold_used": threshold,
                "is_calibrated": False,
                "model_status": "CONFIGURED",
                "model_name": settings.DEEPFAKE_MODEL_NAME
            }
            
        except Exception as e:
            print(f"Error during deepfake inference: {e}")
            return {
                "label": "MODEL_ERROR",
                "confidence": None,
                "model_status": "MODEL_ERROR",
                "model_name": settings.DEEPFAKE_MODEL_NAME
            }

    @classmethod
    def predict_batch(cls, face_crops: list) -> list[dict]:
        """
        Performs batch inference on a list of face crop physical paths.
        Batches valid inputs into a single PyTorch tensor for improved throughput.
        """
        if not face_crops:
            return []

        model = cls.load_model()
        if model is None:
            return [
                {
                    "label": cls._model_status,
                    "confidence": None,
                    "model_status": cls._model_status,
                    "model_name": settings.DEEPFAKE_MODEL_NAME
                }
                for _ in face_crops
            ]

        results = [None] * len(face_crops)
        valid_indices = []
        valid_tensors = []

        for idx, crop_path in enumerate(face_crops):
            if crop_path is None:
                results[idx] = {
                    "label": cls._model_status,
                    "confidence": None,
                    "model_status": cls._model_status,
                    "model_name": settings.DEEPFAKE_MODEL_NAME
                }
                continue

            try:
                img = cv2.imread(str(crop_path))
                if img is None:
                    raise ValueError(f"Failed to read face crop image at {crop_path}")
                img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0
                img_transposed = np.transpose(img_normalized, (2, 0, 1))
                tensor = torch.from_numpy(img_transposed)
                valid_indices.append(idx)
                valid_tensors.append(tensor)
            except Exception as e:
                results[idx] = {
                    "label": "MODEL_ERROR",
                    "confidence": None,
                    "model_status": "MODEL_ERROR",
                    "model_name": settings.DEEPFAKE_MODEL_NAME
                }

        if valid_tensors:
            try:
                device = settings.DEEPFAKE_DEVICE or "cpu"
                batch_tensor = torch.stack(valid_tensors, dim=0).to(device)
                with torch.inference_mode():
                    outputs = model(batch_tensor).squeeze(-1).cpu().numpy()

                threshold = float(getattr(settings, "DEEPFAKE_CLASSIFICATION_THRESHOLD", 0.5))

                for original_idx, p_val in zip(valid_indices, outputs):
                    p = float(p_val)
                    if not np.isfinite(p) or not (0.0 <= p <= 1.0):
                        results[original_idx] = {
                            "label": "MODEL_ERROR",
                            "confidence": None,
                            "boundary_distance": None,
                            "raw_probability": None,
                            "threshold_used": threshold,
                            "is_calibrated": False,
                            "model_status": "MODEL_ERROR",
                            "model_name": settings.DEEPFAKE_MODEL_NAME
                        }
                    else:
                        if p >= threshold:
                            label = "REAL"
                            boundary_dist = (p - threshold) / max(1e-6, 1.0 - threshold)
                        else:
                            label = "FAKE"
                            boundary_dist = (threshold - p) / max(1e-6, threshold)
                        boundary_dist = float(np.clip(boundary_dist, 0.0, 1.0))

                        results[original_idx] = {
                            "label": label,
                            "confidence": boundary_dist,
                            "boundary_distance": boundary_dist,
                            "raw_probability": p,
                            "threshold_used": threshold,
                            "is_calibrated": False,
                            "model_status": "CONFIGURED",
                            "model_name": settings.DEEPFAKE_MODEL_NAME
                        }
            except Exception as e:
                print(f"Error during batched deepfake inference: {e}")
                for original_idx in valid_indices:
                    if results[original_idx] is None:
                        results[original_idx] = {
                            "label": "MODEL_ERROR",
                            "confidence": None,
                            "model_status": "MODEL_ERROR",
                            "model_name": settings.DEEPFAKE_MODEL_NAME
                        }

        # Fallback check for any unfilled slots
        for idx in range(len(results)):
            if results[idx] is None:
                results[idx] = {
                    "label": "MODEL_ERROR",
                    "confidence": None,
                    "model_status": "MODEL_ERROR",
                    "model_name": settings.DEEPFAKE_MODEL_NAME
                }

        return results

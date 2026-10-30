import uuid
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from app.core.config import settings
from app.services.deepfake_model_interface import DeepfakeModelInterface, MesoInception4


class GradCAMService:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for MesoInception-4.
    Highlights facial regions with high activation gradients towards REAL or FAKE classification.
    
    Academic Note: Grad-CAM is an explanatory diagnostic visualization and does NOT constitute
    factual or legal proof of manipulation or authenticity.
    """

    @classmethod
    def generate_heatmap(
        cls,
        face_crop_path: str,
        target_class_is_real: bool = False
    ) -> Dict[str, Any]:
        """
        Generates a Grad-CAM activation heatmap overlay for a face crop image.
        
        Args:
            face_crop_path: Physical path to the face crop image.
            target_class_is_real: If True, computes gradients for REAL (1.0). If False, for FAKE (0.0).
            
        Returns:
            Dictionary with visualization filename, status, and target class description.
        """
        # Ensure model is configured
        model = DeepfakeModelInterface.load_model()
        if model is None:
            return {
                "status": "MODEL_NOT_CONFIGURED",
                "error": "Cannot generate Grad-CAM: deepfake model is not configured with trained weights.",
                "gradcam_filename": None
            }

        crop_path = Path(face_crop_path)
        if not crop_path.is_file():
            return {
                "status": "FAILED",
                "error": f"Face crop image file not found at: {face_crop_path}",
                "gradcam_filename": None
            }

        try:
            # 1. Load and prepare image
            orig_bgr = cv2.imread(str(crop_path))
            if orig_bgr is None:
                raise ValueError("OpenCV failed to decode face crop image.")

            orig_resized = cv2.resize(orig_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_normalized, (2, 0, 1))

            device = settings.DEEPFAKE_DEVICE or "cpu"
            input_tensor = torch.from_numpy(img_transposed).unsqueeze(0).to(device)
            input_tensor.requires_grad = True

            # 2. Hook activations and gradients on the target convolutional layer (conv4)
            target_layer = getattr(model, "spatial_stream", model).conv4
            activations = []
            gradients = []

            def forward_hook(module, input, output):
                activations.append(output)

            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0])

            handle_fwd = target_layer.register_forward_hook(forward_hook)
            handle_bwd = target_layer.register_full_backward_hook(backward_hook)

            # 3. Forward pass (with gradients enabled for Grad-CAM)
            model.zero_grad()
            logits = model.forward_logits(input_tensor)

            # Target score: for FAKE, we compute gradients with respect to -logits (or 1 - prob)
            # For REAL, with respect to logits
            score = logits[0, 0] if target_class_is_real else -logits[0, 0]

            # 4. Backward pass
            score.backward()

            handle_fwd.remove()
            handle_bwd.remove()

            if not activations or not gradients:
                raise RuntimeError("Failed to capture feature activations or gradients during backward hook.")

            act = activations[0].detach()  # Shape: (1, 16, H, W)
            grad = gradients[0].detach()   # Shape: (1, 16, H, W)

            # 5. Global average pooling of gradients to obtain channel importance weights
            alpha = torch.mean(grad, dim=(2, 3), keepdim=True)  # Shape: (1, 16, 1, 1)

            # 6. Weighted combination of activation maps followed by ReLU
            cam = torch.sum(alpha * act, dim=1, keepdim=True)   # Shape: (1, 1, H, W)
            cam = torch.relu(cam)

            # Squeeze to 2D numpy array
            cam_np = cam.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            cam_max = np.max(cam_np)
            if cam_max > 0:
                cam_normalized = cam_np / cam_max
            else:
                cam_normalized = np.zeros_like(cam_np)

            # 7. Resize heatmap to original image dimensions (256x256)
            heatmap = cv2.resize(cam_normalized, (256, 256), interpolation=cv2.INTER_LINEAR)
            heatmap_uint8 = np.uint8(255 * heatmap)

            # Apply COLORMAP_JET
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # 8. Overlay heatmap onto original image (alpha blending)
            alpha_blend = 0.5
            overlay = cv2.addWeighted(heatmap_color, alpha_blend, orig_resized, 1.0 - alpha_blend, 0)

            # 9. Save visualization in GRADCAM_DIR
            settings.GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
            output_filename = f"gradcam_{uuid.uuid4().hex}.jpg"
            output_path = settings.GRADCAM_DIR / output_filename
            cv2.imwrite(str(output_path), overlay)

            return {
                "status": "COMPLETED",
                "gradcam_filename": output_filename,
                "target_class": "REAL" if target_class_is_real else "FAKE",
                "layer_inspected": "conv4"
            }

        except Exception as e:
            return {
                "status": "FAILED",
                "error": f"Grad-CAM generation failed: {str(e)}",
                "gradcam_filename": None
            }

import os
import tempfile
import numpy as np
import torch
import cv2
import traceback
import torch.nn as nn
from pathlib import Path
from app.core.config import settings
from app.services.deepfake_model_interface import DeepfakeModelInterface, MesoInception4

def test_phase7():
    print("=== STARTING VERIDOC FORENSICS PHASE 7 MODEL INTEGRATION TESTS ===")

    # Test 9: Structural verification of PyTorch MesoInception-4 architecture
    print("Test 9: Structural verification of PyTorch MesoInception-4 architecture...")
    try:
        model = MesoInception4(num_classes=1)
        
        # Verify Inception Block 1
        # branch 1: Conv2d(3, 1, 1)
        assert isinstance(model.inception1.branch1[0], nn.Conv2d)
        assert model.inception1.branch1[0].in_channels == 3
        assert model.inception1.branch1[0].out_channels == 1
        assert model.inception1.branch1[0].kernel_size == (1, 1)
        
        # branch 2: Conv2d(3, 4, 1) -> Conv2d(4, 4, 3)
        assert isinstance(model.inception1.branch2[0], nn.Conv2d)
        assert model.inception1.branch2[0].in_channels == 3
        assert model.inception1.branch2[0].out_channels == 4
        assert model.inception1.branch2[0].kernel_size == (1, 1)
        
        assert isinstance(model.inception1.branch2[2], nn.Conv2d)
        assert model.inception1.branch2[2].in_channels == 4
        assert model.inception1.branch2[2].out_channels == 4
        assert model.inception1.branch2[2].kernel_size == (3, 3)
        
        # branch 3: Conv2d(3, 4, 1) -> Conv2d(4, 4, 3, dilation=2)
        assert isinstance(model.inception1.branch3[0], nn.Conv2d)
        assert model.inception1.branch3[0].in_channels == 3
        assert model.inception1.branch3[0].out_channels == 4
        assert model.inception1.branch3[0].kernel_size == (1, 1)
        
        assert isinstance(model.inception1.branch3[2], nn.Conv2d)
        assert model.inception1.branch3[2].in_channels == 4
        assert model.inception1.branch3[2].out_channels == 4
        assert model.inception1.branch3[2].kernel_size == (3, 3)
        assert model.inception1.branch3[2].dilation == (2, 2)
        
        # branch 4: MaxPool2d(3) -> Conv2d(3, 2, 1)
        assert isinstance(model.inception1.branch4[0], nn.MaxPool2d)
        assert model.inception1.branch4[0].kernel_size == 3
        assert model.inception1.branch4[0].stride == 1
        assert model.inception1.branch4[0].padding == 1
        
        assert isinstance(model.inception1.branch4[1], nn.Conv2d)
        assert model.inception1.branch4[1].in_channels == 3
        assert model.inception1.branch4[1].out_channels == 2
        assert model.inception1.branch4[1].kernel_size == (1, 1)
        
        # BatchNorm after concat (bn1)
        assert isinstance(model.bn1, nn.BatchNorm2d)
        assert model.bn1.num_features == 11
        
        # Inception Block 2
        # branch 1: Conv2d(11, 2, 1)
        assert isinstance(model.inception2.branch1[0], nn.Conv2d)
        assert model.inception2.branch1[0].in_channels == 11
        assert model.inception2.branch1[0].out_channels == 2
        
        # branch 2: Conv2d(11, 4, 1) -> Conv2d(4, 4, 3)
        assert isinstance(model.inception2.branch2[0], nn.Conv2d)
        assert model.inception2.branch2[0].in_channels == 11
        assert model.inception2.branch2[0].out_channels == 4
        assert isinstance(model.inception2.branch2[2], nn.Conv2d)
        assert model.inception2.branch2[2].in_channels == 4
        assert model.inception2.branch2[2].out_channels == 4
        
        # branch 3: Conv2d(11, 4, 1) -> Conv2d(4, 4, 3, dilation=2)
        assert isinstance(model.inception2.branch3[0], nn.Conv2d)
        assert model.inception2.branch3[0].in_channels == 11
        assert model.inception2.branch3[0].out_channels == 4
        assert isinstance(model.inception2.branch3[2], nn.Conv2d)
        assert model.inception2.branch3[2].in_channels == 4
        assert model.inception2.branch3[2].out_channels == 4
        assert model.inception2.branch3[2].dilation == (2, 2)
        
        # branch 4: MaxPool2d(3) -> Conv2d(11, 2, 1)
        assert isinstance(model.inception2.branch4[0], nn.MaxPool2d)
        assert isinstance(model.inception2.branch4[1], nn.Conv2d)
        assert model.inception2.branch4[1].in_channels == 11
        assert model.inception2.branch4[1].out_channels == 2
        
        # BatchNorm after concat (bn2)
        assert isinstance(model.bn2, nn.BatchNorm2d)
        assert model.bn2.num_features == 12
        
        # Conv3: input 12, output 16, kernel 5x5
        assert isinstance(model.conv3, nn.Conv2d)
        assert model.conv3.in_channels == 12
        assert model.conv3.out_channels == 16
        assert model.conv3.kernel_size == (5, 5)
        assert isinstance(model.bn3, nn.BatchNorm2d)
        
        # Conv4: input 16, output 16, kernel 5x5
        assert isinstance(model.conv4, nn.Conv2d)
        assert model.conv4.in_channels == 16
        assert model.conv4.out_channels == 16
        assert model.conv4.kernel_size == (5, 5)
        assert isinstance(model.bn4, nn.BatchNorm2d)
        
        # classifier dimensions
        # self.fc contains Flatten (Flatten is first), Dropout, Linear(1024, 16), LeakyReLU, Dropout, Linear(16, 1)
        assert isinstance(model.fc[0], nn.Flatten)
        assert isinstance(model.fc[1], nn.Dropout)
        assert isinstance(model.fc[2], nn.Linear)
        assert model.fc[2].in_features == 1024
        assert model.fc[2].out_features == 16
        assert isinstance(model.fc[3], nn.LeakyReLU)
        assert isinstance(model.fc[4], nn.Dropout)
        assert isinstance(model.fc[5], nn.Linear)
        assert model.fc[5].in_features == 16
        assert model.fc[5].out_features == 1
        
        print("[SUCCESS] PyTorch architecture structural checks completed successfully.")
        
    except Exception as e:
        print(f"[FAIL] PyTorch architecture structural verification failed: {e}")
        traceback.print_exc()
        exit(1)

    # Test 3: Model architecture initialization
    print("Test 3: Initializing model architecture...")
    try:
        model = MesoInception4(num_classes=1)
        print("[SUCCESS] Model initialized successfully.")
    except Exception as e:
        print(f"[FAIL] Model initialization failed: {e}")
        exit(1)

    # Test 4: Input tensor shape = (1,3,256,256)
    print("Test 4: Verifying model forward pass tensor shapes...")
    try:
        dummy_input = torch.randn(1, 3, 256, 256)
        output = model(dummy_input)
        assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"
        print("[SUCCESS] Output shape is correct.")
    except Exception as e:
        print(f"[FAIL] Tensor shape validation failed: {e}")
        exit(1)

    # Test 1: Missing checkpoint -> MODEL_NOT_CONFIGURED
    print("Test 1: Testing missing checkpoint behavior...")
    # Backup settings
    original_path = settings.DEEPFAKE_MODEL_PATH
    original_version = settings.DEEPFAKE_MODEL_VERSION
    
    try:
        # Set to nonexistent file
        settings.DEEPFAKE_MODEL_PATH = "nonexistent_checkpoint.pth"
        settings.DEEPFAKE_MODEL_VERSION = "1.0"
        
        # Reset interface model cache
        DeepfakeModelInterface._model = None
        DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"
        
        # Trigger prediction fallback
        pred = DeepfakeModelInterface.predict(None)
        assert pred["model_status"] == "MODEL_NOT_CONFIGURED"
        assert pred["confidence"] is None
        print("[SUCCESS] Missing checkpoint correctly fell back to MODEL_NOT_CONFIGURED.")
    except Exception as e:
        print(f"[FAIL] Missing checkpoint test failed: {e}")
        exit(1)

    # Test 2: Invalid checkpoint -> graceful failure
    print("Test 2: Testing invalid/corrupted checkpoint behavior...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            f.write(b"corrupted state dict content")
            invalid_path = f.name
            
        settings.DEEPFAKE_MODEL_PATH = invalid_path
        DeepfakeModelInterface._model = None
        
        # Explicitly trigger loading to fail and set status to MODEL_ERROR
        DeepfakeModelInterface.load_model()
        
        pred = DeepfakeModelInterface.predict(None)
        assert pred["model_status"] == "MODEL_ERROR", f"Expected MODEL_ERROR, got {pred['model_status']}"
        assert pred["confidence"] is None
        print("[SUCCESS] Invalid checkpoint handled gracefully with MODEL_ERROR.")
        
        # Cleanup temp file
        os.unlink(invalid_path)
    except Exception as e:
        print(f"[FAIL] Invalid checkpoint test failed: {e}")
        traceback.print_exc()
        exit(1)

    # Test 5: Successful inference using a VERIFIED compatible checkpoint if available
    # We will construct a dummy state_dict checkpoint to simulate a verified compatible checkpoint.
    # Since there are no real weights in the workspace, this test checks weight loading and prediction functionality.
    print("Test 5 & 6 & 7 & 8: Simulating compatible weight checkpoint loading and inference...")
    try:
        # Create a valid state dict
        valid_checkpoint = model.state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            valid_path = f.name
        
        torch.save(valid_checkpoint, valid_path)
        
        # Configure model settings
        settings.DEEPFAKE_MODEL_PATH = valid_path
        settings.DEEPFAKE_MODEL_VERSION = "1.0"
        DeepfakeModelInterface._model = None
        
        # Create a dummy face crop image using numpy and opencv
        dummy_img = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
            dummy_img_path = img_f.name
        cv2.imwrite(dummy_img_path, dummy_img)
        
        # Load the model explicitly and run prediction
        DeepfakeModelInterface.load_model()
        assert DeepfakeModelInterface._model_status == "CONFIGURED"
        
        pred = DeepfakeModelInterface.predict(dummy_img_path)
        print(f"Mock inference result: {pred}")
        
        assert pred["model_status"] == "CONFIGURED"
        assert pred["label"] in ("REAL", "FAKE")
        p_val = pred["confidence"]
        assert 0.0 <= p_val <= 1.0, f"Confidence {p_val} is out of bounds [0, 1]."
        
        print("[SUCCESS] Inference executed successfully, probability and confidence bounds verified.")
        
        # Cleanup
        os.unlink(valid_path)
        os.unlink(dummy_img_path)
        
    except Exception as e:
        print(f"[FAIL] Simulated compatible model inference failed: {e}")
        traceback.print_exc()
        exit(1)

    # Restore settings
    settings.DEEPFAKE_MODEL_PATH = original_path
    settings.DEEPFAKE_MODEL_VERSION = original_version
    DeepfakeModelInterface._model = None
    DeepfakeModelInterface._model_status = "MODEL_NOT_CONFIGURED"

    print("\n=== ALL PHASE 7 MODEL INTEGRATION TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_phase7()

import os
import h5py
import torch
import numpy as np
from pathlib import Path

# We can import the model class from the app
import sys
sys.path.append(str(Path(__file__).parent.parent))
from app.services.deepfake_model_interface import MesoInception4

def convert_h5_to_pytorch(h5_path, pth_path):
    print(f"Loading Keras checkpoint from {h5_path}...")
    f = h5py.File(h5_path, 'r')
    
    # We will build the PyTorch state_dict
    state_dict = {}
    
    # Define mapping from Keras layer to PyTorch parameter prefix
    # Keras group names in H5 are:
    # conv2d_1, conv2d_2, etc.
    # batch_normalization_1, etc.
    # dense_1, dense_2
    
    # Structural inspection maps:
    # Inception 1
    # conv2d_1 -> inception1.branch1.0
    # conv2d_2 -> inception1.branch2.0
    # conv2d_3 -> inception1.branch2.2
    # conv2d_4 -> inception1.branch3.0
    # conv2d_5 -> inception1.branch3.2
    # conv2d_6 -> inception1.branch4.1
    # conv2d_7 -> inception1.branch4.3
    # batch_normalization_1 -> bn1
    
    # Inception 2
    # conv2d_8 -> inception2.branch1.0
    # conv2d_9 -> inception2.branch2.0
    # conv2d_10 -> inception2.branch2.2
    # conv2d_11 -> inception2.branch3.0
    # conv2d_12 -> inception2.branch3.2
    # conv2d_13 -> inception2.branch4.1
    # conv2d_14 -> inception2.branch4.3
    # batch_normalization_2 -> bn2
    
    # Conv 3 / 4
    # conv2d_15 -> conv3
    # batch_normalization_3 -> bn3
    # conv2d_16 -> conv4
    # batch_normalization_4 -> bn4
    
    # Dense 1 / 2
    # dense_1 -> fc.2
    # dense_2 -> fc.5
    
    mapping = {
        # Inception 1 Convolutions
        'conv2d_1/conv2d_1/kernel:0': 'inception1.branch1.0.weight',
        'conv2d_1/conv2d_1/bias:0': 'inception1.branch1.0.bias',
        
        'conv2d_2/conv2d_2/kernel:0': 'inception1.branch2.0.weight',
        'conv2d_2/conv2d_2/bias:0': 'inception1.branch2.0.bias',
        'conv2d_3/conv2d_3/kernel:0': 'inception1.branch2.2.weight',
        'conv2d_3/conv2d_3/bias:0': 'inception1.branch2.2.bias',
        
        'conv2d_4/conv2d_4/kernel:0': 'inception1.branch3.0.weight',
        'conv2d_4/conv2d_4/bias:0': 'inception1.branch3.0.bias',
        'conv2d_5/conv2d_5/kernel:0': 'inception1.branch3.2.weight',
        'conv2d_5/conv2d_5/bias:0': 'inception1.branch3.2.bias',
        
        'conv2d_6/conv2d_6/kernel:0': 'inception1.branch4.1.weight',
        'conv2d_6/conv2d_6/bias:0': 'inception1.branch4.1.bias',
        'conv2d_7/conv2d_7/kernel:0': 'inception1.branch4.3.weight',
        'conv2d_7/conv2d_7/bias:0': 'inception1.branch4.3.bias',
        
        # Inception 1 BatchNorm
        'batch_normalization_1/batch_normalization_1/gamma:0': 'bn1.weight',
        'batch_normalization_1/batch_normalization_1/beta:0': 'bn1.bias',
        'batch_normalization_1/batch_normalization_1/moving_mean:0': 'bn1.running_mean',
        'batch_normalization_1/batch_normalization_1/moving_variance:0': 'bn1.running_var',
        
        # Inception 2 Convolutions
        'conv2d_8/conv2d_8/kernel:0': 'inception2.branch1.0.weight',
        'conv2d_8/conv2d_8/bias:0': 'inception2.branch1.0.bias',
        
        'conv2d_9/conv2d_9/kernel:0': 'inception2.branch2.0.weight',
        'conv2d_9/conv2d_9/bias:0': 'inception2.branch2.0.bias',
        'conv2d_10/conv2d_10/kernel:0': 'inception2.branch2.2.weight',
        'conv2d_10/conv2d_10/bias:0': 'inception2.branch2.2.bias',
        
        'conv2d_11/conv2d_11/kernel:0': 'inception2.branch3.0.weight',
        'conv2d_11/conv2d_11/bias:0': 'inception2.branch3.0.bias',
        'conv2d_12/conv2d_12/kernel:0': 'inception2.branch3.2.weight',
        'conv2d_12/conv2d_12/bias:0': 'inception2.branch3.2.bias',
        
        'conv2d_13/conv2d_13/kernel:0': 'inception2.branch4.1.weight',
        'conv2d_13/conv2d_13/bias:0': 'inception2.branch4.1.bias',
        'conv2d_14/conv2d_14/kernel:0': 'inception2.branch4.3.weight',
        'conv2d_14/conv2d_14/bias:0': 'inception2.branch4.3.bias',
        
        # Inception 2 BatchNorm
        'batch_normalization_2/batch_normalization_2/gamma:0': 'bn2.weight',
        'batch_normalization_2/batch_normalization_2/beta:0': 'bn2.bias',
        'batch_normalization_2/batch_normalization_2/moving_mean:0': 'bn2.running_mean',
        'batch_normalization_2/batch_normalization_2/moving_variance:0': 'bn2.running_var',
        
        # Conv 3
        'conv2d_15/conv2d_15/kernel:0': 'conv3.weight',
        'conv2d_15/conv2d_15/bias:0': 'conv3.bias',
        'batch_normalization_3/batch_normalization_3/gamma:0': 'bn3.weight',
        'batch_normalization_3/batch_normalization_3/beta:0': 'bn3.bias',
        'batch_normalization_3/batch_normalization_3/moving_mean:0': 'bn3.running_mean',
        'batch_normalization_3/batch_normalization_3/moving_variance:0': 'bn3.running_var',
        
        # Conv 4
        'conv2d_16/conv2d_16/kernel:0': 'conv4.weight',
        'conv2d_16/conv2d_16/bias:0': 'conv4.bias',
        'batch_normalization_4/batch_normalization_4/gamma:0': 'bn4.weight',
        'batch_normalization_4/batch_normalization_4/beta:0': 'bn4.bias',
        'batch_normalization_4/batch_normalization_4/moving_mean:0': 'bn4.running_mean',
        'batch_normalization_4/batch_normalization_4/moving_variance:0': 'bn4.running_var',
        
        # Dense 1 / 2
        'dense_1/dense_1/kernel:0': 'fc.2.weight',
        'dense_1/dense_1/bias:0': 'fc.2.bias',
        'dense_2/dense_2/kernel:0': 'fc.5.weight',
        'dense_2/dense_2/bias:0': 'fc.5.bias',
    }
    
    print("\n--- CONVERTING TENSORS ---")
    for keras_name, pytorch_name in mapping.items():
        if keras_name not in f:
            raise KeyError(f"Expected dataset {keras_name} not found in H5 checkpoint.")
            
        data = np.array(f[keras_name])
        
        # Handle transpositions
        if 'kernel' in keras_name:
            if 'dense_1' in keras_name:
                # Keras dense_1 kernel needs permutation from channels-last to channels-first flattening order
                # Keras shape: (1024, 16) -> PyTorch shape: (16, 1024)
                H, W, C = 8, 8, 16
                permuted_data = np.zeros((16, 1024), dtype=np.float32)
                for h in range(H):
                    for w in range(W):
                        for c in range(C):
                            idx_keras = h * W * C + w * C + c
                            idx_pytorch = c * H * W + h * W + w
                            permuted_data[:, idx_pytorch] = data[idx_keras, :]
                data = permuted_data
            elif 'dense_2' in keras_name:
                # Keras Dense kernel shape: (In_Features, Out_Features) -> PyTorch Weight: (Out_Features, In_Features)
                data = np.transpose(data, (1, 0))
            else:
                # Keras Conv2D kernel shape: (H, W, In_C, Out_C) -> PyTorch Weight: (Out_C, In_C, H, W)
                data = np.transpose(data, (3, 2, 0, 1))
                
        tensor = torch.from_numpy(data)
        state_dict[pytorch_name] = tensor
        print(f"Mapped {keras_name} ({f[keras_name].shape}) -> {pytorch_name} ({tensor.shape})")
        
    f.close()
    
    # Save the PyTorch checkpoint
    os.makedirs(os.path.dirname(pth_path), exist_ok=True)
    torch.save(state_dict, pth_path)
    print(f"\nSaved PyTorch state_dict checkpoint to {pth_path} ({os.path.getsize(pth_path) / 1024:.2f} KB)")
    
    # Verify strict loading into PyTorch model
    print("\n--- VERIFYING CHECKPOINT LOADING ---")
    model = MesoInception4(num_classes=1)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[SUCCESS] Checkpoint loaded successfully with strict=True!")
    except Exception as e:
        print(f"[FAIL] Checkpoint loading with strict=True failed: {e}")
        raise e

if __name__ == "__main__":
    h5_path = "data/MesoInception_DF.h5"
    pth_path = "data/mesoinception_4.pth"
    convert_h5_to_pytorch(h5_path, pth_path)

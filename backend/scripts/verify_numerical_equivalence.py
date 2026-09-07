import os
import sys
import numpy as np
import torch
import tensorflow as tf
from pathlib import Path

# Set up python path
sys.path.append(str(Path(__file__).parent.parent))

# Import PyTorch model class
from app.services.deepfake_model_interface import MesoInception4

# Disable GPU for TensorFlow to match CPU PyTorch execution
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Disable TF warnings
import logging
tf.get_logger().setLevel(logging.ERROR)

def build_keras_model():
    from tensorflow.keras.models import Model as KerasModel
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense, LeakyReLU, concatenate
    
    # Layer naming index tracker
    conv_idx = 1
    
    def InceptionLayer(a, b, c, d):
        nonlocal conv_idx
        c1_name = f"conv2d_{conv_idx}"
        c2_name = f"conv2d_{conv_idx+1}"
        c3_name = f"conv2d_{conv_idx+2}"
        c4_name = f"conv2d_{conv_idx+3}"
        c5_name = f"conv2d_{conv_idx+4}"
        c6_name = f"conv2d_{conv_idx+5}"
        c7_name = f"conv2d_{conv_idx+6}"
        conv_idx += 7
        
        def func(x):
            # Branch 1
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu', name=c1_name)(x)
            
            # Branch 2
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu', name=c2_name)(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu', name=c3_name)(x2)
            
            # Branch 3
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu', name=c4_name)(x)
            x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu', name=c5_name)(x3)
            
            # Branch 4
            x4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu', name=c6_name)(x4)
            x4 = Conv2D(d, (3, 3), padding='same', activation='relu', name=c7_name)(x4)
            
            return concatenate([x1, x2, x3, x4], axis=-1)
        return func

    x = Input(shape=(256, 256, 3))
    
    # Block 1
    x1 = InceptionLayer(1, 4, 4, 2)(x)
    x1 = BatchNormalization(name="batch_normalization_1")(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
    
    # Block 2
    x2 = InceptionLayer(2, 4, 4, 2)(x1)
    x2 = BatchNormalization(name="batch_normalization_2")(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
    
    # Block 3
    x3 = Conv2D(16, (5, 5), padding='same', activation='relu', name=f"conv2d_{conv_idx}")(x2)
    x3 = BatchNormalization(name="batch_normalization_3")(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
    conv_idx += 1
    
    # Block 4
    x4 = Conv2D(16, (5, 5), padding='same', activation='relu', name=f"conv2d_{conv_idx}")(x3)
    x4 = BatchNormalization(name="batch_normalization_4")(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
    conv_idx += 1
    
    # Dense Classifier
    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16, name="dense_1")(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation='sigmoid', name="dense_2")(y)
    
    return KerasModel(inputs=x, outputs=y)

def verify_numerical_equivalence():
    print("=== STARTING NUMERICAL EQUIVALENCE TEST ===")
    
    h5_path = "data/MesoInception_DF.h5"
    pth_path = "data/mesoinception_4.pth"
    
    # 1. Initialize PyTorch model and load weights
    print("\nInitializing PyTorch model and loading mesoinception_4.pth...")
    pytorch_model = MesoInception4(num_classes=1)
    pytorch_state = torch.load(pth_path, map_location='cpu', weights_only=True)
    pytorch_model.load_state_dict(pytorch_state, strict=True)
    pytorch_model.eval()
    print("[SUCCESS] PyTorch model loaded strictly.")
    
    # 2. Initialize Keras model and load weights
    print("\nInitializing Keras model and loading MesoInception_DF.h5...")
    keras_model = build_keras_model()
    keras_model.load_weights(h5_path)
    print("[SUCCESS] Keras weights loaded successfully.")
    
    # 3. Generate identical inputs
    print("\nGenerating identical preprocessed input tensors...")
    np.random.seed(42)
    # Generate 5 random images in shape (5, 256, 256, 3) with scale [0, 0.05]
    # to avoid saturation at 1.0 and test dynamic range matching.
    dummy_input_keras = (np.random.rand(5, 256, 256, 3) * 0.05).astype(np.float32)
    
    # Also inject a zero input tensor as the first sample
    dummy_input_keras[0] = 0.0
    
    # PyTorch expects channels-first (N, C, H, W)
    dummy_input_pytorch = np.transpose(dummy_input_keras, (0, 3, 1, 2))
    tensor_pytorch = torch.from_numpy(dummy_input_pytorch)
    
    # 4. Run PyTorch Inference
    print("\nRunning PyTorch Inference...")
    with torch.no_grad():
        output_pytorch = pytorch_model(tensor_pytorch).numpy()
        
    # 5. Run Keras Inference
    print("Running Keras Inference...")
    output_keras = keras_model.predict(dummy_input_keras, verbose=0)
    
    # 6. Compare Outputs
    print("\n--- INFERENCE COMPARISON REPORT ---")
    differences = []
    agreement_count = 0
    
    for i in range(5):
        p_pt = output_pytorch[i][0]
        p_ks = output_keras[i][0]
        
        diff = abs(p_pt - p_ks)
        differences.append(diff)
        
        # Classification agreement check (using 0.5 threshold)
        class_pt = "REAL" if p_pt >= 0.5 else "FAKE"
        class_ks = "REAL" if p_ks >= 0.5 else "FAKE"
        
        agreement = "AGREE" if class_pt == class_ks else "MISMATCH"
        if class_pt == class_ks:
            agreement_count += 1
            
        print(f"Sample {i+1}:")
        print(f"  PyTorch Probability: {p_pt:.6f} ({class_pt})")
        print(f"  Keras Probability:   {p_ks:.6f} ({class_ks})")
        print(f"  Absolute Difference: {diff:.6e} ({agreement})")
        
    max_diff = max(differences)
    mean_diff = sum(differences) / len(differences)
    agreement_percentage = (agreement_count / 5) * 100
    
    print("\n--- STATISTICS SUMMARY ---")
    print(f"Max Absolute Difference:  {max_diff:.6e}")
    print(f"Mean Absolute Difference: {mean_diff:.6e}")
    print(f"Classification Agreement: {agreement_percentage:.1f}%")
    
    # Validation assertions
    if max_diff < 1e-4:
        print("\n[SUCCESS] PyTorch and Keras models are numerically equivalent within tolerance!")
    else:
        print("\n[FAIL] Numerical mismatch exceeded tolerance limits.")
        sys.exit(1)

if __name__ == "__main__":
    verify_numerical_equivalence()

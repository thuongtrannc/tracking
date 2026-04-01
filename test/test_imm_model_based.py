"""
Test script for Model-Based IMM
Demonstrates basic usage and comparison with traditional IMM
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from core.imm import IMM
from core.imm_model_based import ModelBasedIMM


def test_traditional_imm():
    """Test the traditional IMM implementation"""
    print("Testing Traditional IMM...")
    print("-" * 60)
    
    config_file = "configs/imm.json"
    data_file = "data/imm_single.txt"
    
    # Load data
    data = np.loadtxt(data_file, delimiter=',')
    ground_truth = data[:, :8]
    observations = data[:, [0, 1, 6, 7]]
    
    # Initialize IMM
    imm = IMM(config_file)
    
    # Run tracking
    estimates = []
    model_probs = []
    
    for t in range(min(100, len(observations))):  # Test on first 100 samples
        z = observations[t]
        imm.update(z)
        estimates.append(imm.get_estimate())
        model_probs.append(imm.get_model_prob().copy())
    
    estimates = np.array(estimates)
    model_probs = np.array(model_probs)
    
    # Calculate RMSE
    errors = estimates - ground_truth[:len(estimates)]
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    
    print(f"\nProcessed {len(estimates)} samples")
    print(f"Overall RMSE: {np.mean(rmse):.6f}")
    print("\nAverage model probabilities:")
    print(f"  CA: {np.mean(model_probs[:, 0]):.4f}")
    print(f"  CV: {np.mean(model_probs[:, 1]):.4f}")
    print(f"  CT: {np.mean(model_probs[:, 2]):.4f}")
    
    return estimates, model_probs


def test_model_based_imm(model_path=None):
    """Test the model-based IMM implementation"""
    print("\nTesting Model-Based IMM...")
    print("-" * 60)
    
    config_file = "configs/imm.json"
    data_file = "data/imm_single.txt"
    
    # Load data
    data = np.loadtxt(data_file, delimiter=',')
    ground_truth = data[:, :8]
    observations = data[:, [0, 1, 6, 7]]
    
    # Initialize IMM
    imm = ModelBasedIMM(config_file, model_path=model_path, device='cpu')
    
    # Run tracking
    estimates = []
    model_probs = []
    
    for t in range(min(100, len(observations))):  # Test on first 100 samples
        z = observations[t]
        imm.update(z)
        estimates.append(imm.get_estimate())
        model_probs.append(imm.get_model_prob().copy())
    
    estimates = np.array(estimates)
    model_probs = np.array(model_probs)
    
    # Calculate RMSE
    errors = estimates - ground_truth[:len(estimates)]
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    
    print(f"\nProcessed {len(estimates)} samples")
    if model_path:
        print(f"Using trained model: {model_path}")
    else:
        print("Using likelihood-based model selection (no trained model)")
    print(f"Overall RMSE: {np.mean(rmse):.6f}")
    print("\nAverage model probabilities:")
    print(f"  CA: {np.mean(model_probs[:, 0]):.4f}")
    print(f"  CV: {np.mean(model_probs[:, 1]):.4f}")
    print(f"  CT: {np.mean(model_probs[:, 2]):.4f}")
    
    return estimates, model_probs


def test_feature_generation():
    """Test the feature generation for training"""
    print("\nTesting Feature Generation...")
    print("-" * 60)
    
    config_file = "configs/imm.json"
    data_file = "data/imm_single.txt"
    
    # Load data
    data = np.loadtxt(data_file, delimiter=',')
    ground_truth = data[:50, :8]  # Use first 50 samples
    observations = data[:50, [0, 1, 6, 7]]
    
    # Initialize IMM
    imm = ModelBasedIMM(config_file, model_path=None, device='cpu')
    
    # Generate features
    features, labels = imm.get_model_differences_batch(observations, ground_truth)
    
    print(f"Generated features shape: {features.shape}")
    print(f"Generated labels shape: {labels.shape}")
    print(f"\nFeature dimension per model: {features.shape[2]}")
    print(f"Expected: 2 × obs_dim + 2 × state_dim = 2×4 + 2×8 = 24")
    
    print("\nSample features from first timestep:")
    print(f"  Model 0 (CA) features: {features[0, 0, :8]}")
    print(f"  Model 1 (CV) features: {features[0, 1, :8]}")
    print(f"  Model 2 (CT) features: {features[0, 2, :8]}")
    
    print("\nSample labels from first timestep:")
    print(f"  Model probabilities: {labels[0]}")
    print(f"  Sum: {np.sum(labels[0]):.6f} (should be 1.0)")
    
    return features, labels


if __name__ == "__main__":
    print("=" * 60)
    print("Model-Based IMM Testing Suite")
    print("=" * 60)
    
    # Test 1: Traditional IMM
    try:
        test_traditional_imm()
    except Exception as e:
        print(f"Traditional IMM test failed: {e}")
    
    # Test 2: Model-Based IMM without trained model
    try:
        test_model_based_imm(model_path=None)
    except Exception as e:
        print(f"Model-Based IMM test failed: {e}")
    
    # Test 3: Feature generation
    try:
        test_feature_generation()
    except Exception as e:
        print(f"Feature generation test failed: {e}")
    
    # Test 4: Model-Based IMM with trained model (if available)
    model_path = "models/imm_gru_model.pth"
    if os.path.exists(model_path):
        print("\n" + "=" * 60)
        print("Testing with Trained Model")
        print("=" * 60)
        try:
            test_model_based_imm(model_path=model_path)
        except Exception as e:
            print(f"Trained model test failed: {e}")
    else:
        print(f"\n\nNote: Trained model not found at {model_path}")
        print("Run 'python tools/train.py' to train the model first.")
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

#!/usr/bin/env python
"""
Complete working example of Model-Based IMM
Demonstrates both likelihood-based and neural network-based model selection
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from core.imm_model_based import ModelBasedIMM
from core.imm import IMM


def compare_imm_methods():
    """Compare traditional IMM, likelihood-based model-based IMM, and neural network-based IMM"""
    
    print("="*70)
    print("IMM Tracking Comparison")
    print("="*70)
    
    # Load data
    data_file = "data/imm_single.txt"
    config_file = "configs/imm.json"
    
    data = np.loadtxt(data_file, delimiter=',')
    ground_truth = data[:200, :8]  # Use first 200 samples for quick demo
    observations = data[:200, [0, 1, 6, 7]]
    
    print(f"\nDataset: {len(observations)} samples")
    print(f"Observation dimension: {observations.shape[1]}")
    print(f"State dimension: {ground_truth.shape[1]}")
    
    # ========================================================================
    # Method 1: Traditional IMM with transition matrix
    # ========================================================================
    print("\n" + "="*70)
    print("Method 1: Traditional IMM (Transition Matrix)")
    print("="*70)
    
    imm_traditional = IMM(config_file)
    estimates_trad = []
    probs_trad = []
    
    for t in range(len(observations)):
        z = observations[t]
        imm_traditional.update(z)
        estimates_trad.append(imm_traditional.get_estimate())
        probs_trad.append(imm_traditional.get_model_prob().copy())
    
    estimates_trad = np.array(estimates_trad)
    probs_trad = np.array(probs_trad)
    
    errors_trad = estimates_trad - ground_truth
    rmse_trad = np.sqrt(np.mean(errors_trad ** 2, axis=0))
    
    print(f"\nOverall RMSE: {np.mean(rmse_trad):.6f}")
    print("\nRMSE by component:")
    state_names = ['x', 'y', 'v', 'dv', 'w', 'dw', 'width', 'length']
    for i, name in enumerate(state_names):
        print(f"  {name:6s}: {rmse_trad[i]:10.6f}")
    
    print("\nAverage model probabilities:")
    print(f"  CA: {np.mean(probs_trad[:, 0]):.4f}")
    print(f"  CV: {np.mean(probs_trad[:, 1]):.4f}")
    print(f"  CT: {np.mean(probs_trad[:, 2]):.4f}")
    
    # ========================================================================
    # Method 2: Model-Based IMM with likelihood (no neural network)
    # ========================================================================
    print("\n" + "="*70)
    print("Method 2: Model-Based IMM (Likelihood-Based)")
    print("="*70)
    
    imm_likelihood = ModelBasedIMM(config_file, model_path=None, device='cpu')
    estimates_like = []
    probs_like = []
    
    for t in range(len(observations)):
        z = observations[t]
        imm_likelihood.update(z)
        estimates_like.append(imm_likelihood.get_estimate())
        probs_like.append(imm_likelihood.get_model_prob().copy())
    
    estimates_like = np.array(estimates_like)
    probs_like = np.array(probs_like)
    
    errors_like = estimates_like - ground_truth
    rmse_like = np.sqrt(np.mean(errors_like ** 2, axis=0))
    
    print(f"\nOverall RMSE: {np.mean(rmse_like):.6f}")
    print("\nRMSE by component:")
    for i, name in enumerate(state_names):
        print(f"  {name:6s}: {rmse_like[i]:10.6f}")
    
    print("\nAverage model probabilities:")
    print(f"  CA: {np.mean(probs_like[:, 0]):.4f}")
    print(f"  CV: {np.mean(probs_like[:, 1]):.4f}")
    print(f"  CT: {np.mean(probs_like[:, 2]):.4f}")
    
    # ========================================================================
    # Method 3: Model-Based IMM with neural network (if available)
    # ========================================================================
    model_path = "models/imm_gru_model.pth"
    estimates_neural = []
    probs_neural = []
    rmse_neural = None
    
    if os.path.exists(model_path):
        print("\n" + "="*70)
        print("Method 3: Model-Based IMM (Neural Network)")
        print("="*70)
        print("Note: Neural network model loading may fail due to class definition issues.")
        print("      This is expected. Retrain the model if needed: python tools/train.py")
        
        try:
            imm_neural = ModelBasedIMM(config_file, model_path=model_path, device='cpu')
            
            for t in range(len(observations)):
                z = observations[t]
                imm_neural.update(z)
                est = imm_neural.get_estimate()
                prob = imm_neural.get_model_prob().copy()
                
                # Check for NaN
                if np.isnan(est).any() or np.isnan(prob).any():
                    print(f"\nWarning: NaN detected at timestep {t}, stopping...")
                    break
                    
                estimates_neural.append(est)
                probs_neural.append(prob)
            
            if len(estimates_neural) > 0:
                estimates_neural = np.array(estimates_neural)
                probs_neural = np.array(probs_neural)
                
                errors_neural = estimates_neural - ground_truth[:len(estimates_neural)]
                rmse_neural = np.sqrt(np.mean(errors_neural ** 2, axis=0))
                
                print(f"\nProcessed {len(estimates_neural)} samples")
                print(f"Overall RMSE: {np.mean(rmse_neural):.6f}")
                print("\nRMSE by component:")
                for i, name in enumerate(state_names):
                    print(f"  {name:6s}: {rmse_neural[i]:10.6f}")
                
                print("\nAverage model probabilities:")
                print(f"  CA: {np.mean(probs_neural[:, 0]):.4f}")
                print(f"  CV: {np.mean(probs_neural[:, 1]):.4f}")
                print(f"  CT: {np.mean(probs_neural[:, 2]):.4f}")
        except Exception as e:
            print(f"\nNeural network method failed: {e}")
            print("This is expected if the model was trained from a different script.")
    else:
        print(f"\n\nNeural network model not found at: {model_path}")
        print("Train the model first using: python tools/train.py")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Traditional IMM RMSE:     {np.mean(rmse_trad):.6f}")
    print(f"Likelihood-based IMM RMSE: {np.mean(rmse_like):.6f}")
    if rmse_neural is not None and len(estimates_neural) > 0:
        print(f"Neural network IMM RMSE:   {np.mean(rmse_neural):.6f} (first {len(estimates_neural)} samples)")
    else:
        print("Neural network IMM:        Not available")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    compare_imm_methods()

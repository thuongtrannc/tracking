#!/usr/bin/env python
"""
Evaluate MSE between IMM filter output and ground truth
Compares position (x, y) only
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import argparse


def load_data(filepath):
    """Load data from text file"""
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compute_position_mse(predictions, ground_truth):
    """
    Compute MSE on position (x, y) only
    
    Args:
        predictions: (N, 8) array with [x, y, v, dv, w, dw, width, length]
        ground_truth: (N, 8) array with same format
    
    Returns:
        Dictionary with MSE metrics
    """
    # Extract positions (first 2 columns: x, y)
    pred_pos = predictions[:, :2]
    gt_pos = ground_truth[:, :2]
    
    # Ensure same length
    min_len = min(len(pred_pos), len(gt_pos))
    pred_pos = pred_pos[:min_len]
    gt_pos = gt_pos[:min_len]
    
    # Compute errors
    errors = pred_pos - gt_pos
    squared_errors = errors ** 2
    
    # MSE for x and y separately
    mse_x = np.mean(squared_errors[:, 0])
    mse_y = np.mean(squared_errors[:, 1])
    
    # Overall position MSE (average of x and y)
    mse_position = np.mean(squared_errors)
    
    # RMSE
    rmse_x = np.sqrt(mse_x)
    rmse_y = np.sqrt(mse_y)
    rmse_position = np.sqrt(mse_position)
    
    # Per-timestep Euclidean distance
    euclidean_distances = np.sqrt(np.sum(squared_errors, axis=1))
    mean_euclidean_distance = np.mean(euclidean_distances)
    
    return {
        'mse_x': mse_x,
        'mse_y': mse_y,
        'mse_position': mse_position,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'rmse_position': rmse_position,
        'mean_euclidean_distance': mean_euclidean_distance,
        'max_euclidean_distance': np.max(euclidean_distances),
        'min_euclidean_distance': np.min(euclidean_distances),
        'num_samples': min_len
    }


def print_metrics(metrics, title="Evaluation Results"):
    """Print evaluation metrics in a formatted way"""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)
    print(f"Number of samples: {metrics['num_samples']}")
    print("\nMean Squared Error (MSE):")
    print(f"  MSE (x):        {metrics['mse_x']:.6f}")
    print(f"  MSE (y):        {metrics['mse_y']:.6f}")
    print(f"  MSE (position): {metrics['mse_position']:.6f}")
    
    print("\nRoot Mean Squared Error (RMSE):")
    print(f"  RMSE (x):        {metrics['rmse_x']:.6f}")
    print(f"  RMSE (y):        {metrics['rmse_y']:.6f}")
    print(f"  RMSE (position): {metrics['rmse_position']:.6f}")
    
    print("\nEuclidean Distance Statistics:")
    print(f"  Mean:   {metrics['mean_euclidean_distance']:.6f}")
    print(f"  Max:    {metrics['max_euclidean_distance']:.6f}")
    print(f"  Min:    {metrics['min_euclidean_distance']:.6f}")
    print("="*70 + "\n")


def evaluate_imm(imm_file='data/filter_imm.txt', gt_file='data/imm_single_gt.txt'):
    """
    Main evaluation function
    
    Args:
        imm_file: Path to IMM filter output file
        gt_file: Path to ground truth file
    """
    print("="*70)
    print("IMM Position Evaluation")
    print("="*70)
    print(f"IMM output file: {imm_file}")
    print(f"Ground truth file: {gt_file}")
    
    # Load data
    print("\nLoading data...")
    imm_data = load_data(imm_file)
    gt_data = load_data(gt_file)
    
    if imm_data is None or gt_data is None:
        print("Error: Could not load data files")
        return None
    
    print(f"  IMM data shape: {imm_data.shape}")
    print(f"  GT data shape: {gt_data.shape}")
    
    # Check if files exist and have correct format
    if imm_data.shape[1] < 2:
        print(f"Error: IMM file should have at least 2 columns (x, y), got {imm_data.shape[1]}")
        return None
    
    if gt_data.shape[1] < 2:
        print(f"Error: GT file should have at least 2 columns (x, y), got {gt_data.shape[1]}")
        return None
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_position_mse(imm_data, gt_data)
    
    # Print results
    print_metrics(metrics, title="IMM vs Ground Truth - Position Evaluation")
    
    return metrics


def compare_multiple_models(model_files, gt_file='data/imm_single_gt.txt'):
    """
    Compare multiple model outputs against ground truth
    
    Args:
        model_files: Dictionary of {model_name: filepath}
        gt_file: Path to ground truth file
    """
    print("="*70)
    print("Multiple Model Comparison")
    print("="*70)
    
    # Load ground truth
    gt_data = load_data(gt_file)
    if gt_data is None:
        print("Error: Could not load ground truth file")
        return
    
    print(f"Ground truth: {gt_file} (shape: {gt_data.shape})")
    print("\nEvaluating models...\n")
    
    # Store results
    results = {}
    
    # Evaluate each model
    for model_name, model_file in model_files.items():
        if not os.path.exists(model_file):
            print(f"⚠ Warning: {model_file} not found, skipping {model_name}")
            continue
        
        model_data = load_data(model_file)
        if model_data is None:
            print(f"⚠ Warning: Could not load {model_file}, skipping {model_name}")
            continue
        
        metrics = compute_position_mse(model_data, gt_data)
        results[model_name] = metrics
        
        print_metrics(metrics, title=f"{model_name}")
    
    # Summary comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("Summary Comparison")
        print("="*70)
        print(f"{'Model':<30} | {'RMSE Position':>12} | {'Mean Euclidean':>15}")
        print("-"*70)
        
        for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['rmse_position']):
            print(f"{model_name:<30} | {metrics['rmse_position']:>12.6f} | {metrics['mean_euclidean_distance']:>15.6f}")
        
        print("="*70 + "\n")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['rmse_position'])
        print(f"🏆 Best model: {best_model[0]} (RMSE: {best_model[1]['rmse_position']:.6f})")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate IMM position tracking accuracy')
    parser.add_argument('--imm-file', type=str, default='data/filter_imm.txt',
                        help='Path to IMM filter output file')
    parser.add_argument('--gt-file', type=str, default='data/imm_single_gt.txt',
                        help='Path to ground truth file')
    # parser.add_argument('--compare-all', action='store_true',
    #                     help='Compare all available filter outputs')
    
    args = parser.parse_args()
    
    # if args.compare_all:
    #     # Compare multiple models
    #     model_files = {
    #         'Traditional IMM': 'data/filter_imm.txt',
    #         # 'CA Filter': 'data/filter_ca.txt',
    #         # 'CV Filter': 'data/filter_cv.txt',
    #         # 'CT Filter': 'data/filter_ct.txt',
    #     }
        
    #     compare_multiple_models(model_files, args.gt_file)
    # else:
    #     # Single model evaluation
    evaluate_imm(args.imm_file, args.gt_file)

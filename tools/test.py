#!/usr/bin/env python
"""
Test script for Model-Based IMM
Evaluates trained model on whole sequence and saves predictions
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import argparse
import json
from core.imm_model_based import ModelBasedIMM
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SimpleGRUModelNet(nn.Module):
    """
    Simplified GRU-based network following KalmanNet architecture more closely.
    Uses a single GRU that processes all model differences together.
    """
    def __init__(self, obs_dim=4, state_dim=8, hidden_dim=64, num_models=3, num_layers=2):
        super(SimpleGRUModelNet, self).__init__()
        
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_models = num_models
        
        # Input dimension per model
        self.input_dim_per_model = 2 * obs_dim + 2 * state_dim  # 24
        # Total input: concatenate all models
        self.total_input_dim = self.input_dim_per_model * num_models  # 72
        
        # Single GRU to process concatenated model features
        self.gru = nn.GRU(
            self.total_input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_models),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_models, input_dim) for single-step
        
        Returns:
            probs: (batch_size, num_models)
        """
        batch_size = x.shape[0]
        
        # Flatten model dimensions: (batch_size, num_models * input_dim)
        if len(x.shape) == 3:
            x_flat = x.reshape(batch_size, -1)
            # Add sequence dimension: (batch_size, 1, total_input_dim)
            x_flat = x_flat.unsqueeze(1)
        else:
            # Already has sequence dimension
            seq_len = x.shape[1]
            x_flat = x.reshape(batch_size, seq_len, -1)
        
        # Pass through GRU
        gru_out, _ = self.gru(x_flat)
        
        # Take last time step
        last_out = gru_out[:, -1, :]
        
        # Pass through FC to get probabilities
        probs = self.fc(last_out)
        
        return probs


def test_model_based_imm(config_file, data_file, gt_file, model_path, 
                         output_file='data/filter_model_based_imm.txt'):
    """
    Test the trained Model-Based IMM on the whole sequence.
    
    Args:
        config_file: Path to IMM configuration file
        data_file: Path to test data file (observations)
        gt_file: Path to ground truth file
        model_path: Path to trained model
        output_file: Path to save predictions
    
    Returns:
        Dictionary with test metrics
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading Test Data")
    print("="*60)
    
    data = np.loadtxt(data_file, delimiter=',')
    gt_data = np.loadtxt(gt_file, delimiter=',')
    
    observations = data[:, [0, 1, 6, 7]]  # x, y, width, length
    ground_truth = gt_data[:, :8]  # Full state: x, y, v, dv, w, dw, width, length
    gt_positions = gt_data[:, :2]  # x, y only for loss computation
    
    num_samples = len(observations)
    print(f"Number of test samples: {num_samples}")
    
    # Load trained model
    print(f"\nLoading trained model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    # First, try loading the state_dict version (more portable)
    state_dict_path = model_path.replace('.pth', '_state_dict.pth')
    
    if os.path.exists(state_dict_path):
        try:
            print(f"Found state dict file: {state_dict_path}")
            model = SimpleGRUModelNet(obs_dim=4, state_dim=8, hidden_dim=128, num_models=3, num_layers=2)
            model.load_state_dict(torch.load(state_dict_path, map_location=device))
            model = model.to(device)
            model.eval()
            print("✓ Model loaded successfully from state dict!")
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {total_params:,}")
        except Exception as e:
            print(f"✗ Error loading model from state dict: {e}")
            return None
    else:
        # Try loading complete model with proper module reference
        try:
            import sys
            # Add SimpleGRUModelNet to __main__ module so pickle can find it
            sys.modules['__main__'].SimpleGRUModelNet = SimpleGRUModelNet
            
            model = torch.load(model_path, map_location=device)
            model.eval()
            print("✓ Model loaded successfully (complete model)!")
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {total_params:,}")
        except Exception as e:
            print(f"✗ Failed to load complete model: {e}")
            print(f"Hint: State dict file not found at {state_dict_path}")
            print("Please retrain the model to generate the state dict file.")
            return None
    
    # Initialize Model-Based IMM
    print("\nInitializing Model-Based IMM...")
    imm = ModelBasedIMM(config_file, model=model, device=device)
    
    # Initialize with first observation
    initial_state = observations[0]
    imm._init_filters(initial_state)
    
    # Storage for predictions
    predictions = []
    model_probs_history = []
    losses = []
    
    # MSE loss for evaluation
    criterion = torch.nn.MSELoss()
    
    print("\n" + "="*60)
    print("Running Inference on Whole Sequence")
    print("="*60)
    
    # Run through whole sequence
    with torch.no_grad():
        for i in range(num_samples):
            z = observations[i]
            gt_pos = gt_positions[i]
            
            # Update IMM and get prediction
            X_hat = imm.update(z)  # Returns torch tensor (8,) on device
            
            # Compute loss on x, y positions
            gt_pos_tensor = torch.FloatTensor(gt_pos).to(device)
            loss = criterion(X_hat[:2], gt_pos_tensor)
            losses.append(loss.item())
            
            # Get model probabilities (stored internally)
            model_probs = imm.U  # numpy array (3,)
            model_probs_history.append(model_probs.copy())
            
            # Convert prediction to numpy for storage
            X_hat_np = X_hat.cpu().numpy()
            predictions.append(X_hat_np)
            
            # Progress indicator
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Processed {i+1}/{num_samples} samples... "
                      f"Current loss: {loss.item():.6f}")
    
    # Convert to arrays
    predictions = np.array(predictions)  # Shape: (N, 8)
    model_probs_history = np.array(model_probs_history)  # Shape: (N, 3)
    
    # Save predictions in same format as filter_imm.txt
    # Format: x, y, v, dv, w, dw (first 6 columns)
    print(f"\nSaving predictions to: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    np.savetxt(output_file, predictions[:, :6], delimiter=',', fmt='%.18e')
    print(f"Predictions saved successfully!")
    
    # Save model probabilities to uprob_model.txt
    probs_file = 'data/uprob_model.txt'
    print(f"Saving model probabilities to: {probs_file}")
    np.savetxt(probs_file, model_probs_history, delimiter=',', fmt='%.18e')
    print(f"Model probabilities saved successfully!")
    
    # Also save to output-based filename for reference
    probs_file_alt = output_file.replace('.txt', '_probs.txt')
    np.savetxt(probs_file_alt, model_probs_history, delimiter=',', fmt='%.18e')
    
    # Compute metrics
    print("\n" + "="*60)
    print("Computing Metrics")
    print("="*60)
    
    # Position MSE
    pred_pos = predictions[:, :2]
    errors = pred_pos - gt_positions
    squared_errors = errors ** 2
    
    mse_x = np.mean(squared_errors[:, 0])
    mse_y = np.mean(squared_errors[:, 1])
    mse_total = np.mean(squared_errors)
    rmse_total = np.sqrt(mse_total)
    
    # Euclidean distance
    euclidean_errors = np.sqrt(np.sum(squared_errors, axis=1))
    mean_euclidean = np.mean(euclidean_errors)
    max_euclidean = np.max(euclidean_errors)
    min_euclidean = np.min(euclidean_errors)
    
    # Average loss
    avg_loss = np.mean(losses)
    
    # Model probability statistics
    mean_probs = np.mean(model_probs_history, axis=0)
    
    # Print results
    print("\nPosition Metrics (x, y only):")
    print(f"  MSE (x):           {mse_x:.6f}")
    print(f"  MSE (y):           {mse_y:.6f}")
    print(f"  MSE (total):       {mse_total:.6f}")
    print(f"  RMSE (total):      {rmse_total:.6f}")
    
    print("\nEuclidean Distance:")
    print(f"  Mean:              {mean_euclidean:.6f}")
    print(f"  Max:               {max_euclidean:.6f}")
    print(f"  Min:               {min_euclidean:.6f}")
    
    print("\nAverage Loss (MSE on x,y):")
    print(f"  {avg_loss:.6f}")
    
    print("\nModel Probabilities (averaged over sequence):")
    print(f"  CA (Constant Acceleration): {mean_probs[0]:.4f}")
    print(f"  CV (Constant Velocity):     {mean_probs[1]:.4f}")
    print(f"  CT (Constant Turn):         {mean_probs[2]:.4f}")
    
    # Save metrics
    metrics = {
        'num_samples': int(num_samples),
        'position_metrics': {
            'mse_x': float(mse_x),
            'mse_y': float(mse_y),
            'mse_total': float(mse_total),
            'rmse_total': float(rmse_total)
        },
        'euclidean_distance': {
            'mean': float(mean_euclidean),
            'max': float(max_euclidean),
            'min': float(min_euclidean)
        },
        'average_loss': float(avg_loss),
        'model_probabilities': {
            'CA': float(mean_probs[0]),
            'CV': float(mean_probs[1]),
            'CT': float(mean_probs[2])
        }
    }
    
    metrics_file = output_file.replace('.txt', '_metrics.json')
    print(f"\nSaving metrics to: {metrics_file}")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved successfully!")
    
    print("\n" + "="*60)
    print("Testing Completed!")
    print("="*60)
    
    return metrics


def compare_with_original_imm(model_based_file, original_imm_file):
    """
    Compare Model-Based IMM predictions with original IMM predictions
    
    Args:
        model_based_file: Path to model-based IMM predictions
        original_imm_file: Path to original IMM predictions
    """
    print("\n" + "="*60)
    print("Comparing Model-Based IMM with Original IMM")
    print("="*60)
    
    # Load predictions
    model_based = np.loadtxt(model_based_file, delimiter=',')
    original = np.loadtxt(original_imm_file, delimiter=',')
    
    # Extract positions (first 2 columns)
    model_based_pos = model_based[:, :2]
    original_pos = original[:, :2]
    
    # Ensure same length
    min_len = min(len(model_based_pos), len(original_pos))
    model_based_pos = model_based_pos[:min_len]
    original_pos = original_pos[:min_len]
    
    # Compute differences
    errors = model_based_pos - original_pos
    squared_errors = errors ** 2
    
    mse_x = np.mean(squared_errors[:, 0])
    mse_y = np.mean(squared_errors[:, 1])
    mse_total = np.mean(squared_errors)
    rmse_total = np.sqrt(mse_total)
    
    euclidean_errors = np.sqrt(np.sum(squared_errors, axis=1))
    mean_euclidean = np.mean(euclidean_errors)
    max_euclidean = np.max(euclidean_errors)
    
    print("\nDifference Metrics (Model-Based vs Original IMM):")
    print(f"  MSE (x):           {mse_x:.6f}")
    print(f"  MSE (y):           {mse_y:.6f}")
    print(f"  MSE (total):       {mse_total:.6f}")
    print(f"  RMSE (total):      {rmse_total:.6f}")
    print(f"  Mean Euclidean:    {mean_euclidean:.6f}")
    print(f"  Max Euclidean:     {max_euclidean:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Model-Based IMM on whole sequence')
    parser.add_argument('--config', type=str, default='configs/imm.json',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/imm_single.txt',
                        help='Path to test data file (observations)')
    parser.add_argument('--gt-file', type=str, default='data/imm_single_gt.txt',
                        help='Path to ground truth file')
    parser.add_argument('--model-path', type=str, default='models/imm_gru_model.pth',
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='data/filter_model_based_imm.txt',
                        help='Path to save predictions')
    parser.add_argument('--compare-original', action='store_true',
                        help='Compare with original IMM predictions')
    parser.add_argument('--original-imm-file', type=str, default='data/filter_imm.txt',
                        help='Path to original IMM predictions for comparison')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Model-Based IMM Testing")
    print("=" * 60)
    print(f"Configuration:     {args.config}")
    print(f"Test data:         {args.data}")
    print(f"Ground truth:      {args.gt_file}")
    print(f"Model path:        {args.model_path}")
    print(f"Output file:       {args.output}")
    print("=" * 60)
    
    # Run test
    metrics = test_model_based_imm(
        config_file=args.config,
        data_file=args.data,
        gt_file=args.gt_file,
        model_path=args.model_path,
        output_file=args.output
    )
    
    # Compare with original IMM if requested
    if args.compare_original and metrics is not None:
        if os.path.exists(args.original_imm_file):
            compare_with_original_imm(args.output, args.original_imm_file)
        else:
            print(f"\nWarning: Original IMM file not found at {args.original_imm_file}")
            print("Skipping comparison.")

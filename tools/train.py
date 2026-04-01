import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from core.imm_model_based import ModelBasedIMM


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


class GRUModelNet(nn.Module):
    """
    GRU-based neural network for predicting model probabilities in IMM.
    Follows the KalmanNet architecture with GRU cells.
    
    Input: model_i_difference = [delta_y_tilde, delta_y, delta_x_tilde, delta_x]
           for each of the 3 models (CA, CV, CT)
           Shape: (batch_size, num_models=3, input_dim=24)
           where input_dim = 2*obs_dim + 2*state_dim = 2*4 + 2*8 = 24
    
    Output: u ∈ R^3 (model probabilities after softmax)
    """
    def __init__(self, obs_dim=4, state_dim=8, hidden_dim=64, num_models=3):
        super(GRUModelNet, self).__init__()
        
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_models = num_models
        
        # Input dimension: [delta_y_tilde, delta_y, delta_x_tilde, delta_x]
        # = obs_dim + obs_dim + state_dim + state_dim = 4 + 4 + 8 + 8 = 24
        self.input_dim = 2 * obs_dim + 2 * state_dim
        
        # Separate GRU layer for each model to capture model-specific dynamics
        self.gru_layers = nn.ModuleList([
            nn.GRU(self.input_dim, hidden_dim, batch_first=True)
            for _ in range(num_models)
        ])
        
        # Fully connected layers to process each model's GRU output
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
            )
            for _ in range(num_models)
        ])
        
        # Cross-model fusion layer to capture interactions between models
        self.fusion_dim = hidden_dim // 4 * num_models
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Final output layer with softmax to produce probabilities
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_models),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_models, input_dim) for single-step
               OR (batch_size, seq_len, num_models, input_dim) for sequences
        
        Returns:
            u: (batch_size, num_models) - model probabilities
        """
        batch_size = x.shape[0]
        
        # Handle both single-step and sequential inputs
        if len(x.shape) == 3:
            # Single time step: (batch_size, num_models, input_dim)
            # Add sequence dimension: (batch_size, 1, num_models, input_dim)
            x = x.unsqueeze(1)
            single_step = True
        else:
            single_step = False
        
        seq_len = x.shape[1]
        
        # Process each model's features through its dedicated GRU
        model_features = []
        for i in range(self.num_models):
            # Extract features for model i: (batch_size, seq_len, input_dim)
            model_input = x[:, :, i, :]
            
            # Pass through GRU: output shape (batch_size, seq_len, hidden_dim)
            gru_out, _ = self.gru_layers[i](model_input)
            
            # Take the last time step output: (batch_size, hidden_dim)
            last_out = gru_out[:, -1, :]
            
            # Pass through FC layers: (batch_size, hidden_dim//4)
            fc_out = self.fc_layers[i](last_out)
            
            model_features.append(fc_out)
        
        # Concatenate all model features: (batch_size, fusion_dim)
        concatenated = torch.cat(model_features, dim=-1)
        
        # Fusion layer to capture cross-model interactions
        fused = self.fusion_layer(concatenated)
        
        # Output layer with softmax to get probabilities
        probs = self.output_layer(fused)
        
        return probs


class IMMDataset(Dataset):
    """
    Dataset for training the model-based IMM.
    Loads data from imm_single.txt with ground truth and observations.
    """
    def __init__(self, data_file, config_file, sequence_length=10):
        self.data_file = data_file
        self.config_file = config_file
        self.sequence_length = sequence_length
        
        # Load data
        self.data = np.loadtxt(data_file, delimiter=',')
        
        # Parse data: [x, y, v, vx, vy, w, width, length]
        # Observation: [x, y, width, length]
        # State: [x, y, v, dv, w, dw, width, length]
        self.ground_truth = self.data[:, :8]  # Full state
        self.observations = self.data[:, [0, 1, 6, 7]]  # x, y, width, length
        
        print(f"Loaded {len(self.data)} samples from {data_file}")
        print(f"Ground truth shape: {self.ground_truth.shape}")
        print(f"Observations shape: {self.observations.shape}")
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        """
        Returns a sequence of observations and corresponding ground truth
        """
        obs_seq = self.observations[idx:idx+self.sequence_length]
        gt_seq = self.ground_truth[idx:idx+self.sequence_length]
        
        return {
            'observations': torch.FloatTensor(obs_seq),
            'ground_truth': torch.FloatTensor(gt_seq)
        }


def generate_training_data(data_file, config_file):
    """
    Generate training features and labels using ModelBasedIMM.
    Returns features and ground truth x,y positions for MSE loss.
    """
    # Load data
    data = np.loadtxt(data_file, delimiter=',')
    ground_truth = data[:, :8]  # Full state
    observations = data[:, [0, 1, 6, 7]]  # x, y, width, length
    gt_positions = data[:, :2]  # x, y positions only
    
    print(f"Generating training data from {len(data)} samples...")
    
    # Create IMM instance
    imm = ModelBasedIMM(config_file, model_path=None, device='cpu')
    
    # Generate features (model differences for each timestep)
    features, _ = imm.get_model_differences_batch(observations, ground_truth)
    
    print(f"Generated features shape: {features.shape}")
    print(f"Ground truth positions shape: {gt_positions.shape}")
    
    return features, gt_positions


def train_model(config_file, data_file, epochs=100, batch_size=32, 
                learning_rate=0.001, save_path='models/imm_model.pth',
                use_simple_gru=True):
    """
    Train the GRU-based model probability predictor using MSE loss on x,y positions.
    
    The network predicts model probabilities, which are used to compute weighted
    IMM estimates. The loss is the MSE between predicted x,y positions and ground truth.
    
    Args:
        config_file: Path to IMM configuration file
        data_file: Path to training data file
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_path: Path to save the trained model
        use_simple_gru: If True, use SimpleGRUModelNet; otherwise use GRUModelNet
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load full data for computing IMM estimates
    print("\n" + "="*60)
    print("Loading Training Data")
    print("="*60)
    
    data = np.loadtxt(data_file, delimiter=',')
    ground_truth = data[:, :8]  # Full state
    observations = data[:, [0, 1, 6, 7]]  # x, y, width, length
    gt_positions = data[:, :2]  # Ground truth x, y positions
    
    # Generate features
    print("Generating model difference features...")
    features, _ = generate_training_data(data_file, config_file)
    
    # Split into train and validation
    split_idx = int(0.8 * len(features))
    train_features = features[:split_idx]
    train_gt_pos = gt_positions[:split_idx]
    train_observations = observations[:split_idx]
    train_gt_full = ground_truth[:split_idx]
    
    val_features = features[split_idx:]
    val_gt_pos = gt_positions[split_idx:]
    val_observations = observations[split_idx:]
    val_gt_full = ground_truth[split_idx:]
    
    print(f"\nTraining samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    print(f"Feature dimension per model: {train_features.shape[2]}")
    
    # Convert to tensors
    train_features = torch.FloatTensor(train_features).to(device)
    train_gt_pos = torch.FloatTensor(train_gt_pos).to(device)
    val_features = torch.FloatTensor(val_features).to(device)
    val_gt_pos = torch.FloatTensor(val_gt_pos).to(device)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_features, train_gt_pos)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(val_features, val_gt_pos)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print(f"\nInitializing {'Simple' if use_simple_gru else 'Complex'} GRU Model...")
    if use_simple_gru:
        model = SimpleGRUModelNet(obs_dim=4, state_dim=8, hidden_dim=128, num_models=3, num_layers=2)
    else:
        model = GRUModelNet(obs_dim=4, state_dim=8, hidden_dim=64, num_models=3)
    model = model.to(device)
    
    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Initialize 3 Kalman filters to get model-specific state estimates
    print("\nInitializing Kalman filters for each model...")
    # Note: IMM stays on CPU, only features pass through GPU model
    
    # Loss function and optimizer
    # Using MSE loss for x,y position prediction
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    max_patience = 30
    
    print("\n" + "="*60)
    print("Starting Training (MSE Loss on x,y positions)")
    print("="*60)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # Reset IMM for training epoch
        imm_train = ModelBasedIMM(config_file, model_path=None, device='cpu')
        # Don't set model yet - we'll predict probabilities separately
        
        for i in range(len(train_observations)):
            # Get observation and features
            z = train_observations[i]
            gt_pos_i = train_gt_pos[i]  # Already on GPU
            features_i = train_features[i].unsqueeze(0)  # Already on GPU: (1, 3, 24)
            
            # Predict model probabilities using GPU model
            model_probs = model(features_i)  # (1, 3)
            
            # Run IMM update on CPU to get state estimates from each model
            # (We don't use the IMM's model probabilities, just the state estimates)
            imm_train.update(z)
            
            # Get individual model estimates (x, y positions only)
            model_estimates = torch.FloatTensor([
                imm_train.models[0].X_hat[:2],  # CA: x, y
                imm_train.models[1].X_hat[:2],  # CV: x, y
                imm_train.models[2].X_hat[:2],  # CT: x, y
            ]).to(device)  # (3, 2) - move to GPU
            
            # Compute weighted IMM estimate: sum(prob_i * estimate_i)
            # model_probs: (1, 3), model_estimates: (3, 2)
            weighted_estimate = torch.matmul(model_probs, model_estimates)  # (1, 2)
            
            # MSE loss between predicted and ground truth x,y positions
            optimizer.zero_grad()
            loss = criterion(weighted_estimate, gt_pos_i.unsqueeze(0))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_observations)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        imm_val = ModelBasedIMM(config_file, model_path=None, device='cpu')
        
        with torch.no_grad():
            for i in range(len(val_observations)):
                z = val_observations[i]
                gt_pos_i = val_gt_pos[i]  # Already on GPU
                features_i = val_features[i].unsqueeze(0)  # Already on GPU
                
                model_probs = model(features_i)
                imm_val.update(z)
                
                model_estimates = torch.FloatTensor([
                    imm_val.models[0].X_hat[:2],
                    imm_val.models[1].X_hat[:2],
                    imm_val.models[2].X_hat[:2],
                ]).to(device)
                
                weighted_estimate = torch.matmul(model_probs, model_estimates)
                loss = criterion(weighted_estimate, gt_pos_i.unsqueeze(0))
                val_loss += loss.item()
        
        val_loss /= len(val_observations)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Loss (MSE): {train_loss:.6f} | "
                  f"Val Loss (MSE): {val_loss:.6f}")
        
        # Save best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            torch.save(model, save_path)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  → Saved best model (val_loss: {val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Best validation loss (MSE): {best_val_loss:.6f}")
    print(f"Model saved to: {save_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_epoch': len(train_losses),
        'loss_type': 'MSE on x,y positions'
    }
    history_path = save_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to: {history_path}")
    
    return model, history


def test_model(config_file, data_file, model_path):
    """
    Test the trained model on the dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create IMM with trained model (it will load the model internally)
    print(f"Loading model from {model_path}")
    imm = ModelBasedIMM(config_file, model_path=model_path, device=device)
    
    # Load test data
    data = np.loadtxt(data_file, delimiter=',')
    ground_truth = data[:, :8]
    observations = data[:, [0, 1, 6, 7]]
    
    print(f"\nTesting on {len(observations)} samples...")
    
    # Run IMM
    estimates = []
    model_probs = []
    
    try:
        for t in range(len(observations)):
            z = observations[t]
            imm.update(z)
            estimate = imm.get_estimate()
            probs = imm.get_model_prob()
            
            # Check for NaN
            if np.isnan(estimate).any() or np.isnan(probs).any():
                print(f"\n⚠ Warning: NaN detected at timestep {t}")
                print(f"  Estimate: {estimate}")
                print(f"  Probabilities: {probs}")
                break
                
            estimates.append(estimate.copy())
            model_probs.append(probs.copy())
    except Exception as e:
        print(f"\n❌ Error during tracking at timestep {t}: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    estimates = np.array(estimates)
    model_probs = np.array(model_probs)
    
    # Calculate RMSE only on successfully processed samples
    if len(estimates) > 0:
        errors = estimates - ground_truth[:len(estimates)]
        rmse = np.sqrt(np.mean(errors ** 2, axis=0))
        
        print(f"\nProcessed {len(estimates)}/{len(observations)} samples")
        print("\nRMSE per state dimension:")
        state_names = ['x', 'y', 'v', 'dv', 'w', 'dw', 'width', 'length']
        for i, name in enumerate(state_names):
            print(f"  {name}: {rmse[i]:.6f}")
        
        print(f"\nOverall RMSE: {np.mean(rmse):.6f}")
        
        print("\nAverage model probabilities:")
        print(f"  CA: {np.mean(model_probs[:, 0]):.4f}")
        print(f"  CV: {np.mean(model_probs[:, 1]):.4f}")
        print(f"  CT: {np.mean(model_probs[:, 2]):.4f}")
    else:
        print("\n❌ No estimates generated - check model initialization")
    
    return estimates, model_probs
    
    return estimates, model_probs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Model-Based IMM with GRU Network')
    parser.add_argument('--config', type=str, default='configs/imm.json',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/imm_single.txt',
                        help='Path to training data file')
    parser.add_argument('--model-path', type=str, default='models/imm_gru_model.pth',
                        help='Path to save the trained model')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--simple-gru', action='store_true', default=True,
                        help='Use simple GRU architecture')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test a pre-trained model')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Model-Based IMM Training with GRU Network")
    print("=" * 60)
    print(f"Configuration file: {args.config}")
    print(f"Data file: {args.data}")
    print(f"Model save path: {args.model_path}")
    print(f"Architecture: {'Simple GRU' if args.simple_gru else 'Complex GRU'}")
    print("=" * 60)
    
    if not args.test_only:
        # Train the model
        model, history = train_model(
            config_file=args.config,
            data_file=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_path=args.model_path,
            use_simple_gru=args.simple_gru
        )
    
    # Test the model
    if os.path.exists(args.model_path):
        print("\n" + "=" * 60)
        print("Testing Trained Model")
        print("=" * 60)
        
        estimates, model_probs = test_model(
            config_file=args.config,
            data_file=args.data,
            model_path=args.model_path
        )
        
        print("\n" + "=" * 60)
        print("Completed!")
        print("=" * 60)
    else:
        print(f"\nModel not found at {args.model_path}")
        print("Training may have failed or was skipped.")

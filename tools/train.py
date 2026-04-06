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

def train_model(config_file, data_file, gt_file, epochs=100, batch_size=1, 
                learning_rate=0.001, save_path='models/imm_model.pth',
                use_simple_gru=True):
    """
    Train the GRU-based model probability predictor using MSE loss on x,y positions.
    
    The network predicts model probabilities, which are used to compute weighted
    IMM estimates. The loss is the MSE between predicted x,y positions and ground truth.
    
    Args:
        config_file: Path to IMM configuration file
        data_file: Path to training data file
        gt_file: Path to ground truth data file
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
    gt_data = np.loadtxt(gt_file, delimiter=',')

    ground_truth = gt_data[:, [0, 1, 6, 7]]  # x, y, width, length
    observations = data[:, [0, 1, 6, 7]]  # x, y, width, length
    gt_positions = gt_data[:, :2]  # Ground truth x, y positions
    
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

    num_one_train_iters = 100
    num_one_val_iters = 1
    train_sequence_length = 100
    # val_sequence_length = 100
    val_sequence_length = len(observations) - 1  # Use full sequence for validation to get stable estimate of val loss

    # Create IMM instances once (reuse the same model reference)
    # imm_train = ModelBasedIMM(config_file, model=model, device=device)
    # imm_val = ModelBasedIMM(config_file, model=model, device=device)
    imm_model = ModelBasedIMM(config_file, model=model, device=device)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        print('Epoch {:3d}/{:3d} - Training... \n'.format(epoch + 1, epochs))
        for i in range(num_one_train_iters):
            print(f"  Training iteration {i+1}/{num_one_train_iters}...", end='\r')
            
            # Reset Kalman filters for each training sequence
            imm_model._reset_filters()

            # Get random indice for training
            start_idx = np.random.randint(0, len(observations) - train_sequence_length)

            # Initialize IMM with the first observation of the training sequence
            initial_state = observations[start_idx]
            imm_model._init_filters(initial_state)

            for idx in range(start_idx, start_idx + train_sequence_length):
                # Get observation and features
                z = observations[idx]
                gt_pos_i = gt_positions[idx]  # Shape: (2,) - numpy array
                X_hat = imm_model.update(z)  # Get IMM estimate - returns torch tensor on device
            
                # MSE loss between predicted and ground truth x,y positions
                optimizer.zero_grad()
                # X_hat is (8,) on device, extract first 2 elements for x,y
                # gt_pos_i is (2,) numpy array, convert to tensor on same device
                gt_pos_tensor = torch.FloatTensor(gt_pos_i).to(device)
                loss = criterion(X_hat[:2], gt_pos_tensor)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
                train_loss += loss.item() / train_sequence_length

        # Compute average losses
        avg_train_loss = train_loss / num_one_train_iters
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for i in range(num_one_val_iters):
                # Reset Kalman filters for each validation sequence
                imm_model._reset_filters()

                # Get random indice for validation
                start_idx = np.random.randint(0, len(observations) - val_sequence_length)

                # Initialize IMM with the first observation of the validation sequence
                initial_state = observations[start_idx]
                imm_model._init_filters(initial_state)

                for idx in range(start_idx, start_idx + val_sequence_length):
                    z = observations[idx]
                    gt_pos_i = gt_positions[idx]  # Shape: (2,) - numpy array
                    X_hat = imm_model.update(z)  # Get IMM estimate - returns torch tensor on device

                    # MSE loss between predicted and ground truth x,y positions
                    gt_pos_tensor = torch.FloatTensor(gt_pos_i).to(device)
                    loss = criterion(X_hat[:2], gt_pos_tensor)
                    val_loss += loss.item() / val_sequence_length

        # Compute average validation loss
        avg_val_loss = val_loss / num_one_val_iters
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress with AVERAGED losses
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                f"Train Loss (MSE): {avg_train_loss:.6f} | "
                f"Val Loss (MSE): {avg_val_loss:.6f}")
        
        # Save best model and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            torch.save(model, save_path)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  → Saved best model (val_loss: {avg_val_loss:.6f})")
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Model-Based IMM with GRU Network')
    parser.add_argument('--config', type=str, default='configs/imm.json',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/imm_single.txt',
                        help='Path to training data file')
    parser.add_argument('--gt_file', type=str, default='data/imm_single_gt.txt',
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
            gt_file=args.gt_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_path=args.model_path,
            use_simple_gru=args.simple_gru
        )

    



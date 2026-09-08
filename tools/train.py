import os
import sys
sys.path.append(os.getcwd())

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from core.imm_model_based import ModelBasedIMM


class GRUModelNet(nn.Module):
    """
    GRU-based neural network for predicting model probabilities in IMM.
    Follows the KalmanNet architecture with GRU cells.
    
    Input: model_i_difference = [delta_y_tilde, delta_y, delta_x_tilde, delta_x]
           for each of the 3 models (CA, CV, CT)
    Output: u ∈ R^3 (model probabilities after softmax)
    """
    def __init__(self, obs_dim=4, state_dim=8, hidden_dim=64, num_models=3):
        super(GRUModelNet, self).__init__()
        
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_models = num_models
        
        # Input dimension: [delta_y_tilde, delta_y, delta_x_tilde, delta_x]
        # = obs_dim + obs_dim + state_dim + state_dim
        self.input_dim = 2 * obs_dim + 2 * state_dim
        
        # GRU layers for each model
        self.gru_layers = nn.ModuleList([
            nn.GRU(self.input_dim, hidden_dim, batch_first=True)
            for _ in range(num_models)
        ])
        
        # Fully connected layers to process GRU outputs
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_models)
        ])
        
        # Optional: Cross-model attention layer
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Final layer to combine model outputs
        self.output_layer = nn.Sequential(
            nn.Linear(num_models, num_models),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_models, input_dim) 
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
        
        # Process each model's features through its GRU
        model_outputs = []
        for i in range(self.num_models):
            # Extract features for model i: (batch_size, seq_len, input_dim)
            model_input = x[:, :, i, :]
            
            # Pass through GRU: output shape (batch_size, seq_len, hidden_dim)
            gru_out, _ = self.gru_layers[i](model_input)
            
            # Take the last time step output: (batch_size, hidden_dim)
            if single_step:
                last_out = gru_out[:, -1, :]
            else:
                last_out = gru_out[:, -1, :]
            
            model_outputs.append(last_out)
        
        # Stack model outputs: (batch_size, num_models, hidden_dim)
        stacked_outputs = torch.stack(model_outputs, dim=1)
        
        # Optional: Apply cross-model attention
        attended_outputs, _ = self.cross_attention(
            stacked_outputs, stacked_outputs, stacked_outputs
        )
        
        # Process through FC layers: (batch_size, num_models, 1)
        fc_outputs = []
        for i in range(self.num_models):
            fc_out = self.fc_layers[i](attended_outputs[:, i, :])
            fc_outputs.append(fc_out)
        
        # Concatenate and squeeze: (batch_size, num_models)
        logits = torch.cat(fc_outputs, dim=-1)
        
        # Apply softmax to get probabilities
        probs = self.output_layer(logits)
        
        return probs


class IMMDataset(Dataset):
    """
    Dataset for training the model-based IMM.
    Loads one trajectory with paired observation and ground truth files.
    """
    def __init__(self, data_file, config_file, sequence_length=10, gt_file=None):
        self.data_file = data_file
        self.gt_file = gt_file or infer_ground_truth_file(data_file)
        self.config_file = config_file
        self.sequence_length = sequence_length
        
        obs_data, gt_data = load_trajectory_pair(self.data_file, self.gt_file)
        self.data = obs_data
        
        self.ground_truth = gt_data[:, :8]
        self.observations = obs_data[:, [0, 1, 6, 7]]
        
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


def infer_ground_truth_file(data_file):
    if data_file.endswith('_gt.txt'):
        return data_file
    return data_file.replace('.txt', '_gt.txt')


def collect_trajectory_pairs(data_path):
    """
    Return observation and ground-truth file pairs from a single file or a directory.
    """
    is_directory = os.path.isdir(data_path)
    if os.path.isdir(data_path):
        pattern = os.path.join(data_path, '*.txt')
        observation_files = sorted(
            file_path
            for file_path in glob.glob(pattern)
            if not file_path.endswith(('_gt.txt', '_filter.txt', '_uprob.txt', '_probs.txt'))
        )
    else:
        observation_files = [data_path]

    file_pairs = []
    for obs_file in observation_files:
        gt_file = infer_ground_truth_file(obs_file)
        if not os.path.exists(gt_file):
            if is_directory:
                continue
            raise FileNotFoundError(f"Ground truth file not found for {obs_file}: {gt_file}")
        file_pairs.append((obs_file, gt_file))

    if not file_pairs:
        raise FileNotFoundError(f"No trajectory files found in {data_path}")

    return file_pairs


def load_trajectory_pair(data_file, gt_file=None):
    obs_data = np.loadtxt(data_file, delimiter=',')
    gt_path = gt_file or infer_ground_truth_file(data_file)
    gt_data = np.loadtxt(gt_path, delimiter=',')
    return obs_data, gt_data


def generate_training_data(data_path, config_file):
    """
    Generate training features and labels using noisy Monte Carlo trajectories.
    """
    file_pairs = collect_trajectory_pairs(data_path)
    all_features = []
    all_labels = []

    total_samples = 0
    print(f"Generating training data from {len(file_pairs)} trajectory files...")
    for index, (obs_file, _) in enumerate(file_pairs, start=1):
        obs_data = np.loadtxt(obs_file, delimiter=',')
        ground_truth = obs_data[:, :8]
        observations = obs_data[:, [0, 1, 6, 7]]
        total_samples += len(observations)

        imm = ModelBasedIMM(config_file, model_path=None, device='cpu')
        features, labels = imm.get_model_differences_batch(observations, ground_truth)
        all_features.append(features)
        all_labels.append(labels)

        if index <= 3 or index == len(file_pairs) or index % 50 == 0:
            print(f"  Loaded {len(observations)} noisy samples from {os.path.basename(obs_file)}")

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Generated training data from {total_samples} samples")
    print(f"Generated features shape: {features.shape}")
    print(f"Generated labels shape: {labels.shape}")

    return features, labels


def train_model(config_file, data_path, epochs=100, batch_size=32, 
                learning_rate=0.001, save_path='models/imm_model.pth'):
    """
    Train the GRU-based model probability predictor
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate training data
    features, labels = generate_training_data(data_path, config_file)

    permutation = np.random.permutation(len(features))
    features = features[permutation]
    labels = labels[permutation]
    
    # Split into train and validation
    split_idx = int(0.8 * len(features))
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    val_features = features[split_idx:]
    val_labels = labels[split_idx:]
    
    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    
    # Convert to tensors
    train_features = torch.FloatTensor(train_features)
    train_labels = torch.FloatTensor(train_labels)
    val_features = torch.FloatTensor(val_features)
    val_labels = torch.FloatTensor(val_labels)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = GRUModelNet(obs_dim=4, state_dim=8, hidden_dim=64, num_models=3)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.KLDivLoss(reduction='batchmean')  # KL divergence for probability distributions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=10, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_features)
            
            # KL divergence loss (output should be log probabilities for KLDivLoss)
            loss = criterion(torch.log(outputs + 1e-10), batch_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(torch.log(outputs + 1e-10), batch_labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model, save_path)
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    history_path = save_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to: {history_path}")
    
    return model, history


def test_model(config_file, data_path, model_path):
    """
    Test the trained model on the dataset
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    file_pairs = collect_trajectory_pairs(data_path)
    all_estimates = []
    all_ground_truth = []
    all_model_probs = []
    total_samples = 0

    for obs_file, gt_file in file_pairs:
        imm = ModelBasedIMM(config_file, model_path=model_path, device=device)
        obs_data, gt_data = load_trajectory_pair(obs_file, gt_file)
        ground_truth = gt_data[:, :8]
        observations = obs_data[:, [0, 1, 6, 7]]
        total_samples += len(observations)

        estimates = []
        model_probs = []
        for t in range(len(observations)):
            z = observations[t]
            imm.update(z)
            estimates.append(imm.get_estimate())
            model_probs.append(imm.get_model_prob().copy())

        all_estimates.append(np.array(estimates))
        all_ground_truth.append(ground_truth)
        all_model_probs.append(np.array(model_probs))

    estimates = np.concatenate(all_estimates, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    model_probs = np.concatenate(all_model_probs, axis=0)

    print(f"\nTesting on {total_samples} samples from {len(file_pairs)} trajectories...")

    errors = estimates - ground_truth
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    
    print("\nRMSE per state dimension:")
    state_names = ['x', 'y', 'v', 'dv', 'w', 'dw', 'width', 'length']
    for i, name in enumerate(state_names):
        print(f"  {name}: {rmse[i]:.6f}")
    
    print(f"\nOverall RMSE: {np.mean(rmse):.6f}")
    
    print("\nAverage model probabilities:")
    print(f"  CA: {np.mean(model_probs[:, 0]):.4f}")
    print(f"  CV: {np.mean(model_probs[:, 1]):.4f}")
    print(f"  CT: {np.mean(model_probs[:, 2]):.4f}")
    
    return estimates, model_probs


if __name__ == "__main__":
    # Configuration
    config_file = "configs/imm.json"
    data_file = "data/monte_carlo_simulation_data"
    model_save_path = "models/imm_gru_model.pth"
    
    # Training parameters
    EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    print("=" * 60)
    print("Model-Based IMM Training with GRU Network")
    print("=" * 60)
    
    # Train the model
    model, history = train_model(
        config_file=config_file,
        data_path=data_file,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_path=model_save_path
    )
    
    print("\n" + "=" * 60)
    print("Testing Trained Model")
    print("=" * 60)
    
    # Test the model
    estimates, model_probs = test_model(
        config_file=config_file,
        data_path=data_file,
        model_path=model_save_path
    )
    
    print("\n" + "=" * 60)
    print("Training and Testing Completed!")
    print("=" * 60)

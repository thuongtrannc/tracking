import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import json
import torch
import torch.nn as nn
from core.kalman_filter import KamanFilter


class GRUModelNet(nn.Module):
    """Lightweight GRU network matching the trainer architecture.
    Used as a fallback to load state_dict files saved from training.
    """
    def __init__(self, obs_dim=4, state_dim=8, hidden_dim=64, num_models=3):
        super(GRUModelNet, self).__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_models = num_models
        self.input_dim = 2 * obs_dim + 2 * state_dim

        self.gru_layers = nn.ModuleList([
            nn.GRU(self.input_dim, hidden_dim, batch_first=True)
            for _ in range(num_models)
        ])
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_models)
        ])
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(num_models, num_models), nn.Softmax(dim=-1))

    def forward(self, x):
        # Accept (batch, num_models, feature) or (batch, seq, num_models, feature)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch_size, seq_len, _, _ = x.shape
        model_outputs = []
        for i in range(self.num_models):
            model_input = x[:, :, i, :]
            gru_out, _ = self.gru_layers[i](model_input)
            last_out = gru_out[:, -1, :]
            model_outputs.append(last_out)
        stacked = torch.stack(model_outputs, dim=1)
        attended, _ = self.cross_attention(stacked, stacked, stacked)
        fc_outs = [self.fc_layers[i](attended[:, i, :]) for i in range(self.num_models)]
        logits = torch.cat(fc_outs, dim=-1)
        probs = self.output_layer(logits)
        return probs


class ModelBasedIMM:
    """
    Model-Based Interactive Multiple Model (IMM) using neural networks
    to predict model probabilities instead of using transition matrix.
    """
    def __init__(self, config_file, model_path=None, device='cpu') -> None:
        self.configs = self.load_configs(config_file)
        self.device = device

        # Initialize Kalman filters for each motion model
        self.filter_cv = KamanFilter(0, config_file, "cv")
        self.filter_ca = KamanFilter(0, config_file, "ca")
        self.filter_ct = KamanFilter(0, config_file, "ct")

        self.models = [self.filter_ca, self.filter_cv, self.filter_ct]
        self.model_cnt = 3
        self.state_dim = 8
        self.obs_dim = 4

        # Initial model probabilities
        self.U = np.array([1.0/3, 1.0/3, 1.0/3])
        
        # Neural network for model probability prediction
        self.model_net = None
        if model_path is not None:
            self.load_model(model_path)

        # Initialize state history for computing differences
        self.prev_z = None
        self.prev_X_pred = [None, None, None]
        self.prev_X_hat = [None, None, None]
        

    def load_configs(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)

    def load_model(self, model_path):
        """Load the trained neural network model.

        Tries to load a pickled model first. If that fails (e.g. missing
        class definition), attempts to load a state_dict into the local
        `GRUModelNet` fallback.
        """
        try:
            net = torch.load(model_path, map_location=self.device)
            # If the file contains a state_dict under a key, get it
            if isinstance(net, dict) and 'state_dict' in net:
                state = net['state_dict']
                model = GRUModelNet(obs_dim=self.obs_dim, state_dim=self.state_dim)
                model.load_state_dict(state)
                self.model_net = model.to(self.device)
            else:
                # assume the saved object is a model instance
                self.model_net = net
            self.model_net.eval()
        except Exception:
            # fallback: try loading as plain state_dict
            try:
                state = torch.load(model_path, map_location=self.device)
                if isinstance(state, dict):
                    model = GRUModelNet(obs_dim=self.obs_dim, state_dim=self.state_dim)
                    # if state has module prefix, try to strip 'module.' keys
                    new_state = {}
                    for k, v in state.items():
                        new_key = k
                        if k.startswith('module.'):
                            new_key = k[len('module.'):]
                        new_state[new_key] = v
                    model.load_state_dict(new_state)
                    self.model_net = model.to(self.device)
                    self.model_net.eval()
                else:
                    raise
            except Exception as e:
                print(f"Warning: failed to load model from {model_path}: {e}")
                self.model_net = None

    def set_model(self, model):
        """Set the neural network model"""
        self.model_net = model
        self.model_net.to(self.device)

    def compute_model_differences(self, z, model_idx):
        """
        Compute the 4 key differences for each model as in KalmanNet:
        1. delta_y_tilde: innovation difference (observation - prediction)
        2. delta_y: observation difference (current - previous)
        3. delta_x_tilde: forward evolution difference (prediction - previous estimate)
        4. delta_x: forward update difference (current estimate - prediction)
        
        Returns: model_i_difference = [delta_y_tilde, delta_y, delta_x_tilde, delta_x]
        """
        model = self.models[model_idx]
        
        # Get current prediction (before update)
        X_pred = model.X_
        
        # 1. delta_y_tilde: innovation (observation - prediction)
        delta_y_tilde = z - np.matmul(model.H, X_pred)
        
        # 2. delta_y: observation difference
        if self.prev_z is not None:
            delta_y = z - self.prev_z
        else:
            delta_y = np.zeros_like(z)
        
        # 3. delta_x_tilde: forward evolution difference
        if self.prev_X_hat[model_idx] is not None:
            delta_x_tilde = X_pred - self.prev_X_hat[model_idx]
        else:
            delta_x_tilde = np.zeros(self.state_dim)
        
        # 4. delta_x: will be computed after update (placeholder for now)
        delta_x = np.zeros(self.state_dim)
        
        # Concatenate all differences
        model_difference = np.concatenate([
            delta_y_tilde,  # obs_dim
            delta_y,         # obs_dim
            delta_x_tilde,   # state_dim
            delta_x          # state_dim (will be updated later)
        ])
        
        return model_difference, X_pred

    def update_forward_differences(self, model_differences_list):
        """
        Update the delta_x (forward update difference) after all models have been updated.
        delta_x = X_hat (after update) - X_pred (before update)
        """
        for i in range(self.model_cnt):
            X_hat = self.models[i].X_hat
            X_pred = self.prev_X_pred[i]
            
            if X_pred is not None:
                delta_x = X_hat - X_pred
                # Update the last state_dim elements of model_differences_list[i]
                start_idx = 2 * self.obs_dim + self.state_dim
                model_differences_list[i][start_idx:] = delta_x

    def update(self, z):
        """
        Update the IMM filter with a new observation
        """
        # Store predictions before update
        self.prev_X_pred = [model.X_ for model in self.models]
        
        # Compute model differences for all models (before update)
        model_differences_list = []
        for i in range(self.model_cnt):
            model_diff, _ = self.compute_model_differences(z, i)
            model_differences_list.append(model_diff)
        
        # Get current states and covariances
        X = [model.get_estimate() for model in self.models]
        self.P = [model.P for model in self.models]

        # State interaction (mixing)
        if self.model_net is None:
            # Use uniform probabilities if no model is loaded
            u = self.U
        else:
            # Predict model probabilities using neural network (from previous step)
            u = self.U
        
        # Mixing probabilities
        mu = np.zeros((self.model_cnt, self.model_cnt))
        for i in range(self.model_cnt):
            for j in range(self.model_cnt):
                if u[i] > 1e-10:
                    mu[j, i] = self.U[j] / u[i]
                else:
                    mu[j, i] = 1.0 / self.model_cnt
        
        # Mixed initial conditions
        Xmix = [np.zeros(self.state_dim) for _ in range(self.model_cnt)]
        for i in range(self.model_cnt):
            for j in range(self.model_cnt):
                Xmix[i] += mu[j, i] * X[j]

        Pmix = [np.zeros((self.state_dim, self.state_dim)) for _ in range(self.model_cnt)]
        for i in range(self.model_cnt):
            for j in range(self.model_cnt):
                diff = X[j] - Xmix[i]
                Pmix[i] += mu[j, i] * (self.P[j] + np.outer(diff, diff))

        # Update each filter with mixed initial conditions
        for i in range(self.model_cnt):
            self.models[i].update(z, Pmix[i], Xmix[i])

        # Update forward update differences (delta_x)
        self.update_forward_differences(model_differences_list)
        
        # Predict model probabilities using neural network
        if self.model_net is not None:
            with torch.no_grad():
                # Stack all model differences: shape (3, feature_dim)
                input_features = np.stack(model_differences_list, axis=0)
                input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)  # (1, 3, feature_dim)
                
                # Get model probabilities from the network
                u_pred = self.model_net(input_tensor)  # (1, 3)
                u_pred = u_pred.cpu().numpy().squeeze()
                
                # Ensure numeric stability and normalization
                self.U = u_pred / np.sum(u_pred)
        else:
            # Fallback to likelihood-based update
            for i in range(self.model_cnt):
                DZ = z - np.matmul(self.models[i].H, self.models[i].X_hat)
                S = np.matmul(np.matmul(self.models[i].H, self.models[i].P), 
                             self.models[i].H.T) + self.models[i].R
                
                # Likelihood
                Lamda = (np.linalg.det(2 * np.pi * S)) ** (-0.5) * \
                        np.exp(-0.5 * np.dot(np.dot(DZ.T, np.linalg.inv(S)), DZ))
                self.U[i] = Lamda * u[i]
            
            # Normalize
            self.U = self.U / np.sum(self.U)

        # Compute combined estimate
        X = [model.get_estimate() for model in self.models]
        self.X_hat = self.U[0] * X[0] + self.U[1] * X[1] + self.U[2] * X[2]

        # Store current observation and estimates for next iteration
        self.prev_z = z.copy()
        self.prev_X_hat = [X[i].copy() for i in range(self.model_cnt)]

    def get_estimate(self):
        return self.X_hat

    def predict(self):
        return self.X_hat

    def get_model_prob(self):
        return self.U

    def get_model_differences_batch(self, observations, ground_truths=None):
        """
        Compute model differences for a batch of observations.
        Used for training data generation.
        
        Args:
            observations: (T, obs_dim) array of observations
            ground_truths: (T, state_dim) array of ground truth states (optional)
        
        Returns:
            features: (T, 3, feature_dim) array of model differences
            labels: (T, 3) array of model probabilities (if ground_truths provided)
        """
        T = len(observations)
        feature_dim = 2 * self.obs_dim + 2 * self.state_dim
        features = np.zeros((T, self.model_cnt, feature_dim))
        labels = np.zeros((T, self.model_cnt)) if ground_truths is not None else None
        
        # Reset filter state
        self.prev_z = None
        self.prev_X_pred = [None, None, None]
        self.prev_X_hat = [None, None, None]
        
        for t in range(T):
            z = observations[t]
            
            # Store predictions before update
            self.prev_X_pred = [model.X_ for model in self.models]
            
            # Compute model differences
            model_differences_list = []
            for i in range(self.model_cnt):
                model_diff, _ = self.compute_model_differences(z, i)
                model_differences_list.append(model_diff)
            
            # Perform standard update
            X = [model.get_estimate() for model in self.models]
            P = [model.P for model in self.models]
            
            # Simple mixing (no interaction for data generation)
            for i in range(self.model_cnt):
                self.models[i].update(z, P[i], X[i])
            
            # Update forward differences
            self.update_forward_differences(model_differences_list)
            
            # Store features
            features[t] = np.stack(model_differences_list, axis=0)
            
            # Compute labels if ground truth is provided
            if ground_truths is not None:
                gt_state = ground_truths[t]
                errors = []
                for i in range(self.model_cnt):
                    X_hat = self.models[i].X_hat
                    error = np.linalg.norm(X_hat - gt_state)
                    errors.append(error)
                
                # Convert errors to probabilities (inverse exponential)
                errors = np.array(errors)
                probs = np.exp(-errors / np.mean(errors))
                labels[t] = probs / np.sum(probs)
            
            # Update history
            self.prev_z = z.copy()
            self.prev_X_hat = [self.models[i].X_hat.copy() for i in range(self.model_cnt)]
            self.X_hat = self.models[0].X_hat  # Just use one for now
        
        return features, labels

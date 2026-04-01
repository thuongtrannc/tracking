import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import json
import torch
import torch.nn as nn
from core.kalman_filter import KamanFilter


class ModelBasedIMM:
    """
    Model-Based Interactive Multiple Model (IMM) using neural networks
    to predict model probabilities instead of using transition matrix.
    """
    def __init__(self, config_file, model, device='cpu') -> None:
        self.configs = self.load_configs(config_file)
        self.device = device

        # Initialize Kalman filters for each motion model
        self.filter_ca = KamanFilter(0, config_file, "ca")
        self.filter_cv = KamanFilter(0, config_file, "cv")
        self.filter_ct = KamanFilter(0, config_file, "ct")

        # Order: CA, CV, CT
        self.models = [self.filter_ca, self.filter_cv, self.filter_ct]
        self.model_cnt = 3
        self.state_dim = 8
        self.obs_dim = 4

        # Initial model probabilities
        self.U = np.array([1.0/3, 1.0/3, 1.0/3])
        
        # Initialize combined state estimate
        self.X_hat = self.filter_cv.X_hat.copy()
        
        # Neural network for model probability prediction
        self.model_net = model.to(self.device)

        # Initialize state history for computing differences
        self.prev_z = None
        self.prev_X_pred = [None, None, None]
        self.prev_X_hat = [None, None, None]
        

    def load_configs(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)

    def load_model(self, model_path):
        """Load the trained neural network model"""
        self.model_net = torch.load(model_path, map_location=self.device)
        self.model_net.eval()

    def set_model(self, model):
        """Set the neural network model"""
        self.model_net = model
        self.model_net.to(self.device)

    def compute_model_differences(self, z, model_idx):
        model = self.models[model_idx]
        
        # Get current prediction (before update)
        X_pred = model.X_
        
        # 1. delta_y_tilde: innovation (observation - prediction)
        y_pred = np.matmul(model.H, X_pred)
        delta_y_tilde = z - y_pred
        
        # 2. delta_y: observation difference (current observation - previous observation)
        if self.prev_z is not None:
            delta_y = z - self.prev_z
        else:
            delta_y = np.zeros_like(z)
        
        # 3. delta_x_tilde: forward evolution difference (predicted state - previous estimate)
        if self.prev_X_hat[model_idx] is not None:
            delta_x_tilde = X_pred - self.prev_X_hat[model_idx]
        else:
            delta_x_tilde = np.zeros(self.state_dim)
        
        # 4. delta_x: will be computed after update (placeholder for now)
        delta_x = np.zeros(self.state_dim)
        
        # Concatenate all differences: [delta_y_tilde, delta_y, delta_x_tilde, delta_x]
        model_difference = np.concatenate([
            delta_y_tilde,  # obs_dim = 4
            delta_y,         # obs_dim = 4
            delta_x_tilde,   # state_dim = 8
            delta_x          # state_dim = 8 (will be updated later)
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
        Update the IMM filter with a new observation.
        Uses neural network to predict model probabilities if available.
        """
        #1. Get predictions for each model
        X = [model.get_estimate() for model in self.models]
        
        #2. Compute the model differences for each model
        model_differences_list = []
        for i in range(self.model_cnt):
            model_diff, _ = self.compute_model_differences(z, i)
            model_differences_list.append(model_diff)

        #3. Predict model probabilities
        model_differences_input = np.stack(model_differences_list, axis=0)
        model_differences_input = torch.from_numpy(model_differences_input).float().to(self.device)
            
        with torch.no_grad():
            model_probs = self.model_net(model_differences_input)  # Shape: (model_cnt, num_classes)
            model_probs = torch.softmax(model_probs, dim=1).cpu().numpy()  # Convert to probabilities
            
        # Update model probabilities
        self.U = model_probs.mean(axis=0)  # Average over models to get final probabilities

        # 4. Update each model independently (no mixing)
        for i in range(self.model_cnt):
            self.models[i].update(z, self.models[i].P, self.models[i].X_hat)
        
        # 5. Compute combined state estimate as weighted sum of model estimates
        X = [model.get_estimate() for model in self.models]
        self.X_hat = self.U[0] * X[0] + self.U[1] * X[1] + self.U[2] * X[2]

        return self.X_hat

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
        feature_dim = 2 * self.obs_dim + 2 * self.state_dim  # = 2*4 + 2*8 = 24
        features = np.zeros((T, self.model_cnt, feature_dim))
        labels = np.zeros((T, self.model_cnt)) if ground_truths is not None else None
        
        # Reset filter state for clean processing
        self.prev_z = None
        self.prev_X_pred = [None, None, None]
        self.prev_X_hat = [None, None, None]
        
        # Reset each model to initial state
        for model in self.models:
            model.init_estimator(model.P_, model.X_hat)
        
        for t in range(T):
            z = observations[t]
            
            # Store predictions before update
            self.prev_X_pred = [model.X_ for model in self.models]
            
            # Compute model differences for each model
            model_differences_list = []
            for i in range(self.model_cnt):
                model_diff, _ = self.compute_model_differences(z, i)
                model_differences_list.append(model_diff)
            
            # Get current states and covariances
            X = [model.get_estimate() for model in self.models]
            P = [model.P for model in self.models]
            
            # Simple update without mixing for feature generation
            # This keeps each model independent for clearer feature signals
            for i in range(self.model_cnt):
                self.models[i].update(z, P[i], X[i])
            
            # Update forward differences (delta_x)
            self.update_forward_differences(model_differences_list)
            
            # Store features
            features[t] = np.stack(model_differences_list, axis=0)
            
            # Compute labels if ground truth is provided
            if ground_truths is not None:
                gt_state = ground_truths[t]
                
                # Compute error for each model's estimate
                errors = []
                for i in range(self.model_cnt):
                    X_hat = self.models[i].X_hat
                    # Use position and velocity errors (most important states)
                    pos_error = np.linalg.norm(X_hat[:2] - gt_state[:2])  # x, y position
                    vel_error = np.linalg.norm(X_hat[2:4] - gt_state[2:4])  # v, dv
                    
                    # Combined error with position weighted more
                    error = pos_error + 0.5 * vel_error
                    errors.append(error)
                
                # Convert errors to probabilities using softmax-like function
                errors = np.array(errors)
                # Lower error = higher probability
                # Use negative exponential to convert errors to probabilities
                probs = np.exp(-errors / (np.mean(errors) + 1e-6))
                labels[t] = probs / (np.sum(probs) + 1e-10)
            
            # Update history for next iteration
            self.prev_z = z.copy()
            self.prev_X_hat = [self.models[i].X_hat.copy() for i in range(self.model_cnt)]
        
        return features, labels

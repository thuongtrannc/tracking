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
    def __init__(self, config_file, model_path=None, device='cpu') -> None:
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
        # Store predictions before update for computing delta_x later
        self.prev_X_pred = [model.X_.copy() for model in self.models]
        
        # Compute model differences for all models (before update)
        model_differences_list = []
        for i in range(self.model_cnt):
            model_diff, _ = self.compute_model_differences(z, i)
            model_differences_list.append(model_diff)
        
        # Get current states and covariances
        X = [model.get_estimate() for model in self.models]
        self.P = [model.P.copy() for model in self.models]

        # State interaction (mixing) - use current model probabilities
        # In traditional IMM, we would use transition matrix here
        # For model-based, we use the current probabilities
        u = self.U.copy()
        
        # Mixing probabilities: mu[j,i] is probability that model j was active
        # given that model i is active now
        mu = np.zeros((self.model_cnt, self.model_cnt))
        for i in range(self.model_cnt):
            if u[i] > 1e-10:
                for j in range(self.model_cnt):
                    # For model-based IMM without transition matrix,
                    # we assume equal mixing from all previous models
                    mu[j, i] = self.U[j] / u[i]
            else:
                for j in range(self.model_cnt):
                    mu[j, i] = 1.0 / self.model_cnt
        
        # Mixed initial conditions for each model
        Xmix = [np.zeros(self.state_dim) for _ in range(self.model_cnt)]
        for i in range(self.model_cnt):
            for j in range(self.model_cnt):
                Xmix[i] += mu[j, i] * X[j]

        Pmix = [np.zeros((self.state_dim, self.state_dim)) for _ in range(self.model_cnt)]
        for i in range(self.model_cnt):
            for j in range(self.model_cnt):
                diff = X[j] - Xmix[i]
                Pmix[i] += mu[j, i] * (self.P[j] + np.outer(diff, diff))
            
            # Add small regularization to ensure positive definiteness
            Pmix[i] += np.eye(self.state_dim) * 1e-6

        # Update each filter with mixed initial conditions
        for i in range(self.model_cnt):
            try:
                self.models[i].update(z, Pmix[i], Xmix[i])
            except Exception as e:
                print(f"Warning: Model {i} update failed: {e}")
                # Keep previous estimate if update fails
                pass

        # Now update forward update differences (delta_x) after models are updated
        self.update_forward_differences(model_differences_list)
        
        # Clip features to prevent extreme values
        for i in range(len(model_differences_list)):
            model_differences_list[i] = np.clip(model_differences_list[i], -1e6, 1e6)
        
        # Predict model probabilities using neural network
        if self.model_net is not None:
            with torch.no_grad():
                # Stack all model differences: shape (3, 24)
                # where 24 = 2*obs_dim + 2*state_dim = 2*4 + 2*8
                input_features = np.stack(model_differences_list, axis=0)
                
                # Check for NaN/Inf in features
                if np.isnan(input_features).any() or np.isinf(input_features).any():
                    print(f"Warning: NaN/Inf in features, using likelihood-based update")
                    # Fall back to likelihood
                    self.model_net = None
                else:
                    input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(self.device)  # (1, 3, 24)
                    
                    # Get model probabilities from network
                    u_pred = self.model_net(input_tensor)  # (1, 3)
                    u_pred = u_pred.cpu().numpy().squeeze()
                    
                    # Update model probabilities (already normalized by softmax in network)
                    self.U = u_pred
        
        if self.model_net is None:
            # Fallback to likelihood-based update (traditional IMM approach)
            likelihood = np.zeros(self.model_cnt)
            for i in range(self.model_cnt):
                # Innovation
                DZ = z - np.matmul(self.models[i].H, self.models[i].X_hat)
                # Innovation covariance
                S = np.matmul(np.matmul(self.models[i].H, self.models[i].P), 
                             self.models[i].H.T) + self.models[i].R
                
                # Add regularization
                S += np.eye(len(DZ)) * 1e-6
                
                # Gaussian likelihood
                det_S = np.linalg.det(S)
                if det_S > 1e-10:
                    try:
                        inv_S = np.linalg.inv(S)
                        likelihood[i] = (2 * np.pi * det_S) ** (-0.5) * \
                                       np.exp(-0.5 * np.dot(np.dot(DZ.T, inv_S), DZ))
                    except:
                        likelihood[i] = 1e-10
                else:
                    likelihood[i] = 1e-10
                
                # Update model probability
                self.U[i] = likelihood[i] * u[i]
            
            # Normalize probabilities
            sum_U = np.sum(self.U)
            if sum_U > 1e-10:
                self.U = self.U / sum_U
            else:
                self.U = np.ones(self.model_cnt) / self.model_cnt

        # Compute combined estimate: weighted average of model estimates
        X = [model.get_estimate() for model in self.models]
        self.X_hat = np.zeros(self.state_dim)
        for i in range(self.model_cnt):
            self.X_hat += self.U[i] * X[i]
        
        # Check for NaN in final estimate
        if np.isnan(self.X_hat).any():
            print("Warning: NaN in final estimate, resetting to first model")
            self.X_hat = X[0].copy()
            self.U = np.array([1.0, 0.0, 0.0])

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

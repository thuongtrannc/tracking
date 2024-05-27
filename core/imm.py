import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import json
from core.kalman_filter import KamanFilter

class IMM:
    def __init__(self, config_file) -> None:
        self.configs = self.load_configs(config_file)

        self.filter_cv = KamanFilter(0, config_file, "cv")
        self.filter_ca = KamanFilter(0, config_file, "ca")
        self.filter_ct = KamanFilter(0, config_file, "ct")

        self.trans_mtx = np.array(self.configs["data"]["trans_mtx"])
        self.U = np.array([0.9, 0.05, 0.05])
        self.X_cv = self.configs["init_estimation"]["cv"]["X_init"]
        self.X_ca = self.configs["init_estimation"]["ca"]["X_init"]
        self.X_ct = self.configs["init_estimation"]["ct"]["X_init"]

        self.models = [self.filter_cv, self.filter_ca, self.filter_ct]
        self.model_cnt = 3
        self.state_dim = 8
        

    def load_configs(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)


    def update(self, z):

        # Get prediction
        X = [model.get_estimate() for model in self.models]
        self.P = [model.P for model in self.models]

        # State interaction
        u = np.matmul(self.trans_mtx.T, self.U.T)
        u = np.squeeze(u)
        mu = np.zeros(self.trans_mtx.shape)
        for i in range(self.model_cnt):
            mu[:, i] = 1 / u[i] * (self.trans_mtx[:, i] * self.U)
        
        Xmix = [np.zeros(self.state_dim) for _ in range(self.model_cnt)]
        for i in range(self.model_cnt):
            for j in range(self.model_cnt):
                Xmix[i] += mu[j, i] * X[j]

        Pmix = [np.zeros((8,8)) for _ in range(self.model_cnt)]
        for i in range(self.model_cnt):
            for j in range(3):
                Pmix[i] += mu[j, i] * (self.P[j] + np.matmul((X[j] - Xmix[i]), (X[j] - Xmix[i]).T))

        # Update 
        for i in range(self.model_cnt):
            self.models[i].update(z, Pmix[i], Xmix[i])

        # Update model probs
        for i in range(self.model_cnt):
            DZ = z - np.matmul(self.models[i].H, self.models[i].X_hat)
            S = np.matmul(np.matmul(self.models[i].H, self.models[i].P), self.models[i].H.T) + self.models[i].R
            # S = np.matmul(np.matmul(self.models[i].H, Pmix[i]), self.models[i].H.T) + self.models[i].R
            Lamda = (np.linalg.det(2 * np.pi * S)) ** (-0.5) * \
                    np.exp(-0.5 * np.dot(np.dot(DZ.T, np.linalg.inv(S)), DZ))
            self.U[i] = Lamda * u[i]
        self.U = self.U / np.sum(self.U)

        # Get estimate result
        X = [model.get_estimate() for model in self.models]
        self.X_hat = self.U[0] * X[0] + self.U[1] * X[1] + self.U[2] * X[2]


    def get_estimate(self):
        return self.X_hat


    def predict(self):
        return self.X_hat
        

    def get_model_prob(self):
        return self.U


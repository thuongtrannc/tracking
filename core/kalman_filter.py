# Estimate the box with x, y, yaw, vx, vy, yaw_rate, w, l

from mimetypes import init
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import json

class KamanFilter():
    def __init__(self, id, config_file, type="cv"):

        self.configs = self.load_configs(config_file)
        self.dT = self.configs["data"]["dT"]
        self.type = type
        if type == "cv":
            init_estimation = self.configs["init_estimation"]["cv"]
        elif type == "ct":
            init_estimation = self.configs["init_estimation"]["ct"]
        elif type == "ca":
            init_estimation = self.configs["init_estimation"]["ca"]
        else:
            raise TypeError("Motion model is not available in (cv, ct, ca)")
    
        A = np.array(init_estimation["A"])
        H = np.array(init_estimation["H"])
        Q = np.array(init_estimation["Q"])
        R = np.array(init_estimation["R"])
        pii = np.array(init_estimation["pii"])
        X_init = np.array(init_estimation["X_init"])

        self.id = id
        self.init = False
        self.initialize(A, H, Q, R, pii)
        self.init_estimator(pii, X_init)


    def load_configs(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)


    def init_estimator(self, pii, X_init):
        self.X_hat = X_init
        self.X_ = self.X_hat 
        self.init = True


    def initialize(self, A, H, Q, R, pii):
        
        self.A = A
        self.H = H

        ndim = self.A.shape[0]
        mdim = self.H.shape[0]

        self.Q = np.zeros((ndim, ndim))
        for i in range(ndim):
            self.Q[i][i] = Q[i]

        self.R = np.zeros((mdim, mdim))
        for i in range(mdim):
            self.R[i][i] = R[i]

        self.P_ = np.ones((ndim, ndim)) * pii
        self.P = self.P_

        self.ndim = ndim
        self.mdim = mdim
        

    def update(self, Z, P_mix = None, X_mix = None):
        if self.init == False:
            raise KeyError("The estimator is not init yet!")
        
        if self.type == "ct":
            if P_mix is not None and X_mix is not None:
                self.X_hat = X_mix
                self.P = P_mix

            v = self.X_hat[2]
            dv = self.X_hat[3]
            w = self.X_hat[4]
            dw = self.X_hat[5]

            if (np.abs(dw) < 1e-6):
                self.X_[0] = self.X_hat[0] + v * np.cos(self.X_hat[3]) * self.dT
                self.X_[1] = self.X_hat[1] + v * np.sin(self.X_hat[3]) * self.dT
                self.X_[2] = self.X_hat[2] 
                self.X_[3] = self.X_hat[3] 
                self.X_[4] = self.X_hat[4] 
                self.X_[5] = self.X_hat[5] 
                self.X_[6] = self.X_hat[6] 
                # dw = 1e-6

                self.A[0][2] = np.cos(w) * self.dT
                self.A[0][4] = -v * np.sin(w) * self.dT
                self.A[0][5] = 0
                self.A[1][2] = np.sin(w) * self.dT
                self.A[1][4] = v * np.cos(w) * self.dT
                self.A[1][5] = 0

            else:
                self.X_[0] = self.X_hat[0] + v / dw * (-np.sin(w) + np.sin(self.dT * dw + w))
                self.X_[1] = self.X_hat[1] + v / dw * (np.cos(w) - np.cos(self.dT * dw + w))
                self.X_[2] = self.X_hat[2]
                self.X_[3] = self.X_hat[3]
                self.X_[4] = self.X_hat[4] + dw * self.dT
                self.X_[5] = self.X_hat[5]
                self.X_[6] = self.X_hat[6]
                self.X_[7] = self.X_hat[7]

                self.A[0][2] = (1.0 / dw) * (-np.sin(w) + np.sin(self.dT * dw + w))
                self.A[0][4] = (v / dw) * (-np.cos(w) + np.cos(self.dT * dw + w))
                self.A[0][5] = self.dT * (v / dw) * np.cos(self.dT * dw + w) - v / (dw * dw) * (-np.sin(w) + np.sin(self.dT * dw + w))
                self.A[1][2] = (1.0 / dw) * (np.cos(w) - np.cos(self.dT * dw + w))
                self.A[1][4] = (v / dw) * (-np.sin(w) + np.sin(self.dT * dw + w))
                self.A[1][5] = self.dT * (v / dw) * np.sin(self.dT * dw + w) - (v / (dw * dw)) * (np.cos(w) - np.cos(self.dT * dw + w))

            self.P_ = np.matmul(self.A, np.matmul(self.P, self.A.transpose())) + self.Q
        
        elif self.type == "cv":

            if P_mix is not None and X_mix is not None:
                self.X_hat = X_mix
                self.P = P_mix
            
            v = self.X_hat[2]
            dv = self.X_hat[3]
            w = self.X_hat[4]
            dw = self.X_hat[5]

            self.X_[0] = self.X_hat[0] + v * np.cos(w) * self.dT
            self.X_[1] = self.X_hat[1] + v * np.sin(w) * self.dT
            self.X_[2] = self.X_hat[2]
            self.X_[3] = self.X_hat[3]
            self.X_[4] = self.X_hat[4]
            self.X_[5] = self.X_hat[5]
            self.X_[6] = self.X_hat[6]
            self.X_[7] = self.X_hat[7]

            self.A[0][2] = np.cos(w) * self.dT
            self.A[0][4] = -v * np.sin(w) * self.dT
            self.A[1][2] = np.sin(w) * self.dT
            self.A[1][4] = v * np.cos(w) * self.dT

            self.P_ = np.matmul(self.A, np.matmul(self.P, self.A.transpose())) + self.Q

        elif self.type == "ca":
            
            if P_mix is not None and X_mix is not None:
                self.X_hat = X_mix
                self.P = P_mix

            v = self.X_hat[2]
            dv = self.X_hat[3]
            w = self.X_hat[4]
            dw = self.X_hat[5]

            self.X_[0] = self.X_hat[0] + v * np.cos(w) * self.dT + 0.5 * dv * np.cos(w) * self.dT * self.dT 
            self.X_[1] = self.X_hat[1] + v * np.sin(w) * self.dT + 0.5 * dv * np.sin(w) * self.dT * self.dT 
            self.X_[2] = self.X_hat[2]
            # self.X_[2] = self.X_hat[2] + dv * self.dT
            self.X_[3] = self.X_hat[3]
            self.X_[4] = self.X_hat[4]
            self.X_[5] = self.X_hat[5]
            self.X_[6] = self.X_hat[6]
            self.X_[7] = self.X_hat[7]

            self.A[0][2] = np.cos(w) * self.dT
            self.A[0][3] = np.cos(w) * self.dT * self.dT * 0.5 
            self.A[0][4] = -v * np.sin(w) * self.dT - 0.5 * dv * np.sin(w) * self.dT * self.dT
            self.A[1][2] = np.sin(w) * self.dT
            self.A[1][3] = np.sin(w) * self.dT * self.dT * 0.5
            self.A[1][4] = v * np.cos(w) * self.dT + 0.5 * dv * np.cos(w) * self.dT * self.dT

            self.P_ = np.matmul(self.A, np.matmul(self.P, self.A.transpose())) + self.Q
            
        self.K = np.matmul(self.P_, np.matmul(self.H.transpose(), np.linalg.inv(np.matmul(self.H, np.matmul(self.P_, self.H.transpose())) + self.R)))

        self.X_hat = self.X_ + np.matmul(self.K, Z - np.matmul(self.H, self.X_))

        self.P = np.matmul((np.eye(self.ndim) - np.matmul(self.K, self.H)), self.P_)
        
    
    def get_estimate(self):
        if self.init:
            return self.X_hat
        else: return None


    def predict(self, dt):
        if (dt - self.dT) < 1e-2:
            self.A_pred = self.A
        else:
            assert KeyError("Predicting time must not be larger than {self.dT}.")

        return np.matmul(self.A_pred, self.X_hat)

        


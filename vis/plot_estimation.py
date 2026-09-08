import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import json

class VisData:
    def __init__(self, config_file, estimation_file, uprob_file, data_file) -> None:
        self.configs = self.load_configs(config_file)
        self.data_file = data_file
        self.estimation_file = estimation_file
        self.uprob_file = uprob_file


    def load_configs(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)


    def plot_position(self):
        data = np.loadtxt(self.data_file, delimiter=",")
        data_hat = np.loadtxt(self.estimation_file, delimiter=",")

        print(data.shape)
        print(data_hat.shape)
        x = data[:, 0]
        y = data[:, 1]
        x_hat = data_hat[:, 0]
        y_hat = data_hat[:, 1]

        error_x = x - x_hat
        error_y = y - y_hat

        t = np.array(range(0, data.shape[0])) * self.configs["data"]["dT"]

        fig, axs = plt.subplots(2,1)
        axs[0].plot(x, y, 'b', label="real")
        axs[0].plot(x_hat, y_hat, 'r', label="estimate")
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].set_title('position')

        axs[1].plot(t, error_x, 'b', label="real")
        axs[1].plot(t, error_y, 'r', label="estimate")
        axs[1].set_xlabel('error x, y')
        axs[1].set_ylabel('t')
        axs[1].set_title('position error')

        plt.show()


    def plot_others(self):
        data = np.loadtxt(self.data_file, delimiter=",")
        data_hat = np.loadtxt(self.estimation_file, delimiter=",")
        yaw = data[:, 2]
        vx = data[:, 3]
        vy = data[:, 4]
        v = np.sqrt(vx * vx + vy * vy)
        t = np.array(range(0, yaw.shape[0])) * self.configs["data"]["dT"]
        
        vx_hat = data_hat[:, 2]
        vy_hat = data_hat[:, 3]
        tan_yaw = vy_hat / vx_hat
        yaw_hat = np.arctan(tan_yaw)
        v_hat = np.sqrt(vx_hat * vx_hat + vy_hat * vy_hat)
        t = np.array(range(0, yaw.shape[0])) * self.configs["data"]["dT"]
        fig, axs = plt.subplots(2,2)
        
        axs[0,0].plot(t, yaw, 'b', label="real")
        axs[0,0].plot(t, yaw_hat, 'r', label="estimation")
        axs[0,0].set_ylabel('yaw')
        axs[0,0].set_title('yaw')

        axs[0,1].plot(t, v, 'b', label="real")
        axs[0,1].plot(t, v_hat, 'r', label="estimation")
        axs[0,1].set_ylabel('v')
        axs[0,1].set_title('speed')

        axs[1,0].plot(t, vx, 'b', label="real")
        axs[1,0].plot(t, vx_hat, 'r', label="estimation")
        axs[1,0].set_xlabel('t')
        axs[1,0].set_ylabel('vx')
        axs[1,0].set_title('vx')

        axs[1,1].plot(t, vy, 'b', label="real")
        axs[1,1].plot(t, vy_hat, 'r', label="estimation")
        axs[1,1].set_xlabel('t')
        axs[1,1].set_ylabel('vy')
        axs[1,1].set_title('vy')

        plt.show()
    
    def plot_uprob(self):
        data = np.loadtxt(self.uprob_file, delimiter=",")
        t = np.array(range(0, data.shape[0])) * self.configs["data"]["dT"]
        plt.plot(t, data[:,0], 'r')
        plt.plot(t, data[:,1], 'b')
        plt.plot(t, data[:,2], 'g')
        plt.xlabel = "t"
        plt.ylabel = "y"
        plt.show()
    
if __name__ == "__main__":
    # estimation_file = "data/filter_model_based_imm.txt"
    # uprob_file = "data/uprob_model.txt"
    data_file = "data/monte_carlo_simulation_data/imm_mc_0000.txt"
    estimation_file = "data/monte_carlo_simulation_data/imm_mc_000_filter.txt"
    uprob_file = "data/monte_carlo_simulation_data/imm_mc_000_uprob.txt"
    vis_data = VisData("configs/imm.json", estimation_file, uprob_file, data_file)
    vis_data.plot_position()
    vis_data.plot_others()
    vis_data.plot_uprob()
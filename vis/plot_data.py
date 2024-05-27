import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import json

class VisData:
    def __init__(self, config_file) -> None:
        self.configs = self.load_configs(config_file)
        self.data_file = self.configs["save_name"]


    def load_configs(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)


    def plot_position(self):
        data = np.loadtxt(self.data_file, delimiter=",")
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('position')
        plt.show()


    def plot_others(self):
        data = np.loadtxt(self.data_file, delimiter=",")
        yaw = data[:, 2]
        vx = data[:, 3]
        vy = data[:, 4]
        v = np.sqrt(vx * vx + vy * vy)
        t = np.array(range(0, yaw.shape[0])) * self.configs["data"]["dT"]

        fig, axs = plt.subplots(2,2)
        
        axs[0,0].plot(t, yaw)
        axs[0,0].set_xlabel('t')
        axs[0,0].set_ylabel('yaw')
        axs[0,0].set_title('yaw')

        axs[0,1].plot(t, v)
        axs[0,1].set_xlabel('t')
        axs[0,1].set_ylabel('v')
        axs[0,1].set_title('speed')

        axs[1,0].plot(t, vx)
        axs[1,0].set_xlabel('t')
        axs[1,0].set_ylabel('vx')
        axs[1,0].set_title('vx')

        axs[1,1].plot(t, vy)
        axs[1,1].set_xlabel('t')
        axs[1,1].set_ylabel('vy')
        axs[1,1].set_title('vy')


        plt.show()

    
if __name__ == "__main__":
    vis_data = VisData("tracking/configs/imm.json")
    vis_data.plot_position()
    vis_data.plot_others()
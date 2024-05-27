# This code generate the test data for tracking 2D objects.
# The tracking entities include x, y, yaw, vx, vy, yaw_rate, w, l

from asyncio.sslproto import _DO_HANDSHAKE
import os
import sys
sys.path.append(os.getcwd)

import numpy as np
import math
import json


class DataGenerator():

    def __init__(self, config_file):
        self.configs = self.load_configs(config_file)

        self.save_name = self.configs["save_name"]
        self.duration = self.configs["data"]["time"]
        self.sample_time = self.configs["data"]["dT"]
        self.init = self.configs["data"]["init_pos"]

        self.cv = self.configs["cv"]
        self.ct = self.configs["ct"]
        self.ca = self.configs["ca"]

        self.v = self.cv["v"]
        self.a = self.ca["a"]
        self.w = self.ct["w"]

        self.R = self.configs["data"]["R"]
        self.init_v_flag = False

    
    def load_configs(self, config_file):
        with open(config_file, 'r') as f:
            return json.load(f)


    def gennerate(self):
        self.data = []
        self.data.append(self.init)

        num_samples = self.duration / self.sample_time
        for i in range(int(num_samples)):
            curr_time = i * self.sample_time
            x, y, yaw, vx, vy, yaw_rate, w, l = self.data[i]
            x_new, y_new, yaw_new, vx_new, vy_new, yaw_rate_new, w_new, l_new = self.data[i]
            if self.cv["active"] == True:
                if curr_time >= self.cv["time_start"] and curr_time < self.cv["time_end"]:
                    if not self.init_v_flag:
                        v = self.v
                        self.init_v_flag = True
                    else:
                        v = np.sqrt(vx * vx + vy * vy)

                    x, y, yaw, vx, vy, yaw_rate, w, l = self.data[i]
                    yaw_rate_new = 0
                    yaw_new = yaw
                    vx_new = v * np.cos(yaw)
                    vy_new = v * np.sin(yaw)
                    x_new = x + vx_new * self.sample_time
                    y_new = y + vy_new * self.sample_time
                    w_new = w
                    l_new = l

            if self.ca["active"] == True:
                if curr_time >= self.ca["time_start"] and curr_time < self.ca["time_end"]:
                    a = self.a
                    x, y, yaw, vx, vy, yaw_rate, w, l = self.data[i]
                    yaw_rate_new = 0
                    yaw_new = yaw
                    vx_new = vx + a * np.cos(yaw) * self.sample_time
                    vy_new = vy + a * np.sin(yaw) * self.sample_time
                    x_new = x + vx_new * self.sample_time
                    y_new = y + vy_new * self.sample_time
                    w_new = w
                    l_new = l

            if self.ct["active"] == True:
                if curr_time >= self.ct["time_start"] and curr_time < self.ct["time_end"]:
                    omega = self.w
                    x, y, yaw, vx, vy, yaw_rate, w, l = self.data[i]
                    yaw_new = yaw + omega * self.sample_time
                    v = omega * self.ct["radius"]
                    vx_new = v * np.cos(yaw)
                    vy_new = v * np.sin(yaw)
                    x_new = x + vx_new * self.sample_time
                    y_new = y + vy_new * self.sample_time
                    w_new = w
                    l_new = l
        
            entity = [x_new, y_new, yaw_new, vx_new, vy_new, yaw_rate_new, w_new, l_new]
            self.data.append(entity)


    def add_noise(self):
        noise_data = []
        for entity in self.data:
            noise_entity = entity + np.sqrt(self.R)*np.random.randn()
            noise_data.append(noise_entity)

        self.noise_data = noise_data
    
    
    def save_data(self):
        self.gennerate()
        self.add_noise()
        noise_data = np.array(self.noise_data)
        np.savetxt(self.save_name, noise_data, delimiter=',')


if __name__ == '__main__':
    data_generator = DataGenerator(config_file='tracking/configs/imm.json')
    data_generator.save_data()
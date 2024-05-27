# This code generate the test data for tracking 2D objects.
# The tracking entities include x, y, yaw, vx, vy, yaw_rate, w, l

from asyncio.sslproto import _DO_HANDSHAKE
import os
import sys
sys.path.append(os.getcwd)

import numpy as np
import math


class DataGenerator():

    def __init__(self, motion='cv', duration=100, sample_time=0.1, v=5, init = [0, 0, 1.57, 0, 0, 0.0, 2, 4], save_name=''):
        self.motion = motion
        self.duration = duration
        self.sample_time = 0.1
        self.init = init
        self.v = v
        self.R = [0.05, 0.05, 0.01, 0.01, 0.01, 0.0, 0.002, 0.002]
        self.save_name = save_name


    def gennerate(self):
        self.data = []
        self.data.append(self.init)

        if self.motion == 'cv':
            num_samples = self.duration / self.sample_time
            for i in range(int(num_samples)):
                x, y, yaw, vx, vy, yaw_rate, w, l = self.data[i]

                x_new = x + self.v * np.cos(yaw) * self.sample_time
                y_new = y + self.v * np.sin(yaw) * self.sample_time
                yaw_new = yaw
                vx_new = self.v * np.cos(yaw)
                vy_new = self.v * np.sin(yaw)
                yaw_rate_new = 0
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
    data_generator = DataGenerator(save_name='tracking/data/test_cv.txt')
    data_generator.save_data()
# This code generate the test data for tracking 2D objects.
# The tracking entities include x, y, yaw, vx, vy, yaw_rate, w, l
import os
import sys
sys.path.append(os.getcwd)

import numpy as np
import math

from tracking.aux.utils import compute_iou, get_corners
import json
from random import random


class DataGenerator():

    def __init__(self, motion='cv', duration=30, sample_time=0.1, v=5, init =[0, 0, 1.57, 0, 0, 0.0, 2, 4], num_objects=10, save_name=''):
        self.motion = motion
        self.duration = duration
        self.sample_time = 0.1
        self.v = v
        self.num_objects = num_objects
        self.R = [0.05, 0.05, 0.01, 0.01, 0.01, 0.0, 0.02, 0.02]
        self.save_name = save_name
        self.w = 2
        self.l = 4

        self.init = self.init_pose()

    
    def init_pose(self):
        self.init_objects = []
        i = 0
        while i < self.num_objects:
            id = i
            t = np.random.randint(5) * 0.1
            x, y = np.random.randn(2) * 20
            yaw = np.random.randn() * np.pi
            v = np.random.randn() * self.v
            curr_box = [x, y, yaw, self.w, self.l]
            curr_corners = np.array(get_corners(curr_box)).reshape((4,3))

            overlap = False
            for j in range(i):
                box = [self.init_objects[j]['x'], self.init_objects[j]['y'], self.init_objects[j]['yaw'], self.init_objects[j]['w'], self.init_objects[j]['l']]
                corners = np.array(get_corners(box)).reshape((4,3))
                iou = compute_iou(curr_corners, corners)
                if int(iou) > 0:
                    overlap = True

            if overlap == False:
                i += 1
                sim_object = {'id': id, 't': t, 'x': x, 'y': y, 'yaw': yaw, 'w': self.w, 'l': self.l, 'v': v}
                self.init_objects.append(sim_object)


    def gennerate(self):
        self.data = []
        # self.data.append(self.init_objects)

        if self.motion == 'cv':

            num_samples = self.duration / self.sample_time
            start = np.zeros(self.num_objects, dtype=int)
            start = start.tolist()

            for i in range(0, int(num_samples)):
                
                entities = []
                for j in range(self.num_objects):
                    t = i * self.sample_time

                    if t <= self.init_objects[j]['t']:    
                        x_new, y_new, yaw_new, vx_new, vy_new, yaw_rate_new, w_new, l_new = [0, 0, 0, 0, 0, 0, 0, 0]
                    else:
                        if start[j] == 0:
                            start[j] = 1
                            x_new = self.init_objects[j]['x']
                            y_new = self.init_objects[j]['y']
                            yaw_new = self.init_objects[j]['yaw']
                            v_new = self.init_objects[j]['v']
                            w_new = self.init_objects[j]['w']
                            l_new = self.init_objects[j]['l']

                        else:
                            x = self.data[i-1][j]['x']
                            y = self.data[i-1][j]['y']
                            yaw = self.data[i-1][j]['yaw']
                            v = self.data[i-1][j]['v']
                            w = self.data[i-1][j]['w']
                            l = self.data[i-1][j]['l']
                            
                            x_new = x + v * np.cos(yaw) * self.sample_time
                            y_new = y + v * np.sin(yaw) * self.sample_time
                            yaw_new = yaw
                            vx_new = v * np.cos(yaw)
                            vy_new = v * np.sin(yaw)
                            yaw_rate_new = 0
                            w_new = w
                            l_new = l
                    
                    entity = {'id': self.init_objects[j]['id'], 
                              't': t, 
                              'x': x_new, 
                              'y': y_new, 
                              'yaw': yaw_new, 
                              'w': w_new, 
                              'l': l_new, 
                              'v': self.init_objects[j]['v']}
                    entities.append(entity)
                
                self.data.append(entities)


    def add_noise(self):
        noise_data = []
        for i in range(len(self.data)):   
            entities =  []
            for entity in self.data[i]:
                if entity['w'] != 0 | entity['l'] != 0:
                    random_drop = np.random.uniform(1, 100)
                    if random_drop < 10:
                        entity['x'] = 0
                        entity['y'] = 0
                        entity['yaw'] = 0
                        entity['w'] = 0
                        entity['l'] = 0
                    else:   
                        entity['x'] = entity['x'] + np.random.randn() * self.R[0]
                        entity['y'] = entity['y'] + np.random.randn() * self.R[1]
                        entity['yaw'] = entity['yaw'] + np.random.randn() * self.R[2]
                        entity['w'] = entity['w'] + np.random.randn() * self.R[6]
                        entity['l'] = entity['l'] + np.random.randn() * self.R[7]
                entities.append(entity)
            noise_data.append(entities)

        self.noise_data = noise_data
    
    
    def save_data(self):
        print('Gen objects ...')
        self.gennerate()
        print('Add noise ...')
        self.add_noise()
        print('Save ...')
        print('Num fo data ', len(self.noise_data))
        with open(self.save_name, 'w') as f:
            json.dump(self.noise_data, f, indent=2)


    def calc_iou(self, box1, box2):
        corners1 = np.array(get_corners(box1)).reshape((4,3))
        corners2 = np.array(get_corners(box2)).reshape((4,3))
        iou = compute_iou(corners1, corners2)
        return iou


if __name__ == '__main__':
    data_generator = DataGenerator(save_name='tracking/data/test_multi.json')
    data_generator.save_data()
# This code generate the test data for Hungarian matching
# The box information include x, y, yaw, w, l

from asyncio.sslproto import _DO_HANDSHAKE
from cmath import pi
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import math

from tracking.aux.utils import compute_iou

class HungarianGenerator():

    def __init__(self, num_box=15, num_drop=2, box_size=[2,4], boundary=[-40, 40, -40, 40], mean=0.0, std=0.5, save_name='', save_clone=''):
        self.save_name = save_name
        self.save_clone = save_clone
        self.num_box = num_box
        self.num_drop = num_drop
        self.box_size = box_size
        self.boundary = boundary
        self.mean = mean
        self.std = std

        self.x_min = self.boundary[0]
        self.x_max = self.boundary[1]
        self.y_min = self.boundary[2]
        self.y_max = self.boundary[3]


    def gennerate(self):
        boxes = []
        cnt = 0
        while cnt < self.num_box:
            x, y = np.random.randn(2) * 20
            yaw = np.random.randn() * np.pi
            box = [x, y, yaw, self.box_size[0], self.box_size[1]]
            corners = self.get_corners(box)

            corners = np.array(corners).reshape((4,3))

            for i in range(4):
                if (corners[i][0]<self.x_min) | (corners[i][0]>self.x_max) | (corners[i][1]<self.y_min) | (corners[i][1]>self.y_max):
                    continue
            
            for i in range(len(boxes)):
                iou = self.calc_iou(box, boxes[i])
                if iou > 0.0:
                    continue
            
            boxes.append(box)
            cnt += 1

        self.boxes = boxes


    def gen_shaddow(self):
        drop_list = np.random.choice(self.num_box, size=self.num_drop, replace=False)
        shad_boxes = []

        for i in range(self.num_box):
            if i in drop_list:
                continue
            
            box = self.boxes[i]
            x = box[0] + np.random.randn() * self.std
            y = box[1] + np.random.randn() * self.std
            yaw = box[2] + np.random.randn() * self.std / 10.0
            w = box[3] + np.random.randn() * self.std
            l = box[4] + np.random.randn() * self.std

            shad_boxes.append([x, y, yaw, w, l])
        
        self.shad_boxes = shad_boxes
    
    
    def save_data(self):
        self.gennerate()
        self.gen_shaddow()
        np.savetxt(self.save_name, np.array(self.boxes), delimiter=',')
        np.savetxt(self.save_clone, np.array(self.shad_boxes), delimiter=',')


    def calc_iou(self, box1, box2):
        corners1 = np.array(self.get_corners(box1)).reshape((4,3))
        corners2 = np.array(self.get_corners(box2)).reshape((4,3))
        iou = compute_iou(corners1, corners2)
        if (iou > 0):
            print(iou)
        return iou


    def get_corners(self, box):
        corners = []
        x, y, yaw, w, l = box

        top = l/2
        bot = -l/2
        left = w/2
        right = -w/2
        
        corners.append([top, left, 0])
        corners.append([bot, left, 0])
        corners.append([bot, right, 0])
        corners.append([top, right, 0])

        for i in range(4):
            corners[i] = self.rotate_points(corners[i], yaw) + np.array([x, y, 0])
        
        return corners


    def rotate_points(self, corner, yaw):
        rot_mtx = np.array([[np.cos(yaw), -np.sin(yaw), 0], 
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])

        rotate_point = np.matmul(rot_mtx, np.reshape(corner, (3,1)))
        return rotate_point.reshape(1,3)


if __name__ == '__main__':
    data_generator = HungarianGenerator(save_name='tracking/data/test_hungarian.txt', save_clone='tracking/data/test_hungarian_clone.txt', )
    data_generator.save_data()
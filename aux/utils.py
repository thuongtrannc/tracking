import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


def compute_iou(box1, box2):
    """
    box1: corner list
    box2: corner list
    """

    poly_box1 = Polygon([(box1[i, 0], box1[i, 1]) for i in range(4)])
    poly_box2 = Polygon([(box2[i, 0], box2[i, 1]) for i in range(4)])
    
    iou = poly_box1.intersection(poly_box2).area / poly_box1.union(poly_box2).area 

    return iou


def rotate_points(corner, yaw):
        rot_mtx = np.array([[np.cos(yaw), -np.sin(yaw), 0], 
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])

        rotate_point = np.matmul(rot_mtx, np.reshape(corner, (3,1)))
        return rotate_point.reshape(1,3)


def get_corners(box):
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
        corners[i] = rotate_points(corners[i], yaw) + np.array([x, y, 0])
    
    return corners
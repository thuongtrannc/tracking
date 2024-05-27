from asyncio.sslproto import _DO_HANDSHAKE
from cmath import pi
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.aux.utils import compute_iou
from tracking.aux.utils import get_corners


class Matching():
    def __init__(self, method='hungarian'):
        self.method = method


    def calc_affinity(self, det, track):
        affine_mtx = np.zeros((len(det), len(track)), dtype=np.float32)

        for i in range(len(det)):
            for j in range(len(track)):
                corners1 = get_corners(det[i])
                corners2 = get_corners(track[j])
                corners1 = np.array(corners1).reshape((4,3))
                corners2 = np.array(corners2).reshape((4,3))

                iou = compute_iou(corners1, corners2)
                affine_mtx[i,j] = iou

        return affine_mtx


    def data_associate(self, det, track):
        affine_mtx = self.calc_affinity(det, track)

        if self.method == 'hungarian':
            row_idx, col_idx = linear_sum_assignment(-affine_mtx)

        fine_row_idx = []
        fine_col_idx = []
        for i, j in zip(row_idx, col_idx):
            corners1 = get_corners(det[i])
            corners2 = get_corners(track[j])
            corners1 = np.array(corners1).reshape((4,3))
            corners2 = np.array(corners2).reshape((4,3))

            iou = compute_iou(corners1, corners2)
            if iou > 0:
                fine_row_idx.append(i)
                fine_col_idx.append(j)

        return fine_row_idx, fine_col_idx
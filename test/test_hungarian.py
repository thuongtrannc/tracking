from asyncio.sslproto import _DO_HANDSHAKE
from cmath import pi
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from tracking.core.hungarian import Matching


if __name__ == '__main__':
    det_path = 'tracking/data/test_hungarian.txt'
    track_path = 'tracking/data/test_hungarian_clone.txt'

    det = np.loadtxt(det_path, delimiter=',')
    track = np.loadtxt(track_path, delimiter=',')

    matching = Matching()
    row_ind, col_ind = matching.data_associate(det, track)

    print(row_ind)
    print(col_ind)


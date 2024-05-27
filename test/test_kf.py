import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from tracking.core.kalman_filter import KamanFilter

if __name__ == '__main__':

    data_path = 'tracking/data/imm_single.txt'
    save_name = 'tracking/data/filter_imm.txt'
    data = np.loadtxt(data_path, delimiter=',')
    config_file = "tracking/configs/imm.json"
    model = "ca"

    km = KamanFilter(0, config_file, model)

    filter_data = []
    for i in range(data.shape[0]):
        z = np.array([data[i][0], data[i][1], data[i][6], data[i][7]])
        km.update(z)
        x_hat = km.get_estimate()
        filter_data.append(x_hat)

    filter_data = np.array(filter_data)
    save_data = np.zeros((filter_data.shape[0], 6))

    if model == "cv":
        save_data[:, 0] = filter_data[:, 0]
        save_data[:, 1] = filter_data[:, 1]
        save_data[:, 2] = filter_data[:, 2] * np.cos(filter_data[:, 4])
        save_data[:, 3] = filter_data[:, 2] * np.sin(filter_data[:, 4])
        save_data[:, 4] = filter_data[:, 6]
        save_data[:, 5] = filter_data[:, 7] 
    elif model == "ct":
        save_data[:, 0] = filter_data[:, 0]
        save_data[:, 1] = filter_data[:, 1]
        save_data[:, 2] = filter_data[:, 2] * np.cos(filter_data[:, 4])
        save_data[:, 3] = filter_data[:, 2] * np.sin(filter_data[:, 4])
        save_data[:, 4] = filter_data[:, 6]
        save_data[:, 5] = filter_data[:, 7] 
    elif model == "ca":
        save_data[:, 0] = filter_data[:, 0]
        save_data[:, 1] = filter_data[:, 1]
        save_data[:, 2] = filter_data[:, 2] * np.cos(filter_data[:, 4])
        save_data[:, 3] = filter_data[:, 2] * np.sin(filter_data[:, 4])
        save_data[:, 4] = filter_data[:, 6]
        save_data[:, 5] = filter_data[:, 7] 

    np.savetxt(save_name, save_data, delimiter=',')

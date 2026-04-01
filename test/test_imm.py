import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from core.kalman_filter import KamanFilter
from core.imm import IMM

if __name__ == '__main__':

    data_path = 'data/imm_single.txt'
    save_name = 'data/filter_imm.txt'
    u_prob_save_name = 'uprob.txt'
    data = np.loadtxt(data_path, delimiter=',')
    config_file = "configs/imm.json"

    imm = IMM(config_file)

    filter_data = []
    uprob_data = []
    dm = []
    for i in range(data.shape[0]):
        z = np.array([data[i][0], data[i][1], data[i][6], data[i][7]])
        imm.update(z)
        x_hat = imm.get_estimate()
        uprob = imm.get_model_prob()
        filter_data.append(x_hat)
        
        uprob_data.append(uprob.tolist())

    filter_data = np.array(filter_data)
    uprob_data = np.array(uprob_data)
    save_data = np.zeros((filter_data.shape[0], 6))

    save_data[:, 0] = filter_data[:, 0]
    save_data[:, 1] = filter_data[:, 1]
    save_data[:, 2] = filter_data[:, 2] * np.cos(filter_data[:, 4])
    save_data[:, 3] = filter_data[:, 2] * np.sin(filter_data[:, 4])
    save_data[:, 4] = filter_data[:, 6] 
    save_data[:, 5] = filter_data[:, 7] 

    np.savetxt(save_name, save_data, delimiter=',')
    np.savetxt(u_prob_save_name, uprob_data, delimiter=',')

    
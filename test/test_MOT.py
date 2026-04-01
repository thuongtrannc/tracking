import os
import sys
from tkinter import Frame
sys.path.append(os.getcwd())

import numpy as np
import json

from core.mot import MOT

tracks = []


if __name__ == '__main__':
    data_path = 'data/test_multi.json'
    save_path = 'data/mot_multi.json'

    with open(data_path, 'r') as f:
        data = json.load(f)

    mot = MOT()
    timer = 0
    dt = 0.1

    for frame, detection in enumerate(data):
        # if frame > 4:
        #     continue

        # print('Frame ', frame)
        det = []
        for i in range(len(detection)):
            x = detection[i]['x']
            y = detection[i]['y']
            yaw = detection[i]['yaw']
            w = detection[i]['w']
            l = detection[i]['l']

            if w > 0 and l > 0:
                det.append([x, y, yaw, w, l])

        mot.update(det, timer)
        timer += dt

        save_track = []
        for object in mot.tracks:
            new_object = dict(object)
            del new_object['filter']
            save_track.append(new_object)

        tracks.append(save_track)
        print('Num of objects ', len(save_track))

    with open(save_path, 'w') as f:
        json.dump(tracks, f, indent=2)

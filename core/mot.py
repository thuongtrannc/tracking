import os
import sys
from typing import Tuple
sys.path.append(os.getcwd)

import numpy as np
import json
from tracking.core.hungarian import Matching
from tracking.core.kalman_filter import KamanFilter

entity = {  'id': [],
            'last_time': [],
            'age': [],
            'x': [], 
            'y':[],
            'yaw': [],
            'vx': [],
            'vy': [],
            'w': [],
            'l': [],
            'filter': []
         }

class MOT():
    def __init__ (self, max_id=100, alive_frame=5, dT=0.1, overlap_iou=0.2):
        self.max_id = max_id
        self.alive_frame = 5
        self.overlap_iou = 0.2
        self.dT = dT

        self.tracks = []
        self.matching = Matching()


    def update(self, det, curr_time):
        num_detect = len(det)
        num_track = len(self.tracks)

        # track_boxes = self.get_track_boxes()
        track_boxes = self.get_predict_boxes(self.dT)
        row_ind, col_ind = self.data_association(det, track_boxes)

        for i, j in zip(row_ind, col_ind):
            
            z = np.array([det[i][0], det[i][1], det[i][3], det[i][4]])
            self.tracks[j]['filter'].update(z)
            self.tracks[j]['last_time'] = curr_time
            estimator = self.tracks[j]['filter'].get_estimate()

            self.tracks[j]['x'] = estimator[0]
            self.tracks[j]['y'] = estimator[1]
            self.tracks[j]['yaw'] = det[i][2]
            self.tracks[j]['vx'] = estimator[2]
            self.tracks[j]['vy'] = estimator[3]
            self.tracks[j]['w'] = estimator[4]
            self.tracks[j]['l'] = estimator[5]
            self.tracks[j]['age'] += 1

        for i in range(num_detect):
            if i not in row_ind:
                self.add_object(det[i], curr_time)

        rm_list = []
        for i in range(num_track):
            if i not in col_ind:
                remove = self.check_alive(self.tracks[i], curr_time)
                if remove == True:
                    rm_list.append(i)
                else:
                    dt = curr_time - self.tracks[i]['last_time']
                    predictor = self.tracks[i]['filter'].predict(dt)
                    self.tracks[i]['x'] = predictor[0]
                    self.tracks[i]['y'] = predictor[1]
                    self.tracks[i]['yaw'] = self.tracks[i]['yaw']
                    self.tracks[i]['w'] = predictor[4]
                    self.tracks[i]['l'] = predictor[5]

        self.remove_objects(rm_list)
        if len(rm_list) > 0:
            print('Remove object id ', rm_list)

    
    def predict(self):
        return 0


    def data_association(self, det, track):
        row_ind, col_ind = self.matching.data_associate(det, track)
        return row_ind, col_ind
        

    def add_object(self, det, curr_time):
        new_entity = dict(entity)
        ids = self.get_all_ids()
        
        can_add = False
        for i in range(self.max_id):
            if i not in ids:
                new_id = i
                can_add = True
                break
        
        if can_add:
            new_entity['id'] = new_id
            new_entity['age'] = 0
            new_entity['last_time'] = curr_time
            new_entity['x'] = det[0]
            new_entity['y'] = det[1]
            new_entity['yaw'] = det[2]
            new_entity['vx'] = 0
            new_entity['vy'] = 0
            new_entity['w'] = det[3]
            new_entity['l'] = det[4]
            new_entity['filter'] = KamanFilter(new_entity['id'])

            # Init for kalman filter [x, y, 0, 0, w, l]
            x_init = [det[0], det[1], 0, 0, det[3], det[4]]
            new_entity['filter'].init_estimator(x_init)

            self.tracks.append(new_entity)
            print('Add new object')

    
    def get_all_ids(self):
        ids = []
        for i in range(len(self.tracks)):
            ids.append(self.tracks[i]['id'])
        return ids


    def remove_objects(self, rm_list):
        for index in sorted(rm_list, reverse=True):
            del self.tracks[index]


    def check_alive(self, check_entity, curr_time):
        dtime = curr_time - check_entity['last_time']
        if dtime > self.alive_frame * self.dT:
            return True
        else:
            return False


    def get_track_boxes(self):
        track_boxes = []
        for i in range(len(self.tracks)):
            box = [self.tracks[i]['x'], self.tracks[i]['y'], self.tracks[i]['yaw'], self.tracks[i]['w'], self.tracks[i]['l']]
            track_boxes.append(box)        
        return track_boxes


    def get_predict_boxes(self, dt):
        track_boxes = []
        for i in range(len(self.tracks)):
            pred = self.tracks[i]['filter'].predict(dt)
            box = [pred[0], pred[1], self.tracks[i]['yaw'], pred[4], pred[5]]
            track_boxes.append(box)        
        return track_boxes

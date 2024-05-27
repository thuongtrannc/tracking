import json
from operator import index
import os
import sys
sys.path.append(os.getcwd())

import numpy as np

import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas
from vispy import app
from vispy.scene.visuals import Text

import json


class Vis():
    def __init__(self, data_path=''):
        self.data_path = data_path

        self.load_data()

        self.num_frame = len(self.data)
        self.index = 0

        self.canvas = SceneCanvas(keys='interactive',
                                 show=True,
                                 size=(1600, 900))
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)

        self.grid = self.canvas.central_widget.add_grid()
        self.scan_view = vispy.scene.widgets.ViewBox(parent=self.canvas.scene,
                                                    camera=TurntableCamera(distance=30.0))
        self.grid.add_widget(self.scan_view)
        self.scan_vis = visuals.Markers()
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)

        self.bbox = vispy.scene.visuals.Line(parent=self.scan_view.scene)
        
        self.timer_interval = 0.1
        self.timer = app.Timer(interval=self.timer_interval, connect=self.update_frame, start=True)
        self.paused = False
        self.debug = True

    def plot_boxes(self, objects, type = 'gt'):

        object_colors = {0: np.array([1, 0, 0, 1]), 
                         1: np.array([0, 1, 0, 1])}
        
        connect = []
        points = []
        colors = []

        cnt = 0
        for i in range(len(objects)):
            box = [objects[i]['x'], objects[i]['y'], objects[i]['yaw'], objects[i]['w'], objects[i]['l']]
            if box[3] == 0 or box[4] == 0:
                continue
            
            corners = self.get_corners(box)

            k = 4 * cnt
            cnt += 1
            con = [ [k, k + 1],
                    [k + 1, k + 2],
                    [k + 2, k + 3],
                    [k + 3, k]]
            
            print('Test 00')
            connect.append(con)
            points.append(corners)
            colors.append(np.tile(object_colors[0], (4,1)))


        num_points = cnt * 4
        if num_points == 0:
            return
        else:
            points = np.array(points).reshape((num_points, 3))
            connect = np.array(connect).reshape((num_points, 2))
            colors = np.array(colors)

            self.bbox.set_data(pos=points,
                                connect=connect,
                                color=colors)
            return
        

    def load_data(self):
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)


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


    def _key_press(self, event):
        if event.key == 'Right':
            if self.paused:
                self.index += 1
            self.update_frame(event)

        if event.key == 'Left':
            if self.paused and self.index > 0:
                self.index -= 1
            self.update_frame(event)

        if event.key == 'Space':
            if self.paused:
                self.timer.start()
            else:
                self.timer.stop()

            self.paused = not self.paused

        if event.key == "W":
            self.timer_interval -= 0.01
            self.timer_interval = max(self.timer_interval, 0)
            self.timer.interval = self.timer_interval

        if event.key == "S":
            self.timer_interval += 0.01
            self.timer.interval = self.timer_interval

        if event.key == "Down":
            # for saving if needed
            pass

        if event.key == 'Q':
            self.destroy()


    def destroy(self):
        self.canvas.close()
        vispy.app.quit()


    def _draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    
    def run(self):
        self.canvas.app.run()


    def update_frame(self, event):

        print('Frame ', self.index)
        if self.index >= self.num_frame:
            print('Done visualization')
            self.destroy()
            return

        self.canvas.title = f"Frame: {self.index} / {self.num_frame}"

        objects = self.data[self.index]
        self.plot_boxes(objects, 'gt')
        
        if not self.paused:
            self.index += 1
            


if __name__ == '__main__':
    vis = Vis('data/test_multi.json')
    vis.run()
    

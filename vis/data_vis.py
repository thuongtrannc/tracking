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


class Vis():
    def __init__(self, data_path=''):
        self.data_path = data_path

        self.load_data()
        # self.vis_box()

        self.num_frame = self.data.shape[0]
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

    def plot_boxes(self, objects, type = 'gt'):
        if len(objects) == 0:
            self.bbox.set_data(pos=[],
                            connect=[],
                            color=[])
            return

        object_colors = {0: np.array([1, 0, 0, 1]), 
                         1: np.array([0, 1, 0, 1])}
        
        box = [objects[0], objects[1], objects[2], objects[6], objects[7]]

        corners = self.get_corners(box)

        connect = []
        points = []
        colors = []

        if type == 'gt':
            colors.append(np.tile(object_colors[0], (4,1)))
        elif type == 'pred':
            colors.append(np.tile(object_colors[1], (4,1)))

        j = 4 * 0
        con = [ [j, j + 1],
                [j + 1, j + 2],
                [j + 2, j + 3],
                [j + 3, j]]
        con = np.array(con)
        connect = con

        points = corners

        self.bbox.set_data(pos=points,
                            connect=connect,
                            color=colors)
        

    def load_data(self):
        self.data = np.loadtxt(self.data_path, delimiter=',')


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
    vis = Vis('data/test_cv.txt')
    vis.run()
    

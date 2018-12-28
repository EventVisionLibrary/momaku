# Copyright 2018 Event Vision Library.

import time

import cv2
import numpy as np

from objects.cube import SolidCube
from objects.sphere import SolidSphere
from renderer import Renderer
from subjects.simple_walker import SimpleWalker
from subjects.gopigo_simulator import Gopigo

class FallingStone():

    def __init__(self):
        self.time_delta = 1e-3
        _ = self.reset()

    def init_subject(self):
        # subject = SimpleWalker()
        subject = Gopigo()
        subject.initialize_dynamics(
                    position = np.array([-1, -1, -1], dtype=np.float32),
                    direction = np.array([3, 3, 3], dtype=np.float32),
                    velocity = np.array([0.01, 0.01, 0.01], dtype=np.float32)
        )
        return subject

    def init_objects(self):
        objects = []
        cube = SolidCube(size=2.0)
        cube.initialize_dynamics(direction=np.array([np.pi/4, np.pi/4, np.pi/4]),
                                 position=np.array([2, 4, 1]),
                                 velocity=np.array([0, 2, 0]))
        objects.append(cube)
        return objects

    def init_renderer(self):
        renderer = Renderer(camera_position=self.subject.position,
                            target_position=self.subject.direction)
        return renderer

    def move_objects(self):
        for obj in self.objects:
            obj.update_dynamics(dt=0.1, new_velocity=np.array([0, 2, 0]), 
                                angular_velocity=np.array([0, np.pi, 0]))

    def move_subject(self, action):
        # self.subject.position += self.subject.velocity
        getattr(self.subject, action)

    def step(self, action):
        self.timestamp += self.time_delta
        self.move_objects()
        self.move_subject(action)
        self.renderer.update_perspective(self.subject.position, self.subject.direction)

        # obs
        self.current_image = self.renderer.render_objects(self.objects)
        events = self.__calc_events()
        self.prev_image = self.current_image

        r = 0.0                 # reward
        done = False
        info = {}
        return events, r, done, info

    def reset(self):
        self.timestamp = 0.0
        self.objects = self.init_objects()
        self.subject = self.init_subject()
        self.renderer = self.init_renderer()
        self.prev_image = np.zeros([self.renderer.display_height, self.renderer.display_width, 3])
        self.current_image = self.renderer.render_objects(self.objects)
        events = self.__calc_events()
        return events

    def render(self, mode='human'):
        # TODO: enable rendering for debug
        raise NotImplementedError

    def __calc_events(self):
        # TODO: assign time stamp dynamically
        prev_gray = self.__rgb_to_gray(self.prev_image)
        current_gray = self.__rgb_to_gray(self.current_image)
        diff = current_gray - prev_gray
        events = []
        for y in range(self.renderer.display_height):
            for x in range(self.renderer.display_width):
                pol = diff[y, x]
                if pol == 0:
                    continue
                elif pol > 0:
                    events.append((self.timestamp, y, x, 1))
                else:
                    events.append((self.timestamp, y, x, -1))
        return events

    def __rgb_to_gray(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

def events_to_image(events):
    H = 900
    W = 900
    image = np.zeros([H, W, 3])
    for (t, y, x, p) in events:
        if p == 1:
            image[y, x] = (255, 0, 0)
        else:
            image[y, x] = (0, 0, 255)
    return image

if __name__ == '__main__':
    env = FallingStone()
    start = time.time()
    N = 20
    for i in range(0, N):
        events, r, done, info = env.step(action='go')
        image = events_to_image(events)
        cv2.imwrite("../fig/image" + str(i) + ".png", image)
    print("Average Elapsed Time: {} s".format((time.time() - start) / N))

# Copyright 2018 Event Vision Library.

import time

import cv2
import numpy as np

from objects.cube import SolidCube
from objects.sphere import SolidSphere
from renderer import Renderer
from subjects.simple_walker import SimpleWalker
from subjects.gopigo_simulator import Gopigo

COLLISION_THRESHOLD = 1.0

class FallingStone():
    def __init__(self, dt=1e-2, render_width=900, render_height=900):
        self.dt = 1e-2
        self.render_width = render_width
        self.render_height = render_height
        self.reset()

    def init_subject(self):
        # subject = SimpleWalker()
        subject = Gopigo()
        subject.initialize_dynamics(position=np.array([-1, 0, 2], dtype=np.float32),
                                    direction=np.array([1, 1, 1], dtype=np.float32),
                                    velocity=np.array([0.1, 0.1, 0.1], dtype=np.float32))
        return subject

    def init_objects(self):
        objects = []
        cube = SolidCube(size=2.0)
        cube.initialize_dynamics(position=np.array([2, 0, 1]),
                                 direction=np.array([np.pi/4, np.pi/4, np.pi/4]),
                                 velocity=np.array([0, 0, -1]))
        objects.append(cube)
        sphere = SolidSphere(radius=0.5)
        sphere.initialize_dynamics(position=np.array([3, 0, 2]),
                                   velocity=np.array([0, 0, -1]))
        objects.append(sphere)
        return objects

    def init_renderer(self):
        renderer = Renderer(camera_position=self.subject.position,
                            target_position=self.subject.direction,
                            width=self.render_width,
                            height=self.render_height)
        return renderer

    def move_objects(self):
        for obj in self.objects:
            obj.update_dynamics(dt=self.dt, new_velocity=np.array([0, 0, -1]), 
                                angular_velocity=np.array([np.pi, 0, 0]))

    def move_subject(self, action):
        getattr(self.subject, action)(self.dt)

    def step(self, action):
        if self.done:
            print("[Error] the game already finished.")
            raise Exception
        self.timestamp += self.dt
        self.move_objects()
        self.move_subject(action)
        self.renderer.update_perspective(
            self.subject.position, self.subject.direction)

        # obs
        self.current_image = self.renderer.render_objects(self.objects, True)
        events = self.__calc_events()
        self.prev_image = self.current_image

        if self.__is_collision():
            r = -1.0
            self.done = True
        else:
            r = 0.1
            self.done = False
        info = {}
        return events, r, self.done, info

    def reset(self):
        self.done = False
        self.timestamp = 0.0
        self.objects = self.init_objects()
        self.subject = self.init_subject()
        self.renderer = self.init_renderer()
        self.prev_image = np.zeros([self.renderer.display_height, self.renderer.display_width, 3])
        self.current_image = self.renderer.render_objects(self.objects)

    def render(self, mode='human'):
        # TODO: enable rendering for debug
        raise NotImplementedError

    def __is_collision(self):
        for obj in self.objects:
            if isinstance(obj, SolidSphere):
                if self.__check_sphere_collision(obj):
                    return True
            if isinstance(obj, SolidCube):
                if self.__check_cube_collision(obj):
                    return True
        return False

    def __check_sphere_collision(self, sphere):
        d = np.sum((sphere.position - self.subject.position)**2)
        print("distance: ", np.sqrt(d))
        if d < (COLLISION_THRESHOLD+sphere.radius)**2:
            return True
        else:
            return False

    def __check_cube_collision(self, cube):
        pass

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

def events_to_image(events, width, height):
    image = np.zeros([height, width, 3], dtype=np.uint8)
    for (t, y, x, p) in events:
        if p == 1:
            image[y, x, 0] = 255
        else:
            image[y, x, 2] = 255
    return image

if __name__ == '__main__':
    w, h = 400, 400
    env = FallingStone(render_width=w, render_height=h)
    start = time.time()
    N = 20
    for i in range(0, N):
        events, r, done, info = env.step(action='forward')
        image = events_to_image(events, w, h)
        cv2.imwrite("../fig/image" + str(i) + ".png", image)
    print("Average Elapsed Time: {} s".format((time.time() - start) / N))

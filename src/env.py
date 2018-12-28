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
        self.dt = 0.03
        self.render_width = render_width
        self.render_height = render_height
        self.reset()

    def init_subject(self):
        # subject = SimpleWalker()
        subject = Gopigo()
        subject.initialize_dynamics(position=np.array([-2, -6, -1], dtype=np.float32),
                                    direction=np.array([4, 8, 2], dtype=np.float32),
                                    velocity=np.array([4, 8, 0], dtype=np.float32))
        return subject

    def init_objects(self): 
        objects = []
        cube = SolidCube(size=1.0, color=np.array([0.5, 0.5, 0.0]))
        cube.initialize_dynamics(position=np.array([0, 2, -1]),
                                 direction=np.ones(3),
                                 velocity=np.array([0., 0., 0.]))
        objects.append(cube)
        sphere = SolidSphere(radius=0.5)
        sphere.initialize_dynamics(position=np.array([1, 1, -5]),
                                   velocity=np.array([0, 0, 2]))
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
            new_velocity = self.free_fall_velocity(obj)
            obj.update_dynamics(dt=self.dt, new_velocity=new_velocity,
                                angular_velocity=np.array([np.pi, 0, 0]))

    def free_fall_velocity(self, obj):
        return obj.velocity + self.dt * np.array([0, 0, 9.8])

    def move_subject(self, action):
        getattr(self.subject, action)(self.dt)

    def step(self, action):
        if self.done:
            raise Exception("The game already finished.")
        self.timestamp += self.dt
        self.move_objects()
        self.move_subject(action)
        self.renderer.update_perspective(
            self.subject.position, self.subject.position + self.subject.direction)

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
        prev_intensity = self.__rgb_to_intensity(self.prev_image)
        current_intensity = self.__rgb_to_intensity(self.current_image)
        diff = current_intensity - prev_intensity
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

    def __rgb_to_intensity(self, rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        intensity = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return intensity

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
    N = 30
    for i in range(0, N):
        try:
            events, r, done, info = env.step(action='forward')
        except Exception as inst:
            print(inst)
            break
        image = events_to_image(events, w, h)
        cv2.imwrite("../fig/image" + str(i) + ".png", image)
    print("Average Elapsed Time: {} s".format((time.time() - start) / N))

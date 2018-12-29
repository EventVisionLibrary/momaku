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

    def __init_subject(self):
        # subject = SimpleWalker()
        subject = Gopigo()
        subject.initialize_dynamics(position=np.array([-2, -6, -1], dtype=np.float32),
                                    direction=np.array([4, 8, 3], dtype=np.float32),
                                    velocity=np.array([4, 8, 0], dtype=np.float32))
        return subject

    def __init_objects(self): 
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

    def __init_renderer(self):
        renderer = Renderer(camera_position=self.subject.position,
                            target_position=self.subject.direction,
                            width=self.render_width,
                            height=self.render_height)
        return renderer

    def __move_objects(self):
        for obj in self.objects:
            new_velocity = self.__free_fall_velocity(obj)
            obj.update_dynamics(dt=self.dt, new_velocity=new_velocity,
                                angular_velocity=np.array([np.pi, 0, 0]))

    def __free_fall_velocity(self, obj):
        # TODO: implement air resistance
        return obj.velocity + self.dt * np.array([0, 0, 9.8])

    def __move_subject(self, action):
        getattr(self.subject, action)(self.dt)

    def step(self, action):
        if self.done:
            raise Exception("The game already finished.")
        self.timestamp += self.dt
        self.__move_objects()
        self.__move_subject(action)
        self.renderer.update_perspective(
            self.subject.position, self.subject.position + self.subject.direction)

        # obs
        current_image = self.renderer.render_objects(self.objects, True)
        self.current_intensity = self.__rgb_to_intensity(current_image)
        events = self.__calc_events()
        self.prev_intensity = self.current_intensity

        # default reward (relative position to sphere)
        if self.__is_collision():
            r = -10.0
            self.done = True
        else:
            # extract solidsphere
            for obj in self.objects:
                if isinstance(obj, SolidSphere):
                    r = self.__measure_distance(obj)*(-1.0)/10.0
            self.done = False

        # more forward, more reward
        if action == "forward":
            r += 0.5

        # add done judgement from timestamp
        if self.timestamp > 1.0:
            self.done = True

        info = {}
        return events, r, self.done, info

    def reset(self):
        self.done = False
        self.timestamp = 0.0
        self.objects = self.__init_objects()
        self.subject = self.__init_subject()
        self.renderer = self.__init_renderer()
        prev_image = np.zeros([self.renderer.display_height, self.renderer.display_width, 3])
        current_image = self.renderer.render_objects(self.objects)
        self.prev_intensity = self.__rgb_to_intensity(prev_image)
        self.current_intensity = self.__rgb_to_intensity(current_image)

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

    def __measure_distance(self, sphere):
        return np.sum((sphere.position - self.subject.position)**2)

    def __check_sphere_collision(self, sphere):
        d = np.sum((sphere.position - self.subject.position)**2)
        #print("distance: ", np.sqrt(d))
        if d < (COLLISION_THRESHOLD+sphere.radius)**2:
            return True
        else:
            return False

    def __check_cube_collision(self, cube):
        pass

    def __calc_events(self):
        # TODO: assign time stamp dynamically
        diff = np.sign(self.current_intensity - self.prev_intensity).astype(np.int32)
        event_index = np.where(np.abs(diff) > 0)
        events = np.array([np.full(len(event_index[0]), self.timestamp, dtype=np.int32),
                           event_index[0], event_index[1], diff[event_index]]).T
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
    w, h = 800, 800
    env = FallingStone(render_width=w, render_height=h)
    N = 30
    executed_times = []
    for i in range(0, N):
        start = time.time()
        try:
            events, r, done, info = env.step(action='backward')
        except Exception as inst:
            print(inst)
            break
        image = events_to_image(events, w, h)
        executed_times.append(time.time() - start)
        cv2.imwrite("../fig/image" + str(i) + ".png", image)
    print("Average Elapsed Time: {} s".format(np.mean(executed_times)))

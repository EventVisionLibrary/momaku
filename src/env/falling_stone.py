# Copyright 2018 Event Vision Library.

import copy
import time

import cv2
import numpy as np

from env import EnvBase
from env import physics, util

import objects
import subjects
from renderer import Renderer
import numba


COLLISION_THRESHOLD = 1.0

class FallingStone(EnvBase):
    def __init__(self, dt=1e-2, render_width=900, render_height=900, obs_as_img=False):
        self.obs_as_img = obs_as_img
        self.config = {
            "dt": dt,
            "Cp": 0.05, # plus
            "Cm": 0.03, # minus
            "sigma_Cp": 0.0001,
            "sigma_Cm": 0.0001,
            "refractory_period": 1e-4  # time during which a pixel cannot fire events just after it fired one
        }
        super(FallingStone, self).__init__(dt, render_width, render_height)

    def reset(self):
        self.timestamp = 0.0
        self.objects = self.__init_objects()
        self.subject = self.__init_subject()
        self.renderer = self.__init_renderer()

        current_image = self.renderer.render_objects(self.objects)
        self.current_intensity = util.rgb_to_intensity(current_image)
        self.prev_intensity = copy.deepcopy(self.current_intensity)
        self.ref_values = copy.deepcopy(self.current_intensity)
        self.last_event_timestamp = np.zeros([self.render_height, self.render_width])

        events = self.__calc_events(self.timestamp)
        if self.obs_as_img:
            obs = util.events_to_image(events, self.render_width, self.render_height)
        else:
            obs = events
        r = 0
        self.done = False
        info = {}
        return obs, r, self.done, info

    def __init_renderer(self):
        renderer = Renderer(camera_position=self.subject.position,
                            target_position=self.subject.direction,
                            width=self.render_width,
                            height=self.render_height)
        return renderer

    def __init_subject(self):
        # subject = subjects.SimpleWalker()
        subject = subjects.Gopigo()

        _p = np.random.random()
        if _p < 0.33:
            subject.initialize_dynamics(position=np.array([-2, -6, -1], dtype=np.float32),
                                        direction=np.array([4, 8, 3], dtype=np.float32),
                                        velocity=np.array([4, 8, 0], dtype=np.float32))
        elif _p < 0.66:
            subject.initialize_dynamics(position=np.array([4, 5, -1], dtype=np.float32),
                                        direction=np.array([-7, -7, 3], dtype=np.float32),
                                        velocity=np.array([-7, -7, 0], dtype=np.float32))
        else:
            subject.initialize_dynamics(position=np.array([-2.5, 4, -1], dtype=np.float32),
                                        direction=np.array([6, -7, 3], dtype=np.float32),
                                        velocity=np.array([6, -7, 0], dtype=np.float32))

        return subject

    def __init_objects(self): 
        objs = []
        # cube = objects.SolidCube(size=1.0, color=np.array([0.5, 0.5, 0.0]))
        # cube.initialize_dynamics(position=np.array([0, 2, -1]),
        #                          direction=np.ones(3),
        #                          velocity=np.array([0., 0., 0.]))
        # objs.append(cube)
        sphere = objects.SolidSphere(radius=0.5)
        sphere.initialize_dynamics(position=np.array([1, 1, -5]) + np.random.random(3) - 0.5,
                                   velocity=np.array([0, 0, 2 + np.random.random() - 0.5]))
        objs.append(sphere)
        return objs

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
        self.current_intensity = util.rgb_to_intensity(current_image)
        events = self.__calc_events(dynamic_timestamp=False)
        self.prev_intensity = self.current_intensity
        if self.obs_as_img:
            obs = util.events_to_image(events, self.render_width, self.render_height)
        else:
            obs = events

        # default reward (relative position to sphere)
        if self.__is_collision():
            r = -10.0
            self.done = True
        else:
            # extract objects.solidsphere
            for obj in self.objects:
                if isinstance(obj, objects.SolidSphere):
                    r = self.__measure_distance(obj)*(-1.0)/10.0
            self.done = False

        # more forward, more reward
        if action == "forward":
            r += 0.5

        # add done judgement from timestamp
        if self.timestamp > 1.0:
            self.done = True

        info = {}
        return obs, r, self.done, info

    # basic functions for objects and subjects
    def __move_objects(self):
        for obj in self.objects:
            if self.__check_on_the_surface(obj):
                new_velocity = np.zeros((3))
            else:
                new_velocity = physics.free_fall_velocity(self.dt, obj)
            obj.update_dynamics(dt=self.dt, new_velocity=new_velocity,
                                angular_velocity=np.array([np.pi, 0, 0]))

    def __move_subject(self, action):
        getattr(self.subject, action)(self.dt)

    def __check_on_the_surface(self, obj):
        if obj.position[2] > 0:
            return True
        else:
            return False

    def __is_collision(self):
        for obj in self.objects:
            if isinstance(obj, objects.SolidSphere):
                if self.__check_sphere_collision(obj):
                    return True
            if isinstance(obj, objects.SolidCube):
                if self.__check_cube_collision(obj):
                    return True
        return False

    def __check_sphere_collision(self, sphere):
        d = self.__measure_distance(sphere)
        #print("distance: ", np.sqrt(d))
        if d < (COLLISION_THRESHOLD+sphere.radius)**2:
            return True
        else:
            return False

    def __measure_distance(self, sphere):
        return np.sum((sphere.position - self.subject.position)**2)

    def __check_cube_collision(self, cube):
        pass

    @numba.jit
    def __calc_events(self, dynamic_timestamp=True):
        # functions for calculation of events

        if not dynamic_timestamp:
            diff = np.sign(self.current_intensity - self.prev_intensity).astype(np.int32)
            diff = util.add_impulse_noise(diff, prob=1e-3)
            event_index = np.where(np.abs(diff) > 0)
            events = np.array([np.full(len(event_index[0]), self.timestamp, dtype=np.int32),
                               event_index[0], event_index[1], diff[event_index]]).T
            return events

        # compliment interpolation
        events = []
        for y in range(self.render_height):
            for x in range(self.render_width):
                current = self.current_intensity[y, x]
                prev = self.prev_intensity[y, x]
                if current == prev:
                    continue    # no event
                prev_cross = self.ref_values[y, x]

                pol = 1.0 if current > prev else -1.0
                C = self.config["Cp"] if pol > 0 else self.config["Cm"]
                sigma_C = self.config["sigma_Cp"] if pol > 0 else self.config["sigma_Cm"]
                if sigma_C > 0:
                    C += np.random.normal(0, sigma_C)
                current_cross = prev_cross
                all_crossings = False
                while True:
                    # Consider every time when intensity changed over threshold C.
                    current_cross += pol*C
                    #print("pol: {}, current_cross: {}, prev: {}, current: {}".format(pol, current_cross, prev, current))
                    if (pol > 0 and current_cross > prev and current_cross <= current) \
                    or (pol < 0 and current_cross < prev and current_cross >= current):
                        edt = (current_cross - prev) * self.dt / (current - prev)
                        t = self.timestamp + edt
                        last_t = self.last_event_timestamp[y, x]
                        dt = t - last_t
                        assert dt > 0
                        if last_t == 0 or dt >= self.config["refractory_period"]:
                            events.append((t, y, x, pol>0))
                            self.last_event_timestamp[y, x] = t
                        self.ref_values[y, x] = current_cross
                    else:
                        all_crossings = True
                    if all_crossings:
                        break
        return events

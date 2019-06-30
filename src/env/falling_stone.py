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

COLLISION_THRESHOLD = 1.0

class FallingStone(EnvBase):
    def __init__(self, dt=1e-2, render_width=900, render_height=900, obs_as_img=False):
        self.obs_as_img = obs_as_img
        self.config = self.initialize_esim_config(dt, render_width, render_height)
        super(FallingStone, self).__init__(dt, render_width, render_height)

    def reset(self):
        self.timestamp = 0.0
        self.objects = self.__init_objects()
        self.subject = self.__init_subject()
        self.renderer = self.__init_renderer()
        obs = self.reset_esim_param()
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
        sphere = objects.SolidSphere(radius=0.5)
        sphere.initialize_dynamics(position=np.array([1, 1, -5]) + np.random.random(3) - 0.5,
                                   velocity=np.array([0, 0, 2 + np.random.random() - 0.5]))
        objs.append(sphere)
        return objs

    def step(self, action):
        self.check_done()
        self.timestamp += self.dt
        self.move_objects_with_gravity()
        self.move_subject(action)
        self.renderer.update_perspective(
            self.subject.position, self.subject.position + self.subject.direction)

        # obs
        # current_image = self.renderer.render_objects(self.objects, True)
        # current_intensity = util.rgb_to_intensity(current_image)
        # events, self.ref_values, self.last_event_timestamp \
        #     = util.calc_events_from_image(current_intensity, self.prev_intensity,
        #                                   self.ref_values, self.timestamp,
        #                                   self.last_event_timestamp, self.config,
        #                                   dynamic_timestamp=False)

        # self.prev_intensity = current_intensity
        # if self.obs_as_img:
        #     obs = util.events_to_image(events, self.render_width, self.render_height)
        # else:
        #     obs = events

        self.prev_intensity, self.ref_values, self.last_event_timestamp, obs = \
            self.step_esim_param(self.timestamp, self.config, self.prev_intensity,
                                 self.ref_values, self.last_event_timestamp)
        r, self.done = self.get_reward_and_done(action)
        info = {}
        return obs, r, self.done, info

    def get_reward_and_done(self, action):
        # default reward (relative position to sphere)
        if self.__is_collision():
            r = -10.0
            done = True
        else:
            # extract objects.solidsphere
            for obj in self.objects:
                if isinstance(obj, objects.SolidSphere):
                    r = physics.measure_distance(obj.position, self.subject.position) * (-1.0) / 10.0
            done = False

        # more forward, more reward
        if action == "forward":
            r += 0.5

        # add done judgement from timestamp
        if self.timestamp > 1.0:
            done = True
        return r, done


    def __is_collision(self):
        for obj in self.objects:
            if isinstance(obj, objects.SolidSphere):
                if physics.check_sphere_collision(obj, self.subject, COLLISION_THRESHOLD):
                    return True
            if isinstance(obj, objects.SolidCube):
                if physics.check_cube_collision(obj, self.subject):
                    return True
        return False




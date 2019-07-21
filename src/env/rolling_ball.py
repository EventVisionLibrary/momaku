# Copyright 2018 Event Vision Library.

import time

import cv2
import numpy as np

from env import EnvBase
from env import physics, util

import objects
import subjects
from renderer import Renderer

COLLISION_THRESHOLD = 1.0

class RollingBall(EnvBase):
    def __init__(self, dt=1e-2, render_width=900, render_height=900, obs_as_img=False):
        self.obs_as_img = obs_as_img
        self.config = self.initialize_esim_config(dt, render_width, render_height)
        super(RollingBall, self).__init__(dt, render_width, render_height)

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
        subject = subjects.Gopigo()
        _p = np.random.random()
        subject.initialize_dynamics(position=np.array([-2, -6, -1], dtype=np.float32),
                                    direction=np.array([4, 8, 3], dtype=np.float32),
                                    velocity=np.array([4, 8, 0], dtype=np.float32))
        return subject

    def __init_objects(self): 
        objs = []
        _p = np.random.random()
        sphere = objects.SolidSphere(radius=0.5)
        # if _p < 0.5:
        sphere.initialize_dynamics(position=np.array([2, 10, -5]) + np.random.random(3) - 0.5,
                                    velocity=np.array([0, 2, 2 + np.random.random() - 0.5]))
        # else:
        #     sphere.initialize_dynamics(position=np.array([4, 2, -5]) + np.random.random(3) - 0.5,
        #                                velocity=np.array([1, 2, 2 + np.random.random() - 0.5]))
        objs.append(sphere)
        return objs

    def step(self, action):
        self.check_done()
        self.timestamp += self.dt
        self.move_objects_with_gravity()
        self.move_subject(action)
        self.renderer.update_perspective(
            self.subject.position, self.subject.position + self.subject.direction)

        self.prev_intensity, self.ref_values, self.last_event_timestamp, obs = \
            self.step_esim_param(self.timestamp, self.config, self.prev_intensity,
                                 self.ref_values, self.last_event_timestamp)

        r, self.done = self.get_reward_and_done()
        info = {}
        return obs, r, self.done, info

    def get_reward_and_done(self):
        if self.__is_collision():
            done = True
            r = 0.0
        else:
            done = False
            for obj in self.objects:
                r = 1.0 / (physics.measure_angle(obj.position - self.subject.position,
                                                 self.subject.velocity) + 1.)
        if self.timestamp > 1.0:
            done = True
        return r, done

    def __is_collision(self):
        for obj in self.objects:
            if physics.check_sphere_collision(obj, self.subject, COLLISION_THRESHOLD):
                return True
        return False


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
    def __init__(self, dt=1e-2, render_width=900, render_height=900,
                 obs_as_img=False, subject_action_list=['forward', 'stop']):
        self.obs_as_img = obs_as_img
        self.config = self.initialize_esim_config(dt, render_width, render_height)
        super(FallingStone, self).__init__(dt, render_width, render_height, subject_action_list)

        self.setp_i = 0

    def reset(self):
        self.timestamp = 0.0
        self.objects = self.__init_objects()
        self.subject = self.__init_subject(self.subject_action_list)
        self.renderer = self.__init_renderer()
        obs = self.reset_esim_param()
        r = 0
        self.done = False
        info = {}

        self.setp_i = 0

        return obs, r, self.done, info

    def __init_renderer(self):
        renderer = Renderer(camera_position=self.subject.position,
                            target_position=self.subject.direction,
                            width=self.render_width,
                            height=self.render_height)
        return renderer

    def __init_subject(self, action_list):
        # subject = subjects.SimpleWalker()
        subject = subjects.Gopigo(action_list)

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

        image = self.renderer.render_objects(self.objects, show_axis=True)
        cv2.imwrite("../fig/image" + str(self.setp_i) + ".png", image)

        self.prev_intensity, self.ref_values, self.last_event_timestamp, obs = \
            self.step_esim_param(self.timestamp, self.config, self.prev_intensity,
                                 self.ref_values, self.last_event_timestamp)
        r, self.done = self.get_reward_and_done(action)
        info = {}

        cv2.imwrite("../fig/obs" + str(self.setp_i) + ".png", obs)
        self.setp_i += 1

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
            r += 1.0

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




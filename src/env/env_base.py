# Copyright 2018 Event Vision Library.

import copy
import numba
import numpy as np

from env import physics, util

COLLISION_THRESHOLD = 1.0

class EnvBase():
    def __init__(self, dt, render_width, render_height, subject_action_list):
        self.dt = dt
        self.subject_action_list = subject_action_list
        self.render_width = render_width
        self.render_height = render_height
        self.reset()

    # basic functions for environment
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='human'):
        # TODO: enable rendering for debug
        raise NotImplementedError

    def check_done(self):
        if self.done:
            raise Exception("The game already finished.")

    # basic functions for objects and subjects
    def move_objects_with_gravity(self):
        for obj in self.objects:
            new_velocity = physics.free_fall_velocity(self.dt, obj)
            obj.update_dynamics(dt=self.dt, new_velocity=new_velocity,
                                angular_velocity=np.array([np.pi, 0, 0]))

    def move_subject(self, action):
        getattr(self.subject, action)(self.dt)

    # function for ESIM simulator
    def initialize_esim_config(self, dt, render_width, render_height):
        config = {
            "dt": dt,
            "render_width": render_width,
            "render_height": render_height,
            "Cp": 0.05,                # plus
            "Cm": 0.03,                # minus
            "sigma_Cp": 0.0001,
            "sigma_Cm": 0.0001,
            "refractory_period": 1e-4  # time during which a pixel cannot fire events just after it fired one
        }
        return config

    def reset_esim_param(self):
        current_image = self.renderer.render_objects(self.objects)
        self.current_intensity = util.rgb_to_intensity(current_image)
        self.prev_intensity = copy.deepcopy(self.current_intensity)
        self.ref_values = copy.deepcopy(self.current_intensity)
        self.last_event_timestamp = np.zeros([self.render_height, self.render_width])

        events, self.ref_values, self.last_event_timestamp = \
            util.calc_events_from_image(self.current_intensity, self.prev_intensity,
                                        self.ref_values, self.timestamp, 
                                        self.last_event_timestamp, self.config)
        if self.obs_as_img:
            obs = util.events_to_image(events, self.render_width, self.render_height)
        else:
            obs = events
        return obs

    def step_esim_param(self, timestamp, config, prev_intensity, ref_values, last_event_timestamp):
        current_image = self.renderer.render_objects(self.objects, True)
        current_intensity = util.rgb_to_intensity(current_image)
        events, ref_values, last_event_timestamp \
            = util.calc_events_from_image(current_intensity, prev_intensity,
                                          ref_values, timestamp,
                                          last_event_timestamp, config,
                                          dynamic_timestamp=False)

        if self.obs_as_img:
            obs = util.events_to_image(events, self.render_width, self.render_height)
        else:
            obs = events
        return current_intensity, ref_values, last_event_timestamp, obs



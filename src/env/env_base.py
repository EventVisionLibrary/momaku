# Copyright 2018 Event Vision Library.

import numba
import numpy as np

COLLISION_THRESHOLD = 1.0

class EnvBase():
    def __init__(self, dt, render_width, render_height):
        self.dt = dt
        self.render_width = render_width
        self.render_height = render_height
        self.reset()

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='human'):
        # TODO: enable rendering for debug
        raise NotImplementedError


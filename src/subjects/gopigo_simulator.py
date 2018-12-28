# Copyright 2018 Event Vision Library.

import numpy as np
from subjects import SubjectBase

class Gopigo(SubjectBase):
    """
    Simple Simulator for GoPiGo
    """
    def __init__(self, mass=0.6):
        self.action_list = ['go', 'back', 'stop']
        super(Gopigo, self).__init__(mass)

    def go(self, dt, v=np.array([1., 0., 0.])):
        new_position = self.position + dt * self.velocity
        self.update_dynamics(v, new_position, v)

    def go(self, dt, v=np.array([-1., 0., 0.])):
        new_position = self.position + dt * self.velocity
        self.update_dynamics(v, new_position, v)

    def right(self):
        raise NotImplementedError

    def left(self):
        raise NotImplementedError

    def stop(self):
        self.update_dynamics(self.direction, self.position, np.array([0.0, 0.0, 0.0]))

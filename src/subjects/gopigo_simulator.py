# Copyright 2018 Event Vision Library.

import numpy as np
from subjects import SubjectBase

class Gopigo(SubjectBase):
    """
    Simple Simulator for GoPiGo
    """
    def __init__(self, mass=0.6):
        self.action_list = ['forward', 'backward', 'stop']
        super(Gopigo, self).__init__(mass)

    def forward(self, dt):
        self.update_dynamics(dt, self.direction, self.velocity * np.array([1, 1, 0]))

    def backward(self, dt):
        print(self.velocity)
        if self.velocity[0] > 0:
            self.update_dynamics(dt, self.direction,
                                 self.velocity * np.array([-1, -1, 0]))
        else:
            self.update_dynamics(dt, self.direction,
                                self.velocity * np.array([1, 1, 0]))

    def right(self):
        raise NotImplementedError

    def left(self):
        raise NotImplementedError

    def stop(self, dt):
        self.update_dynamics(dt, self.direction, np.array([0.0, 0.0, 0.0]))

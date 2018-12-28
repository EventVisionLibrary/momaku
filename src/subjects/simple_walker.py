# Copyright 2018 Event Vision Library.

import numpy as np
from subjects import SubjectBase

class SimpleWalker(SubjectBase):
    """
    Random walker!
    """
    def __init__(self, mass=1.0):
        self.action_list = ['upward', 'downward', 'rotate', 'stop']
        super(SimpleWalker, self).__init__(mass)

    def upward(self, dt, v=np.array([0., 0., 0.1])):
        new_position = self.position + dt * v
        self.update_dynamics(self.direction, new_position, v)
        
    def downward(self, dt, v=np.array([0., 0., -0.1])):
        new_position = self.position + dt * v
        self.update_dynamics(self.direction, new_position, v)

    def rotate(self, dt, angular_v=0.001):
        dtheta = angular_v * dt
        new_x = self.direction[0] * np.cos(dtheta) + self.direction[1] * np.sin(dtheta)
        new_y = -self.direction[0] * np.sin(dtheta) + self.direction[1] * np.cos(dtheta)
        new_direction = np.array([new_x, new_y, self.direction[2]])
        self.update_dynamics(new_direction, self.position, self.velocity)

    def stop(self):
        pass

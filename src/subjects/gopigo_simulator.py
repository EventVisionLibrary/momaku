# Copyright 2018 Event Vision Library.

import numpy as np
from subjects import SubjectBase

class Gopigo(SubjectBase):
    """
    Simple Simulator for GoPiGo
    """
    def __init__(self, mass=0.6):
        self.action_list = ['forward', 'right', 'left']
        super(Gopigo, self).__init__(mass)

    def forward(self, dt):
        self.update_dynamics(dt, self.direction, self.initial_velocity * np.array([1, 1, 0]))

    def backward(self, dt):
        self.update_dynamics(dt, self.direction, self.initial_velocity * np.array([-1, -1, 0]))

    def right(self, dt):
        rot = self.calc_rotation_matrix(theta=-0.05)
        self.update_dynamics(dt, rot.dot(self.direction), rot.dot(self.initial_velocity))

    def left(self, dt):
        rot = self.calc_rotation_matrix(theta=0.05)
        self.update_dynamics(dt, rot.dot(self.direction), rot.dot(self.initial_velocity))

    def stop(self, dt):
        self.update_dynamics(dt, self.direction, np.array([0.0, 0.0, 0.0]))

    def calc_rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1.0]])

# Copyright 2018 Event Vision Library.

import numpy as np
from objects import ObjectBase

class SolidCube(ObjectBase):
    """
    Cube object.
    """
    def __init__(self, size, mass=1.0, color=np.zeros(3)):
        super(SolidCube, self).__init__(mass, color)
        self.size = size

    def initialize_dynamics(self, direction, position=np.zeros(3), velocity=np.zeros(3)):
        self.direction = direction
        self.position = position    # [m]
        self.velocity = velocity    # [m/s]
        self.vertices = self.calcVertices()

    def calc_vertices(self):
        raise NotImplementedError

    def update_dynamics(self, new_direction, new_poisition, new_velocity):
        raise NotImplementedError

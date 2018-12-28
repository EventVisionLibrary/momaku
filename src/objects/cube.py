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
        self.original_vertices = np.array([[-1, -1, 1],
                                           [-1, 1, 1],
                                           [1, 1, 1],
                                           [1, -1, 1],
                                           [-1, -1, -1],
                                           [-1, 1, -1],
                                           [1, 1, -1],
                                           [1, -1, -1]]) * size / 2.0
        self.vertices = self.original_vertices    # eight vertices of the cube

    def initialize_dynamics(self, direction, position=np.zeros(3), velocity=np.zeros(3)):
        self.direction = direction  # theta for (x, y, z)
        self.position = position    # [m]
        self.velocity = velocity    # [m/s]
        self.vertices = self.set_vertices()

    def set_vertices(self):
        rotated_vertices = self.original_vertices.dot(self.get_rotation_matrix(self.direction))
        return self.position + rotated_vertices

    def get_rotation_matrix(self, theta):
        # rotate from x > y > z axis
        rx = np.array([[1, 0, 0],
                       [0, np.cos(theta[0]), np.sin(theta[0])],
                       [0, -np.sin(theta[0]), np.cos(theta[0])]])
        ry = np.array([[np.cos(theta[1]), 0, -np.sin(theta[1])],
                       [0, 1, 0],
                       [np.sin(theta[1]), 0, np.cos(theta[1])]])
        rz = np.array([[np.cos(theta[2]), np.sin(theta[2]), 0],
                       [-np.sin(theta[2]), np.cos(theta[2]), 0],
                       [0, 0, 1]])
        return rz.dot(ry).dot(rx)

    def update_dynamics(self, dt, new_velocity=None, angular_velocity=None):
        self.position = self.position + self.velocity * dt
        if angular_velocity is None:
            self.vertices = self.vertices + self.velocity * dt
        else:
            self.direction += angular_velocity * dt
            self.vertices = self.set_vertices()
        if new_velocity is not None:
            self.velocity = new_velocity

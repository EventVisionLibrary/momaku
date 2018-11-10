from . import ObjectBase
import numpy as np


class SolidCube(ObjectBase):
    """
    Cube object.
    """
    def __init__(self, size, mass=1.0, color=np.zeros(3)):
        super(SolidCube, self).__init__(mass, color)
        self.size = size

    def initializeDynamics(self, direction, position=np.zeros(3), velocity=np.zeros(3)):
        self.direction = direction
        self.vertices = self.calcVertices()
        self.position = position    # [m]
        self.velocity = velocity    # [m/s**2]

    def calcVertices(self):
        raise NotImplementedError

    def updateDynamics(self, new_direction, new_poisition, new_velocity):
        raise NotImplementedError

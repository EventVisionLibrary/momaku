import numpy as np
from objects import ObjectBase


class SolidSphere(ObjectBase):
    """
    Sphere object.
    """
    def __init__(self, radius, mass=1.0, color=np.zeros(3)):
        super(SolidSphere, self).__init__(mass, color)
        self.radius = radius

    def update_dynamics(self, dt, new_velocity=None, angular_velocity=None):
        self.position += self.velocity * dt
        if new_velocity is not None:
            self.velocity = new_velocity
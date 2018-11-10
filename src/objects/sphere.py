from . import ObjectBase
import numpy as np


class SolidSphere(ObjectBase):
    """
    Sphere object.
    """
    def __init__(self, radius, mass=1.0, color=np.zeros(3)):
        super(SolidSphere, self).__init__(mass, color)
        self.radius = radius


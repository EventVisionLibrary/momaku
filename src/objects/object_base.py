# Copyright 2018 Event Vision Library.

import numpy as np

class ObjectBase(object):
    """
    Object Base class.
    This class has:
        mass [kg]           ... scalar
        color (RGB)         ... 3D vector
        position [m]        ... 3D vector
        velocity [m**2]     ... 3D vector
    """
    def __init__(self, mass=1.0, color=np.zeros(3)):
        self.mass = 1.0             # [kg]
        self.color = color          # [r, g, b]
        self.is_visible = True

    def initialize_dynamics(self, position=np.zeros(3), velocity=np.zeros(3), *args, **kwargs):
        self.position = position    # [m]
        self.velocity = velocity    # [m/s]

    def update_dynamics(self, dt, new_velocity=None, *args, **kargs):
        self.position = self.position + self.velocity * dt
        if new_velocity is not None:
            self.velocity = new_velocity

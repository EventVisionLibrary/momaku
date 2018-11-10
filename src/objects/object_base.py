import numpy as np


class ObjectBase():
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

    def initializeDynamics(self, position=np.zeros(3), velocity=np.zeros(3)):
        self.position = position    # [m]
        self.velocity = velocity    # [m/s**2]

    def updateDynamics(self, new_poisition, new_velocity):
        self.updatePosition(new_poisition)
        self.updateVelocity(new_velocity)

    def updatePosition(self, new_poisition):
        self.position = new_poisition

    def updateVelocity(self, new_velocity):
        self.velocity = new_velocity

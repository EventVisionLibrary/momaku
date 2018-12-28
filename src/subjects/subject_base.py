# Copyright 2018 Event Vision Library.

import numpy as np
from objects import ObjectBase

class SubjectBase(ObjectBase):
    """
    SubjectBase class.
    """
    def __init__(self, mass=1.0):
        super(SubjectBase, self).__init__(mass)
        self.check_action_list_defined()

    def check_action_list_defined(self):
        if hasattr(self, 'action_list'):
            pass
        else:
            raise NotImplementedError('Please define action_list for the subject!')

    def initialize_dynamics(self, position=np.zeros(3), direction=np.ones(3), velocity=np.zeros(3)):
        self.position = position    # [x, y, z] [m] at time _t_
        self.direction = direction  # [x, y, z] [m] at time _t_
        self.velocity = velocity    # [x, y, z] [m/s] at time _t_

    def update_dynamics(self, dt, new_direction, new_velocity):
        self.position += dt * self.velocity
        self.direction = new_direction
        self.velocity = new_velocity

    def act(self, action_name, *args):
        """
        action_name ... string.
        args        ... args for the action, if any.
        """
        try:
            getattr(self, action_name)(*args)
        except:
            raise NotImplementedError('The action', action_name, 'is not defined!')

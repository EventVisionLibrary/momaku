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

    def initialize_dynamics(self, direction=np.ones(3), position=np.zeros(3), velocity=np.zeros(3)):
        self.direction = direction  # [x, y, z] [m]
        self.position = position    # [x, y, z] [m]
        self.velocity = velocity    # [x, y, z] [m/s]

    def update_dynamics(self, new_direction, new_position, new_velocity):
        self.direction = new_direction
        self.position = new_position
        self.velocity = new_velocity

    def check_action_list_defined(self):
        if hasattr(self, 'action_list'):
            pass
        else:
            raise NotImplementedError('Please define action_list for the subject!')

    def act(self, action_name, *args):
        """
        action_name ... string.
        args        ... args for the action, if any.
        """
        self.check_action_defined(action_name)
        getattr(self, action_name)(*args)

    def check_action_defined(self, action_name):
        if action_name in self.action_list:
            pass
        else:
            raise NotImplementedError('The action', action_name, 'is not defined!')

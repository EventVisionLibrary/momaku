# Copyright 2018 Event Vision Library.

import numpy as np
from objects import ObjectBase

class Field(ObjectBase):
    """
    Field object.
    """
    def __init__(self, color=np.zeros(3), size=100000):
        super(Field, self).__init__(0.0, color)
        self.vertices = np.zeros((4, 3))
        self.size = size

    def initialize_dynamics(self, position=np.zeros(3), *args, **kargs):
        self.position = position    # [m]
        self.setVertices()

    def set_vertices(self):
        self.vertices[:, :2] = np.array([[self.size, self.size],
                                         [self.size, -self.size], 
                                         [-self.size, -self.size],
                                         [-self.size, self.size]])
        self.vertices[:, 2] = self.position[2]

    def update_dynamics(self, *args, **kargs):
        pass


class TextureField(Field):
    """
    Textured-field object.
    """
    def __init__(self, texture, size=100000):
        super(TextureField, self).__init__(np.array([0., 0., 0.]), size)
        self.texture = self.setTexture()

    def set_texture(self):
        raise NotImplementedError

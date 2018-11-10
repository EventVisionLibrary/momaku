from . import ObjectBase
import numpy as np


class Field(ObjectBase):
    """
    Field object.
    """
    def __init__(self, color=np.zeros(3), size=100000):
        super(Field, self).__init__(0.0, color)
        self.vertices = np.zeros((4, 3))
        self.size = size

    def initializeDynamics(self, position=np.zeros(3), *args, **kargs):
        self.position = position    # [m]
        self.setVertices()

    def setVertices(self):
        self.vertices[:, :2] = np.array([[size, size],[size, -size], 
                                         [-size, -size], [-size, size]])
        self.vertices[:, 2] = self.position[2]

    def updateDynamics(self, *args, **kargs):
        pass


class TextureField(Field):
    """
    Textured-field object.
    """
    def __init__(self, texture, size=100000):
        super(TextureField, self).__init__(np.array([0., 0., 0.]), size)
        self.texture = self.setTexture()

    def setTexture(self):
        raise NotImplementedError

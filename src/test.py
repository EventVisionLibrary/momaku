import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

from objects.cube import SolidCube
from objects.sphere import SolidSphere

light_ambient = [0.25, 0.25, 0.25]
light_position = [10, 5, 0, 2]


class Renderer():

    def __init__(self, camera_position=None):
        self.display_width = 900
        self.display_height = 900

        self.camera_position = camera_position

        self.setup_display()
        self.setup_light()

    def __del__(self):
        glfw.destroy_window(self.window)
        glfw.terminate()

    def setup_display(self):
        if not glfw.init():
            return
        # Set window hint NOT visible
        glfw.window_hint(glfw.VISIBLE, False)
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(self.display_width, self.display_width, "hidden window", None, None)
        if not self.window:
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(self.window)

        gluPerspective(90, (self.display_width / self.display_height), 0.01, 12)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST) # 隠面消去を有効に
        glDepthFunc(GL_LEQUAL)

    def setup_light(self):
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)  # 環境光
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)  # 光源の位置
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)

    def __draw_sphere(self, sphere):
        raise NotImplementedError

    def __draw_cube(self, cube):
        vertices = cube.vertices
        # 1
        glVertex3f(*vertices[0])
        glVertex3f(*vertices[1])
        glVertex3f(*vertices[2])
        glVertex3f(*vertices[3])
        # 2
        glVertex3f(*vertices[4])
        glVertex3f(*vertices[5])
        glVertex3f(*vertices[6])
        glVertex3f(*vertices[7])
        # 3
        glVertex3f(*vertices[0])
        glVertex3f(*vertices[1])
        glVertex3f(*vertices[5])
        glVertex3f(*vertices[4])
        # 4
        glVertex3f(*vertices[2])
        glVertex3f(*vertices[3])
        glVertex3f(*vertices[7])
        glVertex3f(*vertices[6])
        # 5
        glVertex3f(*vertices[1])
        glVertex3f(*vertices[2])
        glVertex3f(*vertices[6])
        glVertex3f(*vertices[5])
        # 6
        glVertex3f(*vertices[0])
        glVertex3f(*vertices[3])
        glVertex3f(*vertices[7])
        glVertex3f(*vertices[4])

    def render_objects(self, objects):
        glRotatef(-90, 1, 0, 0)  # Straight rotation
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glRotatef(285, 0, 0, 1)  # Rotate yaw
        glTranslatef(-15, -5, -5)  # Move to position

        glBegin(GL_QUADS)
        for obj in objects:
            glColor3f(*obj.color)
            if isinstance(obj, SolidCube):
                self.__draw_cube(obj)
            elif isinstance(obj, SolidSphere):
                self.__draw_sphere(obj)
        glEnd()
        image_buffer = glReadPixels(0, 0, self.display_width, self.display_height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(self.display_width, self.display_height, 3)
        return image

if __name__ == "__main__":

    cube = SolidCube(size=1.0)
    cube.vertices = [[2, 2, 0], [2, 2, 2], [2, 6, 2], [2, 6, 0],
                     [4, 2, 0], [4, 2, 2], [4, 6, 2], [4, 6, 0]
                     ]
    renderer = Renderer()
    image = renderer.render_objects([cube])
    cv2.imwrite("image.png", image)

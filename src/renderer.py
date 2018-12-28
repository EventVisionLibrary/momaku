import cv2
from math import pi, cos, sin
import numpy as np
import time

from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

from objects.cube import SolidCube
from objects.sphere import SolidSphere

# color of materials
ambient = [0.25, 0.25, 0.25]
diffuse = [1.0, 0.0, 0.0]
specular = [1.0, 1.0, 1.0]
shininess = 32.0

# color (RGBA)
red = [1.0, 0.0, 0.0, 1.0]
green = [0.0, 1.0, 0.0, 1.0]
blue = [0.0, 0.0, 1.0, 1.0]

class Renderer():
    def __init__(self, camera_position, target_position, width, height):
        self.display_width = width
        self.display_height = height

        self.light_ambient = [0.25, 0.25, 0.25]
        self.light_position = [10, 5, 0, 2]
        self.light_specular = [1.0, 1.0, 1.0]

        self.camera_position = camera_position
        self.target_position = target_position

        self.setup_display()
        self.setup_light()
        self.setup_perspective()

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
        glLightfv(GL_LIGHT0, GL_AMBIENT, self.light_ambient)  # 環境光
        glLightfv(GL_LIGHT0, GL_POSITION, self.light_position)  # 光源の位置
        glLightfv(GL_LIGHT0, GL_SPECULAR, self.light_specular)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)

    def setup_perspective(self):
        glLoadIdentity() # make gluLookAt non-cumulative
        gluPerspective(90, (self.display_width / self.display_height), 0.01, 12)
        gluLookAt(*self.camera_position, *self.target_position, 0, 0, 1) # the last 3 is fixed

    def update_perspective(self, new_camera_position, new_target_position):
        self.camera_position = new_camera_position
        self.target_position = new_target_position
        self.setup_perspective()

    def __draw_axis(self):
        glLineWidth(3.0)
        glBegin(GL_LINES)
        length = 20.0
        # x
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [1.0, 0.0, 0.0, 1.0])
        glVertex3f(length, 0.0, 0.0)
        glVertex3f(-length, 0.0, 0.0)

        # y
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.0, 1.0, 0.0, 1.0])
        glVertex3f(0.0, length, 0.0)
        glVertex3f(0.0, -length, 0.0)

        # z
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.0, 0.0, 1.0, 1.0])
        glVertex3f(0.0, 0.0, length)
        glVertex3f(0.0, 0.0, -length)

        glEnd()

    def __draw_plane(self):
        x_size = 10.0
        y_size = 10.0
        x_num = 60
        y_num = 60
        glLineWidth(1.0)
        glBegin(GL_LINES)
        # y direction
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.0, 1.0, 0.0, 0.5])
        for x in np.linspace(-x_size, x_size, x_num):
            glVertex3f(x, -y_size, 0.0)
            glVertex3f(x, +y_size, 0.0)
        # x direction
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [1.0, 0.0, 0.0, 0.5])
        for y in np.linspace(-y_size, y_size, y_num):
            glVertex3f(-x_size, y, 0.0)
            glVertex3f(+x_size, y, 0.0)
        glEnd()

    def __draw_sphere(self, sphere):
        # divide into triangles
        def f(u, v):
            r = sphere.radius
            x, y, z = sphere.position
            return np.array([x+r*sin(v)*cos(u), y+r*cos(v), z+r*sin(v)*sin(u)]) # quadratic

        start_u = 0
        start_v = 0
        end_u = 2*pi
        end_v = pi
        resolution_u = 20
        resolution_v = 20
        for u in np.linspace(start_u, end_u, resolution_u):
            for v in np.linspace(start_v, end_v, resolution_v):
                u_next = min(end_u, u+(end_u-start_u)/resolution_u)
                v_next = min(end_v, v+(end_v-start_v)/resolution_v)
                p0 = f(u, v)
                p1 = f(u, v_next)
                p2 = f(u_next, v)
                p3 = f(u_next, v_next)
                # 1
                glBegin(GL_TRIANGLES)
                glMaterialfv(GL_FRONT, GL_AMBIENT, [*sphere.color, 1.0])
                glVertex3f(*p0)
                glVertex3f(*p2)
                glVertex3f(*p1)
                glEnd()
                # 2
                glBegin(GL_TRIANGLES)
                glMaterialfv(GL_FRONT, GL_AMBIENT, [*sphere.color, 1.0])
                glVertex3f(*p3)
                glVertex3f(*p1)
                glVertex3f(*p2)
                glEnd()

    def __draw_cube(self, cube):
        glBegin(GL_QUADS)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [*cube.color, 1.0])
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
        glEnd()

    def render_objects(self, objects, show_axis=False):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # axis and plane
        if show_axis:
            self.__draw_axis()
            self.__draw_plane()

        # objects
        # glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
        # glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
        # glMaterialfv(GL_FRONT, GL_SHININESS, shininess)
        for obj in objects:
            if isinstance(obj, SolidCube):
                self.__draw_cube(obj)
            elif isinstance(obj, SolidSphere):
                self.__draw_sphere(obj)

        image_buffer = glReadPixels(0, 0, self.display_width, self.display_height,
                                    OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(self.display_width, self.display_height, 3)
        return image

if __name__ == "__main__":
    camera_position = [-1, -3, -1]
    target_position = [3, 3, 3]
    renderer = Renderer(camera_position=camera_position, target_position=target_position)

    # objects
    cube = SolidCube(size=1.0, color=np.array([0.5, 0.5, 0.0]))
    cube.vertices = np.array([[2, 2, 0], [2, 2, 2], [2, 6, 2], [2, 6, 0],
                              [4, 2, 0], [4, 2, 2], [4, 6, 2], [4, 6, 0]], dtype=np.float64)
    sphere = SolidSphere(radius=0.5, color=np.array([0.5, 0.0, 0.5]))
    sphere.position = np.array([0.0, 1.0, 0.5])

    start = time.time()
    N = 20
    for i in range(0, N):
        cube.vertices[:, 1] += 0.2
        sphere.position += np.array([0.0, 0.1, 0.0])
        camera_position[0] -= 0.05
        camera_position[1] -= 0.05
        renderer.update_perspective(camera_position, target_position)
        image = renderer.render_objects([cube, sphere], show_axis=True)
        cv2.imwrite("../fig/image" + str(i) + ".png", image)
    print("Average Elapsed Time: {} s".format((time.time()-start)/N))

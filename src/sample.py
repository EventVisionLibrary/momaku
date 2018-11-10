#!/usr/bin/env python
#coding:utf-8
import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

window = None
xrot = yrot = zrot = 0.0
xspeed = yspeed = 0.0
z = -5.0

# 光源の色
light_ambient = [0.25, 0.25, 0.25]
light_diffuse = [1.0, 1.0, 1.0]
light_specular = [1.0, 1.0, 1.0]
# 光源の位置
light_position = [2, 5, 2, 1]

# マテリアルの色
ambient = [0.25, 0.25, 0.25]
diffuse = [1.0, 0.0, 0.0]
specular = [1.0, 1.0, 1.0]
shininess = 32.0


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(100, 100)
    glutCreateWindow("ライティング")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    init(300, 300)
    glutMainLoop()


def init(width, height):
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)

    # ライティングの設定
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glEnable(GL_LIGHTING)  # ライティングを有効にする
    glEnable(GL_LIGHT0)  # 0番目の照明を有効にする

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0.0, 0.0, 1.5, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0)

    # マテリアルの設定
    glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
    glMaterialfv(GL_FRONT, GL_SHININESS, shininess)

    # Teapotの描画
    glutSolidTeapot(1.0)

    # 平面
    glBegin(GL_POLYGON)
    glVertex3f(-10, 10, 0)
    glVertex3f(10, 10, 0)
    glVertex3f(10, -10, 0)
    glVertex3f(-10, -10, 0)
    glEnd()

    glTranslatef(2, 2, 1)
    glutSolidSphere(0.1, 20, 20)

    glutSwapBuffers()


def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width) / float(height), 0.1, 100.0)

if __name__ == '__main__':
    main()

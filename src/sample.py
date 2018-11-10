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
light_position = [0, 5, 0, 1]

# マテリアルの色
ambient = [0.25, 0.25, 0.25]
diffuse = [1.0, 0.0, 0.0]
specular = [1.0, 1.0, 1.0]
shininess = 32.0

def main():
    global window

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutInitWindowPosition(100, 100)
    window = glutCreateWindow("OpenGL")

    glutDisplayFunc(display)
    glutIdleFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutSpecialFunc(keyboard)

    init(640, 480)
    glutFullScreen()
    glutMainLoop()

def init(width, height):
    glClearColor(0.0, 0.0, 1.0, 0.0)

    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)

    # 投影変換
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(1.0, float(width)/float(height), 0.1, 100.0)

    # モデリング変換
    glMatrixMode(GL_MODELVIEW)

    # 光源
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)     # 環境光
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)     # 拡散光
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)   # 鏡面光
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)   # 光源の位置
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)

def display():
    global xrot, yrot, zrot, xspeed, yspeed

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Teapotのマテリアル
    glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
    glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
    glMaterialfv(GL_FRONT, GL_SHININESS, shininess)

    # Teapotの移動・回転
    glLoadIdentity()
    glTranslatef(1, 1, 1)
    glRotatef(xrot, 1, 0, 0)
    glRotatef(yrot, 0, 1, 0)
    glRotatef(zrot, 0, 0, 1)

    # Teapotのレンダリング
    # glutSolidTeapot(1)
    glutSolidSphere(1.0, 20, 20)

    # 平面
    glBegin(GL_POLYGON)
    glVertex3f(-10, 10, 0)
    glVertex3f(10, 10, 0)
    glVertex3f(10, -10, 0)
    glVertex3f(-10, -10, 0)
    glEnd()


    # 回転
    xrot += xspeed
    yrot += yspeed
    zrot += 0.0

    # 視点
    #gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0)

    glutSwapBuffers()

def reshape(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

def keyboard(key, x, y):
    global xspeed, yspeed

    if key == "\033":  # ESC
        glutDestroyWindow(window)
        sys.exit()

    # Teapotの回転速度
    if key == GLUT_KEY_UP:
        xspeed -= 0.1
    elif key == GLUT_KEY_DOWN:
        xspeed += 0.1
    elif key == GLUT_KEY_RIGHT:
        yspeed += 0.1
    elif key == GLUT_KEY_LEFT:
        yspeed -= 0.1

if __name__ == "__main__":
    main()

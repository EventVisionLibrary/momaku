# Copyright 2018 Event Vision Library.

import numpy as np

from env import EnvBase
from env import physics, util

import objects
import subjects
from renderer import Renderer

COLLISION_THRESHOLD = 1.0

class FallingStone(EnvBase):
    def __init__(self, dt=1e-2, render_width=900, render_height=900):
        super(FallingStone, self).__init__(dt, render_width, render_height)

    def reset(self):
        self.done = False
        self.timestamp = 0.0
        self.objects = self.__init_objects()
        self.subject = self.__init_subject()
        self.renderer = self.__init_renderer()
        prev_image = np.zeros([self.renderer.display_height, self.renderer.display_width, 3])
        current_image = self.renderer.render_objects(self.objects)
        self.prev_intensity = util.rgb_to_intensity(prev_image)
        self.current_intensity = util.rgb_to_intensity(current_image)

    def __init_renderer(self):
        renderer = Renderer(camera_position=self.subject.position,
                            target_position=self.subject.direction,
                            width=self.render_width,
                            height=self.render_height)
        return renderer

    def __init_subject(self):
        # subject = subjects.SimpleWalker()
        subject = subjects.Gopigo()
        subject.initialize_dynamics(position=np.array([-2, -6, -1], dtype=np.float32),
                                    direction=np.array([4, 8, 3], dtype=np.float32),
                                    velocity=np.array([4, 8, 0], dtype=np.float32))
        return subject

    def __init_objects(self): 
        objs = []
        cube = objects.SolidCube(size=1.0, color=np.array([0.5, 0.5, 0.0]))
        cube.initialize_dynamics(position=np.array([0, 2, -1]),
                                 direction=np.ones(3),
                                 velocity=np.array([0., 0., 0.]))
        objs.append(cube)
        sphere = objects.SolidSphere(radius=0.5)
        sphere.initialize_dynamics(position=np.array([1, 1, -5]),
                                   velocity=np.array([0, 0, 2]))
        objs.append(sphere)
        return objs

    def step(self, action):
        if self.done:
            raise Exception("The game already finished.")
        self.timestamp += self.dt
        self.__move_objects()
        self.__move_subject(action)
        self.renderer.update_perspective(
            self.subject.position, self.subject.position + self.subject.direction)

        # obs
        current_image = self.renderer.render_objects(self.objects, True)
        self.current_intensity = util.rgb_to_intensity(current_image)
        events = util.calc_events(self.current_intensity, self.prev_intensity, self.timestamp)
        self.prev_intensity = self.current_intensity

        # default reward (relative position to sphere)
        if self.__is_collision():
            r = -10.0
            self.done = True
        else:
            # extract objects.solidsphere
            for obj in self.objects:
                if isinstance(obj, objects.SolidSphere):
                    r = self.__measure_distance(obj)*(-1.0)/10.0
            self.done = False

        # more forward, more reward
        if action == "forward":
            r += 0.5

        # add done judgement from timestamp
        if self.timestamp > 1.0:
            self.done = True

        info = {}
        return events, r, self.done, info

    # basic functions for objects and subjects
    def __move_objects(self):
        for obj in self.objects:
            if self.__check_on_the_surface(obj):
                new_velocity = np.zeros((3))
            else:
                new_velocity = physics.free_fall_velocity(self.dt, obj)
            obj.update_dynamics(dt=self.dt, new_velocity=new_velocity,
                                angular_velocity=np.array([np.pi, 0, 0]))

    def __move_subject(self, action):
        getattr(self.subject, action)(self.dt)

    def __check_on_the_surface(self, obj):
        if obj.position[2] > 0:
            return True
        else:
            return False

    def __is_collision(self):
        for obj in self.objects:
            if isinstance(obj, objects.SolidSphere):
                if self.__check_sphere_collision(obj):
                    return True
            if isinstance(obj, objects.SolidCube):
                if self.__check_cube_collision(obj):
                    return True
        return False

    def __check_sphere_collision(self, sphere):
        d = self.__measure_distance(sphere)
        #print("distance: ", np.sqrt(d))
        if d < (COLLISION_THRESHOLD+sphere.radius)**2:
            return True
        else:
            return False

    def __measure_distance(self, sphere):
        return np.sum((sphere.position - self.subject.position)**2)

    def __check_cube_collision(self, cube):
        pass

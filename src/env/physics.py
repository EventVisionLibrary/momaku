import numpy as np
from numpy import linalg as LA

def free_fall_velocity(dt, obj):
    # TODO: implement air resistance
    return obj.velocity + dt * np.array([0, 0, 9.8])


def measure_distance(pos1, pos2):
    """
    returns euclidean distance of two vectors
    """
    return np.sum((pos1 - pos2) ** 2)

def measure_angle(vec1, vec2):
    """
    returns angle of two vectors
    """
    theta = np.arccos(np.clip(np.dot(vec1, vec2) / (LA.norm(vec1) * LA.norm(vec2)), -1.0, 1.0))
    return np.abs(theta)
def check_sphere_collision(sphere, self_obj, collision_threshold):
    d = measure_distance(sphere.position, self_obj.position)
    if d < (collision_threshold + sphere.radius) ** 2:
        return True
    else:
        return False

def check_cube_collision(cube, self_obj):
    # TODO Implement me
    return False

def check_on_the_surface(obj):
    if obj.position[2] > 0:
        return True
    else:
        return False

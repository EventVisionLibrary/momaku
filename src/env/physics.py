import numpy as np

def free_fall_velocity(dt, obj):
    # TODO: implement air resistance
    return obj.velocity + dt * np.array([0, 0, 9.8])

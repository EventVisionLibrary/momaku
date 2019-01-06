# Copyright 2018 Event Vision Library.

import numpy as np

def add_impulse_noise(intensity, prob):
    """
    Add salt and pepper noise to intensity
    prob: Probability of the noise 
    """ 
    rand_noise = np.random.random(intensity.shape)
    intensity = intensity + (prob > rand_noise).astype(np.int32) - (rand_noise > 1 - prob).astype(np.int32)
    return np.clip(intensity, -1, 1)

def add_gaussian_noise():
    # TODO(shiba) implement gaussian noise around events
    pass

def rgb_to_intensity(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    intensity = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return intensity / 255.0

def events_to_image(events, width, height):
    image = np.zeros([height, width, 3], dtype=np.uint8)
    for (t, y, x, p) in events:
        if p == 1:
            image[y, x, 0] = 255
        else:
            image[y, x, 2] = 255
    return image

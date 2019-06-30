# Copyright 2018 Event Vision Library.

import numba
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

@numba.jit
def calc_events_from_image(current_intensity, prev_intensity, ref_values,
                           timestamp, last_event_timestamp, 
                           config, dynamic_timestamp=True):
    # functions for calculation of events
    if not dynamic_timestamp:
        diff = np.sign(current_intensity - prev_intensity).astype(np.int32)
        diff = add_impulse_noise(diff, prob=1e-3)
        event_index = np.where(np.abs(diff) > 0)
        events = np.array([np.full(len(event_index[0]), timestamp, dtype=np.int32),
                            event_index[0], event_index[1], diff[event_index]]).T
        return events, ref_values, last_event_timestamp

    # compliment interpolation
    events = []
    for y in range(config["render_height"]):
        for x in range(config["render_width"]):
            current = current_intensity[y, x]
            prev = prev_intensity[y, x]
            if current == prev:
                continue    # no event
            prev_cross = ref_values[y, x]

            pol = 1.0 if current > prev else -1.0
            C = config["Cp"] if pol > 0 else config["Cm"]
            sigma_C = config["sigma_Cp"] if pol > 0 else config["sigma_Cm"]
            if sigma_C > 0:
                C += np.random.normal(0, sigma_C)
            current_cross = prev_cross
            all_crossings = False
            while True:
                # Consider every time when intensity changed over threshold C.
                current_cross += pol*C
                #print("pol: {}, current_cross: {}, prev: {}, current: {}".format(pol, current_cross, prev, current))
                if (pol > 0 and current_cross > prev and current_cross <= current) \
                or (pol < 0 and current_cross < prev and current_cross >= current):
                    edt = (current_cross - prev) * config["dt"] / (current - prev)
                    t = timestamp + edt
                    last_t = last_event_timestamp[y, x]
                    dt = t - last_t
                    assert dt > 0
                    if last_t == 0 or dt >= config["refractory_period"]:
                        events.append((t, y, x, pol>0))
                        last_event_timestamp[y, x] = t
                    ref_values[y, x] = current_cross
                else:
                    all_crossings = True
                if all_crossings:
                    break
    return events, ref_values, last_event_timestamp

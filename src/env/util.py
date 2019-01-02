# Copyright 2018 Event Vision Library.

import numpy as np

# functions for calculation of events
def calc_events(current_intensity, prev_intensity, timestamp):
    # TODO: assign time stamp dynamically
    diff = np.sign(current_intensity - prev_intensity).astype(np.int32)
    event_index = np.where(np.abs(diff) > 0)
    events = np.array([np.full(len(event_index[0]), timestamp, dtype=np.int32),
                       event_index[0], event_index[1], diff[event_index]]).T
    return events

def rgb_to_intensity(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    intensity = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return intensity

def events_to_image(events, width, height):
    image = np.zeros([height, width, 3], dtype=np.uint8)
    for (t, y, x, p) in events:
        if p == 1:
            image[y, x, 0] = 255
        else:
            image[y, x, 2] = 255
    return image

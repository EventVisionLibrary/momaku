
import os
import time
import numpy as np
import cv2

from env import FallingStone
from env import events_to_image

if __name__ == '__main__':
    w, h = 800, 800
    env = FallingStone(dt=0.01, render_width=w, render_height=h)
    N = 100
    executed_times = []
    selfdir = os.path.dirname(os.path.abspath(__file__))
    savedir = os.path.join(selfdir, "../fig")
    print("Save to", savedir)
    for i in range(0, N):
        start = time.time()
        action = np.random.choice(env.subject.action_list)
        try:
            events, r, done, info = env.step(action=action)
        except Exception as inst:
            print(inst)
            break
        print(i, action)
        image = events_to_image(events, w, h)
        executed_times.append(time.time() - start)        
        cv2.imwrite(os.path.join(savedir, "image" + str(i) + ".png"), image)
    print("Average Elapsed Time: {} s".format(np.mean(executed_times)))

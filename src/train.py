# Copyright 2018 Event Vision Library.

import time
import cv2

from agents import create_network, Pipeline, select_softmax
from env import FallingStone

def main():
    # display size
    w, h = 80, 80

    # Build network
    network = create_network(action_num=2)

    # Load environment
    environment = FallingStone(render_width=w, render_height=h)
    environment.subject.action_list = ['forward', 'stop'] # experimentally remove 'backward' action

    # Build pipeline from specified components.
    pipeline = Pipeline(network, environment, encoding='bernoulli',
                        action_function=select_softmax, output='Output Layer')

    # Run environment simulation for 100 episodes.
    #pipeline.reset_()
    for i in range(1000):
        # initialize episode reward
        reward = 0
        while True:
            pipeline.step()
            reward += pipeline.reward
            print(reward, pipeline.action_name)
            if pipeline.done:
                pipeline.reset_()
                environment.subject.action_list = ['forward', 'stop']
                break
        print("Episode " + str(i) + " reward:", reward)


if __name__ == "__main__":
    main()

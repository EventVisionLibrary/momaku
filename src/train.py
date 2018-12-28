import time
import cv2

# bindsnet
from bindsnet.encoding import bernoulli

from agents import create_network, Pipeline, select_softmax
from env import FallingStone, events_to_image


"""
Now I'm debugging in following environment...
renderer.display_width = 80
renderer.display_height = 80
"""

def main():
    # Build network
    network = create_network()

    # Load environment
    environment = FallingStone()

    # Build pipeline from specified components.
    pipeline = Pipeline(network, environment, encoding=bernoulli,
                        action_function=select_softmax, output='Output Layer')

    # Run environment simulation for 100 episodes.
    pipeline.reset_()
    for i in range(100):
        # initialize episode reward
        reward = 0
        # pipeline.reset_() # including bug
        pipeline.step()
        #while True:
        #    pipeline.step()
        #    reward += pipeline.reward
        #    if pipeline.done:
        #        break
        print("Episode " + str(i) + " reward:", reward)


if __name__ == "__main__":
    main()
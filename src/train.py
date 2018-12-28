import time
import cv2

# bindsnet
from bindsnet.encoding import bernoulli

from agents import create_network, Pipeline, select_softmax
from env import FallingStone, events_to_image


def main():
    # display size
    w, h = 80, 80

    # Build network
    network = create_network()

    # Load environment
    environment = FallingStone(render_width=w, render_height=h)

    # Build pipeline from specified components.
    pipeline = Pipeline(network, environment, encoding=bernoulli,
                        action_function=select_softmax, output='Output Layer')

    # Run environment simulation for 100 episodes.
    #pipeline.reset_()
    for i in range(10):
        # initialize episode reward
        reward = 0
        # pipeline.reset_() # including bug
        pipeline.step()
        
        # [Future work] after env.done is defined, this part should be implemented!
        #while True:
        #    pipeline.step()
        #    reward += pipeline.reward
        #    if pipeline.done:
        #        break
        print("Episode " + str(i) + " reward:", reward)


if __name__ == "__main__":
    main()
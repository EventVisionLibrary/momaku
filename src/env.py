import gym
from gym import spaces

from renderer import Renderer

class FallingStone(gym.Env):

    def __init__(self):
        self.objects = []
        self.subject = None
        self.renderer = Renderer()

        # action_space
        self.action_space = spaces.Discrete(4)
        # observation_space
        # observation is [(t, x, y, p)]

    def move_objects(self):
        pass

    def move_subject(self):
        pass

    def step(self, action):
        self.move_objects()
        self.move_subject()

        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


if __name__ == '__main__':
    env = FallingStone()
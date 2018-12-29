import cv2
import numpy as np
import pickle


import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

from env import FallingStone

class QFunction(chainer.Chain):

    def __init__(self, n_actions, hidden_dim=100):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=3, out_channels=2, ksize=3)
            self.conv2 = L.Convolution2D(in_channels=2, out_channels=4, ksize=3)
            self.l1 = L.Linear(4*76*76, hidden_dim)
            self.l2 = L.Linear(hidden_dim, hidden_dim)
            self.l3 = L.Linear(hidden_dim, n_actions)
            self.bn1 = L.BatchNormalization(size=2)
            self.bn2 = L.BatchNormalization(size=4)
            self.bn3 = L.BatchNormalization(size=hidden_dim)
            self.bn4 = L.BatchNormalization(size=hidden_dim)

    def __call__(self, x):
        x = F.transpose(x, axes=(0, 3, 1, 2))
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.l1(h)))
        h = F.relu(self.bn4(self.l2(h)))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))


def sampling_func():
    actions = [0, 1]
    return np.random.choice(actions)

def main():
    w = 80
    h = 80
    env = FallingStone(render_width=w, render_height=h, obs_as_img=True)
    env.subject.action_list = ['forward', 'stop']

    q_func = QFunction(n_actions=2)
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    gamma = 0.95
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.1, random_action_func=sampling_func)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)

    # Now create an agent that will interact with the environment.
    agent = chainerrl.agents.DoubleDQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=100, update_interval=1,
        target_update_interval=50, phi=phi)

    n_episodes = 30
    max_episode_len = 34 # 30 fps, 1 s
    R_list = []
    for i in range(1, n_episodes + 1):
        obs, _, _, _ = env.reset()
        reward = 0
        done = False
        R = 0
        t = 0
        while not done and t < max_episode_len:
            action_index = agent.act_and_train(obs, reward)
            action = env.subject.action_list[action_index]
            print("time: {}, action: {}".format(t, action))
            cv2.imshow('image', obs)
            cv2.waitKey(10)
            obs, reward, done, _ = env.step(action)
            R += reward
            t += 1
        if i % 1 == 0:
            print('episode:', i, 'R:', R, 'statistics:', agent.get_statistics())
        agent.stop_episode_and_train(obs, reward, done)
        R_list.append(R)

    with open("../result/dqn/model.pickle", "wb") as f:
        pickle.dump(agent, f)
    with open("../result/dqn/reward.pickel", "wb") as f:
        pickle.dump(R_list, f)
    print('Finished.')

if __name__ == '__main__':
    main()

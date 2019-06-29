import cv2
import numpy as np
import pickle
import signal
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

from env import BouncingBall

class QFunction(chainer.Chain):
    def __init__(self, input_w, input_h, n_actions, hidden_dim=100):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=3, out_channels=2, ksize=3)
            self.conv2 = L.Convolution2D(in_channels=2, out_channels=4, ksize=3)
            self.l1 = L.Linear(4 * (input_w - 4) * (input_h - 4), hidden_dim)
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
    actions = [0, 1, 2]
    return np.random.choice(actions)

def get_save_name(dir):
    i = 0
    agent_dir_name = os.path.join(dir, "model" + str(i) + ".pickle")
    while os.path.exists(agent_dir_name):
        agent_dir_name = os.path.join(dir, "model" + str(i) + ".pickle")
        i += 1
    reward_file_name = os.path.join(dir, "reward" + str(i - 1) + ".pickle")
    return (agent_dir_name, reward_file_name)

def save_all(dir, agent, R_list):
    agent_dir_name, reward_file_name = get_save_name(dir)
    print("Saving models to ", agent_dir_name)
    agent.save(agent_dir_name)
    with open(reward_file_name, "wb") as f:
        pickle.dump(R_list, f)

def draw_reward_fig(dir, R_list):
    plt.plot(np.array(R_list))
    plt.savefig(os.path.join(dir, "reward.png"))
    plt.close()

def main():
    w = 240
    h = 180
    env = BouncingBall(dt=0.03, render_width=w, render_height=h, obs_as_img=True)
    # env.subject.action_list = ['forward', 'stop']
    savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../result/dqn")

    q_func = QFunction(input_w=w, input_h=h, n_actions=len(env.subject.action_list))
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

    n_episodes = 100
    max_episode_len = 30        # 30 fps, 1 s
    R_list = []

    def handler(signal, frame):
        print("Shutting down...")
        save_all(savedir, agent, R_list)
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

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
        draw_reward_fig(savedir, R_list)
        
    save_all(savedir, agent, R_list)
    print('Finished.')

if __name__ == '__main__':
    main()

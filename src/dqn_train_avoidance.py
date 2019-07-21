import cv2
import numpy as np
import signal
import os
import sys

import chainer
import chainerrl

from env import FallingStone
from dqn_agent import *

def sampling_func():
    actions = [0, 1]
    return np.random.choice(actions)

def main():
    w = 240
    h = 180
    env = FallingStone(dt=0.03, render_width=w, render_height=h,
                       obs_as_img=True, subject_action_list=['forward', 'stop'])
    env.subject.set_action_list(['forward', 'stop'])
    savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../result/dqn/avoidance")

    q_func = QFunction(input_w=w, input_h=h, n_actions=2)
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

    n_episodes = 50
    max_episode_len = 34        # 30 fps, 1 s
    R_list_train = []
    R_list_eval = []

    def handler(signal, frame):
        print("Shutting down...")
        save_all(savedir, agent, R_list_eval)
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

    for i in range(1, n_episodes + 1):
        # training session
        obs, reward, done, _ = env.reset()
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
            print('Training episode:', i, 'R:', R, 'statistics:', agent.get_statistics())
        agent.stop_episode_and_train(obs, reward, done)
        R_list_train.append(R)
        draw_reward_fig(savedir, 'reward_train.png' ,R_list_train)

        # evaluation session
        obs, reward, done, _ = env.reset()
        R = 0
        t = 0
        while not done and t < max_episode_len:
            action_index = agent.act(obs)
            action = env.subject.action_list[action_index]
            print("time: {}, action: {}".format(t, action))
            cv2.imshow('image', obs)
            cv2.waitKey(10)
            obs, reward, done, _ = env.step(action)
            R += reward
            t += 1
        if i % 1 == 0:
            print('Evaluation episode:', i, 'R:', R, 'statistics:', agent.get_statistics())
        agent.stop_episode()
        R_list_eval.append(R)
        draw_reward_fig(savedir, 'reward_eval.png' ,R_list_eval)

    save_all(savedir, agent, R_list_eval)
    print('Finished.')

if __name__ == '__main__':
    main()

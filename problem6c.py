#!/usr/bin/python

import numpy as np


def load_episode_list(path):
    return np.load(path)


def offPolicyMonteCarlo(path, b, pi, gamma = 0.99):
    episode_list = load_episode_list(path)

    q = np.zeros((11, 11, 4), dtype=np.float32)
    c = np.zeros((11, 11, 4), dtype=np.float32)
    # loop for each episode
    for i in range(len(episode_list)/3):
        state_list = episode_list[i * 3]
        action_list = episode_list[i * 3 + 1]
        reward_list = episode_list[i * 3 + 2]

        G = 0
        W = 1
        for t in range(len(state_list) -1, -1, -1):
            while W:
                G = gamma * G + reward_list[t]
                c[state_list[t][0], state_list[t][1], action_list[t]] = \
                        c[state_list[t][0], state_list[t][1], action_list[t]] + W
                q[state_list[t][0], state_list[t][1], action_list[t]] = \
                    q[state_list[t][0], state_list[t][1], action_list[t]] + \
                    (W / c[state_list[t][0], state_list[t][1], action_list[t]]) * \
                    (G - q[state_list[t][0], state_list[t][1], action_list[t]])
                W = W * (pi[state_list[t][0], state_list[t][1], action_list[t]] / b[state_list[t][0], state_list[t][1], action_list[t]])

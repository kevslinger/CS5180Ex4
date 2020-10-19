#!/usr/bin/python

import FourRoomsEnv
import problem4a
import numpy as np



if __name__ == '__main__':
    num_trials = 10
    num_episodes = 10000
    average_return_list = np.zeros(num_episodes)
    env = FourRoomsEnv.FourRooms()
    pi = np.zeros(shape=(11, 11, 4), dtype=np.float32)
    # Begin with an equiprobable random policy
    pi.fill(0.25)
    q = np.zeros((11, 11, 4), dtype=np.float32)
    for wall_state in env.WALL_STATES:
        q[wall_state[0], wall_state[1], :] = -50
    returns = problem4a.create_returns_list(11, 11, 4)
    env = FourRoomsEnv.FourRooms()
    return_list, q, episode_list, pi = problem4a.onPolicyFVMC(pi, q=q, epsilon=0.1)
    print(return_list[-50:])
    #problem4a.print_greedy_policy()

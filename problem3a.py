#!/usr/bin/python

import gym
import numpy as np
import matplotlib.pyplot as plt

HIT = 1
STICK = 0


def firstVisitMonteCarlo():
    for i in range(500000):
        # State is a 3-tuple: sum of player hand, dealer showing card, usable ace or no)
        state = env.reset()
        # If our sum is less than 12, we will always hit. This particular case is not interesting, and we will not
        # consider it for our returns or graph
        while state[0] < 12:
            state, reward, done, _ = env.step(HIT)
        # Convert the boolean value of usable ace to binary
        if state[2] == True:
            state = (state[0] - 12, state[1] - 1, 1)
        else:
            state = (state[0] - 12, state[1] - 1, 0)
        action = pi[state]
        s_prime, reward, done, _ = env.step(action)
        if s_prime[2] == True:
            s_prime = (s_prime[0] - 12, s_prime[1] - 1, 1)
        else:
            s_prime = (s_prime[0] - 12, s_prime[1] - 1, 0)
        state_list = [state]
        action_list = [action]
        reward_list = [reward]
        while not done:
            action = pi[s_prime[0], s_prime[1], s_prime[2]]
            state = s_prime
            s_prime, reward, done, _ = env.step(action)
            if s_prime[2] == True:
                s_prime = (s_prime[0] - 12, s_prime[1] - 1, 1)
            else:
                s_prime = (s_prime[0] - 12, s_prime[1] - 1, 0)
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
        G = 0
        # Start looping from the back of our state list.
        for t in range(len(state_list)-1, -1, -1):
            G = gamma * G + reward_list[t]
            #print(state_list[t])
            if state_list[t][0] > 9:
                continue
            if state_list[t] not in state_list[:t]:
                # print(state_list[t])
                # print(returns)
                # print(len(returns[0]))
                # print(returns[state_list[t][0]])
                # print(returns[state_list[t][1]])
                # print(returns[state_list[t][2]])
                returns[state_list[t][0]][state_list[t][1]][state_list[t][2]].append(G)
                V[state_list[t]] = sum(returns[state_list[t][0]][state_list[t][1]][state_list[t][2]]) / \
                    len(returns[state_list[t][0]][state_list[t][1]][state_list[t][2]])
        if i == 9999:
            fig = plt.figure(1)
            ax = plt.axes(projection='3d')
            ax.contour3D(x, y, V[:, :, 0], 500, cmap='autumn')
            ax.set_ylabel('player sum')
            ax.set_xlabel('dealer showing')
            ax.set_zlabel('Value')
            plt.title('No Usable Ace, after 10,000 episodes')
            fig = plt.figure(2)
            ax1 = plt.axes(projection='3d')
            ax1.contour3D(x, y, V[:, :, 1], 500, cmap='autumn')
            ax1.set_ylabel('player sum')
            ax1.set_xlabel('dealer showing')
            ax1.set_zlabel('Value')
            plt.title('Usable Ace, after 10,000 episodes')


if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    # Initialize the Value function and pi.
    V = np.zeros((10, 10, 2), dtype=np.float32)
    pi = np.ones((10, 10, 2), dtype=np.int)
    # Make a list of returns for each state
    # Our returns tensor has dimensions (10, 10, 2)
    returns = [[[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]],
               [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]],
               [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]],
               [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]],
               [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]],
               [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]],
               [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]],
               [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]],
               [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]],
               [[[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []], [[], []]]
               ]
    # in this problem, we only stick on 20 or 21.
    pi[9, :, :] = STICK
    pi[8, :, :] = STICK
    gamma = 1
    x, y = np.meshgrid(np.arange(10), np.arange(10))
    x = x + 1
    y = y + 12
    firstVisitMonteCarlo()

    fig = plt.figure(3)
    ax2 = plt.axes(projection='3d')
    ax2.contour3D(x, y, V[:, :, 0], 500, cmap='autumn')
    ax2.set_ylabel('player sum')
    ax2.set_xlabel('dealer showing')
    ax2.set_zlabel('Value')
    plt.title('No Usable Ace, after 500,000 episodes')
    fig = plt.figure(4)
    ax3 = plt.axes(projection='3d')
    ax3.contour3D(x, y, V[:, :, 1], 500, cmap='autumn')
    ax3.set_ylabel('player sum')
    ax3.set_xlabel('dealer showing')
    ax3.set_zlabel('Value')
    plt.title('Usable Ace, after 500,000 episodes')
    print(V[:, :, 1])
    plt.show()

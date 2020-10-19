#!/usr/bin/python

import gym
import numpy as np
import matplotlib.pyplot as plt

HIT = 1
STICK = 0


def everyVisitMonteCarlo():
    # By calling env.reset(), and then going up to a sum of at least 12, we know that we
    # have a probability of getting to each starting state with probability > 0 (by the way the
    # environment works). Now, we need to make sure that each action has probability > 0.
    # To do this, let's make it so that A(S_0) chooses to stick with probability (sum of player hand) / 22,
    # and give it prob to hit of (22 - sum of player hand) / 22.
    # this means that even if we have a natural blackjack value of 21 at the start, we still hit with
    # probability 1/22. Keep in mind that we only consider states from 12-21, since any value below 12
    # is not interesting (always hit!)
    for i in range(500000):
        # State is a 3-tuple: sum of player hand, dealer showing card, usable ace or no)
        state = env.reset()
        # If our sum is less than 12, we will always hit. This particular case is not interesting, and we will not
        # consider it for our returns or graph
        while state[0] < 12:
            state, reward, done, _ = env.step(HIT)
        # Choose first action with probability as discussed above.
        rand = np.random.random()
        if rand < state[0] / 22:
            action = 1
        else:
            action = 0
        # convert the boolean value of usable ace to binary.
        # also adjust state space to be between 12 and 21.
        if state[2] == True:
            state = (state[0] - 12, state[1] - 1, 1)
        else:
            state = (state[0] - 12, state[1] - 1, 0)
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
        for t in range(len(state_list) - 1, -1, -1):
            G = gamma * G + reward_list[t]
            # print(state_list[t])
            # This means we busted.
            if state_list[t][0] > 9:
                continue
            if (state_list[t], action_list[t]) not in zip(state_list[:t], action_list[:t]):
                # print(state_list[t])
                # print(returns)
                # print(len(returns[0]))
                # print(returns[state_list[t][0]])
                # print(returns[state_list[t][1]])
                # print(returns[state_list[t][2]])
                returns[state_list[t][0]][state_list[t][1]][state_list[t][2]][action_list[t]].append(G)
                Q[state_list[t][0], state_list[t][1], state_list[t][2], action_list[t]] = sum(returns[state_list[t][0]]
                                                       [state_list[t][1]]
                                                       [state_list[t][2]]
                                                       [action_list[t]]) / \
                                            len(returns[state_list[t][0]]
                                                        [state_list[t][1]]
                                                        [state_list[t][2]]
                                                        [action_list[t]])
                # Change policy to be best action in the state
                pi[state_list[t][0], state_list[t][1], state_list[t][2]] = np.argmax(Q[state_list[t]])
        if i == 9999 or i == 499999:
        # x, y = np.meshgrid(np.arange(10), np.arange(10))
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.contour3D(x + 12, y + 1, V[:, :, 0], cmap='binary')
        # ax.set_xlabel('Sum of player hand')
        # ax.set_ylabel('Dealer Showing')
        # ax.set_title('V with no usable ace')
            #plt.imshow(Q[:, :, 0])
            print(Q[:, :, 1])
    #plt.show()


if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    # Initialize the Q function and pi.
    Q = np.zeros((10, 10, 2, 2), dtype=np.float32)
    # Initialize policy randomly (only options are 1 for hit and 0 for stick)
    pi = np.random.randint(2, size=(10, 10, 2))
    # Make a list of returns for each state
    # Our returns tensor has dimensions (10, 10, 2, 2)
    returns = [[[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]]],
               [[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]]],
               [[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]]],
               [[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]]],
               [[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]]],
               [[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]]],
               [[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]]],
               [[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]]],
               [[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]]],
               [[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]],
                [[[], []], [[], []]], [[[], []], [[], []]]]
               ]
    gamma = 1
    everyVisitMonteCarlo()
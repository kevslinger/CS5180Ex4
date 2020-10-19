#!/usr/bin/python

import FourRoomsEnv
import numpy as np
import matplotlib.pyplot as plt


# Function to print out the policy.
def print_greedy_policy(env, pi):
    # store the policy actions in the array
    printout = np.chararray(shape=(11, 11), itemsize=6)
    # Iterate over each state
    for x in range(11):
        for y in range(11):
            # We only care about the states that aren't walls.
            if (x, y) not in env.WALL_STATES:
                # Handle goal state specially
                if (x, y) == env.goal_state:
                    print('GOAL' + ' ' * 2, end='')
                    continue
                # Break ties randomly (only needed for the initial exploring stage)
                a = np.random.choice(np.flatnonzero(pi[x, y] == pi[x, y].max()))
                # Store action in printout at state.
                if a == 0:
                    printout[x, y] = 'up'
                elif a == 1:
                    printout[x, y] = 'down'
                elif a == 2:
                    printout[x, y] = 'left'
                else:
                    printout[x, y] = 'right'
            # Handle wall states specially
            else:
                printout[x, y] = 'WALL' + ' ' * 2
            # Print out the chosen action, with some formatting to make it look good.
            print(str(printout[x, y].decode('utf-8')) + ' ' * (6 - len(printout[x, y])), end='')
        print()


# Function to print out the Q function
def print_q(env, q):
    # Iterate over all states
    for i in range(len(q)):
        for j in range(len(q[i])):
            # Print out wall
            if (i, j) in env.WALL_STATES:
                print('WALL ', end='')
            # Print out goal
            elif (i, j) == env.goal_state:
                print('GOAL' + ' ', end='')
            else:
                # Print out max q value for each non-goal state.
                print(str(round(q[i, j, np.argmax(q[i, j])], 2)) + ' ', end='')
        print()


# Generic function to plot a curve with confidence band
def plot(x, y, label):
    plt.plot(x, y, label=label)
    y_err = 1.96 * (y.std() / np.sqrt(num_trials))
    ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
    #plt.legend(label)


# Function to create a list of lists to store the returns for
# each state-action pair
def create_returns_list(i, j, k):
    returns = np.empty((i, j, k), dtype=object)
    for i in range(len(returns)):
        for j in range(len(returns[i])):
            for k in range(len(returns[i, j])):
                returns[i, j, k] = list()
    return returns


# function to create the Q values. Initialize wall states
# with very low value.
def create_q_function(env):
    q = np.zeros((11, 11, 4), dtype=np.float32)
    for wall_state in env.WALL_STATES:
        q[wall_state[0], wall_state[1], :] = -50
    return q

# Function to select an action from pi
# Based on the e-soft greedy
def select_action_from_pi(pi, state, env):
    prob = np.random.random()
    sum_prob = 0
    for a in env.ACTION_LIST:
        sum_prob += pi[state[0], state[1], a]
        if prob <= sum_prob:
            return a
    # if, somehow nothing is selected, return the last action
    return a


# Function for on policy First-Visit Monte-Carlo with e-soft policy
def onPolicyFVMC(pi, q=create_q_function(FourRoomsEnv.FourRooms()), returns=create_returns_list(11, 11, 4), epsilon=0.01, gamma=0.99, num_episodes=10000, env=FourRoomsEnv.FourRooms()):
    discounted_return_list = np.array([])
    episode_list = []
    # loop for lots of episodes
    for i in range(num_episodes):
        # Reset the environment at the beginning of the episode.
        state = env.reset()
        # Pick an action and then run that step in the environment
        action = select_action_from_pi(pi, state, env)
        s_prime, reward, done = env.step(action)

        # Create lists for the states/actions/rewards to keep in our episode
        # so we can backtrack
        state_list = [state]
        action_list = [action]
        reward_list = [reward]

        # This loop simulates the episode.
        while not done:
            # simulate the environment by selecting actions, observing rewards and next states
            action = select_action_from_pi(pi, state, env)
            state = s_prime
            s_prime, reward, done = env.step(action)

            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward)
        # Calculate return for each state, backtracking.
        G = 0
        for t in range(len(state_list)-1, -1, -1):
            G = gamma * G + reward_list[t]
            # first-visit
            if (state_list[t], action_list[t]) not in zip(state_list[:t], action_list[:t]):
                # append the return to the list
                returns[state_list[t][0], state_list[t][1], action_list[t]].append(G)
                q[state_list[t][0], state_list[t][1], action_list[t]] = sum(returns[state_list[t][0],
                                                                                    state_list[t][1],
                                                                                    action_list[t]]) / \
                                                                        len(returns[state_list[t][0],
                                                                                    state_list[t][1],
                                                                                    action_list[t]])
                # Calculate a*, breaking ties arbitrarily.
                a_star = np.random.choice(np.flatnonzero(q[state_list[t][0], state_list[t][1]] == q[state_list[t][0], state_list[t][1]].max()))
                # Update Pi
                for a in env.ACTION_LIST:
                    if a == a_star:
                        pi[state_list[t][0], state_list[t][1], a] = 1 - epsilon + (epsilon / len(env.ACTION_LIST))
                    else:
                        pi[state_list[t][0], state_list[t][1], a] = epsilon / len(env.ACTION_LIST)
        discounted_return_list = np.append(discounted_return_list, G)
        episode_list.append(state_list)
        episode_list.append(action_list)
        episode_list.append(reward_list)
    print_greedy_policy(env, pi)
    print()
    print_q(env, q)
    print()
    return discounted_return_list, q, episode_list, pi


if __name__ == '__main__':
    num_trials = 10
    num_episodes = 10000
    average_return_dict = dict()
    average_return_list = np.zeros(num_episodes)
    for epsilon in [0, 0.01, 0.1]:
        for f in range(num_trials):
            env = FourRoomsEnv.FourRooms()
            pi = np.zeros(shape=(11, 11, 4), dtype=np.float32)
            # Begin with an equiprobable random policy
            pi.fill(0.25)
            #q = np.zeros((11, 11, 4), dtype=np.float32)
            #for wall_state in env.WALL_STATES:
            #    q[wall_state[0], wall_state[1], :] = -50
            #returns = create_returns_list(11, 11, 4)

            return_list, _, _, _ = onPolicyFVMC(pi, epsilon=epsilon, num_episodes=num_episodes)
            average_return_list = average_return_list + ((return_list - average_return_list) / (f+1))
        average_return_dict[epsilon] = average_return_list

    # Part b
    fig, ax = plt.subplots()
    plt.ylabel('Discounted Return')
    plt.xlabel('Episode')
    plt.title('Discounted Return over Time')
    plot(range(len(average_return_list)), average_return_dict[0], 'e=0')
    plot(range(len(average_return_list)), average_return_dict[0.01], 'e=0.01')
    plot(range(len(average_return_list)), average_return_dict[0.1], 'e=0.1')
    plot(range(len(average_return_list)), np.array([0.99 ** 20] * len(average_return_list)), 'upper bound')

    plt.legend()
    plt.show()


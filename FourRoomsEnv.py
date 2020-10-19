#!/usr/bin/python


import random



class FourRooms():
    def __init__(self, goal_state=(0, 10)):
        self.state = (10, 0)
        self.WALL_STATES = [(5, 0), (5, 2), (5, 3), (5, 4), (5, 5), (0, 5), (1, 5), (3, 5), (4, 5),
                            (6, 5), (7, 5), (8, 5), (10, 5), (6, 6), (6, 7), (6, 9), (6, 10)]
        self.ACTION_LIST = [0, 1, 2, 3]
        self.goal_state = goal_state
        self.n = 10
        self.timeout = 459
        self.steps = 0

    def reset(self):
        self.state = (10, 0)
        self.steps = 0
        return self.state

    def get_state(self):
        return self.state

    # Process an action in the environment
    # x and y represent changes
    # for instance, if we move up, then x would be 1 and y would be 0
    # if we move east, x would be 0 and y would be 1
    def process_action(self, x, y):
        ret_state = (self.state[0] + x, self.state[1] + y)
        if ret_state[0] > self.n or ret_state[0] < 0 or ret_state[1] > self.n or ret_state[1] < 0 or ret_state in self.WALL_STATES:
            return self.state
        # If moving up/down would result in hitting an edge
       # if ret_state[0] > self.n or ret_state[1] < 0:
       #     return self.state
        # If moving left/right would result in hitting an edge
       # elif ret_state[0] > self.n or ret_state[1] < 0:
       #     return self.state
        # If any movement would result in hitting an inner wall
       # elif ret_state in self.WALL_STATES:
       #     return self.state
        # If we don't hit anything, we moved!
        else:
            return ret_state

    # 0 is up
    # 1 is down
    # 2 is left
    # 3 is right
    def step(self, action):
        #if self.state == self.goal_state:
        #    reward = 1
        #    done = True
        #    return self.state, reward, done
        random_slip = random.random()
        slip1 = False
        slip2 = False
        # Determine if we can take the intended action, or if we slip
        if random_slip < 0.1:
            slip1 = True
        elif random_slip < 0.2:
            slip2 = True
        # Handle moving up.
        # slip1 is left
        # slip2 is right
        if action == 0:
            if slip1:
                self.state = self.process_action(0, -1)
            elif slip2:
                self.state = self.process_action(0, 1)
            else:
                self.state = self.process_action(-1, 0)
        # Handle moving down
        # slip1 is right
        # slip2 is left
        elif action == 1:
            if slip1:
                self.state = self.process_action(0, 1)
            elif slip2:
                self.state = self.process_action(0, -1)
            else:
                self.state = self.process_action(1, 0)
        # Handle moving left
        # slip1 is up
        # slip2 is down
        elif action == 2:
            if slip1:
                self.state = self.process_action(-1, 0)
            elif slip2:
                self.state = self.process_action(1, 0)
            else:
                self.state = self.process_action(0, -1)
        # Handle moving right
        # slip1 is down
        # slip2 is up
        elif action == 3:
            if slip1:
                self.state = self.process_action(1, 0)
            elif slip2:
                self.state = self.process_action(-1, 0)
            else:
                self.state = self.process_action(0, 1)

        if self.state == self.goal_state:
            reward = 1
            done = True
        elif self.steps >= self.timeout:
            reward = 0
            done = True
        else:
            reward = 0
            done = False
            self.steps += 1
        return self.state, reward, done


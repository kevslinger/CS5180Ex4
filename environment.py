
import numpy as np
import random


class FourRoomsEnv(object):
    def __init__(self, goal_position=[1, 11]):
        # I'm creating the environment with a buffer for each side to make things easier for choosing actions
        self.n = 13
        self.goal_position = goal_position
        # Initializing fields
        self.grid = None
        self.agent_position = None
        # Instantiating fields
        self.reset()

    # Accessor for state field
    def get_state(self):
        return self.agent_position

    # Reset environment back to the init
    def reset(self):
        self.grid = np.zeros((self.n, self.n))
        # Mark edges as occupied
        self.grid[:, 0] = 1
        self.grid[0, :] = 1
        self.grid[self.n - 1, :] = 1
        self.grid[:, self.n - 1] = 1
        # Mark the dark gray squares as occupied or not available
        self.grid[6, 1] = 1
        self.grid[6, 3:7] = 1

        self.grid[7, 6:12] = 1
        self.grid[7, 9] = 0

        self.grid[:, 6] = 1
        self.grid[3, 6] = 0
        self.grid[10, 6] = 0

        self.agent_position = [11, 1]

    # Simulate four rooms environment
    # Takes in a station-action pair and returns
    # the next state as well as observed reward
    def simulate(self, s, a):
        # take action a
        self.agent_position = self.process_action(s, a)

        if self.agent_position == self.goal_position:
            r = 1
            self.reset()
            end = True
        else:
            r = 0
            end = False
        return self.agent_position.copy(), r, end

    # Handle the dynamics of the environment, like slipping and
    # trying to move into a wall
    def process_action(self, s, a):
        # Determine if we take the chosen action or agent slips.
        chosen_action = False
        slip1 = False
        rand = random.random()
        if rand <= 0.1:
            # Slip with 10% chance
            slip1 = True
        elif rand < 0.9:
            chosen_action = True
        # If rand is >= 0.9, slip the other way with 10% chance

        # HANDLE LEFT ACTION
        if a == 'left' or a == 2:
            if chosen_action:
                # If we would hit a wall, we do nothing.
                if self.grid[s[0], s[1] - 1] != 1:
                    s[1] = s[1] - 1
            # Slip up
            elif slip1:
                if self.grid[s[0] - 1, s[1]] != 1:
                    s[0] = s[0] - 1
            # Slip down
            else:
                if self.grid[s[0] + 1, s[1]] != 1:
                    s[0] = s[0] + 1
        # HANDLE RIGHT ACTION
        elif a == 'right' or a == 3:
            if chosen_action:
                # If we would hit a wall, we do nothing.
                if self.grid[s[0], s[1] + 1] != 1:
                    s[1] = s[1] + 1
            # Slip down
            elif slip1:
                if self.grid[s[0] + 1, s[1]] != 1:
                    s[0] = s[0] + 1
            # Slip up
            else:
                if self.grid[s[0] - 1, s[1]] != 1:
                    s[0] = s[0] - 1
        # HANDLE UP ACTION
        elif a == 'up' or a == 0:
            if chosen_action:
                if self.grid[s[0] - 1, s[1]] != 1:
                    s[0] = s[0] - 1
            # slip left
            elif slip1:
                if self.grid[s[0], s[1] - 1] != 1:
                    s[1] = s[1] - 1
            # slip right
            else:
                if self.grid[s[0], s[1] + 1] != 1:
                    s[1] = s[1] + 1
        # HANDLE DOWN ACTION
        else:
            if chosen_action:
                if self.grid[s[0] + 1, s[1]] != 1:
                    s[0] = s[0] + 1
            # slip down
            elif slip1:
                if self.grid[s[0], s[1] + 1] != 1:
                    s[1] = s[1] + 1
            # slip up
            else:
                if self.grid[s[0], s[1] - 1] != 1:
                    s[1] = s[1] - 1
        return s

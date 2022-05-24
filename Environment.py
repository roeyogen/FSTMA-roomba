import random
import copy
import numpy as np
import gym
from gym import spaces
import itertools


class Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    # metadata = {'render.modes': ['console']}
    metadata = {'render.modes': ['human']}
    ACTIONS = {'STAY': (0, 0), 'UP': (-1, 0), 'DOWN': (1, 0), 'RIGHT': (0, 1), 'LEFT': (0, -1)}
    OUT = -100
    CHARGE = 100

    def __init__(self, num_of_solar_panels=3, height=5, width=10, number_of_agents=2, max_fuel=100):
        super(Env, self).__init__()

        # Initialize the board
        self.height = height
        self.width = width
        self.length = (width + 1) * num_of_solar_panels + 1
        self.max_fuel = max_fuel
        # Initialize the agents positions
        self.number_of_agents = number_of_agents
        self.charging_points = [(self.height // 2, column) for column in range(0, self.length, self.width + 1)]
        self.agents = {}
        self.board, self.agents = self.reset()

        # Define action and observation space
        self.action = list(itertools.product(self.ACTIONS.keys(), repeat=number_of_agents))
        self.num_actions = len(self.action)
        self.action_space = spaces.Discrete(self.num_actions)
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=self.OUT, high=self.CHARGE,
                                            shape=(self.height, self.length), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the board
        self.board = np.full((self.height, self.length), 1, dtype=int)
        for column in range(0, self.length, self.width + 1):
            self.board[:, column] = self.OUT
            self.board[self.height // 2, column] = self.CHARGE

        # Initialize the agents positions
        starting_points = random.sample(self.charging_points, self.number_of_agents)
        for i in range(self.number_of_agents):
            self.agents['Agent_{}'.format(i + 1)] = [starting_points[i], self.max_fuel]

        return self.board, self.agents

    def is_in_board(self, agent_loc):
        if 0 <= agent_loc[0] <= self.length and 0 <= agent_loc[1] <= self.width and \
                not self.board[agent_loc] == self.OUT:
            return True
        return False

    def is_legal_step(self, new_agents):

        locs = [x[0] for x in new_agents.values()]

        # check no overlap
        if not len(set(locs)) == len(locs):
            return False

        for loc in locs:
            # check in in_board
            if not self.is_in_board(loc):
                return False

        return True

    # def is_done(self):

    def step(self, action):
        new_agents = copy.deepcopy(self.agents)
        for i, a in enumerate(action):
            agent = "Agent_{}".format(i + 1)
            if self.agents[agent][1] == 0:
                continue

            new_agents[agent][0] = tuple(map(sum, zip(self.agents[agent][0], self.ACTIONS[a])))
            new_agents[agent][1] -= 1
        if not self.is_legal_step(new_agents):
            for a in self.agents.values():
                if a[1] > 0:
                    a[1] -= 1
        else:
            for a, v in self.agents.items():
                if v[1] > 0:
                    self.board[new_agents[a][0]] -= 1
        self.agents = new_agents

        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        # done = is_done()

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.agent_pos == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([self.agent_pos]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass

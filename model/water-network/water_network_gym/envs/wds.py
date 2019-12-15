import gym
import wntr
import numpy as np
import random

from gym import error, spaces
from pkg_resources import resource_filename

class WDSEnv(gym.Env):

    reward_types = ('gaussian', 'delta', 'abs')
    control_types = ('pressure', 'flowrate')

    def __init__(self, reference, control_type, reward_type):
        if not control_type in self.control_types:
            raise ValueError("Control type not supported")
        if not reward_type in self.reward_types:
            raise ValueError("Reward type not supported")
        self.control_type = control_type
        self.reward_type = reward_type

        inp_file = resource_filename(__name__, 'WDS1.inp')
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.valve = self.wn.get_link('V1')
        self.demand_node = self.wn.get_node('N3')
        self.initial_setting = self.valve.minor_loss
        self.valve.minor_loss = self.initial_setting
        self.sim = wntr.sim.WNTRSimulator(self.wn)
        self.demand_node.add_leak(self.wn, 0.1, 0.1, start_time = 0)
        self.reference = reference
        self.observation_space = spaces.Box(low = np.array([0]), high = np.array([np.inf]))
        self.action_space = spaces.Box(low = np.array([0]), high = np.array([np.inf]))

    def reset(self):
        self.wn.reset_initial_values()
        self.valve.minor_loss = self.initial_setting
        results = self.sim.run_sim()
        if self.control_type == 'flowrate':
            return np.array([float(results.link['flowrate']['P2'])])
        elif self.control_type == 'pressure':
            return np.array([float(results.node['pressure']['N3'])])

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.valve.minor_loss = action[0]
        results = self.sim.run_sim()
        if self.control_type == 'flowrate':
            observation = np.array([float(results.link['flowrate']['P2'])])
        elif self.control_type == 'pressure':
            observation = np.array([float(results.node['pressure']['N3'])])
        if self.reward_type == 'gaussian':
            reward = np.exp(-(self.reference - observation)**2/2e-3)
        elif self.reward_type == 'abs':
            reward = float(-abs(self.reference - observation))
        elif self.reward_type == 'delta':
            reward = 1 if abs(self.reference - observation) < 1e-4 else 0
        self.wn.reset_initial_values()
        return observation, reward, False, {}
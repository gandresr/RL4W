import gym
import wntr
import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding
from pkg_resources import resource_filename

class WDSEnv(gym.Env):

    def __init__(self, flow_reference):
        inp_file = resource_filename(__name__, 'WDS1.inp')
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.valve = self.wn.get_link('V1')
        self.initial_setting = self.valve.minor_loss
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        self.flow_reference = flow_reference

    def reset(self):
        self.valve.minor_loss = self.initial_setting
        results = self.sim.run_sim()
        return float(results.node['demand']['N3'])

    def step(self, action):
        self.valve.minor_loss = action
        results = self.sim.run_sim()
        observation = results.node['demand']['N3']
        reward = -abs(self.flow_reference - observation)
        return float(observation), float(reward), False, None

    def render(self, mode='human', close=False):
        pass
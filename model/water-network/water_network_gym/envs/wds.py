import gym
import wntr
import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding

class WDSEnv(gym.Env):

    def __init__(self, flow_reference):
        self.wn = wntr.network.WaterNetworkModel('WDS1.inp')
        self.valve = self.wn.get_link('V1')
        self.initial_setting = self.valve.minor_loss
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        self.flow_reference = flow_reference

    def reset(self):
        self.valve.minor_loss = self.initial_setting
        results = self.sim.run_sim()
        return results.node['demand']['N3']

    def step(self, action):
        self.valve.minor_loss = action
        results = self.sim.run_sim()
        observation = results.node['demand']['N3']
        reward = -abs(self.flow_reference - observation)
        return observation, reward, False, None

    def render(self, mode='human', close=False):
        pass
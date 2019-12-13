import gym
import wntr
import numpy as np

from gym import error, spaces
from pkg_resources import resource_filename

class WDSEnv(gym.Env):

    def __init__(self, pressure_reference):
        inp_file = resource_filename(__name__, 'WDS1.inp')
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.valve = self.wn.get_link('V1')
        self.initial_setting = self.valve.minor_loss
        self.sim = wntr.sim.WNTRSimulator(self.wn)
        self.demand_node = self.wn.get_node('N3')
        self.demand_node.add_leak(self.wn, 0.1, 0.1, start_time = 0)
        self.pressure_reference = pressure_reference
        self.low = np.array([0])
        self.high = np.array([np.inf])
        self.observation_space = spaces.Box(low = self.low, high = self.high)
        self.action_space = spaces.Box(low = self.low, high = self.high)

    def reset(self):
        self.valve.minor_loss = self.initial_setting
        results = self.sim.run_sim()
        return np.array([float(results.node['head']['N3'])])

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.valve.minor_loss = action[0]
        results = self.sim.run_sim()
        observation = np.array([float(results.node['head']['N3'])])
        reward = np.exp(-(self.pressure_reference - observation)**2/2)
        self.wn.reset_initial_values()
        return observation, reward, False, {}

    def render(self, mode='human', close=False):
        pass
import gym
import water_network_gym
import numpy as np
from time import time
env = gym.make('single-valve-v0', flow_reference = 0.002) # flow_reference == desired demand flow in N3
print(env.reset()) # state == demand at node N3
obs = env.step(np.array([0]))
print(obs)
print('Step', obs) # action == valve loss coefficient
import gym
import water_network_gym

env = gym.make('single-valve-v0', flow_reference = 0.002) # flow_reference == desired demand flow in N3
print(env.reset()) # state == demand at node N3
print(env.step(1000)) # action == valve loss coefficient
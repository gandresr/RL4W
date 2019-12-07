from gym.envs.registration import register

register(
    id='single-valve-v0',
    entry_point='water_network_gym.envs:WDSEnv',
)
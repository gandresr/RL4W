import os

import gym
import water_network_gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

from utils import save_ppo_results, timestamp_name
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO1
from stable_baselines.common.policies import MlpPolicy
from scipy.ndimage.filters import gaussian_filter1d

def callback(_locals, _globals):
    """
    Callback after n steps
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global env, xarr, yarr, n_before, n_now, true_actions, true_observations
    if 'actions' in _locals:
        true_actions = _locals['actions']
        true_observations = _locals['observations']
    episodic = False
    if not episodic:
        if env.rewards != None:
            xarr.append(env.total_steps)
            n_now = env.total_steps
            yarr.append(sum(env.rewards[n_before:n_now]))
            n_before = env.total_steps
    else:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            xarr = x
            yarr = y
    return True

def get_pi(true_observations, true_actions, model):
    '''
    Resturn trained policy
    '''
    true_observations = np.array(true_observations)
    true_actions = np.array(true_actions)

    action_min = np.amin(true_actions)
    action_max = np.amax(true_actions)
    obs_min = np.amin(true_observations)
    obs_max = np.amax(true_observations)
    prob_space = np.zeros((100,100))
    action_space = np.linspace(action_min, action_max, 100)
    obs_space = np.linspace(obs_min, obs_max, 100)

    for i in range(100):
        for j in range(100):
            prob_space[i,j] = model.action_probability(
                observation = np.array( [obs_space[i]] ),
                actions = np.array( [action_space[j]] ))

    return prob_space

if __name__ == "__main__":
    lam = 0.95; gamma = 0.99
    entcoeffs = np.linspace(0, 0.01, 4)
    control_type = sys.argv[1]
    reward_type = sys.argv[2]
    clip = float(sys.argv[3])

    if not control_type in ('flowrate', 'pressure',):
        raise ValueError("Control type not supported")
    if not reward_type in ('gaussian', 'abs', 'delta',):
        raise ValueError("Reward type not supported")

    if control_type == 'flowrate':
        reference = 0.1
    elif control_type == 'pressure':
        reference = 10

    for entcoeff in entcoeffs:
        best_mean_reward, n_steps = -np.inf, 0
        yarr = []; xarr = []
        true_observations = []; true_actions = []

        #print("start")
        log_dir = "tmp/"
        fig_dir = ''.join(['main_results', os.sep, control_type, os.sep, reward_type])
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        #print("make environment")
        env = gym.make(
            'single-valve-v0',
            reference = 0.1,
            control_type = 'flowrate',
            reward_type = 'abs')

        n_before = 0
        n_now = 0
        env = Monitor(env, log_dir, allow_early_resets=True)

        #print("make learning model")
        actor_batch_size = 256
        model = PPO1(MlpPolicy, env, verbose=0, timesteps_per_actorbatch=actor_batch_size,
                    gamma = gamma, clip_param=clip, entcoeff=entcoeff, optim_epochs=4,
                    optim_batchsize=16, optim_stepsize=0.001, lam=lam, adam_epsilon=3e-3,
                    n_cpu_tf_sess = 1)
        time_steps = 2e4

        model.learn(total_timesteps=int(time_steps), callback=callback)

        xarr = np.array(xarr)
        yarr = np.array(yarr)
        y_smooth = gaussian_filter1d(yarr, sigma=2)
        plt.plot(xarr, yarr, 'b.')
        plt.plot(xarr, y_smooth, 'b-')
        plt.title('$\epsilon$ = %.2f, ent = %.2f, $\gamma$ = %.2f, $\lambda$ = %.2f' % \
                (clip, entcoeff, gamma, lam,))
        plt.xlabel('Learning Iterations')
        plt.ylabel('Reward')
        fig_name = ''.join([
            'ppo_', control_type, '_', reward_type, '_',
            '%.2f' % clip, '_', '%.2f' % entcoeff, '_', '%.2f' % gamma, '_', '%.2f' % lam])
        fig_name.replace('.', '-')
        plt.savefig(fig_dir + fig_name, 'png')
        plt.clf()

        pi = get_pi(true_observations, true_actions, model)
        save_ppo_results(fig_name + '.dat', clip, gamma, lam, entcoeff,
            time_steps, xarr, yarr, true_observations, true_actions, pi)
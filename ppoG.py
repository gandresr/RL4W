import os

import gym
import water_network_gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

from utils import save_ppo_results, timestamp_name
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO1
from stable_baselines.common.policies import MlpPolicy
from scipy.ndimage.filters import gaussian_filter1d

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN and others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global env, xarr, yarr, n_before, n_now
    episodic = False
    if not episodic:
        if env.rewards != None:
            xarr.append(env.total_steps)
            n_now = env.total_steps
            yarr.append(sum(env.rewards[n_before:n_now]))
            n_before = env.total_steps
    else:
    #global n_steps, best_mean_reward
    #global xarr, yarr
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            xarr = x
            yarr = y

    # Print stats every 1000 calls
    #if (n_steps + 1) % 10 == 0:
        # Evaluate policy training performance
        #x, y = ts2xy(load_results(log_dir), 'timesteps')
        #if len(x) > 0:
        #    mean_reward = np.mean(y[-100:])
        #    print(x[-1], 'timesteps')
        #    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            #if mean_reward > best_mean_reward:
            #    best_mean_reward = mean_reward
                # Example for saving best model
            #    print("Saving new best model")
            #    _locals['self'].save(log_dir + 'best_model.pkl')
    #n_steps += 1
    return True

if __name__ == "__main__":

    clips = [0.75, 1, 1.5, 5, 10]
    entcoeffs = [1, 5, 10, 100]
    for clip in clips:
        for entcoeff in entcoeffs:
            best_mean_reward, n_steps = -np.inf, 0
            yarr = []
            xarr = []

            #print("start")
            log_dir = "tmp/"
            os.makedirs(log_dir, exist_ok=True)

            #print("make environment")
            env = gym.make('single-valve-v0', flow_reference = 0.1)
            n_before = 0
            n_now = 0
            env = Monitor(env, log_dir, allow_early_resets=True)

            #print("make learning model")
            actor_batch_size = 256
            gamma = 0.99
            lam = 0.95
            model = PPO1(MlpPolicy, env, verbose=1, timesteps_per_actorbatch=actor_batch_size,
                        gamma = gamma, clip_param= clip, entcoeff=entcoeff, optim_epochs=4,
                        optim_batchsize=16, optim_stepsize=0.001, lam=lam, adam_epsilon=1e-05)
            time_steps = 10e3

            model.learn(total_timesteps=int(time_steps), callback=callback)

            print("plotting ", xarr, yarr)
            xarr = np.array(xarr)
            yarr = np.array(yarr)
            save_ppo_results(clip, gamma, lam, entcoeff, time_steps, xarr, yarr)
            y_smooth = gaussian_filter1d(yarr, sigma=2)
            plt.plot(xarr, yarr, 'b.')
            plt.plot(xarr, y_smooth, 'b-')
            plt.title('PPO with eps = ' + str(clip) + ', ent_coeff =' + str(entcoeff))
            plt.xlabel('timesteps during learning')
            plt.ylabel('reward')
            plt.savefig('results/figures/' + timestamp_name('ppo', 'png'))
            plt.clf()

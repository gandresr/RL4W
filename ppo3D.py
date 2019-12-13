import os

import gym
import water_network_gym
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    global env, xarr, yarr, n_before, n_now, true_actions, true_observation
    if 'actions' in _locals:
        true_actions = _locals['actions']
        true_observation = _locals['observations']
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
    best_mean_reward, n_steps = -np.inf, 0
    xarr = []
    yarr = []
    true_observation = []
    true_actions = []

    #print("start")
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    #print("make environment")
    env = gym.make('single-valve-v0', flow_reference = 0.14)
    #env = gym.make('MountainCarContinuous-v0')
    n_before = 0
    n_now = 0
    env = Monitor(env, log_dir, allow_early_resets=True)

    #print("make learning model")
    actor_batch_size = 256
    clip = 10
    gamma = 0.99
    lam = 0.95
    entcoeff = 100
    model = PPO1(MlpPolicy, env, verbose=1, timesteps_per_actorbatch=actor_batch_size,
                 gamma = gamma, clip_param= clip, entcoeff=entcoeff, optim_epochs=4,
                 optim_batchsize=16, optim_stepsize=0.001, lam=lam, adam_epsilon=1e-05)
    time_steps = 2e4
    print(time_steps)
    model.learn(total_timesteps=int(time_steps), callback=callback)

    true_observation = np.array(true_observation)
    true_actions = np.array(true_actions)

    action_min = np.amin(true_actions)
    action_max = np.amax(true_actions)
    obs_min = np.amin(true_observation)
    obs_max = np.amax(true_observation)

    prob_space = np.zeros((100,100))
    action_space = np.linspace(action_min, action_max, 100)
    obs_space = np.linspace(obs_min, obs_max, 100)

    for i in range(100):
        for j in range(100):
            prob_space[i,j] = model.action_probability(observation=np.array([obs_space[i]]), actions=np.array([action_space[j]]))

    fig = plt.figure()
    ha = fig.add_subplot(111, projection='3d')

    X,Y = np.meshgrid(obs_space, action_space)
    ha.plot_surface(X,Y,prob_space)
    ha.set_xlabel('State Space')
    ha.set_ylabel('Action Space')
    plt.title('Probability distribution over state and action space')
    for i in range(100):
        prob = 0
        for j in range(100):
            prob += prob_space[i,j]
        print(prob)
    #ha.scatter3D(obs_space, action_space, prob_space)

    #fig.show()
    plt.show()
    plt.savefig('results3/figures/' + timestamp_name('ppo3D', 'png'))

    print("plotting ", xarr, yarr)
    #results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO CartPole-v0")
    #plt.show()
    xarr = np.array(xarr)
    yarr = np.array(yarr)
    #x_smooth = np.linspace(xarr.min(), xarr.max(), round(time_steps/actor_batch_size))
    #y_smooth = gaussian_filter1d(yarr, sigma=2)
    #plt.plot(xarr, yarr, 'b.')
    #plt.plot(xarr, y_smooth, 'b-')
    #plt.title('PPO with eps = ' + str(clip) + ', batch size =' + str(actor_batch_size))
    #plt.xlabel('timesteps during learning')
    #plt.ylabel('reward')
    #plt.show()

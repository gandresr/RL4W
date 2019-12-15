# Proximal Policy Optimization for Water Distribution Systems Management

Reinforcement Learning project using the Proximal Policy Optimization algorithm to control pressures and flows in a simple water distribution system environment.

## Main Dependencies

* WNTR (for water distribution systems simulation)
```
pip install wntr
```
* Tensorflow v1.14
```
pip install tensorflow==1.14
```
* Bleeding-edge version of Stable Baselines (for core PPO algorithm)
```
git clone https://github.com/hill-a/stable-baselines && cd stable-baselines
pip install -e .[docs,tests]
```
* OpenAI Gym (contains the abstract classes for the water network environment)
```
pip install gym
```

## Structure

* main_results/ -  Grid-search results organized by control type ('flowrate', 'pressure), and type of reward function ('abs', 'gaussian', 'delta')
* model/ - Python scripts for constructing custom water distribution system environment
* plot_results.py - Script to generate plots based on main_results
* run_all.sh - Example of the bash instruction used to run training in parallel
* run_ppo.py - Main script to generate results, receives control_type, reward_type, and clip value as args. A grid search is performed over the space of entropy coefficients.
* utils.py - Utility functions to save results

## How to run

* Create the 'water_network_gym' environment
* In model/ folder, run the following command. This register custom water distribution environment to gym
```
cd model && pip install -e .
```
* To run a specific PPO simulation run the bash command in run_all.sh. If a multi-core processor is used, 'taskset' can be used to run processes in parallel and assign them to a specific worker.
```
taskset -c 0 python ppo_main_results.py flowrate abs 0.1
```
* Run ppo scripts. Note that hyperparameter has to be changed within the script (scripts do not have any inputs)
* Plot results simply running
```
python plot_results.py
```


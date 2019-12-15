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

* Model/ - Python scripts for constructing custom water distribution system environment and raw data for this environment  
* Results/ - Plots and data from first batch of simulations
* Results2/ - Plots and data from second batch of simulations
* Results3/ - Combined plots
* ppo.py - Main script for initial test
* ppo3D.py - Script for generating surface policy plot with estimated best hyperparameters
* ppoG.py - Copy of ppo.py for testing different hyperparameters in one script
* utils.py - Utility functions for saving data 

## How to run

* Create the 'rl_water' environment
* In model/ folder, run the following command. This registers custom water distribution environment to gym 
```
cd model && pip install -e .
```
* Run ppo scripts. Note that hyperparameter has to be changed within the script (scripts do not have any inputs) 


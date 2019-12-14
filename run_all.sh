#!/bin/bash
#----------------------------------------------------
# Example SLURM job script to run pure OpenMP applications
#----------------------------------------------------
#SBATCH -J train_rl        # Job name
#SBATCH -o train_rl.o%j    # Name of stdout output file
#SBATCH -e train_rl.o%j    # Name of stderr output file
#SBATCH -p normal         # Queue name
#SBATCH -N 1              # Total number of nodes requested
#SBATCH -n 1              # Total number of mpi tasks requested
#SBATCH -t 00:10:00       # Run time (hh:mm:ss) - 10 min
#SBATCH -A Urban-Stormwater-Mod       # Project/allocation number

python ppo_main_results.py flowrate abs 0.1
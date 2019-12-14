from mpi4py import MPI
import subprocess

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

combinations = [
    ('flowrate', 'abs', '0.1'),
    ('flowrate', 'abs', '0.2'),
    ('flowrate', 'abs', '0.3'),
    ('pressure', 'abs', '0.1'),
    ('pressure', 'abs', '0.2'),
    ('pressure', 'abs', '0.3'),
    ('flowrate', 'gaussian', '0.1'),
    ('flowrate', 'gaussian', '0.2'),
    ('flowrate', 'gaussian', '0.3'),
    ('pressure', 'gaussian', '0.1'),
    ('pressure', 'gaussian', '0.2'),
    ('pressure', 'gaussian', '0.3')]

subprocess.call(['python', 'ppo_main_results.py', *combinations[rank]])
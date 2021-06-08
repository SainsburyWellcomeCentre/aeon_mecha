#!/bin/bash
#SBATCH -p cpu     # partition (queue)
#SBATCH -N 1       # number of nodes
#SBATCH -n 1       # number of cores
#SBATCH --mem=50G # memory pool for all cores
#SBATCH -t 0 # time (D-HH:MM)
#SBATCH -o "slurm.%N.%j.out" # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

python /ceph/aeon/aeon/code/ProjectAeon/aeon/aeon/preprocess/animal_position_tracking.py

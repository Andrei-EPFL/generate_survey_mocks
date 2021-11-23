#!/bin/bash
#SBATCH -n 1		# Number of tasks
#SBATCH -J firstgen  # Name of the job
#SBATCH -q regular
#SBATCH -C haswell
#SBATCH -N 1          # number of nodes
#SBATCH -c 64          # number of cpus per tasks
#SBATCH --time=48:00:00
#SBATCH -o ./output/out.firstgen_elg_qso.out
#SBATCH -e ./output/err.firstgen_elg_qso.err

source activate desilightcone

srun python main_new.py

#!/bin/bash
#SBATCH -J cont_par
#SBATCH -t 00:60:00
#SBATCH --mem 5000
#SBATCH -n 1
#SBATCH -N 1

module load Python/3.5.1-foss-2016a
python cluster_full_factorial_NFD_computation.py $SLURM_ARRAY_TASK_ID

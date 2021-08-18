#!/bin/bash
#SBATCH -J cont_par
#SBATCH -t 02:00:00
#SBATCH --mem 5000
#SBATCH -n 1
#SBATCH -N 1

module load Python/3.6.6-foss-2018b
python cluster_full_factorial_NFD_computation.py $SLURM_ARRAY_TASK_ID

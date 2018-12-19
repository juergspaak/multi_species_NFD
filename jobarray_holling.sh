#!/bin/bash
#SBATCH -J holling
#SBATCH -t 00:40:00
#SBATCH --mem 1000
#SBATCH -n 1
#SBATCH -N 1

module load Python/3.5.1-foss-2016a
python cluster_holling.py $SLURM_ARRAY_TASK_ID 

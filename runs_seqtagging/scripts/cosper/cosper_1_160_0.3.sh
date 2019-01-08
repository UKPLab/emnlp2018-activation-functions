#!/bin/bash
 
#SBATCH -J Act_cosper_1
#SBATCH --mail-user=eger@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=FAIL
#SBATCH -e /work/scratch/se55gyhe/log/output.err.%j
#SBATCH -o /work/scratch/se55gyhe/log/output.out.%j

#SBATCH -n 1 # Number of cores
#SBATCH --mem-per-cpu=6000
#SBATCH -t 23:59:00 # Hours, minutes and seconds, or '#SBATCH -t 10' -only mins
#module load intel python/3.5

python3 /home/se55gyhe/Act_func/sequence_tagging/arg_min/PE-my.py cosper 346 Nadam 3 0.5000234928706907 0.0019353963823578503 runiform 0.3


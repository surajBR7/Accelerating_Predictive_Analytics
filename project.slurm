#!/bin/bash
#SBATCH --job-name=python
#SBATCH --time=00:45:00
#SBATCH --nodes=1
##SBATCH --nodelist=node4
#SBATCH --ntasks=6
#SATCH --partition=short-single
#SBATCH -o outmsg-%j
#SBATCH -e errmsg-%j
module load conda/2022.05
conda activate project 
python3 main.py

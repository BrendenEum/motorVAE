#!/bin/bash
#SBATCH --account=def-webbr
#SBATCH --time=00-24:00:00 # DD-HH:MM:SS
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=40000M
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out

# Go to project folder
cd /home/beum/scratch/motorVAE

# Load modules
module load StdEnv/2020 
module load python/3.9.6 
module load cuda/11.4 
module load scipy-stack/2021a

# Activate virtual environment
source env/bin/activate

# Train
python motorVAE.py

# Notes
# 200 epochs takes approx 13 hrs.

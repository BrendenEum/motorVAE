#!/bin/bash
#SBATCH --account=def-webbr_gpu
#SBATCH --time=00-20:00:00 # DD-HH:MM:SS
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40000M
#SBATCH --output=logs/%j.out

# Go to project folder
cd /home/beum/scratch/motorVAE

# Load modules
module load StdEnv/2020 
module load python/3.9.6 
module load cuda/12.2
module load scipy-stack/2021a

# Activate virtual environment
source env/bin/activate

# Train
python motorVAE.py

# Notes
# 200 epochs takes approx 13 hrs.

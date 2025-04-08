#!/bin/bash
#SBATCH --account=def-webbr
#SBATCH --time=0-12:00:00 # DD-HH:MM:SS
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32000M
#SBATCH --output=job-logs/%j.out

# Go to project folder
cd /home/beum/scratch/motorVAE

# Load modules
module load StdEnv/2020 python/3.9.6 cuda/11.4

# Activate virtual environment for carVAEna
source env/bin/activate

# Train
python motorVAEGAN.py --data_dir data/evox_256x256_1-3 --dataset motorVAEGAN_256x256_1-3 --img_size 256 --model_path checkpoints/motorVAEGAN_256x256_1-3.pth --train --visualize --extract_latent --sample --latent_dim 128 --kld_weight 0.005 --adv_weight 1.0 --learning_rate 0.0001 --batch_size 128 --epochs 112


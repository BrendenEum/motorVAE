#!/bin/bash
#SBATCH --account=def-webbr
#SBATCH --time=0-12:00:00 # DD-HH:MM:SS
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32000M
#SBATCH --output=job_log-%j.out

# Go to project folder
cd scratch/car-VAE-na

# Load modules
module load StdEnv/2020 python/3.9.6 cuda/11.4

# Activate virtual environment for carVAEna
source env/bin/activate

# Train
python vae.py --data_dir data/evox_256x256_1-3 --img_size 256 --train --visualize --extract_latent --model_path checkpoints/car-VAE-na_256x256_1-3.pth --latent_save_path latents_256x256_1-3/ --latent_dim 128 --kld_weight 0.001 --learning_rate 0.0001 --batch_size 32 --epochs 112


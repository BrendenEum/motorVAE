#!/bin/bash
#SBATCH --account=def-webbr
#SBATCH --time=0-16:00:00 # DD-HH:MM:SS
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32000M
#SBATCH --output=logs/%j.out

# Go to project folder
cd /home/beum/scratch/motorVAE

# Load modules
module load StdEnv/2020 python/3.9.6 cuda/11.4

# Activate virtual environment for carVAEna
source env/bin/activate

# Train
python motorVAEGAN.py --data_dir data/evox_256x256_1-3 --img_size 256 --latent_dim 128 --max_kld_weight 1.0 --adv_weight 1.0 --recon_sample_weight 0.5 --learning_rate 0.0001 --epochs 300 --batch_size 128 --train --reconstructions --extract_latent --sample --traversals 2021_Toyota_CamryHybrid_XLE_sedan_4Door_2.png --interpolate 2007_Toyota_PriusHybrid_nan_hatchback_5Door_3.png 2025_Hyundai_Ioniq5N_nan_CUV_4Door_2.png --track_reconstruction 2007_Honda_Accord_SE_sedan_4Door_1.png
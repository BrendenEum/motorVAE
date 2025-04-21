#!/bin/bash
#SBATCH --account=def-webbr
#SBATCH --time=00-24:00:00 # DD-HH:MM:SS
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32000M
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
python motorVAEGAN-withSupervision.py \
    --data_dir data/evox_256x256_1-4 --img_size 256 \
    --latent_dim 128 \
    --max_kld_weight 1.0 --tc_weight 50.0 \
    --adv_weight 1.0 --recon_sample_weight 0.5 \
    --cls_weight 1.0 --cls_latent_dim 10 \
    --label_file data/labels_evox_256x256_1-4.csv --label_cols Year,Brand,Body,Door \
    --learning_rate 0.0001 --epochs 200 --batch_size 128 \
    --train --reconstructions --extract_latent --sample \
    --classification_accuracy --visualize_latent_class \
    --feature_attribution \
    --traversals 2021_Toyota_CamryHybrid_XLE_sedan_4Door_2.png \
    --interpolate 2007_Toyota_PriusHybrid_nan_hatchback_5Door_3.png 2025_Hyundai_Ioniq5N_nan_CUV_4Door_2.png \
    --track_reconstruction 2018_Audi_Q7_PremiumPlus3.0TFSI_SUV_4Door_2.png

# Notes
# 200 epochs takes approx 13 hrs.

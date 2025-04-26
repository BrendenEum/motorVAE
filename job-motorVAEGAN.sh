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
python superMotorVAEGANpatch.py \
    --data_dir data/evox_256x256_1-4 --img_size 256 \
    --latent_dim 128 \
    --perceptual_weight 1.0 \
    --max_kld_weight 1.0 \
    --mi_weight 1.0 --tc_weight 1.0 --dwkl_weight 1.0 \
    --adv_weight 0.5 --recon_sample_weight 0.5 \
    --cls_weight 0.5 --cls_latent_dim 16 \
    --label_file data/labels_evox_256x256_1-4.csv --label_cols Year,Brand,Body,Door \
    --learning_rate 0.0001 --epochs 200 --batch_size 128 --patch_downsample 4 \
    --train --reconstructions --extract_latent --sample \
    --classification_accuracy --visualize_latent_class --feature_attribution \
    --traversals 2021_Toyota_CamryHybrid_XLE_sedan_4Door_2.png \
    --interpolate 2007_Toyota_PriusHybrid_nan_hatchback_5Door_3.png 2025_Polestar_Polestar4_LongRangeDualMotor_SUV_4Door_1.png \
    --track_reconstruction 2014_Nissan_Versa_SL_sedan_4Door_1.png

# Notes
# 200 epochs takes approx 13 hrs.

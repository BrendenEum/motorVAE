# Variational Autoencoder
Author: Brenden Eum (2025)

This implementation creates a variational autoencoder specifically designed for your 64×64 grayscale vehicle images. 


## Here's a breakdown of what it does:

1. Dataset Handling: The `VehicleDataset` class loads your grayscale PNG images from the specified directory.

2. VAE Architecture:

    - The encoder uses convolutional layers with batch normalization and LeakyReLU activations.
    - The latent space has 128 dimensions by default (configurable).
    - The decoder mirrors the encoder with transposed convolutions for upsampling.
    - Final output uses a sigmoid activation for pixel values between 0 and 1.

3. Key Features:

    - Latent Space Traversal: The `latent_traversal` method allows you to modify specific dimensions and see the generated results.
    - Latent Vector Extraction: The `extract_latent_vectors` function saves both mean and log variance vectors as NumPy files for external analysis.
    - Visualization Tools: Functions to visualize reconstructions and latent space traversals.

4. Optimization: Uses Adam optimizer and a combination of reconstruction loss (MSE) and KL divergence loss, with a configurable weight parameter to balance the two.

5. Command-line Interface: The script provides a flexible CLI with various options.
    ```
    python vae.py --data_dir evox_64x64_1 --train --visualize --extract_latent
    ```


## To get started in an interactive job:

Request an interactive session.

```
salloc --account=def-webbr  --time=00:30:00 --gres=gpu:1 --mem=8000M --ntasks=1 --cpus-per-task=2
```

Load all the modules.

```
cd scratch/car-VAE-na/
module load StdEnv/2020 python/3.9.6 cuda/11.4
# For some reason, you may need to tell the compute node where the cudacore library is. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.4.2/lib/"
```

Activate the environment with `source env/bin/activate`. If it's your first time, you'll need to set up the virtual environment and install all the required libraries.

```
#virtualenv --no-download env
source env/bin/activate
#pip install -r requirements.txt
```

If you have issues with installing the compute canada versions of the packages, then run this.

```
unset PIP_CONFIG_FILE
unset PYTHONPATH
```

Copy-pasta this line of code into the terminal to do all the things!

```
python vae.py --data_dir data/evox_256x256_1-3 --img_size 256 --train --visualize --extract_latent --model_path checkpoints/car-VAE-na.pth --latent_save_path latents/ --latent_dim 128 --kld_weight 0.001 --learning_rate 0.0001 --batch_size 32 --epochs 100
```

Key Arguments

1. Use `--data_dir {path}`to set the location of your training images (default evox_64x64_1).
2. Use `--img_size 64` to set the final image resolution (default 64).
3. Use `--train` to train the VAE.
4. Use `--visualize` to see reconstructions and latent space traversals.
5. Use `--extract_latent` to save latent vectors for external analysis.
6. Use `--model_path {path/fn.pth}` to save/load the model as a .pth file (default pwd/vae_model.pth).
7. Use `--latent_save_path {path/folder_name}` to set the directory to save latent vectors

Other Arguments

- Use `--resume` to resume training from last checkpoint.
- Use `--sample` to generate random samples from the latent space.


Parameters

1. Use `--latent_dim 128` to control the size of your latent space (default is 128). Larger values capture more details but may be harder to train.
2. Use `--kld_weight 0.005` to balance reconstruction quality versus latent space regularity (default is 0.005). Lower values -- like 0.001 -- prioritize reconstruction quality, while higher values -- like 0.01 -- create a more structured latent space.
3. Use `--learning_rate 0.0001` to control how quickly the model learns (default 0.0001). Too high might cause instability, but too low might make training super slow.
4. Use `--batch_size 32` to deal with memory constraints (default 32). Smaller batches help with limited memory, but higher batches speed up training.
5. Use `--epochs 100` to set the number of times the dataset is worked through (default 100). More epochs generally gives better results, but takes longer to train.


## Run it as a job on the cluster

Once you know how to run it interactively, you can just write all of this into shell code and submit it as a SLURM job.

```
sbatch job_car-VAE-na.sh
```
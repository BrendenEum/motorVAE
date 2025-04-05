# motorVAE: Variational Autoencoder for Vehicle Image Reconstruction with Disentangled Latent Space
Author: Brenden Eum (2025)

This implementation creates a variational autoencoder specifically designed for your pixel×pixel grayscale vehicle images. 


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
cd /home/beum/scratch/motorVAE
module load StdEnv/2020 python/3.9.6 cuda/11.4
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
python vae.py --data_dir data/evox_256x256_1-3 --dataset 256x256_1-3 --img_size 256 --model_path checkpoints/motorVAE_256x256_1-3.pth --train --visualize --extract_latent --latent_dim 128 --kld_weight 0.001 --learning_rate 0.0001 --batch_size 32 --epochs 100
```

Key Arguments

1. Use `--data_dir {path}` to set the location of your training images (default evox_64x64_1).
2. Use `--dataset {64x64_1}` to append the name of the dataset to the names of output folders.
3. Use `--img_size 64` to set the final image resolution (default 64).
4. Use `--model_path {path/fn.pth}` to save/load the model as a .pth file (default pwd/vae_model.pth).

What Do You Want To Do

- Use `--train` to train the VAE.
- Use `--resume` to resume training from last checkpoint. This must be used with `--train`.
- Use `--visualize` to see reconstructions and latent space traversals. Only requires `--train` the first time.
- Use `--extract_latent` to save latent vectors for external analysis. Only requires `--train` the first time.
- Use `--sample` to generate random samples from the latent space. Only requires `--train` the first time.

Other Arguments

- Use `--reconstruct_dir {folder_name}` to set the directory to save reconstructions. No need to set this if you specify `--dataset`.
- Use `--latent_save_dir {folder_name}` to set the directory to save latent vectors. No need to set this if you specify `--dataset`
- Use `--latent_traversal_dir {folder_name}` to set the directory to save latent traversals. No need to set this if you specify `--dataset`


Parameters

1. Use `--latent_dim 128` to control the size of your latent space (default is 128). Larger values capture more details but may be harder to train.
2. Use `--kld_weight 0.005` to balance reconstruction quality versus latent space regularity (default is 0.005). Lower values -- like 0.001 -- prioritize reconstruction quality, while higher values -- like 0.01 -- create a more structured latent space.
3. Use `--learning_rate 0.0001` to control how quickly the model learns (default 0.0001). Too high might cause instability, but too low might make training super slow.
4. Use `--batch_size 32` to deal with memory constraints (default 32). Smaller batches help with limited memory, but higher batches speed up training.
5. Use `--epochs 100` to set the number of times the dataset is worked through (default 100). More epochs generally gives better results, but takes longer to train.


## Run it as a job on the cluster

Once you know how to run it interactively, you can just write all of this into shell code and submit it as a SLURM job.

```
sbatch job-motorVAE-evox_256x256_1-3.sh
```

To stream the output, type `tail -f {job_log.out}`. 

To check in on GPU usage, open up a separate terminal, ssh into the cluster, then ssh into the compute node (check the compute node address using `sq`). Then run `watch -n 1 nvidia-smi`.
# motorVAEGAN: Variational Autoencoder for Vehicle Image Reconstruction with Disentangled Latent Space
Author: Brenden Eum (2025)

![motorVAEGAN architecture](<writing/motorVAEGAN-architecture.png>)

For more details on the architecture, see [this .md file](writing/network_architecture.md). This README is only meant to help you get started with the code. I'm writing this as if you're a doofus with coding, which is what I am. 

The code was written to run on Compute Canada's Cedar cluster with a V100 GPU (32GB memory) and 4 worker CPUs. My car dataset contains about 35,000 images at 256x256 pixels. With the settings in the example code below, it takes somewhere between 4-12 hours to train; I usually fall asleep before it finishes and keep forgetting to record training start and end times for an accurate estimate. My hunch is that it takes ~6 hours.

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

## To get started in an interactive job:

Request an interactive session.

```
cd /home/beum/scratch/motorVAE
salloc --account=def-webbr  --time=00:30:00 --gres=gpu:1 --mem=12000M --ntasks=1 --cpus-per-task=4
```

Load all the modules.

```
module load StdEnv/2020 python/3.9.6 cuda/11.4
```

>*First-time setup*: If it's your first time ever running this code, you'll need to set up the virtual environment and install all the required libraries with (1) `virtualenv --no-download env`, (2) `source env/bin/activate`, (3) `pip install -r requirements.txt`. I think `virtualenv` is different on Compute Canada, so you may want to use `venv` on a local computer.
>
>If you have issues with installing Compute Canada's versions of the packages, then you might want to run `unset PIP_CONFIG_FILE` and `unset PYTHONPATH` *right after loading the modules*. This tells Compute Canada to stop installing their own versions of these libraries, and instead downloads the libraries from PyPI.

If it's not your first time, you can skip directly to activating the environment. 

```
source env/bin/activate
```



Copy-pasta this line of code into the terminal to do all the things!

```
python motorVAEGAN.py --data_dir data/evox_256x256_1-3 --img_size 256 --latent_dim 128 --max_kld_weight 0.5 --adv_weight 1.0 --recon_sample_weight 0.5 --learning_rate 0.0001 --epochs 112 --batch_size 128 --train --reconstructions --extract_latent --sample --traversals 2022_Toyota_RAV4Hybrid_SE_CUV_4Door_1.png --interpolate 2022_Chevrolet_BoltEUV_Premier_CUV_4Door_2.png 2022_Volkswagen_Tiguan_SE_CUV_4Door_3.png
```

Key Arguments

- Use `--data_dir {path}` to set the location of your training images (default evox_64x64_1).
- Use `--dataset {64x64_1}` to append the name of the dataset to the names of output folders.
- Use `--img_size 64` to set the final image resolution (default 64).
- Use `--model_path {path/fn.pth}` to save/load the model as a .pth file (default pwd/vae_model.pth).

What Do You Want To Do

- Use `--train` to train the VAE from scratch.
- Use `--resume` to resume training from last checkpoint. Cannot be used with `--train`.
- Use `--reconstructions` to see reconstructions. Only requires `--train` the first time.
- Use `--traversals` to see latent space traversals. Only requires `--train` the first time.
- Use `--extract_latent` to save latent vectors for external analysis. Only requires `--train` the first time.
- Use `--sample` to generate random samples from the latent space. Only requires `--train` the first time.
- Use `--interpolate {img1} {img2}` to interpolate between the two images. Use `--interpolate_steps {#}` with this to specify how many steps you'd like to take between image 1 (z1 in latent space) to image 2 (z2 in latent space).

Parameters

- Use `--latent_dim 128` to control the size of your latent space (default is 128). Larger values capture more details but may be harder to train.
- Use `--kld_weight 0.005` to balance reconstruction quality versus latent space regularity (default is 0.005). Lower values -- like 0.001 -- prioritize reconstruction quality, while higher values -- like 0.01 -- create a more structured latent space.
- Use `--adv_weight 1.0` to control the weight of the adversarial loss term in the loss function.
- Use `--recon_sample_weight 0.7` to adjust weight for reconstruction vs sample discrimination (reconstruction w, sample 1-w).
- Use `--learning_rate 0.0001` to control how quickly the model learns (default 0.0001). Too high might cause instability, but too low might make training super slow.
- Use `--batch_size 128` to deal with memory constraints (default 32). Smaller batches help with limited memory, but higher batches speed up training.
- Use `--epochs 112` to set the number of times the dataset is worked through (default 100). More epochs generally gives better results, but takes longer to train.


## Run it as a job on the cluster

Once you know how to run it interactively, you can just write all of this into shell code and submit it as a SLURM job.

```
sbatch job-motorVAEGAN_256x256_1-3.sh
```

To stream the output, type `tail -f job-logs/{job#}.out`. 

To check in on GPU usage, open up a separate terminal, ssh into the cluster, then ssh into the compute node (check the compute node address using `sq`). Then run `watch -n 1 nvidia-smi`.

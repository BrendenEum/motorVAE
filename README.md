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
salloc --account=def-webbr --time=00:59:00 --gres=gpu:v100l:1 --mem=40000M --ntasks=1 --cpus-per-task=4
```

Load all the modules.

```
module load StdEnv/2020 python/3.9.6 cuda/11.4 scipy-stack/2021a
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
python motorVAEGAN-withSupervision.py \
    --data_dir data/evox_256x256_1-4 --img_size 256 \
    --latent_dim 128 \
    --max_kld_weight 1.0 --tc_weight 50.0 \
    --adv_weight 1.0 --recon_sample_weight 0.5 \
    --cls_weight 1.0 --cls_latent_dim 10 \
    --label_file data/labels_evox_256x256_1-4.csv --label_cols Year,Brand,Body,Door \
    --learning_rate 0.0001 --epochs 2 --batch_size 128 \
    --train --reconstructions --extract_latent --sample \
    --classification_accuracy --visualize_latent_class \
    --feature_attribution \
    --traversals 2021_Toyota_CamryHybrid_XLE_sedan_4Door_2.png \
    --interpolate 2007_Toyota_PriusHybrid_nan_hatchback_5Door_3.png 2025_Hyundai_Ioniq5N_nan_CUV_4Door_2.png \
    --track_reconstruction 2010_Jeep_Compass_Sport_CUV_4Door_4.png
```

Data Parameters

- Use `--data_dir {path}` to set the location of your training images (default data/evox_256x256_1-4).
- Use `--label_file` to tell the code where to find the labels for the training data (default data/labels.csv).
- Use `--label_cols` to input a comma-separated list of column names to use for labels (eg Year,Brand,Body,Door). Column names refer to columns in label_file csv. Default is all columns except Filename.
- Use `--img_size 256` to set the final image resolution (default 64).

Model Parameters

- Use `--latent_dim 128` to control the size of your latent space (default is 128). Larger values capture more details but may be harder to train.
- The weight for L1 reconstruction loss is normalized to 1.0.
- Use `--max_kld_weight 1.0` to balance reconstruction quality versus latent space regularity. We are using a scheduler, so KLD weight starts at 0.01 (to focus on reconstruction), then increases linearly to max_kld_weight (to emphasize latent space structure).
- Use `--mi_weight 1.0` to set the weight for mutual information loss in total KLD loss. This is to reduce the amount of information about x stored in z. Defaults to 1.0.
- Use `--tc_weight 50.0` to set the weight for total correlation loss in total KLD loss. This is to make latent variables z as independent as possible. Defaults to 50.0 based on Sisodia et al (2024).
- Use `--dwkl_weight 1.0` to set the weight for dimension-wise KLD loss in total KLD loss. This is to ensure each latent dimension does not deviate from prior (Gaussian). Defaults to 1.0.
- Use `--adv_weight 1.0` to control the weight of the adversarial loss term in the loss function.
- Use `--recon_sample_weight 0.7` to adjust weight for reconstruction vs sample discrimination (reconstruction w, sample 1-w).
- Use `--cls_weight 1.0` to set the weight for classification loss term. Classification loss is the sum of losses over all labels. Default is 1.0.
- Use `--cls_latent_dim 4` to set the number of latent dimensions to use for classification.This also sets the number of nodes in the intermediate, fully connected layer of the classifier. Default is 4.

Training Parameters

- Use `--train` to train the VAE from scratch.
- Use `--resume` to resume training from last checkpoint. Cannot be used with `--train`.
- Use `--batch_size 128` to deal with memory constraints (default 32). Smaller batches help with limited memory, but higher batches speed up training.
- Use `--learning_rate 0.0001` to control how quickly the model learns (default 0.0001). Too high might cause instability, but too low might make training super slow.
- Use `--epochs 112` to set the number of times the dataset is worked through (default 100). More epochs generally gives better results, but takes longer to train.

Output Parameters

- Use `--out_dir {output subfolder}` to name the subfolder in outputs/. Leaving this blank will automatically give you a detailed subfolder name (recommended).
- Use `--model_path {path/fn.pth}` to save/load the model as a .pth file. Leaving this blank will automatically give you a detailed file name (recommended).

Actions

- Use `--reconstructions` to save visualizations of reconstructions. Randomly selects 10 images to reconstruct, and saves as one file. Only requires `--train` the first time.
- Use `--extract_latent` to save latent vectors for external analysis. Latent vectors are actually matricies: rows are latent dimensions, columns are for each image in dataset. Saves two numpy files: mean and log_variance. Only requires `--train` the first time.
- Use `--sample` to generate 25 images using random samples from the latent space. Saves as one file. Only requires `--train` the first time.
- Use `--track_reconstruction {img}.png` to track reconstruction of a specific image across training epochs. I use this to generate GIFs for oohs and ahhs. Can only run if `--train` is also specified. Default is -unspecified-.
- Use `--traversals {img}.png` to see visualization of latent space traversals. Specify the image you'd like to do this with. Saves in latent_traversals/ subfolder, with one file per dimension. Only requires `--train` the first time.
- Use `--interpolate {img1.png} {img2.png}` to interpolate between the two images. Use `--interpolate_steps {#}` with this to specify how many steps you'd like to take between image 1 (z1 in latent space) to image 2 (z2 in latent space). Saves as one file.
- Use `--classification_accuracy` to evaluate classification accuracy.
- Use `--visualize_latent_class` to visualize latent space by class.
- Use `--feature_attribution` to visualize feature attribution. Use `--feature_samples` to with this to specify the number of samples for feature attribution.


## Run it as a job on the cluster

Once you know how to run it interactively, you can just write all of this into shell code and submit it as a SLURM job.

```
sbatch job-motorVAEGAN.sh
```

To stream the output, type `tail -f job-logs/{job#}.out`. 

To check in on GPU usage, open up a separate terminal, ssh into the cluster, then ssh into the compute node (check the compute node address using `sq`). Then run `watch -n 1 nvidia-smi`.

# motorVAE
Author: Brenden Eum (2025)

![alt text](<motorVAE-architecture.png>)

## Encoder

* Input
    * 256x256x1 greyscale image
* Conv2d (downsampling by sliding kernel)
    * 5 convolutional layers with increasing channels
        * [128x128x32, 64x64x64, 32x32x128, 16x16x256, 8x8x512]
        * stride=2 handles channel dimensionality (what to divide by)
        * kernel_size=3 handles the sliding window for feature detection
        * padding=1 adds a 1-px border of 0s around the input to prevent the output from being smaller (padding 1 with kernel 3 preserves spatial dimensions before stride is applied)
* BatchNorm2d
    * normalizes activations of the previous layer for each batch to keep the activations in a reasonable range and stabilize training
* LeakyReLU
    * allows for all positive or small negative activations (multiply by 0.01 if negative)
* Output
    * 8x8x512 tensor

## Latent Space

* Input
    * Output tensor from encoder (8x8x512) is flattened to 32768-dim vector
* nn.Linear x2
    * Two parallel, fully-connected layers project from this flattened vector to 128-dimensional latent space by learning a 32768x128 weight matrix
* Output:
    * Latent distribution parameterized by two, 128-dim latent vectors (mu, log_var)

## Decoder

* Input
    * Sample from latent distribution to obtain z (reparameterization trick), then do linear projection of 128-dim z to 32768-dim vector and reshape into 8x8x512 tensor to mimic encoder output
* ConvTranspose2d (upsampling with sliding kernel)
    * 4 transposed convolutional layers with decreasing channels
        * [16x16x256, 32x32x128, 64x64x64, 128x128x32]
        * stride=2
        * kernel_size=3
        * padding and output_padding=1 (ensures correct size)
    * Final 5th convolutional layer
        * Uses ConvTranspose2d to go from 128x128x32 to 256x256x32 
        * BatchNorm and LeaklyReLU
        * Uses Conv2d to go from 256x256x32 to 256x256x1
        * nn.Sigmoid to ensure values in [0,1]
* Output
    * 64x64x1 greyscale reconstruction

## Loss Function

* Reconstruction loss: MSE(image, reconstruction)
* $\beta$-weighted KL divergence: sum over all dimensions in latent distribution their deviation from std norm
    * 2-stage training: (stage 1) $\beta \approx 0$, (stage 2) $\beta = 0.05$
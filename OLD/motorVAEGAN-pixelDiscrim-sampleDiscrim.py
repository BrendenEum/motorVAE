import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class VehicleDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image
    
    def get_image_by_filename(self, filename):
        """
        Get an image by its filename
        """
        if filename in self.img_files:
            img_path = os.path.join(self.img_dir, filename)
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            if self.transform:
                image = self.transform(image)
                
            return image
        else:
            raise ValueError(f"File {filename} not found in dataset")
            
    def get_filenames(self):
        """
        Return all filenames in the dataset
        """
        return self.img_files

class VAE(nn.Module):
    def __init__(self, img_size=64, latent_dim=128, hidden_dims=None):
        super(VAE, self).__init__()
        
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.device = device  # Add device attribute for easy reference
        
        # Default architecture if hidden_dims is not provided
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        # Build Encoder
        modules = []
        in_channels = 1  # Grayscale images
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate the size of the feature maps before flattening
        # For an input of size 64, after 5 layers of stride 2, it's 64/(2^5) = 2
        encoder_output_size = img_size // (2 ** len(hidden_dims))
        encoder_output_dim = hidden_dims[-1] * encoder_output_size * encoder_output_size
        
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(encoder_output_dim, latent_dim)
        
        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, encoder_output_dim)
        
        modules = []
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                      hidden_dims[i + 1],
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        # Final layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                  hidden_dims[-1],
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], out_channels=1,
                         kernel_size=3, padding=1),
                nn.Sigmoid())
        )
        
        self.decoder = nn.Sequential(*modules)
        
        # Save the number of hidden dimensions for reshaping
        self.hidden_dims = hidden_dims
        self.encoder_output_size = encoder_output_size
        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        
        # Split the result into mu and var components
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return mu, log_var
    
    def decode(self, z):
        """
        Maps the given latent codes onto the image space.
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], self.encoder_output_size, self.encoder_output_size)
        result = self.decoder(result)
        return result
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, input, compute_loss=False):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z  # Also return z
    
    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def sample_with_latent(self, num_samples):
        """
        Samples from the latent space and returns both the latent vectors
        and the corresponding image space map.
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return z, samples
    
    def reconstruct(self, x):
        """
        Given an input image x, returns the reconstructed image and latent vector
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), z, mu, log_var
    
    def latent_traversal(self, x, dim, start=-3, end=3, steps=10):
        """
        Performs latent space traversal along a specific dimension.
        
        Args:
            x: Input image to encode
            dim: Dimension to traverse
            start: Starting value for traversal
            end: Ending value for traversal
            steps: Number of steps in the traversal
            
        Returns:
            List of decoded images from the traversal
        """
        with torch.no_grad():
            # Encode the input image
            mu, log_var = self.encode(x.unsqueeze(0))
            z = mu  # Use mean for traversal (no sampling)
            
            # Create a list to store the traversal images
            traversal_images = []
            
            # Create values for traversal
            values = np.linspace(start, end, steps)
            
            # Loop through each value and decode
            for value in values:
                z_new = z.clone()
                z_new[0, dim] = value
                decoded = self.decode(z_new)
                traversal_images.append(decoded.squeeze().cpu())
                
            return traversal_images
            
    def interpolate(self, img1, img2, steps=10):
        """
        Performs latent space interpolation between two input images.
        
        Args:
            img1: First input image
            img2: Second input image
            steps: Number of interpolation steps (including endpoints)
            
        Returns:
            List of decoded images from the interpolation
        """
        with torch.no_grad():
            # Encode both images to get their latent representations
            mu1, _ = self.encode(img1.unsqueeze(0))
            mu2, _ = self.encode(img2.unsqueeze(0))
            
            # Use means directly (no sampling) for smooth interpolation
            z1 = mu1
            z2 = mu2
            
            # Create interpolation steps
            interpolation_images = []
            alphas = np.linspace(0, 1, steps)
            
            # Generate and decode each interpolation point
            for alpha in alphas:
                z_interp = (1-alpha) * z1 + alpha * z2
                decoded = self.decode(z_interp)
                interpolation_images.append(decoded.squeeze().cpu())
                
            return interpolation_images

# Discriminator class for GAN component (from script B - pixel-space discriminator)
class Discriminator(nn.Module):
    def __init__(self, img_size=64, hidden_dims=None):
        super(Discriminator, self).__init__()
        
        # Default architecture if hidden_dims is not provided
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]  # Same as encoder for simplicity
        
        modules = []
        in_channels = 1  # Grayscale images
        
        # Build discriminator network (similar to encoder but with different output)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2))
            )
            in_channels = h_dim
        
        self.features = nn.Sequential(*modules)
        
        # Calculate the size of the feature maps before flattening
        encoder_output_size = img_size // (2 ** len(hidden_dims))
        encoder_output_dim = hidden_dims[-1] * encoder_output_size * encoder_output_size
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_output_dim, 1),
            nn.Sigmoid()  # Output probability that the image is real
        )
        
    def forward(self, img):
        features = self.features(img)
        validity = self.classifier(features)
        return validity

def vae_gan_loss(recon_x, x, mu, log_var, d_recon, d_samples, kld_weight=0.005, adv_weight=1.0, recon_sample_weight=0.5):
    """
    VAE-GAN loss function with:
    - Reconstruction loss (MSE)
    - KL Divergence
    - Adversarial loss from discriminator (weighted between reconstructions and samples)
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean vector from encoder
        log_var: Log variance vector from encoder
        d_recon: Discriminator output for reconstructions
        d_samples: Discriminator output for samples from random noise
        kld_weight: Weight for KL divergence term
        adv_weight: Weight for adversarial loss term
        recon_sample_weight: Weight for reconstruction (1-weight for samples) in adversarial loss
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence loss
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Adversarial loss for generator with weighted reconstructions vs samples
    adv_recon_loss = F.binary_cross_entropy(d_recon, torch.ones_like(d_recon))
    adv_samples_loss = F.binary_cross_entropy(d_samples, torch.ones_like(d_samples))
    adv_loss = recon_sample_weight * adv_recon_loss + (1.0 - recon_sample_weight) * adv_samples_loss
    
    # Total loss for VAE (generator)
    vae_loss = recon_loss + kld_weight * kld_loss + adv_weight * adv_loss
    
    return vae_loss, recon_loss, kld_loss, adv_loss

def discriminator_loss(d_real, d_fake_recon, d_fake_samples, recon_sample_weight=0.5):
    """
    Extended GAN discriminator loss that handles both reconstructions and samples
    with configurable weighting
    
    Args:
        d_real: Discriminator output for real images
        d_fake_recon: Discriminator output for reconstructed images
        d_fake_samples: Discriminator output for samples from random noise
        recon_sample_weight: Weight for reconstruction (1-weight for samples) in fake loss
    """
    real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
    fake_recon_loss = F.binary_cross_entropy(d_fake_recon, torch.zeros_like(d_fake_recon))
    fake_samples_loss = F.binary_cross_entropy(d_fake_samples, torch.zeros_like(d_fake_samples))
    
    # Weighted fake loss
    fake_loss = recon_sample_weight * fake_recon_loss + (1.0 - recon_sample_weight) * fake_samples_loss
    d_loss = real_loss + fake_loss
    
    return d_loss

def train_vae_gan(vae_model, discriminator, train_loader, vae_optimizer, d_optimizer, 
                  epochs, kld_weight=0.005, adv_weight=1.0, recon_sample_weight=0.5,
                  save_path="vae_gan_model.pth"):
    """
    Train the VAE-GAN model with both reconstruction and random sample discrimination
    
    Args:
        vae_model: The VAE model
        discriminator: The discriminator model
        train_loader: DataLoader for training data
        vae_optimizer: Optimizer for VAE
        d_optimizer: Optimizer for discriminator
        epochs: Number of training epochs
        kld_weight: Weight for KL divergence term
        adv_weight: Weight for adversarial loss term
        recon_sample_weight: Weight for reconstruction vs sample discrimination (0-1)
        save_path: Path to save model checkpoints
    """
    if not os.path.exists("checkpoints/"):
        os.makedirs("checkpoints/")

    vae_model.train()
    discriminator.train()
    
    # Lists to track all loss components
    total_losses = []
    recon_losses = []
    kld_losses = []
    adv_losses = []
    disc_losses = []
    
    print(f"Training with reconstruction-sample weight: {recon_sample_weight:.2f}")
    
    for epoch in range(epochs):
        epoch_vae_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0
        epoch_adv_loss = 0
        epoch_d_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, data in progress_bar:
            batch_size = data.size(0)
            data = data.to(device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            d_optimizer.zero_grad()
            
            # Real images
            d_real = discriminator(data)
            
            # Fake images - both reconstructions and samples from random noise
            with torch.no_grad():  # Don't backprop through generator here
                # Get reconstructions
                recon_batch, _, _, _ = vae_model(data)
                
                # Get samples from random noise
                z_random = torch.randn(batch_size, vae_model.latent_dim).to(device)
                fake_samples = vae_model.decode(z_random)
            
            # Evaluate both types of fake images
            d_fake_recon = discriminator(recon_batch.detach())
            d_fake_samples = discriminator(fake_samples.detach())
            
            # Discriminator loss with weighting
            d_loss = discriminator_loss(d_real, d_fake_recon, d_fake_samples, recon_sample_weight)
            d_loss.backward()
            d_optimizer.step()
            
            # ---------------------
            # Train VAE (Generator)
            # ---------------------
            vae_optimizer.zero_grad()
            
            # Generate reconstructed images
            recon_batch, mu, log_var, _ = vae_model(data)
            
            # Generate samples from random noise
            z_random = torch.randn(batch_size, vae_model.latent_dim).to(device)
            fake_samples = vae_model.decode(z_random)
            
            # Discriminator output for both types of generated images
            d_fake_recon = discriminator(recon_batch)
            d_fake_samples = discriminator(fake_samples)
            
            # VAE-GAN loss with weighting
            loss, recon_loss, kld_loss, adv_loss = vae_gan_loss(
                recon_batch, data, mu, log_var, d_fake_recon, d_fake_samples, 
                kld_weight, adv_weight, recon_sample_weight
            )
            
            loss.backward()
            vae_optimizer.step()
            
            # Update statistics
            epoch_vae_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kld_loss += kld_loss.item()
            epoch_adv_loss += adv_loss.item()
            epoch_d_loss += d_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'vae_loss': loss.item() / batch_size,
                'recon_loss': recon_loss.item() / batch_size,
                'kld_loss': kld_loss.item() / batch_size,
                'adv_loss': adv_loss.item() / batch_size,
                'd_loss': d_loss.item() / batch_size
            })
        
        # Average losses for the epoch
        avg_vae_loss = epoch_vae_loss / len(train_loader.dataset)
        avg_recon_loss = epoch_recon_loss / len(train_loader.dataset)
        avg_kld_loss = epoch_kld_loss / len(train_loader.dataset)
        avg_adv_loss = epoch_adv_loss / len(train_loader.dataset)
        avg_d_loss = epoch_d_loss / len(train_loader.dataset)

        # Store all loss components
        total_losses.append(avg_vae_loss)
        recon_losses.append(avg_recon_loss)
        kld_losses.append(avg_kld_loss)
        adv_losses.append(avg_adv_loss)
        disc_losses.append(avg_d_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average VAE Loss: {avg_vae_loss:.4f}")
        print(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"Average KLD Loss: {avg_kld_loss:.4f}")
        print(f"Average Adversarial Loss: {avg_adv_loss:.4f}")
        print(f"Average Discriminator Loss: {avg_d_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'vae_model_state_dict': vae_model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'loss': avg_vae_loss,
                'recon_sample_weight': recon_sample_weight
            }, save_path)
            print(f"Checkpoint saved to {save_path}")
    
    # Return all loss components for plotting
    return {
        'total': total_losses,
        'recon': recon_losses,
        'kld': kld_losses,
        'adv': adv_losses,
        'disc': disc_losses
    }

def visualize_reconstructions(model, data_loader, num_images=10, save_dir="output"):
    """
    Visualize original images and their reconstructions
    """
    model.eval()
    
    # Get a batch of images
    dataiter = iter(data_loader)
    images = next(dataiter)[:num_images].to(device)
    
    with torch.no_grad():
        recon_images, _, _, _ = model.reconstruct(images)
    
    # Plot original and reconstructed images
    plt.figure(figsize=(20, 4))
    
    # Original images
    for i in range(num_images):
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].cpu().squeeze().numpy(), cmap='gray')
        plt.title("Original")
        plt.axis('off')
    
    # Reconstructed images
    for i in range(num_images):
        ax = plt.subplot(2, num_images, i + num_images + 1)
        plt.imshow(recon_images[i].cpu().squeeze().numpy(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"reconstructions.png"))
    plt.close()
    print(f"Saved reconstructions to {save_dir}")

def visualize_latent_traversal(model, data_loader, dim=0, num_dims=5, save_dir="output"):
    """
    Visualize latent space traversal for multiple dimensions
    """
    lt_dir = os.path.join(save_dir, "latent_traversals")
    if not os.path.exists(lt_dir):
        os.makedirs(lt_dir)

    model.eval()
    
    # Get a sample image
    dataiter = iter(data_loader)
    image = next(dataiter)[0].to(device)
    
    # For multiple dimensions
    for d in range(dim, dim + num_dims):
        if d >= model.latent_dim:
            break
            
        traversal_images = model.latent_traversal(image, d, start=-3, end=3, steps=10)
        
        # Plot traversal
        plt.figure(figsize=(20, 3))
        for i, img in enumerate(traversal_images):
            ax = plt.subplot(1, len(traversal_images), i + 1)
            plt.imshow(img.numpy(), cmap='gray')
            plt.title(f"z_{d}={-3 + i*0.6:.1f}")
            plt.axis('off')
        
        plt.suptitle(f"Latent Traversal - Dimension {d}")
        plt.tight_layout()
        plt.savefig(os.path.join(lt_dir, f"latent_traversal_dim_{d}.png"))
        plt.close()
        
    print(f"Saved latent traversal visualizations to {lt_dir}")

def visualize_interpolation_between_files(model, dataset, img1_file, img2_file, steps=10, save_dir="output"):
    """
    Visualize interpolation between two specific images identified by filename
    
    Args:
        model: The VAE model
        dataset: The dataset containing the images
        img1_file: Filename of the first image
        img2_file: Filename of the second image
        steps: Number of interpolation steps
        save_dir: Directory to save the visualization
    """
    model.eval()
    
    try:
        # Get the two specified images
        img1 = dataset.get_image_by_filename(img1_file).to(device)
        img2 = dataset.get_image_by_filename(img2_file).to(device)
        
        # Generate interpolation
        interp_images = model.interpolate(img1, img2, steps=steps)
        
        # Plot interpolation
        plt.figure(figsize=(20, 4))
        
        # Add original images at the top with filenames
        plt.subplot(2, steps, 1)
        plt.imshow(img1.cpu().squeeze().numpy(), cmap='gray')
        plt.title(f"Image 1\n{img1_file}")
        plt.axis('off')
        
        plt.subplot(2, steps, steps)
        plt.imshow(img2.cpu().squeeze().numpy(), cmap='gray')
        plt.title(f"Image 2\n{img2_file}")
        plt.axis('off')
        
        # Add interpolated images
        for i, img in enumerate(interp_images):
            ax = plt.subplot(2, steps, steps + i + 1)
            plt.imshow(img.numpy(), cmap='gray')
            plt.title(f"α={i/(steps-1):.1f}")
            plt.axis('off')
        
        plt.suptitle(f"Latent Space Interpolation Between {img1_file} and {img2_file}")
        plt.tight_layout()
        output_name = f"interpolation_{os.path.splitext(img1_file)[0]}_{os.path.splitext(img2_file)[0]}.png"
        plt.savefig(os.path.join(save_dir, output_name))
        plt.close()
        
        print(f"Saved interpolation visualization to {os.path.join(save_dir, output_name)}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Available files in the dataset:")
        for i, filename in enumerate(dataset.get_filenames()):
            print(f"{i}: {filename}")
        return

def extract_latent_vectors(model, data_loader, save_dir="output"):
    """
    Extract and save the latent vectors (mean and log variance) for all images
    """
    model.eval()
    
    all_mu = []
    all_log_var = []
    all_filenames = []
    
    with torch.no_grad():
        for batch_idx, (data) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Extracting latent vectors"):
            data = data.to(device)
            mu, log_var = model.encode(data)
            
            all_mu.append(mu.cpu().numpy())
            all_log_var.append(log_var.cpu().numpy())
            
            # Get filenames for this batch (if available in your dataset)
            # Note: This assumes your dataloader provides filenames. Modify according to your implementation.
            # all_filenames.extend([data_loader.dataset.img_files[i] for i in range(batch_idx*data_loader.batch_size, 
            #                       min((batch_idx+1)*data_loader.batch_size, len(data_loader.dataset)))])
    
    # Concatenate all batches
    all_mu = np.concatenate(all_mu, axis=0)
    all_log_var = np.concatenate(all_log_var, axis=0)
    
    # Save as numpy arrays
    np.save(os.path.join(save_dir, "latent_mu.npy"), all_mu)
    np.save(os.path.join(save_dir, "latent_log_var.npy"), all_log_var)
    
    # Save filenames if available
    if all_filenames:
        np.save(os.path.join(save_dir, "filenames.npy"), np.array(all_filenames))
    
    print(f"Saved latent vectors to {save_dir}")
    print(f"mu shape: {all_mu.shape}, log_var shape: {all_log_var.shape}")
    
    return all_mu, all_log_var

def main(args):

    # Make the folder to save all outputs
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    out_dir = os.path.join("outputs", f"{args.dataset}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    train_dataset = VehicleDataset(img_dir=args.data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Create VAE model
    vae_model = VAE(img_size=args.img_size, latent_dim=args.latent_dim).to(device)
    
    # Create Discriminator (pixel-space from script B)
    discriminator = Discriminator(img_size=args.img_size).to(device)
    print("Using VAE-GAN architecture with pixel-space discrimination")
    
    # Count and print model parameters
    vae_params = sum(p.numel() for p in vae_model.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Total number of VAE parameters: {vae_params:,}")
    print(f"Total number of Discriminator parameters: {disc_params:,}")
    print(f"Total model parameters: {vae_params + disc_params:,}")
    
    # Define optimizers
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=args.learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate * 0.5)
    
    # If resuming from checkpoint
    if args.resume and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path)
        
        # Load VAE-GAN checkpoint
        vae_model.load_state_dict(checkpoint['vae_model_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    if args.train:
        # Train VAE-GAN with weighted reconstruction vs sample discrimination
        losses = train_vae_gan(
            vae_model, discriminator, train_loader, vae_optimizer, d_optimizer,
            args.epochs, kld_weight=args.kld_weight, adv_weight=args.adv_weight, 
            recon_sample_weight=args.recon_sample_weight,
            save_path=args.model_path
        )
        
        # Plot all training losses
        plt.figure(figsize=(12, 7))
        # Plot total loss with thicker line and different color and component losses on log scale
        plt.semilogy(losses['total'], label='Total Loss', linewidth=2.5, color='black')
        plt.semilogy(losses['recon'], label='Reconstruction Loss', alpha=0.7)
        plt.semilogy(losses['kld'], label='KL Divergence Loss', alpha=0.7)
        plt.semilogy(losses['adv'], label='Adversarial Loss', alpha=0.7)
        plt.semilogy(losses['disc'], label='Discriminator Loss', alpha=0.7, linestyle='--')
        # Put it all together
        plt.title('VAE-GAN Training Log-Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Log-Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'training_losses.png'), dpi=300)
        plt.close()
    
    # Load best model for evaluation
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path)
        vae_model.load_state_dict(checkpoint['vae_model_state_dict'])
        print(f"Loaded model from {args.model_path}")
    
    if args.visualize:
        # Visualize reconstructions
        visualize_reconstructions(vae_model, train_loader, num_images=10, save_dir=out_dir)
        
        # Visualize latent space traversal
        visualize_latent_traversal(vae_model, train_loader, dim=0, num_dims=args.latent_dim, save_dir=out_dir)
    
    if args.interpolate:
        # Visualize interpolation between specific files
        visualize_interpolation_between_files(vae_model, train_dataset, 
                                            args.interpolate[0], 
                                            args.interpolate[1],
                                            steps=args.interpolate_steps, 
                                            save_dir=out_dir)
    
    if args.extract_latent:
        # Extract and save latent vectors
        mu, log_var = extract_latent_vectors(vae_model, train_loader, save_dir=out_dir)
    
    if args.sample:
        # Generate random samples
        with torch.no_grad():
            samples = vae_model.sample(num_samples=25)
            
        # Display samples
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.imshow(samples[i].cpu().squeeze().numpy(), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'samples.png'))
        plt.close()
        print(f"Saved random samples to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE-GAN for Vehicle Images with weighted reconstruction vs sample discrimination')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/evox_256x256_1-3', help='Directory containing the dataset')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--dataset', type=str, default='Unspecified-Dataset', help='The properties describing evox: e.g. 64x64_1')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--kld_weight', type=float, default=0.005, help='Weight of KLD loss term')
    parser.add_argument('--adv_weight', type=float, default=1.0, help='Weight of adversarial loss term')
    parser.add_argument('--recon_sample_weight', type=float, default=0.7, 
                        help='Weight for reconstruction vs sample discrimination (default: 0.7, meaning 70% focus on reconstructions, 30% on samples)')
    
    # Training parameters
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    # Output parameters
    parser.add_argument('--model_path', type=str, default='checkpoints/motorVAEGAN_combined_256x256_1-3.pth', help='Path to save/load model')
    
    # Actions
    parser.add_argument('--visualize', action='store_true', help='Visualize reconstructions and latent space')
    parser.add_argument('--extract_latent', action='store_true', help='Extract and save latent vectors')
    parser.add_argument('--sample', action='store_true', help='Generate random samples from the latent space')
    parser.add_argument('--interpolate', nargs=2, metavar=('FILE1', 'FILE2'), 
                        help='Specify two image filenames to interpolate between')
    parser.add_argument('--interpolate_steps', type=int, default=10,
                        help='Number of steps for interpolation (default: 10)')
    
    args = parser.parse_args()
    
    main(args)
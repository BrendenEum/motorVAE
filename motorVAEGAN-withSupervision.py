import os
import time
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
import pandas as pd

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SupervisedVehicleDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, label_cols=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
        # Load labels from CSV
        self.labels_df = pd.read_csv(label_file)
        
        # Filter to only include images that exist in the directory
        self.labels_df = self.labels_df[self.labels_df['Filename'].isin(self.img_files)]
        
        # Keep only existing image files that have labels
        self.img_files = [f for f in self.img_files if f in self.labels_df['Filename'].values]
        
        # Select which label columns to use for classification
        if label_cols:
            self.label_cols = label_cols
        else:  # Default to all label columns except filename
            self.label_cols = [col for col in self.labels_df.columns if col != 'Filename']
        
        print(f"Dataset initialized with {len(self.img_files)} images and {len(self.label_cols)} labels: {self.label_cols}")
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels for this image
        label_row = self.labels_df[self.labels_df['Filename'] == img_file]
        label_values = label_row[self.label_cols].values[0]
        
        # Convert string labels to numerical indices using label encoders
        labels = torch.tensor([self.get_label_index(col, val) for col, val in zip(self.label_cols, label_values)], dtype=torch.long)
        
        return image, labels
    
    def get_label_index(self, col_name, value):
        """Convert string label to numerical index"""
        # Get all unique values for this column
        unique_values = self.labels_df[col_name].unique()
        # Return index of the value in the sorted list of unique values
        return np.where(sorted(unique_values) == value)[0][0]
    
    def get_num_classes(self, col_name):
        """Get number of unique classes for a specific label column"""
        return len(self.labels_df[col_name].unique())
    
    def get_all_num_classes(self):
        """Get number of unique classes for each label column"""
        return {col: self.get_num_classes(col) for col in self.label_cols}
    
    def get_image_by_filename(self, filename):
        """Get an image by its filename"""
        if filename in self.img_files:
            img_path = os.path.join(self.img_dir, filename)
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            
            if self.transform:
                image = self.transform(image)
                
            return image
        else:
            raise ValueError(f"File {filename} not found in dataset")
            
    def get_filenames(self):
        """Return all filenames in the dataset"""
        return self.img_files

class VAEWithClassifier(nn.Module):
    def __init__(self, img_size=64, latent_dim=128, hidden_dims=None, num_classes_dict=None):
        super(VAEWithClassifier, self).__init__()
        
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.device = device
        self.num_classes_dict = num_classes_dict  # Dictionary mapping label names to class counts
        
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
        
        # Build Classifiers (one for each label type)
        self.classifiers = nn.ModuleDict()
        if num_classes_dict:
            for label_name, num_classes in num_classes_dict.items():
                self.classifiers[label_name] = nn.Sequential(
                    nn.Linear(latent_dim * 2, 256),  # Use both mu and log_var for classification
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
        
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
        log_var = torch.clamp(log_var, min=-88, max=88)  # prevent exp overflow/underflow
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def classify(self, mu, log_var):
        """
        Classify the latent representation into different labels
        Returns a dictionary of logits for each label type
        """
        # Concatenate mu and log_var for classification
        z_for_cls = torch.cat([mu, log_var], dim=1)
        
        # Apply each classifier
        logits = {}
        for label_name, classifier in self.classifiers.items():
            logits[label_name] = classifier(z_for_cls)
            
        return logits
    
    def forward(self, input, compute_loss=False):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        
        # Get classification logits
        logits = self.classify(mu, log_var)
        
        return x_recon, mu, log_var, z, logits
    
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

# Discriminator class for GAN component
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

def vae_gan_classification_loss(recon_x, x, mu, log_var, logits, labels, d_recon, d_samples, 
                             kld_weight=0.005, adv_weight=1.0, cls_weight=1.0, recon_sample_weight=0.5,
                             mi_weight=1.0, tc_weight=1.0, dwkl_weight=1.0):
    """
    Supervised VAE-GAN loss function with classification loss:
    - Reconstruction loss (L1)
    - KL Divergence decomposed into three terms
    - Adversarial loss from discriminator
    - Classification loss for each label type
    """
    batch_size = x.size(0)
    tiny_amt = 1e-8  # for numerical stability

    ######################
    # 1. Reconstruction loss
    ######################
    recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    
    ######################
    # 2. KL Divergence components (same as before)
    ######################
    # Reparameterization trick: Sample z from q(z|x)
    log_var = torch.clamp(log_var, min=-88, max=88)  # prevent exp overflow/underflow
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + eps * std
    
    # Compute log q(z|x) for each data point
    log_qz_x = -0.5 * torch.sum(log_var + torch.pow(z - mu, 2) / (torch.exp(log_var) + tiny_amt), dim=1)
    
    # Compute log q(z) as log of average of q(z|x) over batch
    z_expanded = z.unsqueeze(1)  # Shape: [B, 1, D]
    mu_expanded = mu.unsqueeze(0)  # Shape: [1, B, D]
    logvar_expanded = log_var.unsqueeze(0)  # Shape: [1, B, D]
    
    # Compute log q(z|x) for each combination
    log_qz_cross = -0.5 * torch.sum(
        logvar_expanded + torch.pow(z_expanded - mu_expanded, 2) / (torch.exp(logvar_expanded) + tiny_amt), dim=2)
    
    # Compute log q(z) as logsumexp over batch dimension
    log_qz = torch.logsumexp(log_qz_cross, dim=1) - torch.log(torch.tensor(batch_size, dtype=torch.float, device=device))
    
    # Mutual information loss
    mi_loss = torch.mean(log_qz_x - log_qz)
    
    # Total Correlation Loss
    z_perm = z.reshape(batch_size, -1, 1)
    z_perm = z_perm.expand(-1, -1, batch_size)
    z_perm = z_perm.transpose(0, 2)
    z_perm = z_perm.reshape(batch_size * batch_size, -1)
    
    mu_perm = mu.repeat(batch_size, 1)
    logvar_perm = log_var.repeat(batch_size, 1)
    
    log_qzj_xi = -0.5 * (logvar_perm + torch.pow(z_perm - mu_perm, 2) / (torch.exp(logvar_perm) + tiny_amt))
    log_qzj_xi = log_qzj_xi.reshape(batch_size, batch_size, -1)
    
    log_qzj = torch.logsumexp(log_qzj_xi, dim=1) - torch.log(torch.tensor(batch_size, dtype=torch.float, device=device))
    
    log_qz_product = torch.sum(log_qzj, dim=1)
    
    tc_loss = torch.mean(log_qz - log_qz_product)
    
    # Dimension-wise KL Divergence
    log_pz = -0.5 * torch.sum(torch.pow(z, 2), dim=1)
    log_pz_product = -0.5 * torch.sum(torch.pow(z, 2), dim=1)
    
    dwkl_loss = torch.mean(log_qz_product - log_pz_product)
    
    # Weighted sum of the three KL terms
    kld_loss = mi_weight * mi_loss + tc_weight * tc_loss + dwkl_weight * dwkl_loss
    
    ######################
    # 3. Adversarial loss (same as before)
    ######################
    d_recon = torch.clamp(d_recon, min=tiny_amt, max=1-tiny_amt)
    d_samples = torch.clamp(d_samples, min=tiny_amt, max=1-tiny_amt)

    adv_recon_loss = F.binary_cross_entropy(d_recon, torch.ones_like(d_recon))
    adv_samples_loss = F.binary_cross_entropy(d_samples, torch.ones_like(d_samples))
    adv_loss = recon_sample_weight * adv_recon_loss + (1.0 - recon_sample_weight) * adv_samples_loss
    
    ######################
    # 4. Classification loss (NEW)
    ######################
    cls_loss = 0
    cls_losses = {}
    
    for i, (label_name, pred) in enumerate(logits.items()):
        # Get label for this class type
        label = labels[:, i]
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(pred, label)
        cls_losses[label_name] = loss.item()
        cls_loss += loss
    
    ######################
    # Total loss for VAE (generator)
    ######################
    vae_loss = recon_loss + kld_weight * kld_loss + adv_weight * adv_loss + cls_weight * cls_loss
    
    return vae_loss, recon_loss, kld_loss, mi_loss, tc_loss, dwkl_loss, adv_loss, cls_loss, cls_losses

def train_supervised_vaegan(vae_model, discriminator, train_loader, dataset, target_recon_img,
                          vae_optimizer, d_optimizer, epochs, kld_scheduler_fn, kld_scheduler_params,
                          adv_weight=1.0, cls_weight=1.0, recon_sample_weight=0.7,
                          checkpoint_path="supervised_model.pth", recon_path="outputs/supervised"):
    """
    Train the supervised VAE-GAN model
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
    cls_losses = []
    disc_losses = []
    kld_weights = []
    
    # Create per-label classification loss trackers
    label_cls_losses = {label_name: [] for label_name in vae_model.num_classes_dict.keys()}
    
    for epoch in range(epochs):
        # Get the scheduled KLD weight for this epoch
        current_kld_weight = kld_scheduler_fn(epoch, **kld_scheduler_params)
        kld_weights.append(current_kld_weight)
        
        print(f"\nEpoch {epoch+1}/{epochs}, KLD weight: {current_kld_weight:.5f}")
        
        epoch_vae_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0
        epoch_adv_loss = 0
        epoch_cls_loss = 0
        epoch_d_loss = 0
        
        # Per-label classification loss accumulators
        epoch_label_cls_losses = {label: 0 for label in vae_model.num_classes_dict.keys()}
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, labels) in progress_bar:
            batch_size = data.size(0)
            data = data.to(device)
            labels = labels.to(device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            d_optimizer.zero_grad()
            
            # Real images
            d_real = discriminator(data)
            
            # Fake images - both reconstructions and samples
            with torch.no_grad():
                # Get reconstructions
                recon_batch, _, _, _, _ = vae_model(data)
                
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
            
            # Generate reconstructed images and classification logits
            recon_batch, mu, log_var, _, logits = vae_model(data)
            
            # Generate samples from random noise
            z_random = torch.randn(batch_size, vae_model.latent_dim).to(device)
            fake_samples = vae_model.decode(z_random)
            
            # Discriminator output for both types of generated images
            d_fake_recon = discriminator(recon_batch)
            d_fake_samples = discriminator(fake_samples)
            
            # VAE-GAN loss with classification loss
            (loss, recon_loss, kld_loss, mi_loss, tc_loss, dwkl_loss, 
             adv_loss, cls_loss, label_losses) = vae_gan_classification_loss(
                recon_batch, data, mu, log_var, logits, labels, d_fake_recon, d_fake_samples,
                current_kld_weight, adv_weight, cls_weight, recon_sample_weight,
                mi_weight=args.mi_weight, tc_weight=args.tc_weight, dwkl_weight=args.dwkl_weight)
            
            loss.backward()
            vae_optimizer.step()
            
            # Update statistics
            epoch_vae_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kld_loss += kld_loss.item()
            epoch_adv_loss += adv_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_d_loss += d_loss.item()
            
            # Update per-label classification losses
            for label_name, label_loss in label_losses.items():
                epoch_label_cls_losses[label_name] += label_loss
            
            # Update progress bar
            progress_bar.set_postfix({
                'vae_loss': loss.item() / batch_size,
                'recon_loss': recon_loss.item() / batch_size,
                'kld_loss': kld_loss.item() / batch_size,
                'cls_loss': cls_loss.item() / batch_size,
                'adv_loss': adv_loss.item() / batch_size,
                'd_loss': d_loss.item() / batch_size
            })
        
        # Average losses for the epoch
        avg_vae_loss = epoch_vae_loss / len(train_loader.dataset)
        avg_recon_loss = epoch_recon_loss / len(train_loader.dataset)
        avg_kld_loss = epoch_kld_loss / len(train_loader.dataset)
        avg_adv_loss = epoch_adv_loss / len(train_loader.dataset)
        avg_cls_loss = epoch_cls_loss / len(train_loader.dataset)
        avg_d_loss = epoch_d_loss / len(train_loader.dataset)
        
        # Average per-label classification losses
        avg_label_cls_losses = {label: loss / len(train_loader.dataset) 
                               for label, loss in epoch_label_cls_losses.items()}

        # Store all loss components
        total_losses.append(avg_vae_loss)
        recon_losses.append(avg_recon_loss)
        kld_losses.append(avg_kld_loss)
        adv_losses.append(avg_adv_loss)
        cls_losses.append(avg_cls_loss)
        disc_losses.append(avg_d_loss)
        
        # Store per-label classification losses
        for label, loss in avg_label_cls_losses.items():
            label_cls_losses[label].append(loss)
        
        print(f"Average VAE Loss: {avg_vae_loss:.4f}")
        print(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"Average KLD Loss: {avg_kld_loss:.4f} (raw: {avg_kld_loss/current_kld_weight:.4f})")
        print(f"Average Classification Loss: {avg_cls_loss:.4f}")
        for label, loss in avg_label_cls_losses.items():
            print(f"  - {label}: {loss:.4f}")
        print(f"Average Adversarial Loss: {avg_adv_loss:.4f}")
        print(f"Average Discriminator Loss: {avg_d_loss:.4f}")
        
        # Track reconstruction of target image at the current epoch
        if target_recon_img != "-unspecified-":
            track_reconstruction_across_epochs(vae_model, dataset, target_recon_img, epoch+1, save_dir=recon_path)
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'vae_model_state_dict': vae_model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'vae_optimizer_state_dict': vae_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'loss': avg_vae_loss,
                'kld_weight': current_kld_weight,
                'cls_weight': cls_weight,
                'recon_sample_weight': recon_sample_weight,
                'label_cols': dataset.label_cols,
                'num_classes_dict': vae_model.num_classes_dict
            }, os.path.join("checkpoints", checkpoint_path))
            print(f"Checkpoint saved to checkpoints/{checkpoint_path}")
    
    # Return all loss components
    return {
        'total': total_losses,
        'recon': recon_losses,
        'kld': kld_losses,
        'cls': cls_losses,
        'adv': adv_losses,
        'disc': disc_losses,
        'kld_weights': kld_weights,
        'label_cls_losses': label_cls_losses
    }

def discriminator_loss(d_real, d_fake_recon, d_fake_samples, recon_sample_weight=0.5):
    """
    Extended GAN discriminator loss that handles both reconstructions and samples
    with configurable weighting
    """
    # Clamp values slightly away from 0 and 1
    tiny_amt = 1e-8
    d_real = torch.clamp(d_real, min=tiny_amt, max=1-tiny_amt)
    d_fake_recon = torch.clamp(d_fake_recon, min=tiny_amt, max=1-tiny_amt)
    d_fake_samples = torch.clamp(d_fake_samples, min=tiny_amt, max=1-tiny_amt)

    real_loss = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
    fake_recon_loss = F.binary_cross_entropy(d_fake_recon, torch.zeros_like(d_fake_recon))
    fake_samples_loss = F.binary_cross_entropy(d_fake_samples, torch.zeros_like(d_fake_samples))
    
    # Weighted fake loss
    fake_loss = recon_sample_weight * fake_recon_loss + (1.0 - recon_sample_weight) * fake_samples_loss
    d_loss = real_loss + fake_loss
    
    return d_loss

def kld_weight_scheduler(epoch, total_epochs=112, min_weight=0.01, max_weight=0.2, 
                       warmup_epochs=15, schedule_type="linear"):
    """
    A flexible KL divergence weight scheduler.
    """
    # Initial warmup period - keep weight low to establish good reconstruction
    if epoch < warmup_epochs:
        return min_weight
    
    # Calculate progress after warmup period
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)  # Clamp between 0 and 1
    
    if schedule_type == "linear":
        # Linear increase from min to max
        weight = min_weight + progress * (max_weight - min_weight)
        
    elif schedule_type == "step":
        # Step increase at 25%, 50%, and 75% of training
        if progress < 0.25:
            weight = min_weight
        elif progress < 0.5:
            weight = min_weight + (max_weight - min_weight) * 0.33
        elif progress < 0.75:
            weight = min_weight + (max_weight - min_weight) * 0.66
        else:
            weight = max_weight
            
    elif schedule_type == "exp":
        # Exponential increase (slower at first, faster later)
        weight = min_weight + (max_weight - min_weight) * (progress ** 2)
        
    elif schedule_type == "cyclical":
        # Cyclical schedule with 4 cycles
        cycles = 4
        cycle_length = (total_epochs - warmup_epochs) / cycles
        cycle_position = ((epoch - warmup_epochs) % cycle_length) / cycle_length
        
        if cycle_position < 0.5:
            # First half of cycle: linear increase
            cycle_progress = cycle_position * 2
            weight = min_weight + (max_weight - min_weight) * cycle_progress
        else:
            # Second half of cycle: maintain high weight
            weight = max_weight
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return weight

def plot_supervised_training_losses(losses, out_dir):
    """
    Plot training losses including classification loss
    """
    # Plot VAE losses
    plt.figure(figsize=(12, 16))
    
    # VAE Losses
    plt.subplot(4, 1, 1)
    plt.semilogy(losses['total'], label='Total Loss', linewidth=2.5, color='black')
    plt.semilogy(losses['recon'], label='Reconstruction Loss', alpha=0.7)
    plt.semilogy(losses['kld'], label='KL Divergence Loss', alpha=0.7)
    plt.semilogy(losses['adv'], label='Adversarial Loss', alpha=0.7)
    plt.semilogy(losses['cls'], label='Classification Loss', alpha=0.7, color='red')
    plt.title('VAE-GAN with Classification Losses')
    plt.ylabel('Log-Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Discriminator Loss
    plt.subplot(4, 1, 2)
    plt.semilogy(losses['disc'], label='Discriminator Loss', linewidth=2.5, color='purple')
    plt.title('Discriminator Loss')
    plt.ylabel('Log-Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Per-Label Classification Losses
    plt.subplot(4, 1, 3)
    for label, loss_values in losses['label_cls_losses'].items():
        plt.semilogy(loss_values, label=f'{label} Loss', alpha=0.8)
    plt.title('Per-Label Classification Losses')
    plt.ylabel('Log-Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # KLD Weight Schedule
    plt.subplot(4, 1, 4)
    plt.plot(losses['kld_weights'], linewidth=2, color='red')
    plt.title('KL Divergence Weight Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('KLD Weight')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'supervised_training_losses.png'), dpi=300)
    plt.close()

def evaluate_classification_accuracy(model, data_loader):
    """
    Evaluate classification accuracy on the dataset
    """
    model.eval()
    
    correct_preds = {label: 0 for label in model.num_classes_dict.keys()}
    total_samples = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            total_samples += data.size(0)
            
            # Forward pass to get logits
            _, mu, log_var, _, logits = model(data)
            
            # Calculate accuracy for each label
            for i, (label_name, pred) in enumerate(logits.items()):
                # Get ground truth label for this category
                target = labels[:, i]
                
                # Get predictions
                pred_classes = torch.argmax(pred, dim=1)
                
                # Count correct predictions
                correct_preds[label_name] += (pred_classes == target).sum().item()
    
    # Calculate accuracy for each label type
    accuracies = {label: correct / total_samples for label, correct in correct_preds.items()}
    overall_accuracy = sum(correct_preds.values()) / (total_samples * len(model.num_classes_dict))
    
    return accuracies, overall_accuracy

def track_reconstruction_across_epochs(vae_model, dataset, img_name, epoch, save_dir="outputs"):
    """
    Save reconstruction of a specific image at the current epoch
    """
    # Create the reconstructions directory if it doesn't exist
    recon_dir = os.path.join(save_dir, "reconstructions_epochs")
    if not os.path.exists(recon_dir):
        os.makedirs(recon_dir)
    
    # Set model to evaluation mode
    vae_model.eval()
    
    try:
        # Get the specified image
        image = dataset.get_image_by_filename(img_name).to(device)
        
        # Generate reconstruction
        with torch.no_grad():
            recon_image, _, _, _ = vae_model.reconstruct(image.unsqueeze(0))
        
        # Plot original and reconstructed images side by side
        plt.figure(figsize=(8, 4))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image.cpu().squeeze().numpy(), cmap='gray')
        plt.title(f"Original\n{img_name}")
        plt.axis('off')
        
        # Reconstructed image
        plt.subplot(1, 2, 2)
        plt.imshow(recon_image.cpu().squeeze().numpy(), cmap='gray')
        plt.title(f"Reconstruction\nEpoch {epoch}")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save with epoch number in filename
        output_name = f"epoch_{epoch:03d}_recon_{os.path.splitext(img_name)[0]}.png"
        plt.savefig(os.path.join(recon_dir, output_name))
        plt.close()
        
        print(f"Saved epoch {epoch} reconstruction to {os.path.join(recon_dir, output_name)}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Available files in the dataset:")
        for i, filename in enumerate(dataset.get_filenames()[:10]):
            print(f"{i}: {filename}")
        return

def visualize_reconstructions(model, data_loader, num_images=10, save_dir="output"):
    """
    Visualize original images and their reconstructions
    """
    model.eval()
    
    # Get a batch of images
    dataiter = iter(data_loader)
    images, _ = next(dataiter)  # Ignore labels
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        recon_images, _, _, _ = model.forward(images)[:4]  # Ignore logits
    
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

def visualize_latent_traversal(model, dataset, img_name, dim=0, num_dims=5, save_dir="output"):
    """
    Visualize latent space traversal for multiple dimensions
    """
    lt_dir = os.path.join(save_dir, "latent_traversals")
    if not os.path.exists(lt_dir):
        os.makedirs(lt_dir)

    model.eval()
    
    # Get a specific sample image
    image = dataset.get_image_by_filename(img_name).to(device)
    
    # For multiple dimensions
    for d in range(dim, dim + num_dims):
        if d >= model.latent_dim:
            break
            
        # Get the mean and log_var for this image
        with torch.no_grad():
            mu, log_var = model.encode(image.unsqueeze(0))
            z = mu  # Use mean for traversal (no sampling)
            
            # Create a list to store the traversal images
            traversal_images = []
            
            # Create values for traversal
            values = np.linspace(-3, 3, 10)
            
            # Loop through each value and decode
            for value in values:
                z_new = z.clone()
                z_new[0, d] = value
                decoded = model.decode(z_new)
                traversal_images.append(decoded.squeeze().cpu())
        
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
    """
    model.eval()
    
    try:
        # Get the two specified images
        img1 = dataset.get_image_by_filename(img1_file).to(device)
        img2 = dataset.get_image_by_filename(img2_file).to(device)
        
        # Generate interpolation
        with torch.no_grad():
            # Encode both images to get their latent representations
            mu1, _ = model.encode(img1.unsqueeze(0))
            mu2, _ = model.encode(img2.unsqueeze(0))
            
            # Use means directly (no sampling) for smooth interpolation
            z1 = mu1
            z2 = mu2
            
            # Create interpolation steps
            interpolation_images = []
            alphas = np.linspace(0, 1, steps)
            
            # Generate and decode each interpolation point
            for alpha in alphas:
                z_interp = (1-alpha) * z1 + alpha * z2
                decoded = model.decode(z_interp)
                interpolation_images.append(decoded.squeeze().cpu())
        
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
        for i, img in enumerate(interpolation_images):
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
        for batch_idx, (data, _) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Extracting latent vectors"):
            data = data.to(device)
            mu, log_var = model.encode(data)
            
            all_mu.append(mu.cpu().numpy())
            all_log_var.append(log_var.cpu().numpy())
    
    # Concatenate all batches
    all_mu = np.concatenate(all_mu, axis=0)
    all_log_var = np.concatenate(all_log_var, axis=0)
    
    # Save as numpy arrays
    np.save(os.path.join(save_dir, "latent_mu.npy"), all_mu)
    np.save(os.path.join(save_dir, "latent_log_var.npy"), all_log_var)
    
    print(f"Saved latent vectors to {save_dir}")
    print(f"mu shape: {all_mu.shape}, log_var shape: {all_log_var.shape}")
    
    return all_mu, all_log_var

def visualize_feature_attribution(model, dataset, samples=5, save_dir="outputs"):
    """
    Visualize which latent dimensions contribute most to specific classification labels
    using a basic feature attribution technique
    """
    model.eval()
    
    # Create output directory
    attr_dir = os.path.join(save_dir, "feature_attribution")
    if not os.path.exists(attr_dir):
        os.makedirs(attr_dir)
    
    # Get some samples
    sample_indices = np.random.choice(len(dataset), samples, replace=False)
    
    for idx in sample_indices:
        # Get image and labels
        image, labels = dataset[idx]
        image = image.to(device).unsqueeze(0)  # Add batch dimension
        
        # Encode the image
        mu, log_var = model.encode(image)
        z_concat = torch.cat([mu, log_var], dim=1)  # This is what we feed to classifiers
        
        # For each label type
        for i, (label_name, classifier) in enumerate(model.classifiers.items()):
            # Ground truth label for this image
            true_label = labels[i].item()
            true_label_name = dataset.labels_df[label_name].unique()[true_label]
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Original image
            plt.subplot(1, 2, 1)
            plt.imshow(image.cpu().squeeze().numpy(), cmap='gray')
            plt.title(f"Original Image\n{dataset.img_files[idx]}\n{label_name}: {true_label_name}")
            plt.axis('off')
            
            # Feature attribution
            plt.subplot(1, 2, 2)
            
            # Get the weights from the first layer of the classifier
            # This is a very basic attribution method
            weights = classifier[0].weight.data[true_label]  # Weights for the true class
            
            # Normalize weights
            weights = weights.abs().cpu().numpy()
            weights = weights / weights.max()
            
            # Visualize weights for each dimension
            # First half are mu, second half are log_var
            latent_dim = model.latent_dim
            mu_weights = weights[:latent_dim]
            logvar_weights = weights[latent_dim:]
            
            x = np.arange(latent_dim)
            
            plt.bar(x, mu_weights, alpha=0.7, label='μ weights')
            plt.bar(x, -logvar_weights, alpha=0.7, label='log_var weights')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            plt.xlabel('Latent Dimension')
            plt.ylabel('Weight Magnitude')
            plt.title(f'Feature Attribution for {label_name}: {true_label_name}')
            plt.legend()
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(os.path.join(attr_dir, f'attr_{label_name}_{dataset.img_files[idx]}_{true_label_name}.png'))
            plt.close()
    
    print(f"Saved feature attribution visualizations to {attr_dir}")

def visualize_latent_space_by_class(model, dataset, label_name, save_dir="outputs"):
    """
    Create a 2D visualization of the latent space colored by class labels
    """
    from sklearn.decomposition import PCA
    import seaborn as sns
    
    model.eval()
    
    # Create output directory
    latent_dir = os.path.join(save_dir, "latent_space")
    if not os.path.exists(latent_dir):
        os.makedirs(latent_dir)
        
    # Get label index for the selected label
    label_idx = dataset.label_cols.index(label_name)
    
    # Get all unique values for this label
    unique_labels = sorted(dataset.labels_df[label_name].unique())
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # Extract all latent representations
    all_mu = []
    all_log_var = []
    all_labels = []
    
    # Create a dataloader with batch size 1 to iterate through dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc=f"Extracting latent space for {label_name}"):
            data = data.to(device)
            
            # Encode
            mu, log_var = model.encode(data)
            
            # Store
            all_mu.append(mu.cpu().numpy())
            all_log_var.append(log_var.cpu().numpy())
            all_labels.append(labels[:, label_idx].cpu().numpy())
    
    # Concatenate results
    all_mu = np.concatenate(all_mu, axis=0)
    all_log_var = np.concatenate(all_log_var, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(all_mu)
    
    # Convert numerical labels back to string labels for better plot
    label_strings = [unique_labels[label] for label in all_labels]
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    
    # Add legend with actual label values
    handles, _ = scatter.legend_elements()
    plt.legend(handles, unique_labels, title=label_name, loc="best")
    
    plt.title(f"Latent Space Visualization for {label_name}")
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(latent_dir, f'latent_space_{label_name}.png'), dpi=300)
    plt.close()
    
    print(f"Saved latent space visualization for {label_name} to {latent_dir}")

def main(args):
    # Start timing
    start_time = time.time()

    # Make the folder to save all outputs
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # If out_dir is "-unspecified-", generate it from parameters
    subfolder = args.out_dir
    if subfolder == "-unspecified-":
        subfolder = (f"supervised_res{args.img_size}_lat{args.latent_dim}_"
            f"epo{args.epochs}_bat{args.batch_size}_" 
            f"kld{args.max_kld_weight}_cls{args.cls_weight}_"
            f"(mi{args.mi_weight}_tc{args.tc_weight}_dwkl{args.dwkl_weight})_"
            f"adv{args.adv_weight}_rec{args.recon_sample_weight}")
    out_dir = os.path.join("outputs", subfolder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Auto-generated output directory: {subfolder}")

    # Set model path the same way you set output directory
    model_path = args.model_path
    if model_path == "-unspecified-":
        model_path = (f"supervised_res{args.img_size}_lat{args.latent_dim}_"
            f"epo{args.epochs}_bat{args.batch_size}_" 
            f"kld{args.max_kld_weight}_cls{args.cls_weight}_"
            f"(mi{args.mi_weight}_tc{args.tc_weight}_dwkl{args.dwkl_weight})_"
            f"adv{args.adv_weight}_rec{args.recon_sample_weight}.pth")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    
    # Parse label columns to use
    if args.label_cols:
        label_cols = args.label_cols.split(',')
        print(f"Using specified label columns: {label_cols}")
    else:
        label_cols = None
        print("Using all available label columns from CSV")
    
    # Load supervised dataset
    train_dataset = SupervisedVehicleDataset(
        img_dir=args.data_dir,
        label_file=args.label_file,
        transform=transform,
        label_cols=label_cols
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Get number of classes for each label type
    num_classes_dict = train_dataset.get_all_num_classes()
    print("Label categories and their class counts:")
    for label, count in num_classes_dict.items():
        print(f"  - {label}: {count} classes")
    
    # Create VAE model with classifiers
    vae_model = VAEWithClassifier(
        img_size=args.img_size, 
        latent_dim=args.latent_dim,
        num_classes_dict=num_classes_dict
    ).to(device)
    
    # Create Discriminator
    discriminator = Discriminator(img_size=args.img_size).to(device)
    print("Using VAE-GAN architecture with supervised classification")
    
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
    start_epoch = 0
    if args.resume and os.path.exists(os.path.join("checkpoints", model_path)):
        checkpoint = torch.load(os.path.join("checkpoints", model_path))
        
        # Load VAE-GAN checkpoint
        vae_model.load_state_dict(checkpoint['vae_model_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    if args.train:
        # Set up your desired KLD scheduler parameters
        kld_scheduler_params = {
            'total_epochs': args.epochs,
            'min_weight': 0.01,
            'max_weight': args.max_kld_weight,
            'warmup_epochs': 28,
            'schedule_type': "linear"
        }

        # Train with KLD weight scheduling
        losses = train_supervised_vaegan(
            vae_model, discriminator, train_loader, train_dataset, args.track_reconstruction,
            vae_optimizer, d_optimizer, args.epochs, kld_weight_scheduler, kld_scheduler_params,
            adv_weight=args.adv_weight, cls_weight=args.cls_weight, recon_sample_weight=args.recon_sample_weight,
            checkpoint_path=model_path, recon_path=out_dir)
        
        # Plot training losses
        plot_supervised_training_losses(losses, out_dir)
    
    # Load best model for evaluation
    if os.path.exists(os.path.join("checkpoints", model_path)):
        checkpoint = torch.load(os.path.join("checkpoints", model_path))
        vae_model.load_state_dict(checkpoint['vae_model_state_dict'])
        print(f"Loaded model from checkpoints/{model_path}")
    
    # Evaluate classification accuracy
    if args.classification_accuracy:
        accuracies, overall_accuracy = evaluate_classification_accuracy(vae_model, train_loader)
        
        print("\nClassification Accuracy:")
        for label, acc in accuracies.items():
            print(f"  - {label}: {acc:.4f} ({acc*100:.2f}%)")
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # Save accuracy results to file
        with open(os.path.join(out_dir, 'classification_accuracy.txt'), 'w') as f:
            f.write("Classification Accuracy:\n")
            for label, acc in accuracies.items():
                f.write(f"{label}: {acc:.4f} ({acc*100:.2f}%)\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n")
    
    # Visualize latent space by class
    if args.visualize_latent_class:
        for label in train_dataset.label_cols:
            visualize_latent_space_by_class(vae_model, train_dataset, label, save_dir=out_dir)
    
    # Visualize feature attribution
    if args.feature_attribution:
        visualize_feature_attribution(vae_model, train_dataset, samples=args.feature_samples, save_dir=out_dir)
    
    # Restored visualization functions from original VAE
    if args.reconstructions:
        # Visualize reconstructions
        visualize_reconstructions(vae_model, train_loader, num_images=10, save_dir=out_dir)
        
    if args.traversals:
        # Visualize latent space traversal
        visualize_latent_traversal(vae_model, train_dataset, args.traversals, dim=0, num_dims=args.latent_dim, save_dir=out_dir)
    
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
    
    # End timer and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supervised VAE-GAN for Vehicle Images with classification')
    
    #################
    # Data parameters
    #################
    parser.add_argument('--data_dir', type=str, default='data/evox_256x256_1-3', help='Directory containing the image dataset')
    parser.add_argument('--label_file', type=str, default='data/labels.csv', help='CSV file containing image labels')
    parser.add_argument('--label_cols', type=str, default=None, help='Comma-separated list of column names to use for labels. Default is all columns except filename.')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    
    #################
    # Model parameters
    #################
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--max_kld_weight', type=float, default=1.0, 
                        help='Maximum weight for KLD loss term in the scheduler')
    parser.add_argument('--mi_weight', type=float, default=1.0, 
                        help='Weight for mutual information loss term in KLD Loss')
    parser.add_argument('--tc_weight', type=float, default=1.0, 
                        help='Weight for total correlation loss term in KLD Loss')
    parser.add_argument('--dwkl_weight', type=float, default=1.0, 
                        help='Weight for dimension-wise KL divergence loss term in KLD Loss')
    parser.add_argument('--adv_weight', type=float, default=1.0, 
                        help='Weight of adversarial loss term')
    parser.add_argument('--recon_sample_weight', type=float, default=0.7, 
                        help='Weight for reconstruction vs sample discrimination')
    parser.add_argument('--cls_weight', type=float, default=1.0,
                        help='Weight for classification loss term')
    
    #################
    # Training parameters
    #################
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    #################
    # Output parameters
    #################
    parser.add_argument('--out_dir', type=str, default='-unspecified-', help='The output subfolder')
    parser.add_argument('--model_path', type=str, default='-unspecified-', help='Path to save/load model')
    
    #################
    # Actions
    #################
    parser.add_argument('--reconstructions', action='store_true', help='Visualize reconstructions')
    parser.add_argument('--extract_latent', action='store_true', help='Extract and save latent vectors')
    parser.add_argument('--sample', action='store_true', help='Generate random samples from the latent space')
    parser.add_argument('--track_reconstruction', type=str, default='-unspecified-', metavar='FILE', 
                        help='Track reconstruction of a specific image across training epochs')
    parser.add_argument('--traversals', type=str, metavar='FILE1', 
                        help='Visualize latent space traversals. Specify which .png file.')
    parser.add_argument('--interpolate', nargs=2, metavar=('FILE1', 'FILE2'), 
                        help='Specify two image filenames to interpolate between')
    parser.add_argument('--interpolate_steps', type=int, default=10,
                        help='Number of steps for interpolation (default: 10)')
    parser.add_argument('--classification_accuracy', action='store_true', help='Evaluate classification accuracy')
    parser.add_argument('--visualize_latent_class', action='store_true', help='Visualize latent space by class')
    parser.add_argument('--feature_attribution', action='store_true', help='Visualize feature attribution')
    parser.add_argument('--feature_samples', type=int, default=5, help='Number of samples for feature attribution')
    
    args = parser.parse_args()
    
    main(args)
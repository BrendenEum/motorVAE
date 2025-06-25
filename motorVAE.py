import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import umap
import random
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(4)
np.random.seed(4)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 71
EPOCHS = 100
LATENT_DIM = 128
LEARNING_RATE = 0.0001
BETA1 = 0.5 # AI recommended for GAN training
BETA2 = 0.999 # Default for Adam optimizer
TRAIN_PROPORTION = 0.98 # Proportion of data to use for training. Validation is 1-p(train).

# Create weights for different loss components
RECON_WEIGHT = 100.0
PERCEPTUAL_WEIGHT = 5.0
GAN_WEIGHT = 0.2
KLD_WEIGHT_START = 0.00001 # KLD Scheduler
KLD_WEIGHT_END = 0.08
TC_WEIGHT = 0.2  # Total Correlation weight
MI_WEIGHT = 0.1  # Mutual Information weight
DKLD_WEIGHT = 0.00002  # Dimension-wise KL Divergence weight
CLS_WEIGHT = 0.2  # Classifier weight

DISC_WEIGHT = 1.0 # Separate loss function, so this doesn't matter. 

# Number of patches for PatchGAN discriminator
PATCH_SIZE = 16  # Size of each patch

# Checkpoint saving frequency
CHECKPOINT_FREQ = 10  # Save checkpoint every 10 epochs

tc_nan_count = 0
mi_nan_count = 0

# Define custom dataset
class VehicleDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            labels_file (string): Path to the csv file with labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.labels_df = pd.read_csv(labels_file)
        
        # Get all image file names
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
        # Ensure all image files have corresponding labels
        valid_files = []
        for img_file in self.img_files:
            if img_file in self.labels_df['filename'].values:
                valid_files.append(img_file)
            else:
                print(f"{img_file} not found.")
        self.img_files = valid_files
        
        # Count classes for each label type
        self.year_classes = len(self.labels_df['year'].unique())
        self.make_classes = len(self.labels_df['make'].unique())
        self.body_classes = len(self.labels_df['body'].unique())
        self.door_classes = len(self.labels_df['door'].unique())
        
        print(f"Found {len(self.img_files)} valid images")
        print(f"Number of classes - Year: {self.year_classes}, Make: {self.make_classes}, Body: {self.body_classes}, Door: {self.door_classes}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Get labels for this image
        labels = self.labels_df[self.labels_df['filename'] == img_name].iloc[0]
        year = torch.tensor(labels['year'], dtype=torch.long)
        make = torch.tensor(labels['make'], dtype=torch.long)
        body = torch.tensor(labels['body'], dtype=torch.long)
        door = torch.tensor(labels['door'], dtype=torch.long)

        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image, 
            'year': year, 
            'make': make, 
            'body': body, 
            'door': door,
            'filename': img_name
        }

# Define the encoder network
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)  # 128x128
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 64x64
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 32x32
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # 16x16
        self.conv5 = nn.Conv2d(256, 512, 4, stride=2, padding=1)  # 8x8
        
        # Batch normalization
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Fully connected layers
        self.fc = nn.Linear(512 * 8 * 8, 1024)
        self.bn_fc = nn.BatchNorm1d(1024)
        
        # Mean and log variance projections
        self.fc_mean = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        
    def forward(self, x):
        # Convolutional layers with leaky ReLU activations
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.bn_fc(self.fc(x)), 0.2)
        
        # Get mean and log variance
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        return mean, logvar

# Define the decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        
        # Fully connected layers
        self.fc = nn.Linear(latent_dim, 1024)
        self.bn_fc = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512 * 8 * 8)
        self.bn_fc2 = nn.BatchNorm1d(512 * 8 * 8)
        
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)  # 16x16
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 32x32
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)   # 64x64
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)    # 128x128
        self.deconv5 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)     # 256x256
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        
    def forward(self, z):
        # Fully connected layers with ReLU activations
        x = F.relu(self.bn_fc(self.fc(z)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        
        # Reshape for convolutional layers
        x = x.view(x.size(0), 512, 8, 8)
        
        # Transposed convolutional layers with ReLU activations
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        
        # Final layer with sigmoid activation
        x = torch.sigmoid(self.deconv5(x))
        
        return x

# Define the PatchGAN discriminator
class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        
        # Calculate the number of patches
        self.patch_size = PATCH_SIZE
        self.n_patches = (IMAGE_SIZE // self.patch_size) ** 2
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, 4, stride=1, padding=1)
        
        # Batch normalization
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        # Convolutional layers with leaky ReLU activations
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        
        # Final convolution (output is a patch)
        x = self.conv5(x)
        
        # Reshape to get patches
        batch_size = x.size(0)
        spatial_size = x.size(2)
        
        # Return both the raw output and sigmoid output
        return x, torch.sigmoid(x)

# Define classifiers for supervised learning
class Classifier(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super(Classifier, self).__init__()
        
        # latent_dim specifies how many dimensions to use from the latent vector
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        
    def forward(self, z):
        # Use only the first 16 dimensions
        z = z[:, :16]
        
        # Apply batch normalization and leaky ReLU
        x = F.leaky_relu(self.bn1(self.fc1(z)), 0.2)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        
        # Output logits
        x = self.fc3(x)
        
        return x

# Define the perceptual loss using VGG16
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16 model
        vgg = models.vgg16(pretrained=False)
        vgg.load_state_dict(torch.load("vgg16/vgg16-397923af.pth", map_location=device))
        
        # We use the first few layers for perceptual loss
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        
        # Fill with VGG16 layers
        for x in range(4):
            self.slice1.add_module(str(x), vgg.features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg.features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg.features[x])
        
        # Set to evaluation mode and freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        # We need to convert grayscale to RGB (repeat the channel 3 times)
        x = x.repeat(1, 3, 1, 1)
        
        # Get features from different layers
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        
        return [h_relu1, h_relu2, h_relu3]

# Define the full VAE-GAN model
class VAEGAN(nn.Module):
    def __init__(self, latent_dim, year_classes, make_classes, body_classes, door_classes):
        super(VAEGAN, self).__init__()
        
        # Initialize components
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.discriminator = Discriminator()
        
        # Initialize classifiers
        self.year_classifier = Classifier(latent_dim, year_classes)
        self.make_classifier = Classifier(latent_dim, make_classes)
        self.body_classifier = Classifier(latent_dim, body_classes)
        self.door_classifier = Classifier(latent_dim, door_classes) 
        
        # Initialize perceptual loss
        self.perceptual_loss = PerceptualLoss()
        
        # Save dimensions
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def encode(self, x):
        # Get mean and log variance
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
    
    def decode(self, z):
        # Decode latent vector
        return self.decoder(z)
    
    def discriminate(self, x):
        # Discriminate image
        return self.discriminator(x)
    
    def classify(self, z):
        # Classify latent vector
        year_logits = self.year_classifier(z)
        make_logits = self.make_classifier(z)
        body_logits = self.body_classifier(z)
        door_logits = self.door_classifier(z)
        
        return year_logits, make_logits, body_logits, door_logits
    
    def forward(self, x):
        # Full forward pass
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        
        return x_recon, z, mu, logvar

# Calculate KL divergence terms
def kl_divergence(mu, logvar):
    # Clamp logvar to prevent extreme values
    logvar_clamped = torch.clamp(logvar, min=-20, max=20)
    # Standard KL divergence = log(2πσ²) + (x-μ)²/σ² - 1
    # Take mean for batch dimension
    kld = -0.5 * torch.mean(torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1))
    return kld

# Alternative simplified version that's more stable but less theoretically precise
def compute_tc_loss(z, mu, logvar):
    """
    Simplified TC loss that's more numerically stable
    Uses sample-based approximation with better numerical properties
    """
    batch_size, latent_dim = z.shape
    
    # Conservative clamping
    mu = torch.clamp(mu, min=-10, max=10)
    logvar = torch.clamp(logvar, min=-10, max=10)
    
    # Method 1: Use the current batch as an approximation of the full dataset
    # This is what Factor-VAE does and tends to be more stable
    
    # Compute log q(z_i|x_i) - the encoder's output for each sample
    var = torch.exp(logvar).clamp(min=1e-6)
    log_qz_given_x = -0.5 * torch.sum(
        math.log(2 * math.pi) + logvar + (z - mu).pow(2) / var, 
        dim=1
    )  # [B]
    
    # Compute log q(z_i) by marginalizing over the batch
    # For each sample z_i, compute its probability under all encoders in the batch
    z_expanded = z.unsqueeze(1)        # [B, 1, D]
    mu_expanded = mu.unsqueeze(0)      # [1, B, D]
    var_expanded = var.unsqueeze(0)    # [1, B, D]
    
    # Log probabilities of each z under each encoder
    diff = z_expanded - mu_expanded    # [B, B, D]
    log_probs = -0.5 * torch.sum(
        math.log(2 * math.pi) + torch.log(var_expanded) + diff.pow(2) / var_expanded,
        dim=2
    )  # [B, B]
    
    # Marginal log q(z_i) using log-mean-exp
    log_qz = torch.logsumexp(log_probs, dim=1) - math.log(batch_size)  # [B]
    
    # TC loss
    tc_loss = torch.mean(log_qz_given_x - log_qz)
    
    # Stability checks
    if torch.isnan(tc_loss) or torch.isinf(tc_loss):
        print("WARNING: tc_loss nan or infinite. Replacing with 0.0, but you should check this.")
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    
    return tc_loss

def compute_mi_loss(z, mu, logvar, batch_size):
    global mi_nan_count
    eps = 1e-8
    
    # Clamp values for numerical stability
    mu = torch.clamp(mu, min=-10, max=10)
    logvar = torch.clamp(logvar, min=-10, max=10)
    var = torch.clamp(torch.exp(logvar), min=eps, max=100)
    
    # 1. Compute log q(z|x) - conditional distribution (encoder)
    # log N(z; mu, var) = -0.5 * [D*log(2π) + sum_d(log(var_d) + (z_d - mu_d)^2 / var_d)]
    log_2pi = torch.log(torch.tensor(2 * np.pi, device=z.device))
    
    # For each sample i, compute log q(z_i | x_i)
    log_det_cond = torch.sum(torch.log(var), dim=1)          # [B] - sum over dimensions
    z_centered_cond = z - mu                                 # [B, D]
    mahalanobis_cond = torch.sum(z_centered_cond.pow(2) / var, dim=1)  # [B] - sum over dimensions
    
    log_qz_given_x = -0.5 * (z.size(1) * log_2pi + log_det_cond + mahalanobis_cond)  # [B]
    
    # 2. Compute log q(z) - marginal distribution
    # For each sample z_i, compute log q(z_i) = log(1/N * sum_j q(z_i | x_j))
    z_expanded = z.unsqueeze(1)                     # [B, 1, D]
    mu_expanded = mu.unsqueeze(0)                   # [1, B, D]
    var_expanded = var.unsqueeze(0)                 # [1, B, D]
    
    # Compute log q(z_i | x_j) for all i,j pairs
    log_det_marg = torch.sum(torch.log(var_expanded), dim=2)  # [1, B] - sum over dimensions
    z_centered_marg = z_expanded - mu_expanded               # [B, B, D]
    mahalanobis_marg = torch.sum(z_centered_marg.pow(2) / var_expanded, dim=2)  # [B, B] - sum over dimensions
    
    # Compute log probabilities: log q(z_i | x_j)
    log_qz_given_all_x = -0.5 * (z.size(1) * log_2pi + log_det_marg + mahalanobis_marg)  # [B, B]
    
    # Compute log q(z_i) = log(1/N * sum_j q(z_i | x_j)) using log-sum-exp trick
    max_val = torch.max(log_qz_given_all_x, dim=1, keepdim=True)[0]  # [B, 1]
    log_qz = torch.log(torch.mean(torch.exp(log_qz_given_all_x - max_val), dim=1)) + max_val.squeeze(1)  # [B]
    
    # 3. Compute MI = E[log q(z|x) - log q(z)]
    mi_loss = torch.mean(log_qz_given_x - log_qz)
    
    # Check for NaN/Inf values
    if torch.isnan(mi_loss) or torch.isinf(mi_loss):
        mi_nan_count += 1
        print(f"Warning: MI loss is NaN or Inf, returning 0 (Total NaN occurrences: {mi_nan_count})")
        return torch.tensor(0.0, device=z.device)
    
    return mi_loss

def compute_dkld_loss(mu, logvar):
    # More aggressive clamping for stability
    eps = 1e-8
    logvar_clipped = torch.clamp(logvar, min=-15, max=15)
    mu_clipped = torch.clamp(mu, min=-15, max=15)
    var_term = torch.clamp(torch.exp(logvar_clipped), min=eps)

    # Standard KL divergence with better numerical stability
    dkld_loss = -0.5 * torch.mean(torch.sum(1 + logvar_clipped - mu_clipped.pow(2) - var_term, dim=1))
    
    # Handle extreme values
    if torch.abs(dkld_loss) > 1e10: print("Warning: DKLD loss is extreme")
    
    return dkld_loss

def reset_nan_counters():
    global tc_nan_count, mi_nan_count
    tc_nan_count = 0
    mi_nan_count = 0

def print_nan_summary():
    print(f"NaN Summary - TC: {tc_nan_count}, MI: {mi_nan_count}")

# Function to calculate perceptual loss
def perceptual_loss(real_features, fake_features):
    # Calculate L1 loss between real and fake features
    losses = []
    for real_feat, fake_feat in zip(real_features, fake_features):
        losses.append(F.l1_loss(real_feat, fake_feat))
    
    # Return the sum of losses
    return sum(losses)

# Function to create folder based on parameters
def create_output_folder(config):
    folder_name = (
        f"res{IMAGE_SIZE}_lat{LATENT_DIM}_ep{EPOCHS}_bat{BATCH_SIZE}_lrn{LEARNING_RATE}_"
        f"rec{RECON_WEIGHT}_per{PERCEPTUAL_WEIGHT}_gan{GAN_WEIGHT}_"
        f"kld{KLD_WEIGHT_END}(tc{TC_WEIGHT}_mi{MI_WEIGHT}_dk{DKLD_WEIGHT})_cls{CLS_WEIGHT}_pat{PATCH_SIZE}"
    )
    output_dir = os.path.join("outputs", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    recon_epochs_dir = os.path.join(output_dir, "reconstructions_epochs")
    sample_epochs_dir = os.path.join(output_dir, "rsamples_epochs")
    tracking_epochs_dir = os.path.join(output_dir, "tracking_epochs")
    latent_traversals_dir = os.path.join(output_dir, "latent_traversals")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(recon_epochs_dir, exist_ok=True)
    os.makedirs(sample_epochs_dir, exist_ok=True)
    os.makedirs(tracking_epochs_dir, exist_ok=True)
    os.makedirs(latent_traversals_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return output_dir

# Function to save checkpoint
def save_checkpoint(model, optimizers, epoch, losses, output_dir):
    checkpoint_path = os.path.join(output_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth")
    
    # Prepare optimizer states
    optimizer_states = {
        'encoder': optimizers['encoder'].state_dict(),
        'decoder': optimizers['decoder'].state_dict(),
        'discriminator': optimizers['discriminator'].state_dict(),
        'year_classifier': optimizers['year_classifier'].state_dict(),
        'make_classifier': optimizers['make_classifier'].state_dict(),
        'body_classifier': optimizers['body_classifier'].state_dict(),
        'door_classifier': optimizers['door_classifier'].state_dict()
    }
    
    # Create checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_states': optimizer_states,
        'losses': losses
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

# Function to load checkpoint
def load_checkpoint(model, optimizers, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer states
    optimizers['encoder'].load_state_dict(checkpoint['optimizer_states']['encoder'])
    optimizers['decoder'].load_state_dict(checkpoint['optimizer_states']['decoder'])
    optimizers['discriminator'].load_state_dict(checkpoint['optimizer_states']['discriminator'])
    optimizers['year_classifier'].load_state_dict(checkpoint['optimizer_states']['year_classifier'])
    optimizers['make_classifier'].load_state_dict(checkpoint['optimizer_states']['make_classifier'])
    optimizers['body_classifier'].load_state_dict(checkpoint['optimizer_states']['body_classifier'])
    optimizers['door_classifier'].load_state_dict(checkpoint['optimizer_states']['door_classifier'])
    
    start_epoch = checkpoint['epoch'] + 1
    losses = checkpoint['losses']
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return start_epoch, losses

# Function to save images
def save_image_grid(images, path, nrow=10, title=None):
    # Convert tensor images to numpy
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    # Create figure
    n_images = len(images)
    rows = (n_images - 1) // nrow + 1
    
    plt.figure(figsize=(nrow * 2, rows * 2))
    
    # Plot each image
    for i, img in enumerate(images):
        if len(img.shape) == 3:  # Squeeze out channel dim if needed
            img = np.squeeze(img, axis=0)
        
        plt.subplot(rows, nrow, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Function to save reconstructions
def save_reconstructions(model, dataloader, output_dir, epoch):
    model.eval()
    with torch.no_grad():
        # Get a batch of images
        batch = next(iter(dataloader))
        real_images = batch['image'].to(device)
        
        # Generate reconstructions
        recon_images, _, _, _ = model(real_images)
        
        # Save a few examples
        n_examples = min(5, real_images.size(0))
        real_images = real_images[:n_examples].cpu().numpy()
        recon_images = recon_images[:n_examples].cpu().numpy()
        
        # Create comparison
        comparisons = []
        for i in range(n_examples):
            comparisons.extend([real_images[i][0], recon_images[i][0]])
        
        # Save image grid
        save_image_grid(comparisons, os.path.join(output_dir, "reconstructions_epochs", f"recon_epoch_{epoch}.png"), 
                        nrow=2, title=f"Epoch {epoch}")

# Function to save random samples
def save_random_samples(model, output_dir, epoch):
    model.eval()
    with torch.no_grad():
        # Generate random samples
        z = torch.randn(25, LATENT_DIM).to(device)
        samples = model.decode(z)
        
        # Convert to numpy and reshape
        samples = samples.cpu().numpy()
        
        # Save image grid
        save_image_grid(samples, os.path.join(output_dir, "rsamples_epochs", f"sample_epoch_{epoch}.png"), 
                        nrow=5, title="Random Samples")

# Function to save latent traversals
def save_latent_traversals(model, dataloader, output_dir):
    model.eval()
    with torch.no_grad():
        # Get a random image
        batch = next(iter(dataloader))
        img = batch['image'][0:1].to(device)
        
        # Encode the image to get the original latent vector
        z_original, mu, logvar = model.encode(img)
        
        # Use a fixed range around the mean
        n_steps = 7
        traversal_range = torch.linspace(-3, 3, n_steps, device=device)
        
        for dim in range(LATENT_DIM):
            traversal_images = []
            
            for val in traversal_range:
                # Start with the mean
                z_trav = mu.clone()
                
                # Set the current dimension to the traversal value
                z_trav[0, dim] = val
                
                # Decode the modified latent vector
                recon = model.decode(z_trav)
                traversal_images.append(recon[0].cpu().numpy())
            
            # Save traversal for this dimension
            save_image_grid(
                traversal_images, 
                os.path.join(output_dir, "latent_traversals", f"dim_{dim:03d}.png"), 
                nrow=n_steps, 
                title=f"Dimension {dim}: Range -3 to +3"
            )

# Function to save losses
def save_losses(all_losses, output_dir):
    # Create figure with subplots
    plt.figure(figsize=(20, 15))
    
    # Plot total loss
    plt.subplot(3, 2, 1)
    plt.plot(all_losses['total'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    # Plot detailed losses
    plt.subplot(3, 2, 2)
    plt.plot(all_losses['recon'], label='Reconstruction')
    plt.plot(all_losses['perceptual'], label='Perceptual')
    plt.plot(all_losses['kl'], label='KL Divergence')
    plt.plot(all_losses['cls'], label='Classification')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot discriminator loss
    plt.subplot(3, 2, 3)
    plt.plot(all_losses['disc'])
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    # Plot KL decomposition
    plt.subplot(3, 2, 4)
    plt.plot(all_losses['tc'], label='Total Correlation')
    plt.plot(all_losses['mi'], label='Mutual Information')
    plt.plot(all_losses['dkld'], label='Dimension-wise KLD')
    plt.title('KL Decomposition')
    plt.xlabel('Epoch')
    plt.ylim(-3, 3)
    plt.legend()
    plt.grid(True)
    
    # Plot classification losses
    plt.subplot(3, 2, 5)
    plt.plot(all_losses['year_cls'], label='Year')
    plt.plot(all_losses['make_cls'], label='Make')
    plt.plot(all_losses['body_cls'], label='Body')
    plt.plot(all_losses['door_cls'], label='Door')
    plt.title('Classification Losses')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "losses.png"))
    plt.close()

# Function to create UMAP visualizations
def create_umap_visualizations(z_samples, labels, output_dir):
    # Fit UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(z_samples)
    
    # Create separate plots for each label type
    label_types = ['year', 'make', 'body', 'door']
    
    for label_type in label_types:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels[label_type], cmap='tab20', s=5)
        plt.colorbar(scatter, label=label_type)
        plt.title(f'UMAP of Latent Space Colored by {label_type}')
        plt.savefig(os.path.join(output_dir, f"umap_{label_type}.png"))
        plt.close()

"""
# Function to create interpolation between two images
def create_interpolation(model, dataloader, output_dir):
    model.eval()
    with torch.no_grad():
        # Get two random images
        batch = next(iter(dataloader))
        img1, img2 = batch['image'][0:1].to(device), batch['image'][1:2].to(device)
        
        # Encode the images
        z1, _, _ = model.encode(img1)
        z2, _, _ = model.encode(img2)
        
        # Create interpolations
        interpolations = []
        for alpha in np.linspace(0, 1, 5):
            z_interp = alpha * z1 + (1 - alpha) * z2
            recon = model.decode(z_interp)
            interpolations.append(recon[0].cpu().numpy())
        
        # Save interpolation
        save_image_grid(interpolations, os.path.join(output_dir, "interpolation.png"), 
                        nrow=5, title="Interpolation between two random cars")
"""

# Function to create interpolation between two images
def create_interpolation(model, dataloader, output_dir):
    model.eval()
    with torch.no_grad():
        # Get two random images
        batch = next(iter(dataloader))
        img1, img2 = batch['image'][0:1].to(device), batch['image'][1:2].to(device)
        
        # Encode the images
        z1, _, _ = model.encode(img1)
        z2, _, _ = model.encode(img2)
        
        # Start with the first original image (reconstructed from z1)
        interpolations = []
        
        # Add the first original car
        #recon1 = model.decode(z1) # if you want the reconstructed original
        interpolations.append(img1[0].cpu().numpy())
        
        # Create interpolations (excluding endpoints since we're adding them separately)
        for alpha in np.linspace(0.25, 0.75, 3):  # 3 intermediate steps
            z_interp = alpha * z2 + (1 - alpha) * z1  # Note: swapped order for intuitive left-to-right
            recon = model.decode(z_interp)
            interpolations.append(recon[0].cpu().numpy())
        
        # Add the second original car (reconstructed)
        #recon2 = model.decode(z2) # if you want the reconstructed original
        interpolations.append(img2[0].cpu().numpy())
        
        # Save interpolation (now we have 5 images total: original1, interp1, interp2, interp3, original2)
        save_image_grid(interpolations, os.path.join(output_dir, "interpolation.png"), 
                        nrow=5, title="Interpolation between two cars")

# Main training function
def train_vaegan(model, train_loader, val_loader, output_dir):
    # Setup optimizers
    optim_encoder = optim.Adam(model.encoder.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optim_decoder = optim.Adam(model.decoder.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optim_disc = optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optim_year = optim.Adam(model.year_classifier.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optim_make = optim.Adam(model.make_classifier.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optim_body = optim.Adam(model.body_classifier.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optim_door = optim.Adam(model.door_classifier.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)) 
    
    # Loss functions
    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    ce_loss = nn.CrossEntropyLoss()
    
    # Initialize tracking variables
    all_losses = {
        'total': [], 'recon': [], 'perceptual': [], 'disc': [], 'gan': [],
        'kl': [], 'tc': [], 'mi': [], 'dkld': [], 'cls': [],
        'year_cls': [], 'make_cls': [], 'body_cls': [], 'door_cls': []
    }
    
    # Lists to store latent vectors for later analysis
    z_samples = []
    z_labels = {'year': [], 'make': [], 'body': [], 'door': []}
    
    # Select a random car for reconstruction tracking
    random_idx = random.randint(0, len(train_loader.dataset) - 1)
    tracking_batch = train_loader.dataset[random_idx]
    tracking_image = tracking_batch['image'].unsqueeze(0).to(device)
    tracking_filename = tracking_batch['filename']
    print(f"Selected {tracking_filename} for reconstruction tracking")
    
    # Training loop
    for epoch in range(EPOCHS):
        reset_nan_counters()
        model.train()
        epoch_losses = {k: 0.0 for k in all_losses.keys()}
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # Get data
            real_images = batch['image'].to(device)
            year_labels = batch['year'].to(device)
            make_labels = batch['make'].to(device)
            body_labels = batch['body'].to(device)
            door_labels = batch['door'].to(device)
            
            batch_size = real_images.size(0)
            batch_count += 1
            
            # ------------------------------
            # Train Discriminator
            # ------------------------------
            optim_disc.zero_grad()
            
            # Process real images
            real_features = model.perceptual_loss(real_images)
            real_patches, real_preds = model.discriminate(real_images)
            
            # Create labels for real images (1s)
            real_labels = torch.ones_like(real_preds)
            
            # Process reconstructed images (detached)
            z, mu, logvar = model.encode(real_images)
            fake_images = model.decode(z)
            fake_patches, fake_preds = model.discriminate(fake_images.detach())
            
            # Create labels for fake images (0s)
            fake_labels = torch.zeros_like(fake_preds)
            
            # Process random samples (detached)
            z_rand = torch.randn_like(z)
            rand_images = model.decode(z_rand)
            rand_patches, rand_preds = model.discriminate(rand_images.detach())
            
            # Create labels for random images (0s)
            rand_labels = torch.zeros_like(rand_preds)

            # Clamp predictions to (0,1) before BCE loss calculations
            eps = 1e-8
            real_preds = torch.clamp(real_preds, min=0.0 + eps, max=1.0 - eps)
            fake_preds = torch.clamp(fake_preds, min=0.0 + eps, max=1.0 - eps)
            rand_preds = torch.clamp(rand_preds, min=0.0 + eps, max=1.0 - eps)
            
            # Compute discriminator loss
            d_loss_real = bce_loss(real_preds, real_labels)
            d_loss_fake = bce_loss(fake_preds, fake_labels)
            d_loss_rand = bce_loss(rand_preds, rand_labels)
            d_loss = d_loss_real + d_loss_fake + d_loss_rand
            
            # Update discriminator
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
            optim_disc.step()
            
            # ------------------------------
            # Train Encoder & Decoder
            # ------------------------------
            optim_encoder.zero_grad()
            optim_decoder.zero_grad()
            optim_year.zero_grad()
            optim_make.zero_grad()
            optim_body.zero_grad()
            optim_door.zero_grad()
            
            # Reconstruct images
            z, mu, logvar = model.encode(real_images)
            fake_images = model.decode(z)
            
            # Generate random samples
            z_rand = torch.randn_like(z)
            rand_images = model.decode(z_rand)
            
            # Compute features for perceptual loss
            fake_features = model.perceptual_loss(fake_images)
            
            # Compute discriminator predictions for generator training
            _, fake_preds = model.discriminate(fake_images)
            _, rand_preds = model.discriminate(rand_images)
            
            # Create labels for generated images (1s for generator training)
            gen_labels = torch.ones_like(fake_preds)
            rand_labels = torch.ones_like(rand_preds)
            
            # Compute reconstruction loss
            recon_loss = l1_loss(fake_images, real_images)
            
            # Compute perceptual loss
            percep_loss = perceptual_loss(real_features, fake_features)
            
            # Compute GAN losses
            gen_loss_fake = bce_loss(fake_preds, gen_labels)
            gen_loss_rand = bce_loss(rand_preds, rand_labels)
            gen_loss = gen_loss_fake + gen_loss_rand
            
            # Compute KL divergence components
            tc_loss = compute_tc_loss(z.detach(), mu.detach(), logvar.detach())
            mi_loss = compute_mi_loss(z.detach(), mu.detach(), logvar.detach(), batch_size)
            dkld_loss = compute_dkld_loss(mu.detach(), logvar.detach())

            # Total KL Divergence loss schedule
            # Start small for first 25 epochs, then increase for remaining epochs.
            kl_weight = KLD_WEIGHT_START if epoch < 25 else min(KLD_WEIGHT_END, (epoch - 25) / (EPOCHS - 25))
            kl_loss = kl_weight * (tc_loss * TC_WEIGHT + mi_loss * MI_WEIGHT + dkld_loss * DKLD_WEIGHT)
            
            # Compute classification losses
            z_cls = z.detach().clone()
            z_cls.requires_grad = True

            # For the other classifiers, keep as is:
            year_logits = model.year_classifier(z_cls)
            make_logits = model.make_classifier(z_cls)
            body_logits = model.body_classifier(z_cls)
            door_logits = model.door_classifier(z_cls)

            year_loss = ce_loss(year_logits, year_labels)
            make_loss = ce_loss(make_logits, make_labels)
            body_loss = ce_loss(body_logits, body_labels)
            door_loss = ce_loss(door_logits, door_labels)
            
            cls_loss = year_loss + make_loss + body_loss + door_loss
            
            # Compute total loss
            total_loss = (
                RECON_WEIGHT * recon_loss +
                PERCEPTUAL_WEIGHT * percep_loss +
                GAN_WEIGHT * gen_loss +
                kl_loss +
                CLS_WEIGHT * cls_loss
            )
            
            # Update encoder & decoder
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.year_classifier.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.make_classifier.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.body_classifier.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.door_classifier.parameters(), max_norm=1.0)
            optim_encoder.step()
            optim_decoder.step()
            optim_year.step()
            optim_make.step()
            optim_body.step()
            optim_door.step()
            
            # Record losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item() * RECON_WEIGHT
            epoch_losses['perceptual'] += percep_loss.item() * PERCEPTUAL_WEIGHT
            epoch_losses['gan'] += gen_loss.item() * GAN_WEIGHT
            epoch_losses['disc'] += d_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            epoch_losses['tc'] += tc_loss.item() * TC_WEIGHT
            epoch_losses['mi'] += mi_loss.item() * MI_WEIGHT
            epoch_losses['dkld'] += dkld_loss.item() * DKLD_WEIGHT
            epoch_losses['cls'] += cls_loss.item() * CLS_WEIGHT
            epoch_losses['year_cls'] += year_loss.item() * CLS_WEIGHT
            epoch_losses['make_cls'] += make_loss.item() * CLS_WEIGHT
            epoch_losses['body_cls'] += body_loss.item() * CLS_WEIGHT
            epoch_losses['door_cls'] += door_loss.item() * CLS_WEIGHT
            
            # Store latent vectors and labels for last epoch
            if epoch == EPOCHS - 1:
                z_samples.append(z.detach().cpu().numpy())
                z_labels['year'].extend(year_labels.cpu().numpy())
                z_labels['make'].extend(make_labels.cpu().numpy())
                z_labels['body'].extend(body_labels.cpu().numpy())
                z_labels['door'].extend(door_labels.cpu().numpy())
        
        # Calculate average losses for this epoch
        for k in epoch_losses.keys():
            epoch_losses[k] /= batch_count
            all_losses[k].append(epoch_losses[k])
        
        # Print progress
        print(f"Epoch {epoch+1}/{EPOCHS} - " +
              f"Total: {epoch_losses['total']:.4f} " +
              f"| Recon: {epoch_losses['recon']:.4f}, " +
              f"Percept: {epoch_losses['perceptual']:.4f}, " +
              f"GAN: {epoch_losses['gan']:.4f}, " +
              f"KL: {epoch_losses['kl']:.4f}, " +
              f"tc: {epoch_losses['tc']:.4f}, " +
              f"mi: {epoch_losses['mi']:.4f}, " +
              f"dkld: {epoch_losses['dkld']:.4f}, " +
              f"Cls: {epoch_losses['cls']:.4f} " +
              f"| Disc: {epoch_losses['disc']:.4f}")
        
        # Print NaN summary at the end of each epoch
        print_nan_summary()
        
        # Save reconstructions, samples, and losses for tracking
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            save_reconstructions(model, train_loader, output_dir, epoch)
            save_random_samples(model, output_dir, epoch)
            save_losses(all_losses, output_dir)
            
        # Also save tracking reconstruction
        #with torch.no_grad():
        #    model.eval()  # Set to evaluation mode
        #    tracking_recon, _, _, _ = model(tracking_image)
        #    model.train()
        #    tracking_img = tracking_image[0].cpu().numpy()
        #    tracking_rec = tracking_recon[0].cpu().numpy()
        #    comparisons = [tracking_img[0], tracking_rec[0]]
        #    save_image_grid(comparisons, os.path.join(output_dir, "tracking_epochs", f"tracking_{epoch}.png"), 
        #                    nrow=2, title=f"{tracking_filename}: Epoch {epoch}")
                
        # Save checkpoint every CHECKPOINT_FREQ epochs
        if (epoch + 1) % CHECKPOINT_FREQ == 0:
            optimizers = {'encoder': optim_encoder, 'decoder': optim_decoder, 'discriminator': optim_disc, 
                  'year_classifier': optim_year, 'make_classifier': optim_make,
                  'body_classifier': optim_body, 'door_classifier': optim_door}
            save_checkpoint(model, optimizers, epoch, all_losses, output_dir)
    
    # Save final checkpoint
    save_checkpoint(model, optimizers, EPOCHS-1, all_losses, output_dir)
    
    # Concatenate all latent samples
    if z_samples:
        z_samples = np.concatenate(z_samples, axis=0)
    
    # Save final outputs
    save_latent_traversals(model, train_loader, output_dir)
    
    # Save latent vectors
    if len(z_samples) > 0:
        mu_path = os.path.join(output_dir, "latent_mu.npy")
        logvar_path = os.path.join(output_dir, "latent_logvar.npy")
        z_path = os.path.join(output_dir, "latent_z.npy")
        
        # Extract and save the means, log variances, and sampled z
        with torch.no_grad():
            batch = next(iter(train_loader))
            images = batch['image'].to(device)
            mu, logvar = model.encoder(images)
            z = model.reparameterize(mu, logvar)
            
            np.save(mu_path, mu.cpu().numpy())
            np.save(logvar_path, logvar.cpu().numpy())
            np.save(z_path, z.cpu().numpy())
        
        # Create UMAP visualizations
        create_umap_visualizations(z_samples, z_labels, output_dir)
    
    # Create interpolations
    create_interpolation(model, train_loader, output_dir)
    
    # Calculate and save classification accuracy
    model.eval()
    correct = {'year': 0, 'make': 0, 'body': 0, 'door': 0}
    total = {'year': 0, 'make': 0, 'body': 0, 'door': 0}  # Track totals separately

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            year_labels = batch['year'].to(device)
            make_labels = batch['make'].to(device)
            body_labels = batch['body'].to(device)
            door_labels = batch['door'].to(device)
            
            z, _, _ = model.encode(images)
            
            # For year, make, body, door - all samples are valid
            year_logits, make_logits, body_logits, door_logits = model.classify(z)
            
            _, year_preds = torch.max(year_logits, 1)
            _, make_preds = torch.max(make_logits, 1)
            _, body_preds = torch.max(body_logits, 1)
            _, door_preds = torch.max(door_logits, 1)
            
            batch_size = year_labels.size(0)
            
            correct['year'] += (year_preds == year_labels).sum().item()
            correct['make'] += (make_preds == make_labels).sum().item()
            correct['body'] += (body_preds == body_labels).sum().item()
            correct['door'] += (door_preds == door_labels).sum().item()
            
            total['year'] += batch_size
            total['make'] += batch_size
            total['body'] += batch_size
            total['door'] += batch_size

    # Calculate and save accuracies
    with open(os.path.join(output_dir, "classification_accuracy.txt"), "w") as f:
        for label_type in ['year', 'make', 'body', 'door']:
            if total[label_type] > 0:
                accuracy = 100 * correct[label_type] / total[label_type]
                f.write(f"{label_type} accuracy: {accuracy:.2f}% ({correct[label_type]}/{total[label_type]})\n")
                print(f"{label_type} accuracy: {accuracy:.2f}% ({correct[label_type]}/{total[label_type]})")
            else:
                f.write(f"{label_type} accuracy: N/A (no valid samples)\n")
                print(f"{label_type} accuracy: N/A (no valid samples)")
    
    return model, all_losses

# Main function
def main():
    # Start timing
    start_time = time.time()

    # Create output directory
    output_dir = create_output_folder({})
    print(f"Outputs will be saved to: {output_dir}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = VehicleDataset(
        img_dir="data/evox_256x256_1-4",
        labels_file="data/labels_evox_256x256_1-4.csv",
        transform=transform
    )

    # Split into train and validation sets
    train_size = int(TRAIN_PROPORTION * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Get number of classes for each label type
    year_classes = train_dataset.dataset.year_classes
    make_classes = train_dataset.dataset.make_classes
    body_classes = train_dataset.dataset.body_classes
    door_classes = train_dataset.dataset.door_classes
    
    # Initialize model
    model = VAEGAN(
        latent_dim=LATENT_DIM,
        year_classes=year_classes,
        make_classes=make_classes,
        body_classes=body_classes,
        door_classes=door_classes
    ).to(device)

    # Create optimizers for checkpoint loading
    optimizers = {
        'encoder': optim.Adam(model.encoder.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)),
        'decoder': optim.Adam(model.decoder.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)),
        'discriminator': optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)),
        'year_classifier': optim.Adam(model.year_classifier.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)),
        'make_classifier': optim.Adam(model.make_classifier.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)),
        'body_classifier': optim.Adam(model.body_classifier.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2)),
        'door_classifier': optim.Adam(model.door_classifier.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    }
    
    # Check for latest checkpoint
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
    
    start_epoch = 0
    all_losses = None
    
    if checkpoint_files:
        # Sort checkpoints by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
        print(f"Found checkpoint: {latest_checkpoint}")
        
        # Load checkpoint
        start_epoch, all_losses = load_checkpoint(model, optimizers, latest_checkpoint)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")
    
    # Train model
    model, losses = train_vaegan(model, train_loader, val_loader, output_dir)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    print(f"Model saved to {os.path.join(output_dir, 'model.pth')}")

    # End timer and print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

if __name__ == "__main__":
    main()
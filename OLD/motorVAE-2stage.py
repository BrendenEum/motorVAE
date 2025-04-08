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

class VAE(nn.Module):
    def __init__(self, img_size=64, latent_dim=128, hidden_dims=None):
        super(VAE, self).__init__()
        
        self.img_size = img_size
        self.latent_dim = latent_dim
        
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
        return self.decode(z), mu, log_var
    
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

def vae_loss(recon_x, x, mu, log_var, kld_weight=0.005):
    """
    VAE loss function with KL Divergence and reconstruction loss
    """
    # Reconstruction loss (using binary cross entropy for image pixels)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence loss
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    loss = recon_loss + kld_weight * kld_loss
    
    return loss, recon_loss, kld_loss

def train_vae(model, train_loader, optimizer, epochs, kld_weight=0.005, save_path="vae_model.pth"):
    """
    Train the VAE model
    """
    if not os.path.exists("checkpoints/"):
        os.makedirs("checkpoints/")

    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, data in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss, recon_loss, kld_loss = vae_loss(recon_batch, data, mu, log_var, kld_weight)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kld_loss += kld_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() / len(data),
                'recon_loss': recon_loss.item() / len(data),
                'kld_loss': kld_loss.item() / len(data)
            })
        
        # Average losses for the epoch
        avg_loss = epoch_loss / len(train_loader.dataset)
        avg_recon_loss = epoch_recon_loss / len(train_loader.dataset)
        avg_kld_loss = epoch_kld_loss / len(train_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"Average KLD Loss: {avg_kld_loss:.4f}")
        
        train_losses.append(avg_loss)
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Checkpoint saved to {save_path}")
    
    return train_losses

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
    model = VAE(img_size=args.img_size, latent_dim=args.latent_dim).to(device)
    
    # Count and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: {total_params:,}")
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # If resuming from checkpoint
    if args.resume and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    if args.train:
        # Train the model at 0 kld weight
        train_losses = train_vae(model, train_loader, optimizer, args.epochs, kld_weight=0.001, save_path=args.model_path)

        # Retrain again with higher weight
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate * 0.5)  # Often using lower LR for fine-tuning
        train_losses = train_vae(model, train_loader, optimizer, args.epochs, kld_weight=0.05, save_path=args.model_path)
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(out_dir, f'training_loss.png'))
        plt.close()
    
    # Load best model for evaluation
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_path}")
    
    if args.visualize:
        # Visualize reconstructions
        visualize_reconstructions(model, train_loader, num_images=10, save_dir=out_dir)
        
        # Visualize latent space traversal
        visualize_latent_traversal(model, train_loader, dim=0, num_dims=args.latent_dim, save_dir=out_dir)
    
    if args.extract_latent:
        # Extract and save latent vectors
        mu, log_var = extract_latent_vectors(model, train_loader, save_dir=out_dir)
    
    if args.sample:
        # Generate random samples
        with torch.no_grad():
            samples = model.sample(num_samples=25)
            
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
    parser = argparse.ArgumentParser(description='VAE for Vehicle Images')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/evox_256x256_1-3', help='Directory containing the dataset')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--dataset', type=str, default='Unspecified-Dataset', help='The properties describing evox: e.g. 64x64_1')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--kld_weight', type=float, default=0.005, help='Weight of KLD loss term')
    
    # Training parameters
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    # Output parameters
    parser.add_argument('--model_path', type=str, default='checkpoints/vae_model.pth', help='Path to save/load model')
    
    # Actions
    parser.add_argument('--visualize', action='store_true', help='Visualize reconstructions and latent space')
    parser.add_argument('--extract_latent', action='store_true', help='Extract and save latent vectors')
    parser.add_argument('--sample', action='store_true', help='Generate random samples from the latent space')
    
    args = parser.parse_args()
    
    main(args)

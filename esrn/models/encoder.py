"""
Symbolic State Encoder
Encodes visual inputs into discrete symbolic states using VQ-VAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for discrete representation learning
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z):
        """
        Args:
            z: (B, C, H, W) - continuous features
        Returns:
            quantized: (B, C, H, W) - quantized features
            loss: VQ loss
            perplexity: codebook usage
        """
        # Flatten spatial dimensions
        z = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        z_flat = z.view(-1, self.embedding_dim)  # (B*H*W, C)
        
        # Calculate distances to codebook vectors
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.weight.t())
        )
        
        # Get closest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(z.shape)  # (B, H, W, C)
        
        # VQ Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        # Calculate perplexity (codebook usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        return quantized, loss, perplexity


class SymbolicEncoder(nn.Module):
    """
    Encodes visual inputs into symbolic discrete states
    """
    def __init__(
        self,
        in_channels=3,
        hidden_dims=[64, 128, 256],
        vocab_size=512,
        embed_dim=64,
        commitment_cost=0.25
    ):
        super().__init__()
        
        # Convolutional encoder
        modules = []
        prev_channels = in_channels
        
        for hidden_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels, hidden_dim, 
                             kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True)
                )
            )
            prev_channels = hidden_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Project to embedding dimension
        self.pre_quantize = nn.Conv2d(hidden_dims[-1], embed_dim, 
                                      kernel_size=1)
        
        # Vector Quantizer
        self.vq = VectorQuantizer(vocab_size, embed_dim, commitment_cost)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - input images
        Returns:
            symbols: (B, D, H', W') - discrete symbolic states
            vq_loss: VQ loss
            perplexity: codebook usage
        """
        # Encode
        z = self.encoder(x)
        z = self.pre_quantize(z)
        
        # Quantize to discrete symbols
        symbols, vq_loss, perplexity = self.vq(z)
        
        return symbols, vq_loss, perplexity
    
    def get_codebook_indices(self, x):
        """
        Get discrete codebook indices for visualization
        """
        z = self.encoder(x)
        z = self.pre_quantize(z)
        
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.vq.embedding_dim)
        
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.vq.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.vq.embeddings.weight.t())
        )
        
        indices = torch.argmin(distances, dim=1)
        return indices.view(x.size(0), -1)


if __name__ == "__main__":
    # Test the encoder
    encoder = SymbolicEncoder(
        in_channels=1,
        hidden_dims=[64, 128],
        vocab_size=512,
        embed_dim=64
    )
    
    # Dummy input (grayscale RAVEN image)
    x = torch.randn(4, 1, 160, 160)
    
    symbols, vq_loss, perplexity = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output symbols shape: {symbols.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.2f}")
    
    indices = encoder.get_codebook_indices(x)
    print(f"Codebook indices shape: {indices.shape}")
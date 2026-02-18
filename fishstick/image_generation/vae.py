"""
Variational Autoencoder (VAE) implementations for image generation.

This module provides various VAE architectures:
- Vanilla VAE: Standard variational autoencoder with Gaussian latent space
- VQ-VAE: Vector Quantized VAE using discrete latent codes
- Conditional VAE: VAE conditioned on additional information
- Beta-VAE: VAE with adjustable latent channel capacity
- Factor-VAE: VAE with disentanglement regularization
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VAEEncoder(nn.Module):
    """Encoder network for VAE that maps images to latent distribution parameters."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        image_size: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        layers = []
        channels = in_channels

        for hidden_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU(0.2),
                )
            )
            channels = hidden_dim

        self.encoder = nn.Sequential(*layers)

        final_size = image_size // (2 ** len(hidden_dims))
        self.fc_mu = nn.Linear(channels * final_size * final_size, latent_dim)
        self.fc_logvar = nn.Linear(channels * final_size * final_size, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder network for VAE that maps latent codes back to images."""

    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 3,
        hidden_dims: Optional[List[int]] = None,
        image_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.image_size = image_size

        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        final_size = image_size // (2 ** len(hidden_dims))

        self.fc = nn.Linear(latent_dim, hidden_dims[0] * final_size * final_size)

        layers = []
        channels = hidden_dims[0]

        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        channels,
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2),
                )
            )
            channels = hidden_dims[i + 1]

        layers.append(
            nn.ConvTranspose2d(
                channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        )
        layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), -1, self.image_size // 4, self.image_size // 4)
        return self.decoder(h)


class VAE(nn.Module):
    """Vanilla Variational Autoencoder for image generation.

    Args:
        in_channels: Number of input image channels
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions
        image_size: Input image size
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        image_size: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.encoder = VAEEncoder(in_channels, latent_dim, hidden_dims, image_size)
        self.decoder = VAEDecoder(latent_dim, in_channels, hidden_dims, image_size)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def generate(self, num_samples: int, device: torch.device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


class VectorQuantizer(nn.Module):
    """Vector Quantizer for VQ-VAE using learned codebook."""

    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z_flattened = z.view(-1, self.embedding_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = F.mse_loss(z_q.detach(), z) + self.commitment_cost * F.mse_loss(
            z_q, z.detach()
        )

        z_q = z + (z_q - z).detach()

        return z_q, loss, min_encoding_indices


class VQVAE(nn.Module):
    """Vector Quantized VAE with discrete latent codes.

    Args:
        in_channels: Number of input image channels
        latent_dim: Dimension of latent embeddings
        num_embeddings: Number of discrete codebook entries
        commitment_cost: Weight for commitment loss
        image_size: Input image size
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        num_embeddings: int = 256,
        commitment_cost: float = 0.25,
        image_size: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        hidden_dims = [128, 256]

        encoder_layers = []
        channels = in_channels
        for hidden_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                )
            )
            channels = hidden_dim

        encoder_layers.append(nn.Conv2d(channels, latent_dim, kernel_size=1))

        self.encoder = nn.Sequential(*encoder_layers)

        self.quantize = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

        decoder_layers = [
            nn.ConvTranspose2d(latent_dim, channels, kernel_size=1),
        ]

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        channels, hidden_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                )
            )
            channels = hidden_dim

        decoder_layers.append(
            nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encoder(x)
        z_q, loss, indices = self.quantize(z)
        recon = self.decoder(z_q)
        return recon, loss

    def encode(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z_flattened = z.view(-1, self.latent_dim)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.quantize.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.quantize.embedding.weight.t())
        )
        return torch.argmin(d, dim=1).view(x.size(0), -1)

    def decode(self, indices: Tensor) -> Tensor:
        z_q = self.quantize.embedding(indices)
        if len(z_q.shape) == 2:
            z_q = z_q.view(-1, self.latent_dim, 8, 8)
        return self.decoder(z_q)

    def generate(self, num_samples: int, device: torch.device) -> Tensor:
        indices = torch.randint(
            0, self.num_embeddings, (num_samples, 64), device=device
        )
        return self.decode(indices)


class ConditionalVAE(nn.Module):
    """Conditional VAE conditioned on class labels or other information.

    Args:
        in_channels: Number of input image channels
        latent_dim: Dimension of latent space
        num_classes: Number of conditioning classes
        hidden_dims: List of hidden layer dimensions
        image_size: Input image size
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        num_classes: int = 10,
        hidden_dims: Optional[List[int]] = None,
        image_size: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = VAEEncoder(in_channels, latent_dim, hidden_dims, image_size)

        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.decoder = VAEDecoder(latent_dim * 2, in_channels, hidden_dims, image_size)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        label_emb = self.label_emb(labels)

        mu_concat = torch.cat([mu, label_emb], dim=1)
        logvar_concat = torch.cat([logvar, label_emb], dim=1)

        z = self.reparameterize(mu_concat, logvar_concat)
        recon = self.decoder(z)

        return recon, mu, logvar, z

    def generate(
        self,
        labels: Tensor,
        num_samples: Optional[int] = None,
        device: torch.device = None,
    ) -> Tensor:
        if device is None:
            device = labels.device

        if num_samples is not None:
            labels = labels.repeat(num_samples)

        label_emb = self.label_emb(labels)
        z = torch.randn_like(label_emb) * 0.5 + label_emb
        return self.decoder(z)


class BetaVAE(nn.Module):
    """Beta-VAE with adjustable latent channel capacity for disentanglement.

    The beta parameter controls the trade-off between reconstruction and
    latent channel capacity, with higher beta promoting disentanglement.

    Args:
        in_channels: Number of input image channels
        latent_dim: Dimension of latent space
        beta: Weight for KL divergence term (higher = more disentanglement)
        hidden_dims: List of hidden layer dimensions
        image_size: Input image size
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        beta: float = 4.0,
        hidden_dims: Optional[List[int]] = None,
        image_size: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = VAEEncoder(in_channels, latent_dim, hidden_dims, image_size)
        self.decoder = VAEDecoder(latent_dim, in_channels, hidden_dims, image_size)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss(self, x: Tensor, recon: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss


class FactorVAE(nn.Module):
    """Factor-VAE with Total Correlation (TC) loss for improved disentanglement.

    Uses minimum discrimination information (MDI) regularization to encourage
    factorized latent space.

    Args:
        in_channels: Number of input image channels
        latent_dim: Dimension of latent space
        gamma: Weight for TC loss
        hidden_dims: List of hidden layer dimensions
        image_size: Input image size
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        gamma: float = 6.0,
        hidden_dims: Optional[List[int]] = None,
        image_size: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.gamma = gamma

        self.encoder = VAEEncoder(in_channels, latent_dim, hidden_dims, image_size)
        self.decoder = VAEDecoder(latent_dim, in_channels, hidden_dims, image_size)

        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2),
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def tc_loss(self, z: Tensor) -> Tensor:
        """Compute Total Correlation loss using density ratio trick."""
        batch_size = z.size(0)

        z_perm = z[torch.randperm(batch_size)]

        logits = self.discriminator(z)
        logits_perm = self.discriminator(z_perm)

        tc_loss = F.cross_entropy(
            logits, torch.zeros(batch_size, dtype=torch.long, device=z.device)
        ) + F.cross_entropy(
            logits_perm, torch.ones(batch_size, dtype=torch.long, device=z.device)
        )

        return tc_loss

    def loss(
        self, x: Tensor, recon: Tensor, mu: Tensor, logvar: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        z = self.reparameterize(mu, logvar)
        tc_loss = self.tc_loss(z)

        total_loss = recon_loss + kl_loss + self.gamma * tc_loss
        return total_loss, recon_loss, tc_loss

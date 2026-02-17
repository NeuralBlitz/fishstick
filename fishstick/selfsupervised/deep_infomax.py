"""
Deep InfoMax: Learning Deep Representations by Mutual Information Maximization

Based on "Learning Deep Representations by Mutual Information Maximization"
(Global InfoMax variant) and its extensions (Local InfoMax).

Key ideas:
- Maximize mutual information between input and learned representations
- Use contrastive bound on mutual information
- Global and local variants for different granularity
"""

from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class GlobalInfoMaxDiscriminator(nn.Module):
    """Discriminator for Global InfoMax (f(x, E(x)) -> scalar)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        combined = torch.cat([x, h], dim=-1)
        return self.net(combined)


class LocalInfoMaxDiscriminator(nn.Module):
    """Discriminator for Local InfoMax (f(x_i, h_j) -> scalar)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        B, N, D = h.shape
        x_exp = x.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([x_exp, h], dim=-1)
        return self.net(combined)


class DeepInfoMax(nn.Module):
    """Base Deep InfoMax model."""

    def __init__(
        self,
        encoder: nn.Module,
        discriminator: nn.Module,
        embed_dim: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.discriminator = discriminator
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def get_embeddings(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class GlobalInfoMax(DeepInfoMax):
    """Global InfoMax: Maximize mutual information between input and global representation.

    Args:
        encoder: Backbone network producing global representation
        embed_dim: Dimension of embedding space
        hidden_dim: Hidden dimension for discriminator
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        hidden_dim: int = 1024,
    ):
        discriminator = GlobalInfoMaxDiscriminator(embed_dim, hidden_dim)
        super().__init__(encoder, discriminator, embed_dim)

    def forward(self, x: Tensor, use_shadow: bool = True) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]

        h = self.encoder(x)

        if use_shadow:
            x_shadow = x[torch.randperm(batch_size)]
            h_shadow = self.encoder(x_shadow)
        else:
            h_shadow = h[torch.randperm(batch_size)]

        pos_score = self.discriminator(x, h)
        neg_score = self.discriminator(x, h_shadow)

        return pos_score, neg_score

    def get_embeddings(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class LocalInfoMax(DeepInfoMax):
    """Local InfoMax: Maximize mutual information between input and local representations.

    Useful for learning fine-grained representations where the input is
    partitioned into multiple regions.

    Args:
        encoder: Backbone network producing local representations
        embed_dim: Dimension of embedding space
        hidden_dim: Hidden dimension for discriminator
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        hidden_dim: int = 256,
    ):
        discriminator = LocalInfoMaxDiscriminator(embed_dim, hidden_dim)
        super().__init__(encoder, discriminator, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, n_regions = x.shape[0], x.shape[1] if x.dim() > 2 else 1

        if x.dim() == 4:
            x = x.reshape(batch_size, n_regions, -1)

        h = self.encoder(x)

        x_flat = x.reshape(batch_size * n_regions, -1)
        h_flat = h.reshape(batch_size * n_regions, -1)

        pos_score = self.discriminator(x_flat, h_flat)

        perm = torch.randperm(batch_size * n_regions)
        neg_h = h_flat[perm]
        neg_score = self.discriminator(x_flat, neg_h)

        return pos_score, neg_score

    def get_embeddings(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class ConvolutionalInfoMaxEncoder(nn.Module):
    """Convolutional encoder for local InfoMax.

    Uses convolutional layers to produce local feature maps.

    Args:
        in_channels: Number of input channels
        embed_dim: Output embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, embed_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x.flatten(2).transpose(1, 2)


def info_max_loss(pos_score: Tensor, neg_score: Tensor) -> Tensor:
    """Compute InfoMax loss using Jensen-Shannon divergence bound.

    Args:
        pos_score: Scores for positive pairs
        neg_score: Scores for negative pairs

    Returns:
        Scalar loss
    """
    pos_loss = F.softplus(-pos_score).mean()
    neg_loss = F.softplus(neg_score).mean()
    return pos_loss + neg_loss

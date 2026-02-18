"""
BigGAN: Large-Scale GAN for Conditional Image Generation.

Implements BigGAN as described in:
Brock, Donahue, and Simonyan (2019) "Large Scale GAN Training for High Fidelity Natural Image Synthesis"

Key features:
- Class-conditional generation
- Self-attention in both generator and discriminator
- Projection discriminator
- Spectral normalization
- Class-conditional batch normalization
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class ConditionalBatchNorm2d(nn.Module):
    """
    Class-conditional batch normalization.

    Embeds class labels and modulates batch norm parameters.
    """

    def __init__(self, num_features: int, num_classes: int, embedding_dim: int = 256):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.bn = nn.BatchNorm2d(num_features, affine=False)

        self.gamma = nn.Linear(embedding_dim, num_features)
        self.beta = nn.Linear(embedding_dim, num_features)

    def forward(self, x: Tensor, class_labels: Tensor) -> Tensor:
        """
        Apply conditional batch norm.

        Args:
            x: Input features [B, C, H, W]
            class_labels: Class indices [B]

        Returns:
            Normalized features
        """
        emb = self.embedding(class_labels)

        gamma = self.gamma(emb) + 1
        beta = self.beta(emb)

        x = self.bn(x)

        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)

        return x * gamma + beta


class SelfAttention(nn.Module):
    """Self-attention module for BigGAN."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply self-attention.

        Args:
            x: Input [B, C, H, W]

        Returns:
            Attended features
        """
        batch_size, C, H, W = x.shape

        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        out = self.gamma * out + x
        return out


class ResBlock(nn.Module):
    """Residual block with optional self-attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool = False,
        downsample: bool = False,
        num_classes: int = 0,
        attention: bool = False,
    ):
        super().__init__()

        self.conditional_norm = (
            ConditionalBatchNorm2d(in_channels, num_classes)
            if num_classes > 0
            else nn.BatchNorm2d(in_channels)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.upsample = nn.Identity()
        self.downsample = nn.Identity()

        if upsample:
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
        if downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, 3, stride=2, padding=1
            )

        self.skip = nn.Identity()
        if in_channels != out_channels or upsample:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)

        self.attention = SelfAttention(out_channels) if attention else None

    def forward(
        self,
        x: Tensor,
        class_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through residual block.

        Args:
            x: Input features
            class_labels: Class labels for conditional BN

        Returns:
            Output features
        """
        h = x

        if class_labels is not None and hasattr(self, "conditional_norm"):
            h = self.conditional_norm(h, class_labels)
        else:
            h = self.norm1(h)

        h = F.relu(h)

        if self.upsample is not None:
            h = self.upsample(h)

        h = self.conv1(h)

        if class_labels is not None and hasattr(self, "conditional_norm"):
            h = self.norm2(h)
        else:
            h = self.norm2(h)

        h = F.relu(h)
        h = self.conv2(h)

        if self.downsample is not None:
            h = self.downsample(h)

        x = self.skip(x)

        if self.upsample is not None:
            x = self.upsample(x)

        out = x + h

        if self.attention is not None:
            out = self.attention(out)

        return out


class ConditionalGenerator(nn.Module):
    """
    Conditional generator for BigGAN.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 1000,
        base_channels: int = 96,
        channels: Tuple[int, ...] = (96, 96, 96, 96, 96, 96),
        num_attention_layers: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.base_channels = base_channels

        self.embedding = nn.Embedding(num_classes, latent_dim)

        self.fc = nn.Linear(latent_dim * 2, channels[0] * 4 * 4)

        self.blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            attention = i >= len(channels) - num_attention_layers

            self.blocks.append(
                ResBlock(
                    in_ch,
                    out_ch,
                    upsample=True,
                    num_classes=num_classes,
                    attention=attention,
                )
            )

        self.final = nn.Sequential(
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(),
            nn.Conv2d(channels[-1], 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self,
        z: Tensor,
        class_labels: Tensor,
    ) -> Tensor:
        """
        Generate images from latent and class labels.

        Args:
            z: Latent codes [B, latent_dim]
            class_labels: Class indices [B]

        Returns:
            Generated images [B, 3, H, W]
        """
        class_emb = self.embedding(class_labels)

        h = torch.cat([z, class_emb], dim=-1)
        h = self.fc(h)
        h = h.view(h.size(0), -1, 4, 4)

        for block in self.blocks:
            h = block(h, class_labels)

        h = self.final(h)

        return h


class ProjectionDiscriminator(nn.Module):
    """
    Projection discriminator for class-conditional GANs.

    Projects both features and class embeddings for improved conditioning.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        base_channels: int = 64,
        channels: Tuple[int, ...] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.num_classes = num_classes

        self.conv_in = nn.Conv2d(3, base_channels, 3, padding=1)

        self.blocks = nn.ModuleList()

        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2 if i > 0 else 1, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(0.2),
                )
            )
            in_ch = out_ch

        self.embedding = nn.Embedding(num_classes, channels[-1])

        self.fc = nn.Sequential(
            nn.Linear(channels[-1] * 4 * 4 + channels[-1], channels[-1]),
            nn.LeakyReLU(0.2),
            nn.Linear(channels[-1], 1),
        )

    def forward(
        self,
        x: Tensor,
        class_labels: Tensor,
    ) -> Tensor:
        """
        Compute discriminator output.

        Args:
            x: Input images [B, 3, H, W]
            class_labels: Class indices [B]

        Returns:
            Discriminator logits
        """
        h = self.conv_in(x)

        for block in self.blocks:
            h = block(h)

        h = h.view(h.size(0), -1)

        proj = self.embedding(class_labels)

        h = torch.cat([h, proj], dim=-1)

        out = self.fc(h)

        return out


class BigGAN(nn.Module):
    """
    BigGAN model for large-scale conditional image generation.

    Args:
        latent_dim: Dimension of latent code
        num_classes: Number of output classes
        base_channels: Base number of channels
        channels: Channel configuration per layer
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 1000,
        base_channels: int = 96,
        channels: Tuple[int, ...] = (96, 96, 96, 96, 96, 96),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.generator = ConditionalGenerator(
            latent_dim=latent_dim,
            num_classes=num_classes,
            base_channels=base_channels,
            channels=channels,
        )

        self.discriminator = ProjectionDiscriminator(
            num_classes=num_classes,
            base_channels=base_channels,
        )

    def forward(
        self,
        z: Tensor,
        class_labels: Tensor,
        mode: str = "generate",
    ) -> Tensor:
        """
        Forward pass for either generation or discrimination.

        Args:
            z: Latent code
            class_labels: Class labels
            mode: 'generate' or 'discriminate'

        Returns:
            Generated images or discriminator logits
        """
        if mode == "generate":
            return self.generator(z, class_labels)
        elif mode == "discriminate":
            return self.discriminator(z, class_labels)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        class_labels: Optional[Tensor] = None,
        device: str = "cpu",
    ) -> Tensor:
        """
        Sample random images.

        Args:
            batch_size: Number of samples
            class_labels: Optional specific classes, otherwise random
            device: Device

        Returns:
            Generated images
        """
        z = torch.randn(batch_size, self.latent_dim, device=device)

        if class_labels is None:
            class_labels = torch.randint(
                0, self.num_classes, (batch_size,), device=device
            )

        return self.generator(z, class_labels)


class BigGANGenerator(nn.Module):
    """Simplified BigGAN generator for compatibility."""

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 1000,
        channels: Tuple[int, ...] = (96, 96, 96, 96, 96, 96),
    ):
        super().__init__()

        self.generator = ConditionalGenerator(
            latent_dim=latent_dim,
            num_classes=num_classes,
            channels=channels,
        )

    def forward(
        self,
        z: Tensor,
        class_labels: Tensor,
    ) -> Tensor:
        return self.generator(z, class_labels)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        class_labels: Optional[Tensor] = None,
        device: str = "cpu",
    ) -> Tensor:
        z = torch.randn(batch_size, self.generator.latent_dim, device=device)

        if class_labels is None:
            class_labels = torch.randint(
                0, self.generator.num_classes, (batch_size,), device=device
            )

        return self.generator(z, class_labels)

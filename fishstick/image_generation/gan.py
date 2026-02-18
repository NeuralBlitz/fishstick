"""
GAN (Generative Adversarial Network) variants for image generation.

This module provides various GAN architectures:
- DCGAN: Deep Convolutional GAN
- WGAN-GP: Wasserstein GAN with Gradient Penalty
- StyleGAN: Style-based generator with progressive growing
- ProgressiveGAN: GAN that progressively grows during training
"""

from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DCGenerator(nn.Module):
    """Deep Convolutional Generator for DCGAN.

    Args:
        latent_dim: Dimension of input noise vector
        ngf: Number of generator feature maps
        nc: Number of output channels
        image_size: Size of output images
    """

    def __init__(
        self,
        latent_dim: int = 100,
        ngf: int = 64,
        nc: int = 3,
        image_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.nc = nc

        layers = [
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        ]

        current_size = 4
        mult = 8
        while current_size < image_size // 2:
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        ngf * mult, ngf * mult // 2, 4, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(ngf * mult // 2),
                    nn.ReLU(True),
                ]
            )
            mult //= 2
            current_size *= 2

        layers.append(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.main(input)


class DCDiscriminator(nn.Module):
    """Deep Convolutional Discriminator for DCGAN.

    Args:
        ndf: Number of discriminator feature maps
        nc: Number of input channels
        image_size: Size of input images
    """

    def __init__(
        self,
        ndf: int = 64,
        nc: int = 3,
        image_size: int = 64,
    ):
        super().__init__()
        self.ndf = ndf
        self.nc = nc

        layers = [
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        current_size = image_size // 2
        mult = 1
        while current_size > 4 and mult < 8:
            layers.extend(
                [
                    nn.Conv2d(ndf * mult, ndf * mult * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * mult * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            mult *= 2
            current_size //= 2

        layers.append(nn.Conv2d(ndf * mult, 1, 4, 1, 0, bias=False))

        self.main = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.main(input).view(-1, 1).squeeze(1)


class DCGAN(nn.Module):
    """Deep Convolutional GAN combining generator and discriminator.

    Args:
        latent_dim: Dimension of latent noise vector
        ngf: Number of generator feature maps
        ndf: Number of discriminator feature maps
        nc: Number of image channels
        image_size: Size of images
    """

    def __init__(
        self,
        latent_dim: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        nc: int = 3,
        image_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.generator = DCGenerator(latent_dim, ngf, nc, image_size)
        self.discriminator = DCDiscriminator(ndf, nc, image_size)

    def generate(self, num_samples: int, device: torch.device) -> Tensor:
        noise = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        return self.generator(noise)

    def discriminate(self, images: Tensor) -> Tensor:
        return self.discriminator(images)


class WGANGenerator(nn.Module):
    """Wasserstein GAN Generator.

    Args:
        latent_dim: Dimension of input noise
        ngf: Number of generator feature maps
        nc: Number of output channels
        image_size: Size of output images
    """

    def __init__(
        self,
        latent_dim: int = 100,
        ngf: int = 64,
        nc: int = 3,
        image_size: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.main(input)


class WGANDiscriminator(nn.Module):
    """Wasserstein GAN Discriminator without sigmoid at output.

    Args:
        ndf: Number of discriminator feature maps
        nc: Number of input channels
        image_size: Size of input images
    """

    def __init__(
        self,
        ndf: int = 64,
        nc: int = 3,
        image_size: int = 64,
    ):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.main(input).view(-1)


class WGAN_GP(nn.Module):
    """Wasserstein GAN with Gradient Penalty.

    Args:
        latent_dim: Dimension of latent noise
        ngf: Number of generator feature maps
        ndf: Number of discriminator feature maps
        nc: Number of channels
        image_size: Image size
        gp_weight: Weight for gradient penalty
    """

    def __init__(
        self,
        latent_dim: int = 100,
        ngf: int = 64,
        ndf: int = 64,
        nc: int = 3,
        image_size: int = 64,
        gp_weight: float = 10.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight

        self.generator = WGANGenerator(latent_dim, ngf, nc, image_size)
        self.discriminator = WGANDiscriminator(ndf, nc, image_size)

    def gradient_penalty(
        self, real_images: Tensor, fake_images: Tensor, device: torch.device
    ) -> Tensor:
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)

        interpolated_logit = self.discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_logit,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def generate(self, num_samples: int, device: torch.device) -> Tensor:
        noise = torch.randn(num_samples, self.latent_dim, 1, 1, device=device)
        return self.generator(noise)


class StyleGANGenerator(nn.Module):
    """Style-based Generator for StyleGAN.

    Args:
        latent_dim: Dimension of intermediate latent
        style_dim: Dimension of style vector
        ngf: Number of generator feature maps
        nc: Number of output channels
    """

    def __init__(
        self,
        latent_dim: int = 512,
        style_dim: int = 512,
        ngf: int = 64,
        nc: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim

        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
        )

        self.initial_conv = nn.Conv2d(style_dim, ngf * 8, 3, padding=1)
        self.initial_to_rgb = nn.Conv2d(ngf, nc, 1)

        self.conv_blocks = nn.ModuleList()
        self.to_rgb_blocks = nn.ModuleList()

        channels = ngf * 8
        for i in range(3):
            self.conv_blocks.append(StyledConvBlock(channels, channels // 2, style_dim))
            self.to_rgb_blocks.append(nn.Conv2d(channels // 2, nc, 1))
            channels //= 2

        for i in range(3, 5):
            self.conv_blocks.append(StyledConvBlock(channels, channels, style_dim))
            self.to_rgb_blocks.append(nn.Conv2d(channels, nc, 1))

    def forward(
        self,
        z: Tensor,
        noise: Optional[Tensor] = None,
        mixing_range: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        styles = self.mapping(z)

        if mixing_range is not None:
            latent_z2 = torch.randn_like(z)
            styles2 = self.mapping(latent_z2)
            styles = torch.cat(
                [
                    styles2[:, : mixing_range[0]],
                    styles[:, mixing_range[0] : mixing_range[1]],
                    styles2[:, mixing_range[1] :],
                ],
                dim=1,
            )

        out = self.initial_conv(styles.view(-1, self.style_dim, 1, 1))

        for i, conv_block in enumerate(self.conv_blocks):
            out = conv_block(out, styles[:, i])

        return self.to_rgb_blocks[-1](out)


class StyledConvBlock(nn.Module):
    """Styled convolution block with adaptive instance normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.style = StyleMod(out_channels, style_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(
        self, x: Tensor, style: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        out = self.conv(x)

        if noise is not None:
            out = out + self.noise_strength * noise

        out = self.style(out, style)
        out = self.activation(out)
        return out


class StyleMod(nn.Module):
    """Style modulation with adaptive instance normalization."""

    def __init__(
        self,
        channels: int,
        style_dim: int,
    ):
        super().__init__()
        self.scale = nn.Linear(style_dim, channels)
        self.bias = nn.Linear(style_dim, channels)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        scale = self.scale(style).unsqueeze(-1).unsqueeze(-1)
        bias = self.bias(style).unsqueeze(-1).unsqueeze(-1)
        return x * (scale + 1) + bias


class StyleGANDiscriminator(nn.Module):
    """StyleGAN Discriminator with progressive feature extraction.

    Args:
        ndf: Number of discriminator feature maps
        nc: Number of input channels
    """

    def __init__(
        self,
        ndf: int = 64,
        nc: int = 3,
    ):
        super().__init__()

        self.from_rgb = nn.Conv2d(nc, ndf, 1)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ndf, ndf, 3, padding=1),
                    nn.LeakyReLU(0.2),
                ),
                nn.Sequential(
                    nn.Conv2d(ndf, ndf * 2, 3, padding=1),
                    nn.LeakyReLU(0.2),
                ),
                nn.Sequential(
                    nn.Conv2d(ndf * 2, ndf * 4, 3, padding=1),
                    nn.LeakyReLU(0.2),
                ),
                nn.Sequential(
                    nn.Conv2d(ndf * 4, ndf * 8, 3, padding=1),
                    nn.LeakyReLU(0.2),
                ),
            ]
        )

        self.final = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 4, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 8, 1, 1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.from_rgb(x)

        for block in self.blocks:
            x = F.avg_pool2d(block(x), 2)

        return self.final(x).squeeze()


class StyleGAN(nn.Module):
    """StyleGAN combining Style generator and discriminator.

    Args:
        latent_dim: Dimension of latent space
        style_dim: Dimension of style vectors
        ngf: Number of generator feature maps
        ndf: Number of discriminator feature maps
        nc: Number of channels
    """

    def __init__(
        self,
        latent_dim: int = 512,
        style_dim: int = 512,
        ngf: int = 64,
        ndf: int = 64,
        nc: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim

        self.generator = StyleGANGenerator(latent_dim, style_dim, ngf, nc)
        self.discriminator = StyleGANDiscriminator(ndf, nc)

    def generate(self, num_samples: int, device: torch.device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z)


class ProgressiveGAN(nn.Module):
    """Progressive GAN that grows during training.

    Args:
        latent_dim: Dimension of latent space
        ngf: Number of generator feature maps
        ndf: Number of discriminator feature maps
        nc: Number of channels
        max_size: Maximum image size (will be powers of 2: 4, 8, 16, 32, 64, etc.)
    """

    def __init__(
        self,
        latent_dim: int = 512,
        ngf: int = 512,
        ndf: int = 512,
        nc: int = 3,
        max_size: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_size = max_size

        self.current_size = 4
        self.alpha = 1.0

        self.generator = ProgressiveGenerator(latent_dim, ngf, nc)
        self.discriminator = ProgressiveDiscriminator(ndf, nc)

    def grow(self):
        """Grow the network to the next resolution."""
        if self.current_size < self.max_size:
            self.current_size *= 2

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def generate(self, num_samples: int, device: torch.device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z, self.current_size)


class ProgressiveGenerator(nn.Module):
    """Progressive Generator with fade-in layers."""

    def __init__(
        self,
        latent_dim: int,
        ngf: int,
        nc: int,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.progressive_blocks = nn.ModuleList()
        self.to_rgb_blocks = nn.ModuleList()

        channels = ngf
        for _ in range(5):
            self.progressive_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(latent_dim, channels, 4, 1, 0),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.2),
                )
            )
            self.to_rgb_blocks.append(nn.Conv2d(channels, nc, 1))
            channels //= 2

    def forward(self, z: Tensor, size: int) -> Tensor:
        out = z.view(z.size(0), self.latent_dim, 1, 1)

        num_blocks = int(math.log2(size) - 1)

        for i in range(num_blocks):
            out = self.progressive_blocks[i](out)
            if out.shape[2] < size:
                out = F.interpolate(out, scale_factor=2, mode="nearest")

        return self.to_rgb_blocks[num_blocks - 1](out)


class ProgressiveDiscriminator(nn.Module):
    """Progressive Discriminator with fade-in layers."""

    def __init__(
        self,
        ndf: int,
        nc: int,
    ):
        super().__init__()

        self.progressive_blocks = nn.ModuleList()
        self.from_rgb_blocks = nn.ModuleList()

        channels = ndf
        for _ in range(5):
            self.from_rgb_blocks.append(nn.Conv2d(nc, channels, 1))
            self.progressive_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(channels, channels * 2, 3, padding=1, stride=2),
                    nn.LeakyReLU(0.2),
                )
            )
            channels *= 2

        self.final = nn.Sequential(
            nn.Linear(channels, 1),
        )

    def forward(self, x: Tensor, size: int) -> Tensor:
        num_blocks = int(math.log2(size) - 1)

        out = self.from_rgb_blocks[num_blocks - 1](x)

        for i in range(num_blocks):
            out = self.progressive_blocks[i](out)

        return self.final(out.view(out.size(0), -1))

"""
Image-to-image translation models.

This module provides implementations for:
- Pix2Pix: Conditional GAN for image translation
- CycleGAN: Unpaired image translation with cycle consistency
- UNIT: Unsupervised Image-to-Image Translation
"""

from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class UNetGenerator(nn.Module):
    """U-Net based generator for image-to-image translation.

    Args:
        in_channels: Number of input image channels
        out_channels: Number of output image channels
        base_channels: Number of base channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 8)

        self.dec4 = self._upconv_block(base_channels * 8, base_channels * 8)
        self.dec3 = self._upconv_block(base_channels * 16, base_channels * 4)
        self.dec2 = self._upconv_block(base_channels * 8, base_channels * 2)
        self.dec1 = self._upconv_block(base_channels * 4, base_channels)

        self.final = nn.Conv2d(base_channels * 2, out_channels, 1)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def _upconv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        return self.final(d1)


class ResNetGenerator(nn.Module):
    """ResNet-based generator with residual blocks.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_residual_blocks: Number of residual blocks
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_residual_blocks: int = 9,
    ):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        )

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(256) for _ in range(num_residual_blocks)]
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.up1(x)
        x = self.up2(x)

        return self.final(x)


class ResidualBlock(nn.Module):
    """Residual block for generator."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class Pix2PixGenerator(ResNetGenerator):
    """Generator for Pix2Pix model.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__(in_channels, out_channels, num_residual_blocks=9)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator with sigmoid output.

    Args:
        in_channels: Number of input image channels
        base_channels: Number of base channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
        ]

        channels = base_channels
        for i in range(3):
            layers.extend(
                [
                    nn.Conv2d(channels, channels * 2, 4, 2, 1),
                    nn.InstanceNorm2d(channels * 2),
                    nn.LeakyReLU(0.2),
                ]
            )
            channels *= 2

        layers.append(nn.Conv2d(channels, 1, 4, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Pix2PixDiscriminator(PatchGANDiscriminator):
    """Discriminator for Pix2Pix model."""

    def __init__(
        self,
        in_channels: int = 3,
    ):
        super().__init__(in_channels, base_channels=64)


class Pix2Pix(nn.Module):
    """Pix2Pix model for paired image-to-image translation.

    Args:
        generator: Image generator network
        discriminator: PatchGAN discriminator
        lambda_l1: Weight for L1 reconstruction loss
    """

    def __init__(
        self,
        generator: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        lambda_l1: float = 100.0,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1

        self.gen = generator or Pix2PixGenerator()
        self.disc = discriminator or Pix2PixDiscriminator()

        self.gen_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, x: Tensor, y: Tensor, training: bool = True) -> Dict[str, Tensor]:
        fake_y = self.gen(x)

        if training:
            disc_fake = self.disc(torch.cat([x, fake_y], dim=1))
            gen_loss = self.gen_loss(disc_fake, torch.ones_like(disc_fake))

            l1_loss = self.l1_loss(fake_y, y)
            total_loss = gen_loss + self.lambda_l1 * l1_loss

            return {
                "gen_loss": total_loss,
                "l1_loss": l1_loss,
                "fake_y": fake_y,
            }
        else:
            return {"fake_y": fake_y}


class CycleGANGenerator(ResNetGenerator):
    """Generator for CycleGAN."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__(in_channels, out_channels, num_residual_blocks=9)


class CycleGANDiscriminator(nn.Module):
    """Discriminator for CycleGAN."""

    def __init__(
        self,
        in_channels: int = 3,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class CycleGAN(nn.Module):
    """CycleGAN for unpaired image-to-image translation.

    Args:
        lambda_cycle: Weight for cycle consistency loss
        lambda_identity: Weight for identity loss
    """

    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
    ):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        self.gen_XtoY = CycleGANGenerator()
        self.gen_YtoX = CycleGANGenerator()

        self.disc_X = CycleGANDiscriminator()
        self.disc_Y = CycleGANDiscriminator()

        self.adv_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        fake_y = self.gen_XtoY(x)
        recon_x = self.gen_YtoX(fake_y)

        fake_x = self.gen_YtoX(y)
        recon_y = self.gen_XtoY(fake_x)

        return {
            "fake_X": fake_x,
            "fake_Y": fake_y,
            "recon_X": recon_x,
            "recon_Y": recon_y,
        }

    def compute_losses(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        fake_y = self.gen_XtoY(x)
        fake_x = self.gen_YtoX(y)

        disc_x_loss = self.adv_loss(self.disc_X(x), torch.ones_like(self.disc_X(x)))
        disc_y_loss = self.adv_loss(self.disc_Y(y), torch.ones_like(self.disc_Y(y)))

        gen_x_loss = self.adv_loss(
            self.disc_X(fake_x), torch.ones_like(self.disc_X(fake_x))
        )
        gen_y_loss = self.adv_loss(
            self.disc_Y(fake_y), torch.ones_like(self.disc_Y(fake_y))
        )

        cycle_x_loss = self.l1_loss(self.gen_YtoX(fake_y), x)
        cycle_y_loss = self.l1_loss(self.gen_XtoY(fake_x), y)

        id_x = self.gen_YtoX(x)
        id_y = self.gen_XtoY(y)
        id_x_loss = self.l1_loss(id_x, x)
        id_y_loss = self.l1_loss(id_y, y)

        gen_total = (
            gen_x_loss
            + gen_y_loss
            + self.lambda_cycle * (cycle_x_loss + cycle_y_loss)
            + self.lambda_identity * (id_x_loss + id_y_loss)
        )

        return {
            "gen_loss": gen_total,
            "disc_x_loss": disc_x_loss,
            "disc_y_loss": disc_y_loss,
            "cycle_loss": cycle_x_loss + cycle_y_loss,
        }


class UNITGenerator(nn.Module):
    """Generator for Unsupervised Image-to-Image Translation (UNIT).

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.shared = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.shared(self.encoder(x))

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        return self.decode(z)


class UNITDiscriminator(nn.Module):
    """Discriminator for UNIT."""

    def __init__(
        self,
        in_channels: int = 3,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class UNIT(nn.Module):
    """Unsupervised Image-to-Image Translation (UNIT).

    Args:
        lambda_kl: Weight for KL divergence loss
        lambda_recon: Weight for reconstruction loss
    """

    def __init__(
        self,
        lambda_kl: float = 0.01,
        lambda_recon: float = 10.0,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_recon = lambda_recon

        self.gen_XtoY = UNITGenerator()
        self.gen_YtoX = UNITGenerator()

        self.disc_X = UNITDiscriminator()
        self.disc_Y = UNITDiscriminator()

        self.adv_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        z_x = self.gen_XtoY.encode(x)
        z_y = self.gen_YtoX.encode(y)

        fake_y = self.gen_XtoY.decode(z_x)
        fake_x = self.gen_YtoX.decode(z_y)

        recon_x = self.gen_YtoX.decode(z_x)
        recon_y = self.gen_XtoY.decode(z_y)

        return {
            "fake_X": fake_x,
            "fake_Y": fake_y,
            "recon_X": recon_x,
            "recon_Y": recon_y,
            "z_X": z_x,
            "z_Y": z_y,
        }

    def compute_losses(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        z_x = self.gen_XtoY.encode(x)
        z_y = self.gen_YtoX.encode(y)

        fake_y = self.gen_XtoY.decode(z_x)
        fake_x = self.gen_YtoX.decode(z_y)

        disc_x_real = self.disc_X(x)
        disc_x_fake = self.disc_X(fake_x.detach())
        disc_y_real = self.disc_Y(y)
        disc_y_fake = self.disc_Y(fake_y.detach())

        disc_x_loss = 0.5 * (
            self.mse_loss(disc_x_real, torch.ones_like(disc_x_real))
            + self.mse_loss(disc_x_fake, torch.zeros_like(disc_x_fake))
        )
        disc_y_loss = 0.5 * (
            self.mse_loss(disc_y_real, torch.ones_like(disc_y_real))
            + self.mse_loss(disc_y_fake, torch.zeros_like(disc_y_fake))
        )

        gen_x_loss = self.mse_loss(disc_X(fake_x), torch.ones_like(disc_X(fake_x)))
        gen_y_loss = self.mse_loss(disc_Y(fake_y), torch.ones_like(disc_Y(fake_y)))

        recon_x = self.gen_YtoX.decode(z_x)
        recon_y = self.gen_XtoY.decode(z_y)
        recon_loss = self.lambda_recon * (
            self.mse_loss(recon_x, x) + self.mse_loss(recon_y, y)
        )

        gen_total = gen_x_loss + gen_y_loss + recon_loss

        return {
            "gen_loss": gen_total,
            "disc_x_loss": disc_x_loss,
            "disc_y_loss": disc_y_loss,
            "recon_loss": recon_loss,
        }

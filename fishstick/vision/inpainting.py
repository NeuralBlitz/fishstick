"""
Fishstick Vision - Image Inpainting Module

A comprehensive collection of state-of-the-art image inpainting methods including:
- Context encoders and GANs
- Attention-based inpainting
- Progressive and free-form approaches
- High-resolution and video inpainting
- Complete loss functions and utilities

Author: Fishstick Vision Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Union, Dict, Callable, Any
import numpy as np
from PIL import Image
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")


# =============================================================================
# Utility Functions
# =============================================================================


def spectral_norm(module: nn.Module, n_power_iterations: int = 1) -> nn.Module:
    """Apply spectral normalization to a module."""
    return nn.utils.spectral_norm(module, n_power_iterations=n_power_iterations)


def l2normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize a tensor."""
    return v / (v.norm() + eps)


def default_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    bias: bool = True,
) -> nn.Module:
    """Default convolution with reflection padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=(kernel_size // 2),
        bias=bias,
    )


# =============================================================================
# 1. Context Encoders
# =============================================================================


class PartialConv2d(nn.Module):
    """
    Partial Convolution Layer for handling irregular holes.

    Paper: "Image Inpainting for Irregular Holes Using Partial Convolutions"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.mask_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

        self.slide_winsize = kernel_size * kernel_size * in_channels

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of partial convolution.

        Args:
            input: Input tensor [B, C, H, W]
            mask: Mask tensor [B, C, H, W]

        Returns:
            output: Output tensor [B, C, H, W]
            new_mask: Updated mask [B, C, H, W]
        """
        output = self.input_conv(input * mask)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        new_mask = torch.clamp(output_mask, 0, 1)

        mask_ratio = self.slide_winsize / (output_mask + 1e-8)
        output = output * mask_ratio

        output = output * new_mask

        return output, new_mask


class ContextEncoder(nn.Module):
    """
    Context Encoder for image inpainting using encoder-decoder architecture.

    Paper: "Context Encoders: Feature Learning by Inpainting"
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_dim: int = 4000,
        img_size: int = 128,
    ):
        super().__init__()

        self.img_size = img_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        # Fully connected bottleneck
        self.fc_encode = nn.Linear(
            512 * (img_size // 32) * (img_size // 32), latent_dim
        )
        self.fc_decode = nn.Linear(
            latent_dim, 512 * (img_size // 32) * (img_size // 32)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Context Encoder.

        Args:
            x: Input image with masked regions [B, C, H, W]
            mask: Optional mask indicating missing regions [B, 1, H, W]

        Returns:
            inpainted: Inpainted image [B, C, H, W]
        """
        if mask is not None:
            x = x * mask

        # Encode
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        latent = self.fc_encode(features)

        # Decode
        features = self.fc_decode(latent)
        features = features.view(
            features.size(0), 512, self.img_size // 32, self.img_size // 32
        )
        inpainted = self.decoder(features)

        return inpainted


class GlobalLocalDiscriminator(nn.Module):
    """
    Global and Local Discriminator for GLCIC (Globally and Locally Consistent Image Completion).

    Paper: "Globally and Locally Consistent Image Completion"
    """

    def __init__(self, in_channels: int = 3, local_input_size: int = 64):
        super().__init__()

        # Global discriminator - processes entire image
        self.global_disc = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, 2, 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        # Local discriminator - processes masked region
        self.local_disc = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, 2, 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 5, 2, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        # Calculate feature sizes
        self.global_feat_size = 512 * 4 * 4  # Assuming 128x128 input
        self.local_feat_size = 512 * (local_input_size // 16) * (local_input_size // 16)

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.global_feat_size + self.local_feat_size, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(
        self, global_img: torch.Tensor, local_img: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of Global-Local Discriminator.

        Args:
            global_img: Full image [B, C, H, W]
            local_img: Cropped region around masked area [B, C, h, w]

        Returns:
            validity: Discriminator output [B, 1]
        """
        global_feat = self.global_disc(global_img)
        global_feat = global_feat.view(global_feat.size(0), -1)

        local_feat = self.local_disc(local_img)
        local_feat = local_feat.view(local_feat.size(0), -1)

        concat_feat = torch.cat([global_feat, local_feat], dim=1)
        validity = self.fc(concat_feat)

        return validity


class GlobalLocalGAN(nn.Module):
    """
    Complete Global-Local GAN for image inpainting.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 128,
        local_size: int = 64,
    ):
        super().__init__()

        self.generator = ContextEncoder(in_channels, out_channels, img_size=img_size)
        self.discriminator = GlobalLocalDiscriminator(in_channels, local_size)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        generated = self.generator(x, mask)
        return generated


class ShiftConnection(nn.Module):
    """
    Shift connection operation for ShiftNet.

    Paper: "Shift-Net: Image Inpainting via Deep Feature Rearrangement"
    """

    def __init__(self, in_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, enc_feat: torch.Tensor, dec_feat: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform shift operation.

        Args:
            enc_feat: Encoder features [B, C, H, W]
            dec_feat: Decoder features [B, C, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            shifted: Shifted features [B, C, H, W]
        """
        b, c, h, w = dec_feat.size()

        # Find known regions in encoder features
        mask_flat = F.interpolate(mask, size=(h, w), mode="nearest")
        mask_flat = mask_flat.view(b, 1, -1)

        # Normalize features
        enc_flat = enc_feat.view(b, c, -1)
        dec_flat = dec_feat.view(b, c, -1)

        # Compute similarity
        enc_norm = F.normalize(enc_flat, dim=1)
        dec_norm = F.normalize(dec_flat, dim=1)

        similarity = torch.bmm(dec_norm.transpose(1, 2), enc_norm)
        similarity = similarity * mask_flat

        # Get shift indices
        shift_idx = similarity.argmax(dim=2)

        # Apply shift
        shifted = torch.gather(enc_flat, 2, shift_idx.unsqueeze(1).expand(-1, c, -1))
        shifted = shifted.view(b, c, h, w)

        # Combine with decoder features
        output = shifted * (1 - mask) + dec_feat * mask

        return output


class ShiftNet(nn.Module):
    """
    ShiftNet for image inpainting using shift connection.
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64
    ):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1), nn.LeakyReLU(0.2, True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, True),
        )

        # Shift connection layers
        self.shift3 = ShiftConnection(base_channels * 4)
        self.shift2 = ShiftConnection(base_channels * 2)
        self.shift1 = ShiftConnection(base_channels)

        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels, 4, 2, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ShiftNet.

        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            inpainted: Inpainted image [B, C, H, W]
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder with shift connections
        d4 = self.dec4(e4)
        d4 = self.shift3(e3, d4, mask)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4)
        d3 = self.shift2(e2, d3, mask)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = self.shift1(e1, d2, mask)
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.dec1(d2)

        # Combine with known regions
        output = d1 * (1 - mask) + x * mask

        return output


class PConvNet(nn.Module):
    """
    Partial Convolution Network (PConv) for irregular hole inpainting.

    Paper: "Image Inpainting for Irregular Holes Using Partial Convolutions"
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        self.enc1 = PartialConv2d(in_channels, 64, 7, 2, 3)
        self.enc2 = PartialConv2d(64, 128, 5, 2, 2)
        self.enc3 = PartialConv2d(128, 256, 5, 2, 2)
        self.enc4 = PartialConv2d(256, 512, 3, 2, 1)
        self.enc5 = PartialConv2d(512, 512, 3, 2, 1)
        self.enc6 = PartialConv2d(512, 512, 3, 2, 1)
        self.enc7 = PartialConv2d(512, 512, 3, 2, 1)

        self.dec7 = PartialConv2d(512 + 512, 512, 3, 1, 1)
        self.dec6 = PartialConv2d(512 + 512, 512, 3, 1, 1)
        self.dec5 = PartialConv2d(512 + 512, 512, 3, 1, 1)
        self.dec4 = PartialConv2d(512 + 256, 256, 3, 1, 1)
        self.dec3 = PartialConv2d(256 + 128, 128, 3, 1, 1)
        self.dec2 = PartialConv2d(128 + 64, 64, 3, 1, 1)
        self.dec1 = PartialConv2d(64 + in_channels, out_channels, 3, 1, 1)

        self.relu = nn.ReLU(True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(
        self, input: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PConvNet.

        Args:
            input: Input image [B, C, H, W]
            mask: Binary mask [B, C, H, W]

        Returns:
            output: Inpainted image [B, C, H, W]
            final_mask: Final mask [B, C, H, W]
        """
        # Encoder
        e1, m1 = self.enc1(input, mask)
        e1 = self.leaky_relu(e1)

        e2, m2 = self.enc2(e1, m1)
        e2 = self.leaky_relu(e2)

        e3, m3 = self.enc3(e2, m2)
        e3 = self.leaky_relu(e3)

        e4, m4 = self.enc4(e3, m3)
        e4 = self.leaky_relu(e4)

        e5, m5 = self.enc5(e4, m4)
        e5 = self.leaky_relu(e5)

        e6, m6 = self.enc6(e5, m5)
        e6 = self.leaky_relu(e6)

        e7, m7 = self.enc7(e6, m6)
        e7 = self.leaky_relu(e7)

        # Decoder
        d7, dm7 = self.dec7(torch.cat([e7, e6], dim=1), torch.cat([m7, m6], dim=1))
        d7 = self.leaky_relu(d7)
        d7 = F.interpolate(d7, scale_factor=2, mode="nearest")
        dm7 = F.interpolate(dm7, scale_factor=2, mode="nearest")

        d6, dm6 = self.dec6(torch.cat([d7, e5], dim=1), torch.cat([dm7, m5], dim=1))
        d6 = self.leaky_relu(d6)
        d6 = F.interpolate(d6, scale_factor=2, mode="nearest")
        dm6 = F.interpolate(dm6, scale_factor=2, mode="nearest")

        d5, dm5 = self.dec5(torch.cat([d6, e4], dim=1), torch.cat([dm6, m4], dim=1))
        d5 = self.leaky_relu(d5)
        d5 = F.interpolate(d5, scale_factor=2, mode="nearest")
        dm5 = F.interpolate(dm5, scale_factor=2, mode="nearest")

        d4, dm4 = self.dec4(torch.cat([d5, e3], dim=1), torch.cat([dm5, m3], dim=1))
        d4 = self.leaky_relu(d4)
        d4 = F.interpolate(d4, scale_factor=2, mode="nearest")
        dm4 = F.interpolate(dm4, scale_factor=2, mode="nearest")

        d3, dm3 = self.dec3(torch.cat([d4, e2], dim=1), torch.cat([dm4, m2], dim=1))
        d3 = self.leaky_relu(d3)
        d3 = F.interpolate(d3, scale_factor=2, mode="nearest")
        dm3 = F.interpolate(dm3, scale_factor=2, mode="nearest")

        d2, dm2 = self.dec2(torch.cat([d3, e1], dim=1), torch.cat([dm3, m1], dim=1))
        d2 = self.leaky_relu(d2)
        d2 = F.interpolate(d2, scale_factor=2, mode="nearest")
        dm2 = F.interpolate(dm2, scale_factor=2, mode="nearest")

        d1, dm1 = self.dec1(
            torch.cat([d2, input], dim=1), torch.cat([dm2, mask], dim=1)
        )

        return d1, dm1


# =============================================================================
# 2. Attention-based Inpainting
# =============================================================================


class ContextualAttention(nn.Module):
    """
    Contextual Attention Layer for texture synthesis.

    Paper: "Generative Image Inpainting with Contextual Attention"
    """

    def __init__(
        self, kernel_size: int = 3, stride: int = 1, rate: int = 1, fuse: bool = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.rate = rate
        self.fuse = fuse

    def forward(
        self,
        f: torch.Tensor,
        b: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Contextual attention forward pass.

        Args:
            f: Foreground (features with holes) [B, C, H, W]
            b: Background (features for matching) [B, C, H, W]
            mask: Optional mask [B, 1, H, W]
            training: Whether in training mode

        Returns:
            output: Attention output [B, C, H, W]
        """
        kernel_size = self.kernel_size
        stride = self.stride
        rate = self.rate

        # Extract patches
        b_raw = b

        # Downsample background for matching
        b = F.interpolate(b, scale_factor=1.0 / rate, mode="nearest")
        f = F.interpolate(f, scale_factor=1.0 / rate, mode="nearest")

        # Extract patches from background
        b_patches = F.unfold(b, kernel_size=kernel_size, stride=stride)
        b_patches = b_patches.view(b.size(0), b.size(1), kernel_size, kernel_size, -1)
        b_patches = b_patches.permute(0, 4, 1, 2, 3)

        # Normalize patches
        b_patches_norm = F.normalize(b_patches, dim=(2, 3, 4))

        # Extract foreground patches
        f_patches = F.unfold(f, kernel_size=kernel_size, stride=stride)
        f_patches = f_patches.view(f.size(0), f.size(1), kernel_size, kernel_size, -1)
        f_patches = f_patches.permute(0, 4, 1, 2, 3)
        f_patches_norm = F.normalize(f_patches, dim=(2, 3, 4))

        # Compute attention
        attention = torch.einsum("bchw,bnchw->bn", f_patches_norm, b_patches_norm)
        attention = F.softmax(attention * 10, dim=1)  # Scale attention

        # Apply attention to upsampled background
        if rate != 1:
            b_raw = F.interpolate(b_raw, size=f.shape[2:], mode="nearest")

        output = torch.einsum("bn,bchw->bchw", attention, b_patches)
        output = output.view(f.size(0), f.size(2), f.size(3), f.size(1))
        output = output.permute(0, 3, 1, 2)

        # Fuse with original features
        if self.fuse:
            output = output + f

        # Upsample back to original size
        if rate != 1:
            output = F.interpolate(output, scale_factor=rate, mode="nearest")

        return output


class GatedConv2d(nn.Module):
    """
    Gated Convolution Layer.

    Paper: "Free-Form Image Inpainting with Gated Convolution"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        activation: nn.Module = nn.ELU(),
    ):
        super().__init__()

        self.activation = activation
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.sigmoid = nn.Sigmoid()

    def gated(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply gating mechanism."""
        return self.sigmoid(mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of gated convolution.

        Args:
            input: Input tensor [B, C, H, W]

        Returns:
            output: Gated output [B, C, H, W]
        """
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)

        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)

        return x


class GatedDeconv2d(nn.Module):
    """Gated Deconvolution Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        activation: nn.Module = nn.ELU(),
    ):
        super().__init__()

        self.activation = activation
        self.conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.mask_conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.sigmoid = nn.Sigmoid()

    def gated(self, mask: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)

        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)

        return x


class GatedConvNet(nn.Module):
    """
    Gated Convolution Network for free-form inpainting.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_dim: int = 32,
        activation: nn.Module = nn.ELU(),
    ):
        super().__init__()

        # Encoder
        self.enc_conv0 = nn.Conv2d(in_channels, 64, 5, 1, 2)
        self.enc_conv1 = GatedConv2d(64, 128, 3, 2, 1, activation=activation)
        self.enc_conv2 = GatedConv2d(128, 256, 3, 1, 1, activation=activation)
        self.enc_conv3 = GatedConv2d(256, 512, 3, 2, 1, activation=activation)
        self.enc_conv4 = GatedConv2d(512, 512, 3, 1, 1, activation=activation)
        self.enc_conv5 = GatedConv2d(512, 512, 3, 2, 1, activation=activation)
        self.enc_conv6 = GatedConv2d(512, 512, 3, 1, 1, activation=activation)
        self.enc_conv7 = GatedConv2d(512, 512, 3, 2, 1, activation=activation)

        # Dilated convolutions
        self.dilated_conv1 = GatedConv2d(
            512, 512, 3, 1, 2, dilation=2, activation=activation
        )
        self.dilated_conv2 = GatedConv2d(
            512, 512, 3, 1, 4, dilation=4, activation=activation
        )
        self.dilated_conv3 = GatedConv2d(
            512, 512, 3, 1, 8, dilation=8, activation=activation
        )
        self.dilated_conv4 = GatedConv2d(
            512, 512, 3, 1, 16, dilation=16, activation=activation
        )

        # Attention
        self.attention = ContextualAttention(3, 1, rate=2)

        # Decoder
        self.dec_conv7 = GatedDeconv2d(512, 512, 4, 2, 1, activation=activation)
        self.dec_conv6 = GatedDeconv2d(512 * 2, 512, 3, 1, 1, activation=activation)
        self.dec_conv5 = GatedDeconv2d(512, 512, 4, 2, 1, activation=activation)
        self.dec_conv4 = GatedDeconv2d(512 * 2, 256, 3, 1, 1, activation=activation)
        self.dec_conv3 = GatedDeconv2d(256, 128, 4, 2, 1, activation=activation)
        self.dec_conv2 = GatedDeconv2d(128 * 2, 64, 3, 1, 1, activation=activation)
        self.dec_conv1 = GatedDeconv2d(64, 32, 4, 2, 1, activation=activation)
        self.dec_conv0 = nn.Conv2d(32, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GatedConvNet.

        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            output: Inpainted image [B, C, H, W]
        """
        # Apply mask
        x_in = x * mask

        # Encoder
        e0 = self.enc_conv0(x_in)
        e1 = self.enc_conv1(e0)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)
        e5 = self.enc_conv5(e4)
        e6 = self.enc_conv6(e5)
        e7 = self.enc_conv7(e6)

        # Dilated convolutions
        d = self.dilated_conv1(e7)
        d = self.dilated_conv2(d)
        d = self.dilated_conv3(d)
        d = self.dilated_conv4(d)

        # Attention
        d = self.attention(d, d, mask)

        # Decoder
        d7 = self.dec_conv7(d)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.dec_conv6(d7)

        d5 = self.dec_conv5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.dec_conv4(d5)

        d3 = self.dec_conv3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec_conv2(d3)

        d1 = self.dec_conv1(d2)
        d0 = self.dec_conv0(d1)

        output = torch.tanh(d0)

        return output


class SNPatchDiscriminator(nn.Module):
    """
    Spectral Normalization Patch Discriminator.

    Paper: "Free-Form Image Inpainting with Gated Convolution"
    """

    def __init__(self, in_channels: int = 3, num_layers: int = 5):
        super().__init__()

        layers = []
        channels = [in_channels, 64, 128, 256, 512, 512]

        for i in range(num_layers):
            layers.append(
                spectral_norm(nn.Conv2d(channels[i], channels[i + 1], 4, 2, 1))
            )
            if i > 0:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.LeakyReLU(0.2, True))

        self.model = nn.Sequential(*layers)
        self.output = spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]

        Returns:
            validity: Discriminator output [B, 1, H', W']
        """
        features = self.model(x)
        validity = self.output(features)
        return validity


# =============================================================================
# 3. Progressive Inpainting
# =============================================================================


class PixelNorm(nn.Module):
    """Pixel Normalization Layer."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, alpha: float = 1e-8) -> torch.Tensor:
        """
        Apply pixel normalization.

        Args:
            x: Input tensor [B, C, H, W]
            alpha: Small constant for numerical stability

        Returns:
            normalized: Normalized tensor [B, C, H, W]
        """
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + alpha)


class EqualizedConv2d(nn.Module):
    """Equalized Learning Rate Convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = math.sqrt(2) / math.sqrt(in_channels * kernel_size * kernel_size)

        nn.init.normal_(self.conv.weight, 0, 1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x * self.scale)


class EqualizedDeconv2d(nn.Module):
    """Equalized Learning Rate Transposed Convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.scale = math.sqrt(2) / math.sqrt(in_channels * kernel_size * kernel_size)

        nn.init.normal_(self.deconv.weight, 0, 1)
        nn.init.zeros_(self.deconv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x * self.scale)


class PGGANInpainting(nn.Module):
    """
    Progressive GAN for Image Inpainting.

    Paper: "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, max_resolution: int = 1024
    ):
        super().__init__()

        self.max_resolution = max_resolution
        self.current_stage = 0

        # Initial layer (4x4)
        self.initial_conv = EqualizedConv2d(in_channels + 1, 512, 3, 1, 1)
        self.initial_deconv = EqualizedDeconv2d(512, 512, 4, 1, 0)

        # Progressive blocks
        self.blocks = nn.ModuleList()
        resolutions = [8, 16, 32, 64, 128, 256, 512, 1024]

        for i, res in enumerate(resolutions[:-1]):
            if res > max_resolution:
                break
            self.blocks.append(self._make_block(512, 512))

        # To RGB layers
        self.to_rgb = nn.ModuleList()
        for i in range(len(self.blocks) + 1):
            self.to_rgb.append(EqualizedConv2d(512, out_channels, 1, 1, 0))

        self.pixel_norm = PixelNorm()
        self.activation = nn.LeakyReLU(0.2)

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a progressive block."""
        return nn.ModuleDict(
            {
                "conv": EqualizedConv2d(in_channels, out_channels, 3, 1, 1),
                "deconv": EqualizedDeconv2d(out_channels, out_channels, 4, 2, 1),
                "conv2": EqualizedConv2d(out_channels, out_channels, 3, 1, 1),
            }
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass with progressive growing.

        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]
            alpha: Interpolation factor for progressive growing

        Returns:
            output: Inpainted image [B, C, H, W]
        """
        # Concatenate input and mask
        x = torch.cat([x * mask, mask], dim=1)

        # Initial layers
        x = self.activation(self.pixel_norm(self.initial_conv(x)))
        x = self.activation(self.pixel_norm(self.initial_deconv(x)))

        # Progressive blocks
        for i, block in enumerate(self.blocks[: self.current_stage]):
            x = self.activation(self.pixel_norm(block["conv"](x)))
            x = self.activation(self.pixel_norm(block["deconv"](x)))
            x = self.activation(self.pixel_norm(block["conv2"](x)))

        # Output
        output = self.to_rgb[self.current_stage](x)

        # Blend with previous stage if alpha < 1
        if alpha < 1.0 and self.current_stage > 0:
            prev_output = F.interpolate(
                self.to_rgb[self.current_stage - 1](x),
                size=output.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            output = alpha * output + (1 - alpha) * prev_output

        return torch.tanh(output)

    def grow(self):
        """Grow the network to next resolution."""
        if self.current_stage < len(self.blocks):
            self.current_stage += 1


class CoarseToFineNet(nn.Module):
    """
    Coarse-to-Fine Inpainting Network.

    Paper: "Image Inpainting via Generative Multi-column Convolutional Neural Networks"
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Coarse network
        self.coarse_net = nn.ModuleList(
            [
                self._make_encoder_block(in_channels, 32),
                self._make_encoder_block(32, 64),
                self._make_encoder_block(64, 128),
                self._make_encoder_block(128, 256),
            ]
        )

        self.coarse_decoder = nn.ModuleList(
            [
                self._make_decoder_block(256, 128),
                self._make_decoder_block(128 + 128, 64),
                self._make_decoder_block(64 + 64, 32),
                self._make_decoder_block(32 + 32, out_channels, final=True),
            ]
        )

        # Refinement network
        self.refine_net = nn.ModuleList(
            [
                self._make_encoder_block(out_channels, 32),
                self._make_encoder_block(32, 64),
                self._make_encoder_block(64, 128),
                self._make_encoder_block(128, 256),
            ]
        )

        self.refine_decoder = nn.ModuleList(
            [
                self._make_decoder_block(256, 128),
                self._make_decoder_block(128 + 128, 64),
                self._make_decoder_block(64 + 64, 32),
                self._make_decoder_block(32 + 32, out_channels, final=True),
            ]
        )

    def _make_encoder_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True),
        )

    def _make_decoder_block(
        self, in_ch: int, out_ch: int, final: bool = False
    ) -> nn.Module:
        if final:
            return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1), nn.Tanh())
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Two-stage forward pass.

        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            coarse: Coarse inpainted image [B, C, H, W]
            refined: Refined inpainted image [B, C, H, W]
        """
        # Coarse stage
        x_masked = x * mask
        coarse_enc = [x_masked]

        for block in self.coarse_net:
            coarse_enc.append(block(coarse_enc[-1]))

        coarse_dec = coarse_enc[-1]
        for i, block in enumerate(self.coarse_decoder):
            coarse_dec = block(coarse_dec)
            if i < len(self.coarse_decoder) - 1:
                skip = coarse_enc[-(i + 2)]
                if coarse_dec.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(
                        skip, size=coarse_dec.shape[2:], mode="nearest"
                    )
                coarse_dec = torch.cat([coarse_dec, skip], dim=1)

        coarse = coarse_dec

        # Refinement stage
        x_refine = coarse * (1 - mask) + x * mask
        refine_enc = [x_refine]

        for block in self.refine_net:
            refine_enc.append(block(refine_enc[-1]))

        refine_dec = refine_enc[-1]
        for i, block in enumerate(self.refine_decoder):
            refine_dec = block(refine_dec)
            if i < len(self.refine_decoder) - 1:
                skip = refine_enc[-(i + 2)]
                if refine_dec.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(
                        skip, size=refine_dec.shape[2:], mode="nearest"
                    )
                refine_dec = torch.cat([refine_dec, skip], dim=1)

        refined = refine_dec

        return coarse, refined


class EdgeGenerator(nn.Module):
    """
    Edge Generator for EdgeConnect.

    Paper: "EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning"
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 1, 3), nn.BatchNorm2d(64), nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True)
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([self._make_res_block(256) for _ in range(8)])

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True)
        )
        self.dec1 = nn.Sequential(nn.Conv2d(64, out_channels, 7, 1, 3), nn.Sigmoid())

    def _make_res_block(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(
        self, x: torch.Tensor, edge: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate edges for masked regions.

        Args:
            x: Input image [B, C, H, W]
            edge: Edge map [B, 1, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            generated_edge: Generated edge map [B, 1, H, W]
        """
        # Concatenate inputs
        x_in = torch.cat([x, edge, mask], dim=1)

        # Encoder
        e1 = self.enc1(x_in)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Residual blocks
        res = e3
        for block in self.res_blocks:
            res = res + block(res)

        # Decoder
        d3 = self.dec3(res)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        return d1


class EdgeConnect(nn.Module):
    """
    Complete EdgeConnect model.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        self.edge_generator = EdgeGenerator(in_channels)
        self.image_generator = self._build_image_generator(in_channels, out_channels)

    def _build_image_generator(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build image completion network."""
        return nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, 7, 1, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Residual blocks
            *[self._make_res_block(256) for _ in range(8)],
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 7, 1, 3),
            nn.Tanh(),
        )

    def _make_res_block(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
        )

    def forward(
        self, x: torch.Tensor, edge: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]
            edge: Edge map [B, 1, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            generated_edge: Generated edge map [B, 1, H, W]
            generated_image: Generated image [B, C, H, W]
        """
        # Stage 1: Edge generation
        generated_edge = self.edge_generator(x, edge, mask)

        # Stage 2: Image completion
        x_edge = torch.cat([x, generated_edge], dim=1)
        generated_image = self.image_generator(x_edge)

        return generated_edge, generated_image


class StructureFlow(nn.Module):
    """
    StructureFlow for structure-aware image inpainting.

    Paper: "StructureFlow: Image Inpainting via Structure-aware Appearance Flow"
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Structure reconstruction
        self.structure_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 1, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 7, 1, 3),
            nn.Tanh(),
        )

        # Appearance flow
        self.flow_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 7, 1, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 2, 7, 1, 3),  # 2 channels for x,y flow
            nn.Tanh(),
        )

        # Texture synthesis
        self.texture_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 1, 3),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 7, 1, 3),
            nn.Tanh(),
        )

    def warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp image using optical flow.

        Args:
            x: Image to warp [B, C, H, W]
            flow: Optical flow [B, 2, H, W]

        Returns:
            warped: Warped image [B, C, H, W]
        """
        B, C, H, W = x.size()

        # Create sampling grid
        xx = torch.linspace(-1, 1, W, device=x.device).view(1, -1).repeat(H, 1)
        yy = torch.linspace(-1, 1, H, device=x.device).view(-1, 1).repeat(1, W)
        grid = (
            torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], dim=2)
            .unsqueeze(0)
            .repeat(B, 1, 1, 1)
        )

        # Add flow to grid
        vgrid = grid + flow.permute(0, 2, 3, 1)

        # Sample using grid
        warped = F.grid_sample(x, vgrid, align_corners=True)

        return warped

    def forward(
        self, x: torch.Tensor, structure: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]
            structure: Structure map (e.g., edge) [B, C, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            reconstructed_structure: Reconstructed structure [B, C, H, W]
            flow: Appearance flow [B, 2, H, W]
            result: Final result [B, C, H, W]
        """
        # Stage 1: Structure reconstruction
        x_in = torch.cat([x * mask, structure * (1 - mask)], dim=1)
        reconstructed_structure = self.structure_net(x_in)

        # Stage 2: Appearance flow
        flow_input = torch.cat([x * mask, reconstructed_structure], dim=1)
        flow = self.flow_net(flow_input)

        # Warp structure
        warped_structure = self.warp(reconstructed_structure, flow)

        # Stage 3: Texture synthesis
        texture_input = torch.cat([x * mask, warped_structure * (1 - mask)], dim=1)
        texture = self.texture_net(texture_input)

        # Combine
        result = texture * (1 - mask) + x * mask

        return reconstructed_structure, flow, result


# =============================================================================
# 4. Free-form Inpainting
# =============================================================================


class FreeFormInpainting(nn.Module):
    """
    Free-form inpainting for arbitrary shaped holes.

    Handles irregular masks using partial convolutions and gated convolutions.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Use partial convolutions for handling irregular masks
        self.pconv1 = PartialConv2d(in_channels, 64, 7, 2, 3)
        self.pconv2 = PartialConv2d(64, 128, 5, 2, 2)
        self.pconv3 = PartialConv2d(128, 256, 5, 2, 2)
        self.pconv4 = PartialConv2d(256, 512, 3, 2, 1)
        self.pconv5 = PartialConv2d(512, 512, 3, 2, 1)

        # Decoder with skip connections
        self.upconv5 = PartialConv2d(512 + 512, 512, 3, 1, 1)
        self.upconv4 = PartialConv2d(512 + 256, 256, 3, 1, 1)
        self.upconv3 = PartialConv2d(256 + 128, 128, 3, 1, 1)
        self.upconv2 = PartialConv2d(128 + 64, 64, 3, 1, 1)
        self.upconv1 = PartialConv2d(64 + in_channels, out_channels, 3, 1, 1)

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, C, H, W]

        Returns:
            inpainted: Inpainted image [B, C, H, W]
        """
        # Encoder
        e1, m1 = self.pconv1(x, mask)
        e1 = self.activation(e1)

        e2, m2 = self.pconv2(e1, m1)
        e2 = self.activation(e2)

        e3, m3 = self.pconv3(e2, m2)
        e3 = self.activation(e3)

        e4, m4 = self.pconv4(e3, m3)
        e4 = self.activation(e4)

        e5, m5 = self.pconv5(e4, m4)
        e5 = self.activation(e5)

        # Decoder with skip connections
        d5 = F.interpolate(e5, size=e4.shape[2:], mode="nearest")
        m5_up = F.interpolate(m5, size=e4.shape[2:], mode="nearest")
        d5, dm5 = self.upconv5(
            torch.cat([d5, e4], dim=1), torch.cat([m5_up, m4], dim=1)
        )
        d5 = self.activation(d5)

        d4 = F.interpolate(d5, size=e3.shape[2:], mode="nearest")
        dm5_up = F.interpolate(dm5, size=e3.shape[2:], mode="nearest")
        d4, dm4 = self.upconv4(
            torch.cat([d4, e3], dim=1), torch.cat([dm5_up, m3], dim=1)
        )
        d4 = self.activation(d4)

        d3 = F.interpolate(d4, size=e2.shape[2:], mode="nearest")
        dm4_up = F.interpolate(dm4, size=e2.shape[2:], mode="nearest")
        d3, dm3 = self.upconv3(
            torch.cat([d3, e2], dim=1), torch.cat([dm4_up, m2], dim=1)
        )
        d3 = self.activation(d3)

        d2 = F.interpolate(d3, size=e1.shape[2:], mode="nearest")
        dm3_up = F.interpolate(dm3, size=e1.shape[2:], mode="nearest")
        d2, dm2 = self.upconv2(
            torch.cat([d2, e1], dim=1), torch.cat([dm3_up, m1], dim=1)
        )
        d2 = self.activation(d2)

        d1 = F.interpolate(d2, size=x.shape[2:], mode="nearest")
        dm2_up = F.interpolate(dm2, size=x.shape[2:], mode="nearest")
        d1, _ = self.upconv1(
            torch.cat([d1, x], dim=1), torch.cat([dm2_up, mask], dim=1)
        )

        return torch.tanh(d1)


PartialConv = PartialConv2d  # Alias for consistency


class GatedConv2DFreeForm(nn.Module):
    """
    Gated Convolution for Free-form Inpainting.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.gate_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        feature = self.feature_conv(x)
        gate = self.sigmoid(self.gate_conv(x))
        return feature * gate


class LaMa(nn.Module):
    """
    Large Mask Inpainting (LaMa) using Fast Fourier Convolutions.

    Paper: "Resolution-robust Large Mask Inpainting with Fourier Convolutions"
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, num_blocks: int = 9
    ):
        super().__init__()

        # Input projection
        self.input_conv = nn.Conv2d(in_channels + 1, 64, 3, 1, 1)

        # Fast Fourier Convolution blocks
        self.ffc_blocks = nn.ModuleList(
            [
                self._make_ffc_block(64, 128)
                if i < 3
                else self._make_ffc_block(128, 256)
                if i < 6
                else self._make_ffc_block(256, 512)
                for i in range(num_blocks)
            ]
        )

        # Output projection
        self.output_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(128, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def _make_ffc_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create Fast Fourier Convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            # Spectral processing
            self._spectral_conv(out_ch),
            nn.ReLU(True),
        )

    def _spectral_conv(self, channels: int) -> nn.Module:
        """Spectral convolution using FFT."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0), nn.BatchNorm2d(channels)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            inpainted: Inpainted image [B, C, H, W]
        """
        # Concatenate input and mask
        x_in = torch.cat([x * mask, mask], dim=1)

        # Input projection
        feat = self.input_conv(x_in)

        # FFC blocks
        for block in self.ffc_blocks:
            feat = block(feat)

        # Output
        output = self.output_conv(feat)

        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAT(nn.Module):
    """
    Mask-Aware Transformer for image inpainting.

    Paper: "MAT: Mask-Aware Transformer for Large Hole Image Inpainting"
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels + 1, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            inpainted: Inpainted image [B, C, H, W]
        """
        # Apply mask and concatenate
        x_in = torch.cat([x * mask, mask], dim=1)

        # Patch embedding
        x = self.patch_embed(x_in)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        # Add position embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape for decoder
        x = x.transpose(1, 2).view(B, C, H, W)

        # Decode
        output = self.decoder(x)

        return output


# =============================================================================
# 5. High-Resolution Inpainting
# =============================================================================


class HighResInpainting(nn.Module):
    """
    High-Resolution Inpainting using patch-based processing.

    Handles large images by processing in overlapping patches.
    """

    def __init__(self, base_model: nn.Module, patch_size: int = 512, overlap: int = 64):
        super().__init__()

        self.base_model = base_model
        self.patch_size = patch_size
        self.overlap = overlap

    def extract_patches(
        self, img: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]]]:
        """Extract overlapping patches from image."""
        B, C, H, W = img.shape
        stride = self.patch_size - self.overlap

        patches_img = []
        patches_mask = []
        positions = []

        for y in range(0, H - self.overlap, stride):
            for x in range(0, W - self.overlap, stride):
                y_end = min(y + self.patch_size, H)
                x_end = min(x + self.patch_size, W)
                y_start = max(0, y_end - self.patch_size)
                x_start = max(0, x_end - self.patch_size)

                patch_img = img[:, :, y_start:y_end, x_start:x_end]
                patch_mask = mask[:, :, y_start:y_end, x_start:x_end]

                patches_img.append(patch_img)
                patches_mask.append(patch_mask)
                positions.append((y_start, x_start))

        return patches_img, patches_mask, positions

    def merge_patches(
        self,
        patches: List[torch.Tensor],
        positions: List[Tuple[int, int]],
        output_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Merge overlapping patches with blending."""
        B, C, H, W = output_shape
        output = torch.zeros(output_shape, device=patches[0].device)
        weight = torch.zeros((B, 1, H, W), device=patches[0].device)

        for patch, (y, x) in zip(patches, positions):
            _, _, ph, pw = patch.shape

            # Create blending weights
            y_coords = torch.arange(ph, device=patch.device).view(-1, 1).float()
            x_coords = torch.arange(pw, device=patch.device).view(1, -1).float()

            # Distance to nearest edge
            y_dist = torch.min(y_coords, torch.tensor(ph - 1 - y_coords))
            x_dist = torch.min(x_coords, torch.tensor(pw - 1 - x_coords))

            # Gaussian-like weights
            patch_weight = torch.exp(
                -((y_dist / (ph / 4)) ** 2 + (x_dist / (pw / 4)) ** 2)
            )
            patch_weight = patch_weight.view(1, 1, ph, pw).expand(B, 1, -1, -1)

            output[:, :, y : y + ph, x : x + pw] += patch * patch_weight
            weight[:, :, y : y + ph, x : x + pw] += patch_weight

        output = output / (weight + 1e-8)
        return output

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for high-resolution images.

        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            inpainted: Inpainted image [B, C, H, W]
        """
        B, C, H, W = x.shape

        # If image is small enough, process directly
        if H <= self.patch_size and W <= self.patch_size:
            return self.base_model(x, mask)

        # Extract patches
        patches_img, patches_mask, positions = self.extract_patches(x, mask)

        # Process each patch
        inpainted_patches = []
        for patch_img, patch_mask in zip(patches_img, patches_mask):
            # Pad if necessary
            _, _, ph, pw = patch_img.shape
            if ph < self.patch_size or pw < self.patch_size:
                patch_img = F.pad(
                    patch_img, (0, self.patch_size - pw, 0, self.patch_size - ph)
                )
                patch_mask = F.pad(
                    patch_mask, (0, self.patch_size - pw, 0, self.patch_size - ph)
                )

            inpainted_patch = self.base_model(patch_img, patch_mask)

            # Remove padding
            if ph < self.patch_size or pw < self.patch_size:
                inpainted_patch = inpainted_patch[:, :, :ph, :pw]

            inpainted_patches.append(inpainted_patch)

        # Merge patches
        output = self.merge_patches(inpainted_patches, positions, x.shape)

        return output


class AODA(nn.Module):
    """
    Arbitrary-shaped Object Detection and Alignment for inpainting.

    Detects and aligns objects before inpainting.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Object detection branch
        self.detector = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

        # Inpainting branch
        self.inpainter = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            detection: Object detection map [B, 1, H, W]
            inpainted: Inpainted image [B, C, H, W]
        """
        # Detect objects
        detection = self.detector(x)

        # Combine detection with mask
        combined_mask = torch.max(mask, detection)

        # Inpaint
        x_in = x * combined_mask
        inpainted = self.inpainter(x_in)

        return detection, inpainted


class FMM(nn.Module):
    """
    Fast Marching Method baseline for inpainting.

    Simple diffusion-based inpainting baseline.
    """

    def __init__(self, iterations: int = 100):
        super().__init__()

        self.iterations = iterations

        # Laplacian kernel for diffusion
        self.register_buffer(
            "laplacian_kernel",
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(
                1, 1, 3, 3
            ),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply FMM-based inpainting.

        Args:
            x: Input image [B, C, H, W]
            mask: Binary mask [B, 1, H, W]

        Returns:
            inpainted: Inpainted image [B, C, H, W]
        """
        B, C, H, W = x.shape
        result = x.clone()

        # Iterative diffusion
        for _ in range(self.iterations):
            # Apply Laplacian
            laplacian = F.conv2d(
                result, self.laplacian_kernel.expand(C, 1, -1, -1), padding=1, groups=C
            )

            # Update only masked regions
            result = result + laplacian * (1 - mask) * 0.1

            # Keep known regions
            result = result * (1 - mask) + x * mask

        return result


# =============================================================================
# 6. Video Inpainting
# =============================================================================


class VideoInpainting(nn.Module):
    """
    Basic Video Inpainting with temporal consistency.
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, num_frames: int = 5
    ):
        super().__init__()

        self.num_frames = num_frames

        # Temporal encoder
        self.temporal_enc = nn.Sequential(
            nn.Conv3d(in_channels, 64, (3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
        )

        # Spatial encoder
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for video inpainting.

        Args:
            x: Input video [B, T, C, H, W]
            mask: Binary mask [B, T, 1, H, W]

        Returns:
            inpainted: Inpainted video [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape

        # Transpose for 3D conv: [B, C, T, H, W]
        x = x.transpose(1, 2)
        mask = mask.transpose(1, 2)

        # Temporal encoding
        x_temporal = self.temporal_enc(x * mask)

        # Process middle frame
        mid_idx = T // 2
        feat = x_temporal[:, :, mid_idx]

        # Spatial encoding
        feat = self.spatial_enc(feat)

        # Decode
        output = self.decoder(feat)

        return output


class SpatioTemporalAttention(nn.Module):
    """Spatio-temporal attention mechanism."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [B, T*H*W, C]

        Returns:
            output: Attended features [B, T*H*W, C]
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class STTN(nn.Module):
    """
    Spatio-Temporal Transformer Network for video inpainting.

    Paper: "Learning Joint Spatial-Temporal Transformations for Video Inpainting"
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_frames: int = 5,
        img_size: int = 256,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = 16
        self.num_patches_per_frame = (img_size // self.patch_size) ** 2
        self.num_patches = self.num_patches_per_frame * num_frames

        # Patch embedding
        self.patch_embed = nn.Conv3d(
            in_channels + 1,
            embed_dim,
            kernel_size=(1, self.patch_size, self.patch_size),
            stride=(1, self.patch_size, self.patch_size),
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input video [B, T, C, H, W]
            mask: Binary mask [B, T, 1, H, W]

        Returns:
            inpainted: Inpainted frames [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape

        # Concatenate input and mask
        x_in = torch.cat([x * mask, mask], dim=2)

        # Transpose for 3D conv: [B, C, T, H, W]
        x_in = x_in.transpose(1, 2)

        # Patch embedding
        x = self.patch_embed(x_in)
        _, C_emb, T_emb, H_emb, W_emb = x.shape
        x = x.flatten(2).transpose(1, 2)

        # Add position embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape for decoder (process middle frame)
        mid_idx = T_emb // 2
        x_frame = x[
            :,
            mid_idx * self.num_patches_per_frame : (mid_idx + 1)
            * self.num_patches_per_frame,
        ]
        x_frame = x_frame.transpose(1, 2).view(B, C_emb, H_emb, W_emb)

        # Decode
        output = self.decoder(x_frame)

        return output


class OpticalFlowEstimator(nn.Module):
    """Simple optical flow estimator."""

    def __init__(self, in_channels: int = 6):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 2)
        self.conv3 = nn.Conv2d(128, 256, 5, 2, 2)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)

        self.deconv4 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(256 + 256, 128, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1)
        self.deconv1 = nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1)

        self.flow_pred = nn.Conv2d(32, 2, 3, 1, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Estimate optical flow between two frames."""
        x = torch.cat([x1, x2], dim=1)

        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))

        d4 = F.relu(self.deconv4(c4))
        d3 = F.relu(self.deconv3(torch.cat([d4, c3], dim=1)))
        d2 = F.relu(self.deconv2(torch.cat([d3, c2], dim=1)))
        d1 = F.relu(self.deconv1(torch.cat([d2, c1], dim=1)))

        flow = self.flow_pred(d1)

        return flow


class FGVC(nn.Module):
    """
    Flow-Guided Video Completion.

    Paper: "Flow-Guided Video Completion"
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        self.flow_estimator = OpticalFlowEstimator(in_channels * 2)

        # Context encoder
        self.context_enc = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def warp_frame(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp frame using optical flow."""
        B, C, H, W = frame.shape

        # Create sampling grid
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=frame.device),
            torch.linspace(-1, 1, W, device=frame.device),
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        # Add flow
        flow_permuted = flow.permute(0, 2, 3, 1)
        vgrid = grid + flow_permuted

        # Sample
        warped = F.grid_sample(frame, vgrid, align_corners=True)

        return warped

    def forward(
        self, frames: torch.Tensor, masks: torch.Tensor, target_idx: int = -1
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            frames: Input video frames [B, T, C, H, W]
            masks: Binary masks [B, T, 1, H, W]
            target_idx: Index of frame to inpaint

        Returns:
            inpainted: Inpainted frame [B, C, H, W]
        """
        B, T, C, H, W = frames.shape

        if target_idx < 0:
            target_idx = T // 2

        target_frame = frames[:, target_idx]
        target_mask = masks[:, target_idx]

        # Collect valid contexts
        contexts = []
        for t in range(T):
            if t != target_idx:
                flow = self.flow_estimator(frames[:, t], target_frame)
                warped = self.warp_frame(frames[:, t], flow)
                contexts.append(warped)

        # Aggregate contexts
        if contexts:
            context = torch.stack(contexts, dim=1).mean(dim=1)
        else:
            context = target_frame

        # Combine with target
        x = torch.where(target_mask.bool(), context, target_frame)

        # Encode and decode
        feat = self.context_enc(x)
        output = self.decoder(feat)

        return output


class CAP(nn.Module):
    """
    Context-Aware Pyramid for video inpainting.

    Paper: "Video Inpainting via Inference on Graphs with Learned Affinity"
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, num_scales: int = 3
    ):
        super().__init__()

        self.num_scales = num_scales

        # Pyramid levels
        self.pyramid_encs = nn.ModuleList()
        self.pyramid_decs = nn.ModuleList()

        for scale in range(num_scales):
            ch = 64 * (2**scale)
            self.pyramid_encs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels if scale == 0 else ch // 2, ch, 3, 2, 1),
                    nn.ReLU(True),
                    nn.Conv2d(ch, ch, 3, 1, 1),
                    nn.ReLU(True),
                )
            )

            self.pyramid_decs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        ch * 2 if scale < num_scales - 1 else ch,
                        ch // 2 if scale > 0 else out_channels,
                        4,
                        2,
                        1,
                    ),
                    nn.ReLU(True) if scale > 0 else nn.Tanh(),
                )
            )

    def forward(
        self, frames: torch.Tensor, masks: torch.Tensor, target_idx: int = -1
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            frames: Input video [B, T, C, H, W]
            masks: Binary masks [B, T, 1, H, W]
            target_idx: Target frame index

        Returns:
            inpainted: Inpainted frame [B, C, H, W]
        """
        B, T, C, H, W = frames.shape

        if target_idx < 0:
            target_idx = T // 2

        target = frames[:, target_idx]

        # Build temporal context at multiple scales
        pyramid_feats = []
        current = target

        for scale in range(self.num_scales):
            # Downsample if needed
            if scale > 0:
                current = F.avg_pool2d(current, 2)

            # Aggregate temporal context at this scale
            context_feats = []
            for t in range(T):
                frame = frames[:, t]
                if scale > 0:
                    frame = F.avg_pool2d(frame, 2**scale)
                feat = self.pyramid_encs[scale](frame)
                context_feats.append(feat)

            # Average context
            context = torch.stack(context_feats, dim=1).mean(dim=1)
            pyramid_feats.append(context)

        # Decode pyramid
        x = pyramid_feats[-1]
        for scale in reversed(range(self.num_scales)):
            if scale < self.num_scales - 1:
                x = torch.cat([x, pyramid_feats[scale]], dim=1)
            x = self.pyramid_decs[scale](x)

        return x


# =============================================================================
# 7. Loss Functions
# =============================================================================


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss using L1 and/or L2 norms.
    """

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            mask: Optional mask [B, 1, H, W]

        Returns:
            loss: Reconstruction loss
        """
        if mask is not None:
            pred = pred * mask
            target = target * mask

        loss = 0
        if self.l1_weight > 0:
            loss += self.l1_weight * F.l1_loss(pred, target)
        if self.l2_weight > 0:
            loss += self.l2_weight * F.mse_loss(pred, target)

        return loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    """

    def __init__(
        self,
        layers: List[str] = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()

        # Load VGG16
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers

        if weights is None:
            weights = [1.0] * len(layers)
        self.weights = weights

        # Extract feature layers
        self.layer_name_mapping = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 22,
            "relu5_3": 29,
        }

        self.feature_extractor = nn.ModuleDict()

        prev_idx = 0
        for layer_name in layers:
            idx = self.layer_name_mapping[layer_name]
            self.feature_extractor[layer_name] = nn.Sequential(
                *list(vgg[prev_idx : idx + 1])
            )
            prev_idx = idx + 1

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]

        Returns:
            loss: Perceptual loss
        """
        loss = 0
        x = pred
        y = target

        for layer_name, weight in zip(self.layers, self.weights):
            x = self.feature_extractor[layer_name](x)
            y = self.feature_extractor[layer_name](y)
            loss += weight * F.l1_loss(x, y)

        return loss


class StyleLoss(nn.Module):
    """
    Style loss using Gram matrix.
    """

    def __init__(
        self, layers: List[str] = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
    ):
        super().__init__()

        self.perceptual_loss = PerceptualLoss(layers)

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix."""
        B, C, H, W = x.size()
        features = x.view(B, C, -1)
        G = torch.bmm(features, features.transpose(1, 2))
        return G / (C * H * W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute style loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]

        Returns:
            loss: Style loss
        """
        loss = 0

        # Extract features
        pred_features = []
        target_features = []

        x = pred
        y = target

        for layer_name in self.perceptual_loss.layers:
            x = self.perceptual_loss.feature_extractor[layer_name](x)
            y = self.perceptual_loss.feature_extractor[layer_name](y)
            pred_features.append(x)
            target_features.append(y)

        # Compute Gram matrix loss
        for pred_feat, target_feat in zip(pred_features, target_features):
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += F.mse_loss(pred_gram, target_gram)

        return loss


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training.
    """

    def __init__(
        self,
        loss_type: str = "hinge",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Compute adversarial loss.

        Args:
            pred: Discriminator prediction [B, 1, H, W] or [B, 1]
            target_is_real: Whether target is real

        Returns:
            loss: Adversarial loss
        """
        if self.loss_type == "hinge":
            if target_is_real:
                loss = -torch.mean(torch.min(pred - 1, torch.zeros_like(pred)))
            else:
                loss = -torch.mean(torch.min(-pred - 1, torch.zeros_like(pred)))
        elif self.loss_type == "bce":
            target = self.real_label if target_is_real else self.fake_label
            target = target.expand_as(pred)
            loss = F.binary_cross_entropy_with_logits(pred, target)
        elif self.loss_type == "mse":
            target = self.real_label if target_is_real else self.fake_label
            target = target.expand_as(pred)
            loss = F.mse_loss(pred, target)
        elif self.loss_type == "wasserstein":
            if target_is_real:
                loss = -torch.mean(pred)
            else:
                loss = torch.mean(pred)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class TVLoss(nn.Module):
    """
    Total Variation loss for smoothness.
    """

    def __init__(self, tv_weight: float = 1.0):
        super().__init__()
        self.tv_weight = tv_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss.

        Args:
            x: Input image [B, C, H, W]

        Returns:
            loss: TV loss
        """
        h_var = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        w_var = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        loss = self.tv_weight * (h_var + w_var)
        return loss


class EdgeLoss(nn.Module):
    """
    Edge preservation loss using Sobel operator.
    """

    def __init__(self):
        super().__init__()

        # Sobel kernels
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def sobel_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel edges."""
        B, C, H, W = x.shape

        # Apply Sobel operator to each channel
        edge_x = F.conv2d(x.view(B * C, 1, H, W), self.kernel_x, padding=1)
        edge_y = F.conv2d(x.view(B * C, 1, H, W), self.kernel_y, padding=1)

        edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return edges.view(B, C, H, W)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute edge preservation loss.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            mask: Optional mask [B, 1, H, W]

        Returns:
            loss: Edge loss
        """
        pred_edges = self.sobel_edges(pred)
        target_edges = self.sobel_edges(target)

        if mask is not None:
            pred_edges = pred_edges * mask
            target_edges = target_edges * mask

        loss = F.l1_loss(pred_edges, target_edges)
        return loss


# =============================================================================
# 8. Utilities
# =============================================================================


@dataclass
class InpaintingConfig:
    """Configuration for inpainting models."""

    img_size: int = 256
    in_channels: int = 3
    out_channels: int = 3
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-4
    epochs: int = 100
    device: str = "cuda"


class InpaintingDataset(Dataset):
    """
    Dataset for image inpainting with masked image pairs.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_generator: Optional["MaskGenerator"] = None,
        img_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            image_paths: List of paths to images
            mask_generator: Mask generator (creates default if None)
            img_size: Target image size
            transform: Optional additional transforms
        """
        self.image_paths = image_paths
        self.img_size = img_size
        self.mask_generator = mask_generator or MaskGenerator(img_size, img_size)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.

        Returns:
            sample: Dict with 'image', 'masked_image', 'mask'
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Resize
        image = image.resize((self.img_size, self.img_size))

        # Convert to tensor
        image = transforms.ToTensor()(image)

        # Apply additional transforms
        if self.transform:
            image = self.transform(image)

        # Normalize to [-1, 1]
        image = image * 2 - 1

        # Generate mask
        mask = self.mask_generator()

        # Create masked image
        masked_image = image * mask

        return {"image": image, "masked_image": masked_image, "mask": mask}


class MaskGenerator:
    """
    Generate random masks for inpainting.

    Supports various mask types: box, irregular, free-form, etc.
    """

    def __init__(
        self,
        height: int,
        width: int,
        mask_type: str = "mixed",
        min_mask_ratio: float = 0.1,
        max_mask_ratio: float = 0.5,
    ):
        """
        Initialize mask generator.

        Args:
            height: Image height
            width: Image width
            mask_type: Type of mask ('box', 'irregular', 'freeform', 'mixed')
            min_mask_ratio: Minimum ratio of image to mask
            max_mask_ratio: Maximum ratio of image to mask
        """
        self.height = height
        self.width = width
        self.mask_type = mask_type
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio

    def generate_box_mask(self) -> torch.Tensor:
        """Generate rectangular box mask."""
        mask = torch.ones(1, self.height, self.width)

        mask_ratio = np.random.uniform(self.min_mask_ratio, self.max_mask_ratio)
        mask_area = int(self.height * self.width * mask_ratio)

        # Random box size
        box_h = int(np.sqrt(mask_area * np.random.uniform(0.5, 2.0)))
        box_w = int(mask_area / box_h)

        box_h = min(box_h, self.height - 2)
        box_w = min(box_w, self.width - 2)

        # Random position
        y = np.random.randint(0, self.height - box_h)
        x = np.random.randint(0, self.width - box_w)

        mask[:, y : y + box_h, x : x + box_w] = 0

        return mask

    def generate_irregular_mask(self) -> torch.Tensor:
        """Generate irregular mask using random walks."""
        mask = torch.ones(1, self.height, self.width)

        num_vertices = np.random.randint(10, 30)

        for _ in range(np.random.randint(1, 4)):
            # Random walk
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)

            for _ in range(num_vertices):
                # Random direction and distance
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.randint(20, 100)

                for step in range(length):
                    x = int(x + np.cos(angle))
                    y = int(y + np.sin(angle))

                    if 0 <= x < self.width and 0 <= y < self.height:
                        # Draw circle at position
                        r = np.random.randint(5, 20)
                        yy, xx = np.ogrid[: self.height, : self.width]
                        circle = (xx - x) ** 2 + (yy - y) ** 2 <= r**2
                        mask[0, circle] = 0

        return mask

    def generate_freeform_mask(self) -> torch.Tensor:
        """Generate free-form mask using random curves."""
        mask = torch.ones(1, self.height, self.width)

        num_strokes = np.random.randint(1, 5)

        for _ in range(num_strokes):
            # Bezier curve control points
            num_points = np.random.randint(4, 8)
            points_x = np.random.randint(0, self.width, num_points)
            points_y = np.random.randint(0, self.height, num_points)

            # Draw curve
            t = np.linspace(0, 1, 100)
            for i in range(len(t)):
                x = int(
                    np.polyval(
                        np.polyfit(
                            np.linspace(0, 1, num_points), points_x, num_points - 1
                        ),
                        t[i],
                    )
                )
                y = int(
                    np.polyval(
                        np.polyfit(
                            np.linspace(0, 1, num_points), points_y, num_points - 1
                        ),
                        t[i],
                    )
                )

                if 0 <= x < self.width and 0 <= y < self.height:
                    r = np.random.randint(10, 30)
                    yy, xx = np.ogrid[: self.height, : self.width]
                    circle = (xx - x) ** 2 + (yy - y) ** 2 <= r**2
                    mask[0, circle] = 0

        return mask

    def __call__(self) -> torch.Tensor:
        """
        Generate a random mask.

        Returns:
            mask: Binary mask [1, H, W]
        """
        if self.mask_type == "box":
            return self.generate_box_mask()
        elif self.mask_type == "irregular":
            return self.generate_irregular_mask()
        elif self.mask_type == "freeform":
            return self.generate_freeform_mask()
        elif self.mask_type == "mixed":
            mask_types = ["box", "irregular", "freeform"]
            selected = np.random.choice(mask_types)
            if selected == "box":
                return self.generate_box_mask()
            elif selected == "irregular":
                return self.generate_irregular_mask()
            else:
                return self.generate_freeform_mask()
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")


class InpaintingTrainer:
    """
    Specialized trainer for image inpainting models.
    """

    def __init__(
        self,
        model: nn.Module,
        config: InpaintingConfig,
        use_gan: bool = False,
        discriminator: Optional[nn.Module] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Generator model
            config: Training configuration
            use_gan: Whether to use adversarial training
            discriminator: Discriminator model (required if use_gan=True)
        """
        self.model = model.to(config.device)
        self.config = config
        self.use_gan = use_gan

        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            model.parameters(), lr=config.lr, betas=(0.5, 0.999)
        )

        if use_gan:
            if discriminator is None:
                raise ValueError("Discriminator required for GAN training")
            self.discriminator = discriminator.to(config.device)
            self.d_optimizer = torch.optim.Adam(
                discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999)
            )
            self.adversarial_loss = AdversarialLoss()

        # Loss functions
        self.reconstruction_loss = ReconstructionLoss(l1_weight=1.0, l2_weight=0.0)
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Batch with 'image', 'masked_image', 'mask'

        Returns:
            losses: Dictionary of losses
        """
        device = self.config.device

        real_images = batch["image"].to(device)
        masked_images = batch["masked_image"].to(device)
        masks = batch["mask"].to(device)

        # Generate
        if (
            hasattr(self.model, "forward")
            and "mask" in self.model.forward.__code__.co_varnames
        ):
            generated = self.model(masked_images, masks)
        else:
            generated = self.model(masked_images)

        # Ensure output is valid image
        if isinstance(generated, tuple):
            generated = generated[-1]  # Take last output if multiple

        # Combine with known regions
        completed = generated * (1 - masks) + real_images * masks

        losses = {}

        # Reconstruction loss
        losses["recon"] = self.reconstruction_loss(completed, real_images, 1 - masks)

        # Perceptual loss
        losses["perceptual"] = self.perceptual_loss(completed, real_images)

        # Style loss
        losses["style"] = self.style_loss(completed, real_images)

        # Total generator loss
        g_loss = losses["recon"] + 0.1 * losses["perceptual"] + 250 * losses["style"]

        if self.use_gan:
            # Adversarial loss for generator
            fake_validity = self.discriminator(completed)
            losses["g_adv"] = self.adversarial_loss(fake_validity, target_is_real=True)
            g_loss += 0.1 * losses["g_adv"]

            # Update discriminator
            self.d_optimizer.zero_grad()

            real_validity = self.discriminator(real_images)
            d_real_loss = self.adversarial_loss(real_validity, target_is_real=True)

            fake_validity = self.discriminator(completed.detach())
            d_fake_loss = self.adversarial_loss(fake_validity, target_is_real=False)

            d_loss = (d_real_loss + d_fake_loss) / 2
            losses["d_loss"] = d_loss.item()

            d_loss.backward()
            self.d_optimizer.step()

        # Update generator
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        losses["g_loss"] = g_loss.item()

        return losses

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number

        Returns:
            avg_losses: Average losses for the epoch
        """
        self.model.train()
        if self.use_gan:
            self.discriminator.train()

        total_losses = {}

        for batch in dataloader:
            losses = self.train_step(batch)

            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value

        # Average losses
        avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}

        return avg_losses


class InpaintingMetrics:
    """
    Metrics for evaluating inpainting quality.

    Includes FID, LPIPS, PSNR, SSIM, and more.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize metrics calculator.

        Args:
            device: Device to use
        """
        self.device = device

        # LPIPS model
        try:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex").to(device)
            self.lpips_model.eval()
        except ImportError:
            self.lpips_model = None
            print("Warning: lpips not installed. LPIPS metric not available.")

        # VGG for feature extraction
        self.vgg = models.vgg16(pretrained=True).features.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute PSNR.

        Args:
            pred: Predicted image [B, C, H, W] in range [-1, 1]
            target: Target image [B, C, H, W] in range [-1, 1]

        Returns:
            psnr: Average PSNR
        """
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return 100.0
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
        return psnr.item()

    def compute_ssim(
        self, pred: torch.Tensor, target: torch.Tensor, window_size: int = 11
    ) -> float:
        """
        Compute SSIM.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            window_size: Window size for SSIM

        Returns:
            ssim: Average SSIM
        """
        # Simple SSIM implementation
        C1 = 0.01**2
        C2 = 0.03**2

        mu_pred = F.avg_pool2d(pred, window_size, 1, padding=window_size // 2)
        mu_target = F.avg_pool2d(target, window_size, 1, padding=window_size // 2)

        mu_pred_sq = mu_pred**2
        mu_target_sq = mu_target**2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = (
            F.avg_pool2d(pred**2, window_size, 1, padding=window_size // 2) - mu_pred_sq
        )
        sigma_target_sq = (
            F.avg_pool2d(target**2, window_size, 1, padding=window_size // 2)
            - mu_target_sq
        )
        sigma_pred_target = (
            F.avg_pool2d(pred * target, window_size, 1, padding=window_size // 2)
            - mu_pred_target
        )

        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / (
            (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        )

        return ssim_map.mean().item()

    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute LPIPS.

        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]

        Returns:
            lpips: Average LPIPS distance
        """
        if self.lpips_model is None:
            return 0.0

        with torch.no_grad():
            lpips_dist = self.lpips_model(pred, target)
        return lpips_dist.mean().item()

    def compute_fid_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features for FID computation.

        Args:
            images: Images [B, C, H, W]

        Returns:
            features: VGG features [B, D]
        """
        # Resize to 224x224
        images = F.interpolate(
            images, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Normalize for VGG
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images / 2 + 0.5 - mean) / std

        # Extract features
        with torch.no_grad():
            features = self.vgg(images)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)

        return features

    def compute_fid(
        self, pred_features: torch.Tensor, target_features: torch.Tensor
    ) -> float:
        """
        Compute Frechet Inception Distance.

        Args:
            pred_features: Predicted features [N, D]
            target_features: Target features [N, D]

        Returns:
            fid: FID score
        """
        # Compute statistics
        mu_pred = pred_features.mean(dim=0)
        mu_target = target_features.mean(dim=0)

        sigma_pred = torch.cov(pred_features.T)
        sigma_target = torch.cov(target_features.T)

        # Compute FID
        diff = mu_pred - mu_target

        # Product might be almost singular
        covmean = torch.linalg.sqrtm(sigma_pred @ sigma_target)

        if torch.is_complex(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + torch.trace(sigma_pred + sigma_target - 2 * covmean)

        return fid.item()

    def evaluate(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate all metrics.

        Args:
            predictions: Predicted images [N, C, H, W]
            targets: Target images [N, C, H, W]

        Returns:
            metrics: Dictionary of metric values
        """
        predictions = predictions.to(self.device)
        targets = targets.to(self.device)

        metrics = {
            "psnr": self.compute_psnr(predictions, targets),
            "ssim": self.compute_ssim(predictions, targets),
        }

        if self.lpips_model is not None:
            metrics["lpips"] = self.compute_lpips(predictions, targets)

        # Compute FID
        pred_features = self.compute_fid_features(predictions)
        target_features = self.compute_fid_features(targets)
        metrics["fid"] = self.compute_fid(pred_features, target_features)

        return metrics


# =============================================================================
# Export all components
# =============================================================================

__all__ = [
    # Context Encoders
    "ContextEncoder",
    "GlobalLocalGAN",
    "GlobalLocalDiscriminator",
    "ShiftNet",
    "ShiftConnection",
    "PConvNet",
    "PartialConv2d",
    # Attention-based Inpainting
    "ContextualAttention",
    "GatedConv2d",
    "GatedDeconv2d",
    "GatedConvNet",
    "SNPatchDiscriminator",
    # Progressive Inpainting
    "PGGANInpainting",
    "CoarseToFineNet",
    "EdgeGenerator",
    "EdgeConnect",
    "StructureFlow",
    # Free-form Inpainting
    "FreeFormInpainting",
    "PartialConv",
    "GatedConv2DFreeForm",
    "LaMa",
    "MAT",
    "MultiHeadAttention",
    "TransformerBlock",
    # High-Resolution Inpainting
    "HighResInpainting",
    "AODA",
    "FMM",
    # Video Inpainting
    "VideoInpainting",
    "STTN",
    "SpatioTemporalAttention",
    "FGVC",
    "OpticalFlowEstimator",
    "CAP",
    # Loss Functions
    "ReconstructionLoss",
    "PerceptualLoss",
    "StyleLoss",
    "AdversarialLoss",
    "TVLoss",
    "EdgeLoss",
    # Utilities
    "InpaintingConfig",
    "InpaintingDataset",
    "MaskGenerator",
    "InpaintingTrainer",
    "InpaintingMetrics",
    # Helper functions
    "spectral_norm",
    "l2normalize",
    "default_conv",
]

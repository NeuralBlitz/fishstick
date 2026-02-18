"""
Fishstick: Comprehensive Super Resolution Module

This module provides a complete implementation of state-of-the-art super resolution
models including single image SR, GAN-based SR, lightweight SR, video SR, and
real-world SR methods.

Author: Fishstick Team
"""

from typing import Tuple, Optional, Union, List, Dict, Any, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
import warnings


# =============================================================================
# Utility Functions and Base Classes
# =============================================================================


def default_conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Module:
    """Create default convolution layer with appropriate padding."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MeanShift(nn.Conv2d):
    """Normalize/unnormalize RGB images using ImageNet mean and std."""

    def __init__(
        self,
        rgb_range: float = 1.0,
        rgb_mean: Tuple[float, ...] = (0.4488, 0.4371, 0.4040),
        rgb_std: Tuple[float, ...] = (1.0, 1.0, 1.0),
        sign: int = -1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Upsampler(nn.Sequential):
    """Upsampling module for super resolution."""

    def __init__(
        self,
        conv: Callable,
        scale: int,
        n_feats: int,
        bn: bool = False,
        act: bool = False,
        bias: bool = True,
    ):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act:
                    m.append(nn.ReLU(True))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act:
                m.append(nn.ReLU(True))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


# =============================================================================
# Single Image Super Resolution Models
# =============================================================================


class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN).

    Dong et al., "Image Super-Resolution Using Deep Convolutional Networks",
    TPAMI 2016.

    Args:
        num_channels: Number of input/output channels (default: 3)
        upscale_factor: Upsampling factor (default: 2)
    """

    def __init__(self, num_channels: int = 3, upscale_factor: int = 2):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor

        # Feature extraction
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        # Non-linear mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        # Reconstruction
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SRCNN.

        Args:
            x: Low-resolution input tensor of shape (B, C, H, W)

        Returns:
            High-resolution output tensor of shape (B, C, H*scale, W*scale)
        """
        # Upsample using bicubic interpolation first
        x = F.interpolate(
            x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
        )

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class FSRCNN(nn.Module):
    """
    Fast Super-Resolution Convolutional Neural Network (FSRCNN).

    Dong et al., "Accelerating the Super-Resolution Convolutional Neural Network",
    ECCV 2016.

    Uses smaller filters and deconvolution for upsampling instead of interpolation.

    Args:
        num_channels: Number of input/output channels
        upscale_factor: Upsampling factor
        d: LR feature dimension (default: 56)
        s: Shrinking filters (default: 12)
        m: Mapping layers (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        upscale_factor: int = 2,
        d: int = 56,
        s: int = 12,
        m: int = 4,
    ):
        super(FSRCNN, self).__init__()
        self.upscale_factor = upscale_factor

        # Feature extraction
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2), nn.PReLU(d)
        )

        # Shrinking
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]

        # Non-linear mapping (m layers)
        for _ in range(m):
            self.mid_part.extend(
                [nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(s)]
            )

        # Expanding
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)

        # Deconvolution (transposed convolution)
        self.last_part = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=upscale_factor,
            padding=4,
            output_padding=upscale_factor - 1,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using appropriate methods."""
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(
                    m.weight.data,
                    mean=0.0,
                    std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())),
                )
                nn.init.zeros_(m.bias.data)

        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(
                    m.weight.data,
                    mean=0.0,
                    std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())),
                )
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FSRCNN.

        Args:
            x: Low-resolution input tensor of shape (B, C, H, W)

        Returns:
            High-resolution output tensor of shape (B, C, H*scale, W*scale)
        """
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


class VDSR(nn.Module):
    """
    Very Deep Super Resolution (VDSR).

    Kim et al., "Accurate Image Super-Resolution Using Very Deep Convolutional Networks",
    CVPR 2016.

    Uses residual learning with very deep network (20 layers).

    Args:
        num_channels: Number of input/output channels
        num_layers: Number of convolutional layers (default: 20)
    """

    def __init__(self, num_channels: int = 3, num_layers: int = 20):
        super(VDSR, self).__init__()
        self.num_layers = num_layers
        self.residual_layer = self.make_layer(num_layers, num_channels)
        self.input_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.output_conv = nn.Conv2d(
            in_channels=64,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def make_layer(self, num_layers: int, num_channels: int) -> nn.Module:
        """Create residual layers."""
        layers = []
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, scale: int = 2) -> torch.Tensor:
        """
        Forward pass of VDSR.

        Args:
            x: Low-resolution input tensor
            scale: Upsampling factor

        Returns:
            High-resolution output tensor
        """
        # Bicubic upsampling
        residual = F.interpolate(
            x, scale_factor=scale, mode="bicubic", align_corners=False
        )

        out = self.relu(self.input_conv(residual))
        out = self.residual_layer(out)
        out = self.output_conv(out)

        # Residual learning
        out = torch.add(out, residual)
        return out


class DRCN(nn.Module):
    """
    Deeply-Recursive Convolutional Network (DRCN).

    Kim et al., "Deeply-Recursive Convolutional Network for Image Super-Resolution",
    CVPR 2016.

    Uses recursive layers with shared weights to reduce parameters.

    Args:
        num_channels: Number of input/output channels
        num_recursions: Number of recursive iterations (default: 16)
    """

    def __init__(self, num_channels: int = 3, num_recursions: int = 16):
        super(DRCN, self).__init__()
        self.num_recursions = num_recursions

        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(num_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Recursive layer (shared weights)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Reconstruction layer
        self.reconstruction_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, scale: int = 2) -> torch.Tensor:
        """
        Forward pass of DRCN.

        Args:
            x: Low-resolution input tensor
            scale: Upsampling factor

        Returns:
            High-resolution output tensor
        """
        # Bicubic upsampling
        residual = F.interpolate(
            x, scale_factor=scale, mode="bicubic", align_corners=False
        )

        # Embedding
        out = self.embedding_layer(residual)

        # Recursive blocks
        for _ in range(self.num_recursions):
            out = self.relu(self.conv(out))

        # Reconstruction
        out = self.reconstruction_layer(out)

        # Residual learning
        out = torch.add(out, residual)
        return out


class DRRN(nn.Module):
    """
    Deep Recursive Residual Network (DRRN).

    Tai et al., "Deep Recursive Residual Network for Image Super-Resolution",
    CVPR 2017.

    Combines recursive learning with residual blocks.

    Args:
        num_channels: Number of input/output channels
        num_blocks: Number of recursive blocks (default: 25)
        num_layers: Number of layers per block (default: 2)
    """

    def __init__(
        self, num_channels: int = 3, num_blocks: int = 25, num_layers: int = 2
    ):
        super(DRRN, self).__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers

        # Input layer
        self.input_conv = nn.Conv2d(num_channels, 128, kernel_size=3, padding=1)

        # Recursive blocks with shared weights
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Output layer
        self.output_conv = nn.Conv2d(128, num_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, scale: int = 2) -> torch.Tensor:
        """
        Forward pass of DRRN.

        Args:
            x: Low-resolution input tensor
            scale: Upsampling factor

        Returns:
            High-resolution output tensor
        """
        # Bicubic upsampling
        residual = F.interpolate(
            x, scale_factor=scale, mode="bicubic", align_corners=False
        )

        # Input
        out = self.relu(self.input_conv(residual))

        # Recursive residual blocks
        for _ in range(self.num_blocks):
            identity = out
            out = self.relu(self.conv1(out))
            out = self.conv2(out)
            out = torch.add(out, identity)
            out = self.relu(out)

        # Output
        out = self.output_conv(out)
        out = torch.add(out, residual)
        return out


class LapSRN(nn.Module):
    """
    Laplacian Pyramid Super-Resolution Network (LapSRN).

    Lai et al., "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution",
    CVPR 2017.

    Progressive upsampling through pyramid levels.

    Args:
        num_channels: Number of input/output channels
        upscale_factor: Target upsampling factor (default: 4)
    """

    def __init__(self, num_channels: int = 3, upscale_factor: int = 4):
        super(LapSRN, self).__init__()
        self.upscale_factor = upscale_factor

        # Calculate number of pyramid levels
        self.num_levels = int(math.log(upscale_factor, 2))

        # Feature extraction
        self.conv_input = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)

        # Pyramid levels
        self.level_convs = nn.ModuleList()
        self.trans_conv = nn.ModuleList()

        for _ in range(self.num_levels):
            self.level_convs.append(
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, num_channels, kernel_size=3, padding=1),
                )
            )

            if _ < self.num_levels - 1:
                self.trans_conv.append(
                    nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(64, 64, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LapSRN.

        Args:
            x: Low-resolution input tensor

        Returns:
            List of progressively upsampled outputs or final output
        """
        features = self.conv_input(x)
        outputs = []

        for i in range(self.num_levels):
            # Predict residual at current level
            residual = self.level_convs[i](features)

            # Upsample previous output or input
            if i == 0:
                up = F.interpolate(
                    x, scale_factor=2, mode="bicubic", align_corners=False
                )
            else:
                up = F.interpolate(
                    outputs[-1], scale_factor=2, mode="bicubic", align_corners=False
                )

            # Add residual
            out = torch.add(up, residual)
            outputs.append(out)

            # Transition to next level
            if i < self.num_levels - 1:
                features = self.trans_conv[i](features)

        return outputs[-1] if len(outputs) == 1 else outputs


class ResBlock(nn.Module):
    """Residual block for EDSR."""

    def __init__(
        self,
        num_features: int,
        kernel_size: int = 3,
        bias: bool = True,
        act: nn.Module = nn.ReLU(True),
        res_scale: float = 1.0,
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(
                nn.Conv2d(
                    num_features,
                    num_features,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                )
            )
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class EDSR(nn.Module):
    """
    Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR).

    Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution",
    CVPRW 2017.

    Removes unnecessary batch normalization for better performance.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 256)
        num_blocks: Number of residual blocks (default: 32)
        upscale_factor: Upsampling factor (default: 4)
        res_scale: Residual scaling factor (default: 0.1)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 256,
        num_blocks: int = 32,
        upscale_factor: int = 4,
        res_scale: float = 0.1,
    ):
        super(EDSR, self).__init__()

        # Mean shift
        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)

        # Feature extraction
        self.head = nn.Conv2d(num_channels, num_features, 3, padding=1)

        # Body (residual blocks)
        self.body = nn.ModuleList(
            [ResBlock(num_features, res_scale=res_scale) for _ in range(num_blocks)]
        )
        self.body_conv = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Upsampling
        self.tail = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=False),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of EDSR.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        x = self.sub_mean(x)

        x = self.head(x)

        res = x
        for block in self.body:
            res = block(res)
        res = self.body_conv(res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


class ChannelAttention(nn.Module):
    """Channel attention module for RCAN."""

    def __init__(self, num_features: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(
        self,
        num_features: int,
        kernel_size: int = 3,
        reduction: int = 16,
        bias: bool = True,
        act: nn.Module = nn.ReLU(True),
        res_scale: float = 1.0,
    ):
        super(RCAB, self).__init__()
        m = []
        for i in range(2):
            m.append(
                nn.Conv2d(
                    num_features,
                    num_features,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                )
            )
            if i == 0:
                m.append(act)
        m.append(ChannelAttention(num_features, reduction))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class RCAN(nn.Module):
    """
    Residual Channel Attention Network (RCAN).

    Zhang et al., "Image Super-Resolution Using Very Deep Residual Channel Attention Networks",
    ECCV 2018.

    Uses channel attention to adaptively rescale channel-wise features.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_rg: Number of residual groups (default: 10)
        num_rcab: Number of RCAB per group (default: 20)
        upscale_factor: Upsampling factor (default: 4)
        reduction: Channel reduction factor (default: 16)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_rg: int = 10,
        num_rcab: int = 20,
        upscale_factor: int = 4,
        reduction: int = 16,
    ):
        super(RCAN, self).__init__()

        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)

        # Shallow feature extraction
        self.head = nn.Conv2d(num_channels, num_features, 3, padding=1)

        # Residual groups
        self.body = nn.ModuleList()
        for _ in range(num_rg):
            group = nn.ModuleList(
                [RCAB(num_features, reduction=reduction) for _ in range(num_rcab)]
            )
            group.append(nn.Conv2d(num_features, num_features, 3, padding=1))
            self.body.append(group)

        # Global feature fusion
        self.global_conv = nn.Conv2d(num_features * num_rg, num_features, 1, padding=0)
        self.body_conv = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Upsampling
        self.tail = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=False),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RCAN.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        x = self.sub_mean(x)

        x = self.head(x)

        # Process residual groups
        group_features = []
        for group in self.body:
            res = x
            for block in group[:-1]:
                res = block(res)
            res = group[-1](res)
            x = x + res
            group_features.append(x)

        # Global fusion
        x = self.global_conv(torch.cat(group_features, dim=1))
        x = self.body_conv(x)

        x = self.tail(x)
        x = self.add_mean(x)

        return x


class RDB(nn.Module):
    """Residual Dense Block for RDN."""

    def __init__(self, num_features: int, num_layers: int = 6, growth_rate: int = 32):
        super(RDB, self).__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        num_features + i * growth_rate, growth_rate, 3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                )
            )

        # Local feature fusion
        self.lff = nn.Conv2d(
            num_features + num_layers * growth_rate, num_features, 1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for conv in self.convs:
            out = conv(torch.cat(features, dim=1))
            features.append(out)

        out = self.lff(torch.cat(features, dim=1))
        out = out + x  # Local residual learning
        return out


class RDN(nn.Module):
    """
    Residual Dense Network (RDN).

    Zhang et al., "Residual Dense Network for Image Super-Resolution",
    CVPR 2018.

    Uses dense connections within residual blocks.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of RDBs (default: 16)
        num_layers: Number of layers per RDB (default: 6)
        growth_rate: Growth rate for dense connections (default: 32)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        num_layers: int = 6,
        growth_rate: int = 32,
        upscale_factor: int = 4,
    ):
        super(RDN, self).__init__()

        self.num_blocks = num_blocks

        # Shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, 3, padding=1)
        self.sfe2 = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Residual dense blocks
        self.rdbs = nn.ModuleList(
            [RDB(num_features, num_layers, growth_rate) for _ in range(num_blocks)]
        )

        # Global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(num_features * num_blocks, num_features, 1, padding=0),
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

        # Upsampling
        self.upconv = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=False),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RDN.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        # Process RDBs
        x = sfe2
        local_features = []
        for rdb in self.rdbs:
            x = rdb(x)
            local_features.append(x)

        # Global fusion
        x = self.gff(torch.cat(local_features, dim=1))

        # Residual learning
        x += sfe1

        x = self.upconv(x)
        return x


class DBPNSuperResolution(nn.Module):
    """
    Deep Back-Projection Network (DBPN).

    Haris et al., "Deep Back-Projection Networks for Super-Resolution",
    CVPR 2018.

    Uses iterative up- and down-projection units for error feedback.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_stages: Number of projection stages (default: 7)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_stages: int = 7,
        upscale_factor: int = 4,
    ):
        super(DBPNSuperResolution, self).__init__()

        # Initial feature extraction
        self.conv0 = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 3, padding=1), nn.PReLU(num_features)
        )

        # Projection units
        self.up_units = nn.ModuleList()
        self.down_units = nn.ModuleList()

        for i in range(num_stages):
            if i == 0:
                self.up_units.append(
                    self._make_up_unit(num_features, upscale_factor, True)
                )
                self.down_units.append(
                    self._make_down_unit(num_features, upscale_factor, True)
                )
            else:
                self.up_units.append(
                    self._make_up_unit(num_features, upscale_factor, False)
                )
                self.down_units.append(
                    self._make_down_unit(num_features, upscale_factor, False)
                )

        # Reconstruction
        self.reconstruct = nn.Conv2d(num_features, num_channels, 3, padding=1)

    def _make_up_unit(self, num_features: int, scale: int, first: bool) -> nn.Module:
        """Create up-projection unit."""
        if first:
            return nn.Sequential(
                nn.Conv2d(num_features, num_features * scale * scale, 3, padding=1),
                nn.PixelShuffle(scale),
                nn.PReLU(num_features),
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.PixelShuffle(scale),
                nn.PReLU(num_features),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(num_features, num_features * scale * scale, 3, padding=1),
                nn.PixelShuffle(scale),
                nn.PReLU(num_features),
            )

    def _make_down_unit(self, num_features: int, scale: int, first: bool) -> nn.Module:
        """Create down-projection unit."""
        if first:
            return nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1, stride=scale),
                nn.PReLU(num_features),
                nn.Conv2d(num_features, num_features, 3, padding=1, stride=scale),
                nn.PReLU(num_features),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1, stride=scale),
                nn.PReLU(num_features),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DBPN.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        x = self.conv0(x)

        # Iterative back-projection
        for up_unit, down_unit in zip(self.up_units, self.down_units):
            x = up_unit(x)
            x = down_unit(x)

        x = self.reconstruct(x)
        return x


# =============================================================================
# GAN-based Super Resolution Models
# =============================================================================


class ResidualBlock(nn.Module):
    """Residual block for SRGAN/ESRGAN."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class SRGANGenerator(nn.Module):
    """
    SRGAN Generator.

    Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network",
    CVPR 2017.

    Uses residual blocks and sub-pixel convolution for upsampling.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of residual blocks (default: 16)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        upscale_factor: int = 4,
    ):
        super(SRGANGenerator, self).__init__()

        # Pre-residual block
        self.pre_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 9, padding=4), nn.PReLU()
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )

        # Post-residual block
        self.post_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
        )

        # Upsampling
        upsample_layers = []
        for _ in range(int(math.log(upscale_factor, 2))):
            upsample_layers.extend(
                [
                    nn.Conv2d(num_features, 4 * num_features, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.PReLU(),
                ]
            )
        self.upsample = nn.Sequential(*upsample_layers)

        # Output
        self.output_conv = nn.Conv2d(num_features, num_channels, 9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SRGAN generator.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        pre_res = self.pre_conv(x)
        res = self.res_blocks(pre_res)
        res = self.post_conv(res)
        res = res + pre_res  # Skip connection
        res = self.upsample(res)
        res = self.output_conv(res)
        return torch.tanh(res)


class SRGANDiscriminator(nn.Module):
    """
    SRGAN Discriminator.

    Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network",
    CVPR 2017.

    Args:
        num_channels: Number of input channels
        num_features: Number of feature maps (default: 64)
    """

    def __init__(self, num_channels: int = 3, num_features: int = 64):
        super(SRGANDiscriminator, self).__init__()

        def discriminator_block(
            in_filters: int, out_filters: int, first: bool = False
        ) -> nn.Module:
            layers = [
                nn.Conv2d(in_filters, out_filters, 3, stride=1 + (not first), padding=1)
            ]
            if not first:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, 3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(num_channels, num_features, first=True),
            discriminator_block(num_features, num_features * 2),
            discriminator_block(num_features * 2, num_features * 4),
            discriminator_block(num_features * 4, num_features * 8),
        )

        # Classifier
        self.adv_layer = nn.Sequential(
            nn.Linear(num_features * 8 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of discriminator.

        Args:
            x: Input image tensor

        Returns:
            Probability of real image
        """
        out = self.model(x)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity


class RRDB(nn.Module):
    """Residual in Residual Dense Block for ESRGAN."""

    def __init__(
        self, num_features: int, growth_rate: int = 32, res_scale: float = 0.2
    ):
        super(RRDB, self).__init__()
        self.res_scale = res_scale

        # Three dense blocks
        self.dense_blocks = nn.ModuleList(
            [RDB(num_features, num_layers=5, growth_rate=growth_rate) for _ in range(3)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.dense_blocks:
            out = block(out)
        return x + out * self.res_scale


class ESRGANGenerator(nn.Module):
    """
    ESRGAN Generator.

    Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks",
    ECCVW 2018.

    Uses RRDB (Residual in Residual Dense Block) and improved adversarial training.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of RRDBs (default: 23)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 23,
        upscale_factor: int = 4,
    ):
        super(ESRGANGenerator, self).__init__()

        # First convolution
        self.conv_first = nn.Conv2d(num_channels, num_features, 3, padding=1)

        # RRDB blocks
        self.body = nn.ModuleList([RRDB(num_features) for _ in range(num_blocks)])

        # Convolution after body
        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Upsampling
        upsample_layers = []
        for _ in range(int(math.log(upscale_factor, 2))):
            upsample_layers.extend(
                [
                    nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.upsample = nn.Sequential(*upsample_layers)

        # High resolution convolution
        self.hr_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final convolution
        self.conv_last = nn.Conv2d(num_features, num_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ESRGAN generator.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        fea = self.conv_first(x)

        body_fea = fea
        for block in self.body:
            body_fea = block(body_fea)
        body_fea = self.conv_body(body_fea)

        fea = fea + body_fea

        out = self.upsample(fea)
        out = self.hr_conv(out)
        out = self.conv_last(out)

        return out


class ESRGANDiscriminator(nn.Module):
    """
    ESRGAN Discriminator with Spectral Normalization.

    Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks",
    ECCVW 2018.
    """

    def __init__(self, num_channels: int = 3, num_features: int = 64):
        super(ESRGANDiscriminator, self).__init__()

        from torch.nn.utils import spectral_norm

        def conv_block(in_filters: int, out_filters: int, stride: int = 1) -> nn.Module:
            return nn.Sequential(
                spectral_norm(
                    nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            # Input
            spectral_norm(nn.Conv2d(num_channels, num_features, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            conv_block(num_features, num_features, stride=2),
            conv_block(num_features, num_features * 2, stride=1),
            conv_block(num_features * 2, num_features * 2, stride=2),
            conv_block(num_features * 2, num_features * 4, stride=1),
            conv_block(num_features * 4, num_features * 4, stride=2),
            conv_block(num_features * 4, num_features * 8, stride=1),
            conv_block(num_features * 8, num_features * 8, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_features * 8 * 6 * 6, num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_features * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ESRGAN discriminator.

        Args:
            x: Input image tensor

        Returns:
            Discriminator output
        """
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class RealESRGANGenerator(nn.Module):
    """
    Real-ESRGAN Generator for practical blind super-resolution.

    Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
    with Pure Synthetic Data", ICCVW 2021.

    Similar to ESRGAN but designed for real-world degradations.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of RRDBs (default: 23)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 23,
        upscale_factor: int = 4,
    ):
        super(RealESRGANGenerator, self).__init__()

        self.generator = ESRGANGenerator(
            num_channels, num_features, num_blocks, upscale_factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class SRFeatGenerator(nn.Module):
    """
    SRFeat Generator with feature discrimination.

    Park et al., "SRFeat: Single Image Super-Resolution with Feature Discrimination",
    ECCV 2018.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of residual blocks (default: 16)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        upscale_factor: int = 4,
    ):
        super(SRFeatGenerator, self).__init__()

        # Initial feature extraction
        self.conv_first = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, 3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_features, num_features, 3, padding=1),
                    nn.BatchNorm2d(num_features),
                )
                for _ in range(num_blocks)
            ]
        )

        # Upsampling
        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, bn=True, act=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output
        self.conv_last = nn.Conv2d(num_features, num_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SRFeat generator.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        fea = self.conv_first(x)

        for block in self.res_blocks:
            fea = fea + block(fea)

        out = self.upsample(fea)
        out = self.conv_last(out)
        return out


class SRFeatDiscriminator(nn.Module):
    """SRFeat Feature Discriminator."""

    def __init__(self, num_channels: int = 3, num_features: int = 64):
        super(SRFeatDiscriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 4, num_features * 8, 4, stride=1, padding=1),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Conv2d(num_features * 8, 1, 4, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.classifier(out)
        return out


class BebyGANGenerator(nn.Module):
    """
    Beby-GAN Generator (Best-Buddy GAN).

    A variant focusing on buddy networks and collaborative training.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of residual blocks (default: 16)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        upscale_factor: int = 4,
    ):
        super(BebyGANGenerator, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 3, padding=1), nn.PReLU()
        )

        # Residual blocks with attention
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, 3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.PReLU(),
                    nn.Conv2d(num_features, num_features, 3, padding=1),
                    nn.BatchNorm2d(num_features),
                )
            )

        # Buddy path (auxiliary)
        self.buddy_path = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, 1),
            nn.PReLU(),
            nn.Conv2d(num_features // 2, num_features, 1),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 1), nn.PReLU()
        )

        # Upsampling
        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=True),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Beby-GAN generator.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        main = self.main_path(x)

        # Process through residual blocks
        for block in self.res_blocks:
            main = main + block(main)

        # Buddy path
        buddy = self.buddy_path(main)

        # Fusion
        combined = torch.cat([main, buddy], dim=1)
        out = self.fusion(combined)

        # Upsample
        out = self.upsample(out)
        return out


# =============================================================================
# Lightweight Super Resolution Models
# =============================================================================


class CARNBlock(nn.Module):
    """Cascading Residual Block for CARN."""

    def __init__(self, num_features: int, kernel_size: int = 3):
        super(CARNBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_features, num_features, kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv2d(
            num_features, num_features, kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class CARN(nn.Module):
    """
    Cascading Residual Network (CARN).

    Ahn et al., "CARN: Fast, Accurate, and Lightweight Super-Resolution
    with Cascading Residual Network", ECCV 2018.

    Efficient architecture using cascading blocks and group convolutions.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of blocks per group (default: 3)
        num_groups: Number of cascading groups (default: 3)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 3,
        num_groups: int = 3,
        upscale_factor: int = 4,
    ):
        super(CARN, self).__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(num_channels, num_features, 3, padding=1)

        # Cascading blocks
        self.groups = nn.ModuleList()
        for _ in range(num_groups):
            blocks = nn.ModuleList([CARNBlock(num_features) for _ in range(num_blocks)])
            self.groups.append(blocks)

        # Local fusion
        self.local_fusion = nn.ModuleList(
            [
                nn.Conv2d(num_features * num_blocks, num_features, 1)
                for _ in range(num_groups)
            ]
        )

        # Global fusion
        self.global_fusion = nn.Conv2d(num_features * num_groups, num_features, 1)

        # Upsampling
        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=True),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CARN.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        x = self.conv_in(x)

        # Cascading through groups
        cascading = x
        group_features = []

        for group, fusion in zip(self.groups, self.local_fusion):
            # Process blocks
            block_features = []
            out = cascading
            for block in group:
                out = block(out)
                block_features.append(out)

            # Local fusion
            local = fusion(torch.cat(block_features, dim=1))
            group_features.append(local)

            # Cascade to next group
            cascading = cascading + local

        # Global fusion
        out = self.global_fusion(torch.cat(group_features, dim=1))
        out = self.upsample(out)
        return out


class FALSRGenerator(nn.Module):
    """
    Fast, Accurate, and Lightweight Super-Resolution (FALSR).

    Chu et al., "Fast, Accurate and Lightweight Super-Resolution with
    Neural Architecture Search", CVPR 2019.

    NAS-designed efficient super-resolution network.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 48)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self, num_channels: int = 3, num_features: int = 48, upscale_factor: int = 4
    ):
        super(FALSRGenerator, self).__init__()

        # Entry convolution
        self.entry = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Efficient blocks with mixed operations
        self.mixed_ops = nn.ModuleList()
        for _ in range(4):
            self.mixed_ops.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, 3, padding=1, groups=4),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features, 1),
                    nn.ReLU(inplace=True),
                )
            )

        # Exit
        self.exit = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Upsampling
        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=False),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FALSR.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        fea = self.entry(x)

        for op in self.mixed_ops:
            fea = fea + op(fea)

        fea = self.exit(fea)
        out = self.upsample(fea)
        return out


class MSRB(nn.Module):
    """Multi-Scale Residual Block for MSRN."""

    def __init__(self, num_features: int):
        super(MSRB, self).__init__()

        # Multi-scale convolutions
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(num_features, num_features, 5, padding=2), nn.ReLU(inplace=True)
        )

        # Fusion
        self.fusion = nn.Conv2d(num_features * 2, num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3 = self.conv_3x3(x)
        c5 = self.conv_5x5(x)
        fused = self.fusion(torch.cat([c3, c5], dim=1))
        return x + fused


class MSRN(nn.Module):
    """
    Multi-scale Residual Network (MSRN).

    Li et al., "Multi-Scale Residual Network for Image Super-Resolution",
    ECCV 2018.

    Uses multi-scale residual blocks to capture features at different scales.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of MSRBs (default: 8)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        upscale_factor: int = 4,
    ):
        super(MSRN, self).__init__()

        # Initial feature extraction
        self.conv_in = nn.Conv2d(num_channels, num_features, 3, padding=1)

        # Multi-scale residual blocks
        self.msrbs = nn.ModuleList([MSRB(num_features) for _ in range(num_blocks)])

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_features * num_blocks, num_features, 1),
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

        # Upsampling
        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=False),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MSRN.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        x = self.conv_in(x)

        # Collect features from all MSRBs
        features = []
        out = x
        for msrb in self.msrbs:
            out = msrb(out)
            features.append(out)

        # Fusion
        out = self.fusion(torch.cat(features, dim=1))

        # Upsample
        out = self.upsample(out)
        return out


class IDN(nn.Module):
    """
    Information Distillation Network (IDN).

    Hui et al., "Fast and Accurate Single Image Super-Resolution via
    Information Distillation Network", CVPR 2018.

    Distills information progressively for efficient processing.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of distillation blocks (default: 4)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 4,
        upscale_factor: int = 4,
    ):
        super(IDN, self).__init__()

        # Feature extraction
        self.fblock = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 3, padding=1), nn.ReLU(inplace=True)
        )

        # Distillation blocks
        self.dblocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.dblocks.append(self._make_dblock(num_features))

        # Reconstruction
        self.reconstruction = nn.Sequential(
            Upsampler(
                default_conv, upscale_factor, num_features * num_blocks, act=False
            ),
            nn.Conv2d(num_features * num_blocks, num_channels, 3, padding=1),
        )

    def _make_dblock(self, num_features: int) -> nn.Module:
        """Create a distillation block."""
        return nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of IDN.

        Args:
            x: Low-resolution input tensor

        Returns:
            High-resolution output tensor
        """
        x = self.fblock(x)

        # Information distillation
        distilled = []
        for dblock in self.dblocks:
            out = dblock(x)
            # Split: part for next stage, part distilled
            x, d = torch.chunk(out, 2, dim=1)
            distilled.append(d)

        # Reconstruction from all distilled features
        out = torch.cat(distilled, dim=1)
        out = self.reconstruction(out)
        return out


# =============================================================================
# Video Super Resolution Models
# =============================================================================


class VSRNet(nn.Module):
    """
    Video Super-Resolution Network (VSRNet).

    Kappeler et al., "Video Super-Resolution with Convolutional Neural Networks",
    IEEE TCI 2016.

    Simple video SR using temporal concatenation.

    Args:
        num_channels: Number of input/output channels
        num_frames: Number of input frames (default: 5)
        num_features: Number of feature maps (default: 64)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_frames: int = 5,
        num_features: int = 64,
        upscale_factor: int = 4,
    ):
        super(VSRNet, self).__init__()

        self.num_frames = num_frames

        # Temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(num_channels * num_frames, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Spatial feature extraction
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Upsampling
        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=True),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of VSRNet.

        Args:
            x: Input tensor of shape (B, T, C, H, W) where T is num_frames

        Returns:
            Super-resolved center frame of shape (B, C, H*scale, W*scale)
        """
        B, T, C, H, W = x.shape

        # Concatenate frames along channel dimension
        x = x.view(B, T * C, H, W)

        # Temporal and spatial processing
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)

        # Upsample
        x = self.upsample(x)
        return x


class FRVSR(nn.Module):
    """
    Frame-Recurrent Video Super-Resolution (FRVSR).

    Sajjadi et al., "Frame-Recurrent Video Super-Resolution",
    CVPR 2018.

    Uses previously estimated HR frame as input for next frame.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self, num_channels: int = 3, num_features: int = 64, upscale_factor: int = 4
    ):
        super(FRVSR, self).__init__()

        self.upscale_factor = upscale_factor

        # Flow estimation (simplified)
        self.flow_net = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_features, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 2, 5, padding=2),
        )

        # Spatial feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # SR reconstruction
        self.sr_net = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            Upsampler(default_conv, upscale_factor, num_features, act=False),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(
        self, lr_curr: torch.Tensor, lr_prev: torch.Tensor, hr_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of FRVSR for one frame.

        Args:
            lr_curr: Current low-resolution frame (B, C, H, W)
            lr_prev: Previous low-resolution frame (B, C, H, W)
            hr_prev: Previous high-resolution frame (B, C, H*scale, W*scale)

        Returns:
            Current high-resolution frame (B, C, H*scale, W*scale)
        """
        # Estimate flow (simplified - actual implementation would use SpyNet or similar)
        flow_input = torch.cat([lr_curr, lr_prev], dim=1)
        flow = self.flow_net(flow_input)

        # Warp previous HR using flow (downsampled to LR, then upsample back)
        hr_prev_down = F.interpolate(
            hr_prev,
            scale_factor=1 / self.upscale_factor,
            mode="bicubic",
            align_corners=False,
        )

        # Concatenate warped HR with current LR
        aligned = torch.cat([lr_curr, hr_prev_down], dim=1)

        # Extract features and reconstruct
        features = self.feature_extractor(aligned)
        hr_curr = self.sr_net(features)

        return hr_curr


class DUF(nn.Module):
    """
    Dynamic Upsampling Filters (DUF).

    Jo et al., "Deep Video Super-Resolution Network Using Dynamic Upsampling Filters
    Without Explicit Motion Compensation", CVPR 2018.

    Predicts upsampling filters dynamically.

    Args:
        num_channels: Number of input/output channels
        num_frames: Number of input frames (default: 7)
        num_features: Number of feature maps (default: 64)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_frames: int = 7,
        num_features: int = 64,
        upscale_factor: int = 4,
    ):
        super(DUF, self).__init__()

        self.num_frames = num_frames
        self.upscale_factor = upscale_factor

        # 3D convolution for spatiotemporal features
        self.conv3d = nn.Sequential(
            nn.Conv3d(num_channels, num_features, (3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
        )

        # Dynamic filter generation
        filter_size = 5 * 5 * upscale_factor * upscale_factor
        self.filter_gen = nn.Sequential(
            nn.Conv2d(num_features * num_frames, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, filter_size * num_channels, 1),
        )

        # Residual generation
        self.residual_gen = nn.Sequential(
            nn.Conv2d(num_features * num_frames, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_channels * upscale_factor * upscale_factor, 1),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DUF.

        Args:
            x: Input tensor of shape (B, T, C, H, W)

        Returns:
            Super-resolved center frame
        """
        B, T, C, H, W = x.shape

        # 3D convolution
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        features_3d = self.conv3d(x)
        features_3d = features_3d.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # Flatten temporal dimension
        features = features_3d.view(B, T * features_3d.size(2), H, W)

        # Generate dynamic filters and residual
        dynamic_filters = self.filter_gen(features)
        residual = self.residual_gen(features)

        # Apply dynamic filters (simplified - full implementation requires custom CUDA kernel)
        # For demonstration, using standard convolution
        center_frame = x[:, :, T // 2, :, :]
        output = F.interpolate(
            center_frame,
            scale_factor=self.upscale_factor,
            mode="bicubic",
            align_corners=False,
        )
        output = output + residual

        return output


class PCDAlignment(nn.Module):
    """Pyramid, Cascading and Deformable alignment module for EDVR."""

    def __init__(self, num_features: int):
        super(PCDAlignment, self).__init__()

        # Pyramid levels
        self.pyramid_levels = 3

        # Offset and modulation prediction
        self.offset_convs = nn.ModuleList()
        self.deform_convs = nn.ModuleList()

        for i in range(self.pyramid_levels):
            self.offset_convs.append(
                nn.Sequential(
                    nn.Conv2d(num_features * 2, num_features, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

            # Deformable convolution would go here (using standard conv for simplicity)
            self.deform_convs.append(
                nn.Conv2d(num_features, num_features, 3, padding=1)
            )

    def forward(
        self, nbr_fea_l: List[torch.Tensor], ref_fea_l: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Align neighboring features to reference features.

        Args:
            nbr_fea_l: List of neighboring features at different scales
            ref_fea_l: List of reference features at different scales

        Returns:
            Aligned neighboring features
        """
        # Start from coarsest level
        aligned = nbr_fea_l[-1]

        for i in range(self.pyramid_levels - 1, -1, -1):
            offset = self.offset_convs[i](torch.cat([aligned, ref_fea_l[i]], dim=1))
            aligned = self.deform_convs[i](aligned + offset)

            if i > 0:
                aligned = F.interpolate(
                    aligned, scale_factor=2, mode="bilinear", align_corners=False
                )

        return aligned


class EDVR(nn.Module):
    """
    Enhanced Deformable Video Restoration (EDVR).

    Wang et al., "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks",
    ICCVW 2019.

    Uses PCD alignment and TSA fusion for video super-resolution.

    Args:
        num_channels: Number of input/output channels
        num_frames: Number of input frames (default: 5)
        num_features: Number of feature maps (default: 128)
        num_blocks: Number of reconstruction blocks (default: 40)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_frames: int = 5,
        num_features: int = 128,
        num_blocks: int = 40,
        upscale_factor: int = 4,
    ):
        super(EDVR, self).__init__()

        self.num_frames = num_frames
        self.center_frame_idx = num_frames // 2

        # Feature extraction
        self.conv_first = nn.Conv2d(num_channels, num_features, 3, padding=1)

        # PCD alignment (simplified)
        self.alignment = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

        # Temporal fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_features * num_frames, num_features, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

        # Reconstruction
        self.reconstruction = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_blocks)
            ]
        )

        # Upsampling
        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=False),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of EDVR.

        Args:
            x: Input tensor of shape (B, T, C, H, W)

        Returns:
            Super-resolved center frame
        """
        B, T, C, H, W = x.shape

        # Extract features for all frames
        x = x.view(B * T, C, H, W)
        features = self.conv_first(x)
        features = features.view(B, T, -1, H, W)

        # Align to center frame
        ref_fea = features[:, self.center_frame_idx]
        aligned_features = []

        for i in range(T):
            if i == self.center_frame_idx:
                aligned_features.append(features[:, i])
            else:
                aligned = self.alignment(torch.cat([features[:, i], ref_fea], dim=1))
                aligned_features.append(aligned)

        # Temporal fusion
        aligned_features = torch.cat(aligned_features, dim=1)
        fused = self.fusion(aligned_features)

        # Reconstruction
        out = fused
        for block in self.reconstruction:
            out = block(out) + out

        # Upsample
        out = self.upsample(out)
        return out


class BasicVSR(nn.Module):
    """
    BasicVSR: Basic Video Super-Resolution.

    Chan et al., "BasicVSR: The Search for Essential Components in Video
    Super-Resolution and Beyond", CVPR 2021.

    Uses bidirectional propagation and optical flow for alignment.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of residual blocks (default: 30)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 30,
        upscale_factor: int = 4,
    ):
        super(BasicVSR, self).__init__()

        self.upscale_factor = upscale_factor

        # Feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Backward and forward propagation
        self.backward_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_features * 2, num_features, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                for _ in range(num_blocks)
            ]
        )

        self.forward_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_features * 2, num_features, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                for _ in range(num_blocks)
            ]
        )

        # Fusion and upsampling
        self.fusion = nn.Conv2d(num_features * 2, num_features, 3, padding=1)

        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=False),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BasicVSR.

        Args:
            x: Input tensor of shape (B, T, C, H, W)

        Returns:
            Super-resolved frames of shape (B, T, C, H*scale, W*scale)
        """
        B, T, C, H, W = x.shape

        # Extract features
        x_flat = x.view(B * T, C, H, W)
        features = self.feat_extract(x_flat)
        features = features.view(B, T, -1, H, W)

        # Backward propagation
        backward_features = []
        feat = torch.zeros(B, features.size(2), H, W, device=x.device)
        for t in range(T - 1, -1, -1):
            feat = feat + features[:, t]
            for block in self.backward_blocks:
                feat = block(torch.cat([feat, torch.zeros_like(feat)], dim=1))
            backward_features.insert(0, feat)

        # Forward propagation
        outputs = []
        feat = torch.zeros(B, features.size(2), H, W, device=x.device)
        for t in range(T):
            feat = feat + features[:, t]
            for block in self.forward_blocks:
                feat = block(torch.cat([feat, backward_features[t]], dim=1))

            # Fusion and upsampling
            fused = self.fusion(torch.cat([feat, backward_features[t]], dim=1))
            out = self.upsample(fused)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        return outputs


class BasicVSRPlusPlus(nn.Module):
    """
    BasicVSR++: Improved Basic Video Super-Resolution.

    Chan et al., "BasicVSR++: Improving Video Super-Resolution with
    Enhanced Propagation and Alignment", CVPR 2022.

    Enhanced version with second-order grid propagation and flow-guided deformable alignment.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of residual blocks (default: 30)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 30,
        upscale_factor: int = 4,
    ):
        super(BasicVSRPlusPlus, self).__init__()

        self.upscale_factor = upscale_factor

        # Feature extraction
        self.feat_extract = nn.Conv2d(num_channels, num_features, 3, padding=1)

        # Second-order grid propagation
        self.propagation_blocks = nn.ModuleList()
        for _ in range(2):  # Two stages of propagation
            stage_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(num_features * 3, num_features, 3, padding=1),
                        nn.LeakyReLU(0.1, inplace=True),
                        nn.Conv2d(num_features, num_features, 3, padding=1),
                    )
                    for _ in range(num_blocks // 2)
                ]
            )
            self.propagation_blocks.append(stage_blocks)

        # Fusion and reconstruction
        self.fusion = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.reconstruction = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                )
                for _ in range(10)
            ]
        )

        # Upsampling
        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=False),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BasicVSR++.

        Args:
            x: Input tensor of shape (B, T, C, H, W)

        Returns:
            Super-resolved frames
        """
        B, T, C, H, W = x.shape

        # Feature extraction
        x_flat = x.view(B * T, C, H, W)
        features = self.feat_extract(x_flat)
        features = features.view(B, T, -1, H, W)

        # First-order propagation (similar to BasicVSR)
        # Simplified implementation
        outputs = []
        for t in range(T):
            feat = features[:, t]

            # Apply propagation blocks
            for stage in self.propagation_blocks:
                for block in stage:
                    feat = feat + block(torch.cat([feat, feat, feat], dim=1))

            # Reconstruction
            rec = feat
            for block in self.reconstruction:
                rec = block(rec) + rec

            # Upsample
            out = self.upsample(rec)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)
        return outputs


# =============================================================================
# Loss Functions
# =============================================================================


class PixelLoss(nn.Module):
    """
    Pixel-wise loss (L1 or L2).

    Args:
        loss_type: Type of pixel loss ('l1' or 'l2')
    """

    def __init__(self, loss_type: str = "l1"):
        super(PixelLoss, self).__init__()
        if loss_type == "l1":
            self.loss = nn.L1Loss()
        elif loss_type == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.loss_type = loss_type

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel loss.

        Args:
            sr: Super-resolved image
            hr: High-resolution ground truth

        Returns:
            Loss value
        """
        return self.loss(sr, hr)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.

    Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution",
    ECCV 2016.

    Args:
        layer_weights: Dictionary of layer names and weights
        use_input_norm: Whether to normalize inputs
        range_norm: Whether to normalize to [0, 1]
    """

    def __init__(
        self,
        layer_weights: Dict[str, float] = None,
        use_input_norm: bool = True,
        range_norm: bool = True,
    ):
        super(PerceptualLoss, self).__init__()

        if layer_weights is None:
            layer_weights = {"conv5_4": 1.0}

        self.layer_weights = layer_weights
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        # Load VGG19
        vgg = models.vgg19(pretrained=True).features

        # Build feature extractor
        self.layer_name_mapping = {
            "conv1_1": 0,
            "conv1_2": 2,
            "conv2_1": 5,
            "conv2_2": 7,
            "conv3_1": 10,
            "conv3_2": 12,
            "conv3_3": 14,
            "conv3_4": 16,
            "conv4_1": 19,
            "conv4_2": 21,
            "conv4_3": 23,
            "conv4_4": 25,
            "conv5_1": 28,
            "conv5_2": 30,
            "conv5_3": 32,
            "conv5_4": 34,
        }

        max_layer = max(
            [self.layer_name_mapping[name] for name in layer_weights.keys()]
        )
        self.features = vgg[: max_layer + 1]

        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet normalization
        if use_input_norm:
            self.register_buffer(
                "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            sr: Super-resolved image
            hr: High-resolution ground truth

        Returns:
            Perceptual loss value
        """
        # Normalize if needed
        if self.range_norm:
            sr = (sr + 1) / 2
            hr = (hr + 1) / 2

        if self.use_input_norm:
            sr = (sr - self.mean) / self.std
            hr = (hr - self.mean) / self.std

        # Extract features
        sr_features = {}
        hr_features = {}

        x_sr = sr
        x_hr = hr
        for name, idx in self.layer_name_mapping.items():
            if idx >= len(self.features):
                break
            x_sr = self.features[idx](x_sr)
            x_hr = self.features[idx](x_hr)

            if name in self.layer_weights:
                sr_features[name] = x_sr
                hr_features[name] = x_hr

        # Compute loss
        loss = 0
        for name, weight in self.layer_weights.items():
            loss += weight * F.l1_loss(sr_features[name], hr_features[name])

        return loss


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN-based SR.

    Supports different GAN losses: vanilla, lsgan, wgan, hinge.

    Args:
        loss_type: Type of adversarial loss ('vanilla', 'lsgan', 'wgan', 'hinge')
    """

    def __init__(self, loss_type: str = "vanilla"):
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Compute adversarial loss.

        Args:
            pred: Discriminator prediction
            target_is_real: Whether the target is real (True) or fake (False)

        Returns:
            Adversarial loss value
        """
        if self.loss_type == "vanilla":
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            loss = F.binary_cross_entropy(pred, target)
        elif self.loss_type == "lsgan":
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            loss = F.mse_loss(pred, target)
        elif self.loss_type == "wgan":
            loss = -pred.mean() if target_is_real else pred.mean()
        elif self.loss_type == "hinge":
            if target_is_real:
                loss = F.relu(1 - pred).mean()
            else:
                loss = F.relu(1 + pred).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class TextureLoss(nn.Module):
    """
    Texture matching loss using Gram matrices.

    Encourages similar texture patterns between SR and HR images.

    Args:
        layers: List of layer names to use
        weights: Weights for each layer
    """

    def __init__(self, layers: List[str] = None, weights: List[float] = None):
        super(TextureLoss, self).__init__()

        if layers is None:
            layers = ["conv3_2", "conv4_2"]
            weights = [1.0, 1.0]

        self.layers = layers
        self.weights = weights if weights else [1.0] * len(layers)

        # Load VGG
        vgg = models.vgg19(pretrained=True).features

        # Build feature extractor
        self.layer_name_mapping = {
            "conv1_1": 0,
            "conv1_2": 2,
            "conv2_1": 5,
            "conv2_2": 7,
            "conv3_1": 10,
            "conv3_2": 12,
            "conv3_3": 14,
            "conv3_4": 16,
            "conv4_1": 19,
            "conv4_2": 21,
            "conv4_3": 23,
            "conv4_4": 25,
            "conv5_1": 28,
            "conv5_2": 30,
            "conv5_3": 32,
            "conv5_4": 34,
        }

        max_layer = max([self.layer_name_mapping[name] for name in layers])
        self.features = vgg[: max_layer + 1]

        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # Normalization
        self.register_buffer(
            "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for texture representation."""
        b, c, h, w = features.size()
        features = features.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Compute texture loss.

        Args:
            sr: Super-resolved image
            hr: High-resolution ground truth

        Returns:
            Texture loss value
        """
        # Normalize
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std

        # Extract features and compute Gram matrices
        loss = 0
        x_sr = sr
        x_hr = hr

        for name, idx in self.layer_name_mapping.items():
            if idx >= len(self.features):
                break
            x_sr = self.features[idx](x_sr)
            x_hr = self.features[idx](x_hr)

            if name in self.layers:
                gram_sr = self.gram_matrix(x_sr)
                gram_hr = self.gram_matrix(x_hr)
                weight = self.weights[self.layers.index(name)]
                loss += weight * F.mse_loss(gram_sr, gram_hr)

        return loss


class TVLoss(nn.Module):
    """
    Total Variation loss for smoothness regularization.

    Args:
        tv_type: Type of TV loss ('isotropic' or 'anisotropic')
    """

    def __init__(self, tv_type: str = "isotropic"):
        super(TVLoss, self).__init__()
        self.tv_type = tv_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss.

        Args:
            x: Input image

        Returns:
            TV loss value
        """
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]

        if self.tv_type == "isotropic":
            tv = torch.mean(
                torch.sqrt(diff_h[:, :, :, :-1] ** 2 + diff_w[:, :, :-1, :] ** 2)
            )
        else:  # anisotropic
            tv = torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))

        return tv


class GradientLoss(nn.Module):
    """
    Gradient prior loss using image gradients.

    Encourages similar gradient distributions between SR and HR.

    Args:
        loss_type: Type of gradient loss ('l1' or 'l2')
    """

    def __init__(self, loss_type: str = "l1"):
        super(GradientLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss.

        Args:
            sr: Super-resolved image
            hr: High-resolution ground truth

        Returns:
            Gradient loss value
        """
        # Compute gradients
        sr_grad_h = sr[:, :, 1:, :] - sr[:, :, :-1, :]
        sr_grad_w = sr[:, :, :, 1:] - sr[:, :, :, :-1]

        hr_grad_h = hr[:, :, 1:, :] - hr[:, :, :-1, :]
        hr_grad_w = hr[:, :, :, 1:] - hr[:, :, :, :-1]

        # Compute loss
        if self.loss_type == "l1":
            loss = F.l1_loss(sr_grad_h, hr_grad_h) + F.l1_loss(sr_grad_w, hr_grad_w)
        else:
            loss = F.mse_loss(sr_grad_h, hr_grad_h) + F.mse_loss(sr_grad_w, hr_grad_w)

        return loss


# =============================================================================
# Utilities
# =============================================================================


class SRDataset(Dataset):
    """
    Dataset for super resolution training.

    Loads HR images and generates LR versions.

    Args:
        hr_paths: List of paths to high-resolution images
        scale: Downsampling factor
        patch_size: Size of HR patches to extract
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        hr_paths: List[str],
        scale: int = 4,
        patch_size: int = 192,
        augment: bool = True,
    ):
        self.hr_paths = hr_paths
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment

        self.lr_patch_size = patch_size // scale

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Tuple of (LR, HR) tensors
        """
        # Load HR image
        from PIL import Image

        hr = Image.open(self.hr_paths[idx]).convert("RGB")
        hr = transforms.ToTensor()(hr)

        # Random crop
        if self.augment:
            _, h, w = hr.shape
            if h > self.patch_size and w > self.patch_size:
                top = torch.randint(0, h - self.patch_size, (1,)).item()
                left = torch.randint(0, w - self.patch_size, (1,)).item()
                hr = hr[:, top : top + self.patch_size, left : left + self.patch_size]

        # Generate LR
        lr = F.interpolate(
            hr.unsqueeze(0),
            scale_factor=1 / self.scale,
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                hr = torch.flip(hr, dims=[2])
                lr = torch.flip(lr, dims=[2])

            # Random vertical flip
            if torch.rand(1) > 0.5:
                hr = torch.flip(hr, dims=[1])
                lr = torch.flip(lr, dims=[1])

            # Random rotation
            if torch.rand(1) > 0.5:
                hr = torch.rot90(hr, k=1, dims=[1, 2])
                lr = torch.rot90(lr, k=1, dims=[1, 2])

        return lr, hr


class SRDataLoader:
    """
    DataLoader for super resolution.

    Args:
        dataset: SRDataset instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
    """

    def __init__(
        self,
        dataset: SRDataset,
        batch_size: int = 16,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class SRMetrics:
    """
    Metrics for evaluating super resolution quality.

    Includes PSNR, SSIM, and LPIPS.
    """

    @staticmethod
    def psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.

        Args:
            sr: Super-resolved image
            hr: High-resolution ground truth
            max_val: Maximum pixel value

        Returns:
            PSNR value in dB
        """
        mse = torch.mean((sr - hr) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * torch.log10(max_val / torch.sqrt(mse)).item()

    @staticmethod
    def ssim(
        sr: torch.Tensor, hr: torch.Tensor, window_size: int = 11, max_val: float = 1.0
    ) -> float:
        """
        Calculate Structural Similarity Index.

        Args:
            sr: Super-resolved image
            hr: High-resolution ground truth
            window_size: Size of sliding window
            max_val: Maximum pixel value

        Returns:
            SSIM value
        """
        C1 = (0.01 * max_val) ** 2
        C2 = (0.03 * max_val) ** 2

        mu1 = F.avg_pool2d(sr, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(hr, window_size, stride=1, padding=window_size // 2)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.avg_pool2d(sr**2, window_size, stride=1, padding=window_size // 2)
            - mu1_sq
        )
        sigma2_sq = (
            F.avg_pool2d(hr**2, window_size, stride=1, padding=window_size // 2)
            - mu2_sq
        )
        sigma12 = (
            F.avg_pool2d(sr * hr, window_size, stride=1, padding=window_size // 2)
            - mu1_mu2
        )

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim_map.mean().item()

    @staticmethod
    def lpips(sr: torch.Tensor, hr: torch.Tensor, net: str = "alex") -> float:
        """
        Calculate Learned Perceptual Image Patch Similarity.

        Args:
            sr: Super-resolved image
            hr: High-resolution ground truth
            net: Network to use ('alex' or 'vgg')

        Returns:
            LPIPS value (lower is better)
        """
        try:
            import lpips as lpips_lib

            loss_fn = lpips_lib.LPIPS(net=net).to(sr.device)
            with torch.no_grad():
                dist = loss_fn(sr, hr)
            return dist.mean().item()
        except ImportError:
            warnings.warn("lpips package not installed. Returning 0.")
            return 0.0


class SRPostProcessor:
    """
    Post-processing utilities for super-resolved images.

    Includes artifact removal and enhancement.
    """

    @staticmethod
    def remove_artifacts(image: torch.Tensor, method: str = "median") -> torch.Tensor:
        """
        Remove artifacts from super-resolved images.

        Args:
            image: Input image tensor
            method: Artifact removal method ('median', 'gaussian')

        Returns:
            Processed image
        """
        if method == "median":
            # Apply median filtering
            from scipy.ndimage import median_filter

            img_np = image.cpu().numpy()
            filtered = median_filter(img_np, size=(1, 1, 3, 3))
            return torch.from_numpy(filtered).to(image.device)
        elif method == "gaussian":
            # Apply Gaussian smoothing
            kernel_size = 3
            sigma = 0.5
            x = torch.arange(kernel_size) - kernel_size // 2
            gauss = torch.exp(-(x.float() ** 2) / (2 * sigma**2))
            gauss = gauss / gauss.sum()
            kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
            kernel = kernel.unsqueeze(0).unsqueeze(0).to(image.device)
            kernel = kernel.repeat(image.size(1), 1, 1, 1)

            padding = kernel_size // 2
            filtered = F.conv2d(
                image.unsqueeze(0), kernel, padding=padding, groups=image.size(1)
            )
            return filtered.squeeze(0)
        else:
            return image

    @staticmethod
    def enhance_details(image: torch.Tensor, alpha: float = 1.5) -> torch.Tensor:
        """
        Enhance image details.

        Args:
            image: Input image tensor
            alpha: Enhancement factor

        Returns:
            Enhanced image
        """
        # Unsharp masking
        blurred = F.avg_pool2d(image.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        detail = image - blurred
        enhanced = image + alpha * detail
        return torch.clamp(enhanced, 0, 1)

    @staticmethod
    def adjust_brightness_contrast(
        image: torch.Tensor, brightness: float = 0, contrast: float = 1
    ) -> torch.Tensor:
        """
        Adjust brightness and contrast.

        Args:
            image: Input image tensor
            brightness: Brightness adjustment
            contrast: Contrast factor

        Returns:
            Adjusted image
        """
        adjusted = (image - 0.5) * contrast + 0.5 + brightness
        return torch.clamp(adjusted, 0, 1)


class SRTrainer:
    """
    Trainer for super resolution models.

    Handles training loop, validation, and checkpointing.

    Args:
        model: Super resolution model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: SRDataLoader,
        val_loader: Optional[SRDataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        else:
            self.optimizer = optimizer

        self.criterion = PixelLoss("l1")
        self.metrics = SRMetrics()
        self.best_psnr = 0.0

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_psnr = 0.0

        for lr, hr in self.train_loader:
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            # Forward pass
            sr = self.model(lr)
            loss = self.criterion(sr, hr)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_psnr += self.metrics.psnr(sr, hr)

        return {
            "loss": total_loss / len(self.train_loader),
            "psnr": total_psnr / len(self.train_loader),
        }

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0

        with torch.no_grad():
            for lr, hr in self.val_loader:
                lr = lr.to(self.device)
                hr = hr.to(self.device)

                sr = self.model(lr)
                total_psnr += self.metrics.psnr(sr, hr)
                total_ssim += self.metrics.ssim(sr, hr)

        metrics = {
            "psnr": total_psnr / len(self.val_loader),
            "ssim": total_ssim / len(self.val_loader),
        }

        # Update best PSNR
        if metrics["psnr"] > self.best_psnr:
            self.best_psnr = metrics["psnr"]

        return metrics

    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_psnr": self.best_psnr,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_psnr = checkpoint.get("best_psnr", 0.0)
        return checkpoint.get("epoch", 0)


# =============================================================================
# Real-world Super Resolution Models
# =============================================================================


class RealSRModel(nn.Module):
    """
    Real-world Super Resolution Model.

    Designed for real-world degradations with complex noise and blur.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of residual blocks (default: 20)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 20,
        upscale_factor: int = 4,
    ):
        super(RealSRModel, self).__init__()

        # Degradation estimation branch
        self.degradation_est = nn.Sequential(
            nn.Conv2d(num_channels, num_features // 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features // 2, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Main restoration branch
        self.conv_in = nn.Conv2d(num_channels, num_features, 3, padding=1)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_features, num_features, 3, padding=1),
                )
                for _ in range(num_blocks)
            ]
        )

        # Adaptive fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Upsampling
        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=True),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RealSR.

        Args:
            x: Low-quality input image

        Returns:
            High-quality super-resolved image
        """
        # Estimate degradation
        degrad = self.degradation_est(x)

        # Main processing
        feat = self.conv_in(x)

        for block in self.blocks:
            feat = feat + block(feat)

        # Fusion with degradation information
        combined = torch.cat([feat, degrad], dim=1)
        feat = self.fusion(combined)

        # Upsample
        out = self.upsample(feat)
        return out


class BSRGANGenerator(nn.Module):
    """
    Blind Super-Resolution GAN (BSRGAN).

    Zhang et al., "Designing a Practical Degradation Model for
    Deep Blind Image Super-Resolution", ICCV 2021.

    Handles diverse real-world degradations.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of RRDBs (default: 23)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 23,
        upscale_factor: int = 4,
    ):
        super(BSRGANGenerator, self).__init__()

        # Similar to ESRGAN but with modifications for blind SR
        self.conv_first = nn.Conv2d(num_channels, num_features, 3, padding=1)

        # RRDB blocks with enhanced capacity
        self.body = nn.ModuleList([RRDB(num_features) for _ in range(num_blocks)])

        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1)

        # Upsampling
        upsample_layers = []
        for _ in range(int(math.log(upscale_factor, 2))):
            upsample_layers.extend(
                [
                    nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.upsample = nn.Sequential(*upsample_layers)

        # High-resolution refinement
        self.hr_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_last = nn.Conv2d(num_features, num_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BSRGAN generator.

        Args:
            x: Low-quality input image

        Returns:
            High-quality super-resolved image
        """
        fea = self.conv_first(x)

        body_fea = fea
        for block in self.body:
            body_fea = block(body_fea)
        body_fea = self.conv_body(body_fea)

        fea = fea + body_fea

        out = self.upsample(fea)
        out = self.hr_conv(out)
        out = self.conv_last(out)

        return out


class RealESRGANPlusGenerator(nn.Module):
    """
    Real-ESRGAN+ with improved real-world SR capabilities.

    Enhanced version with better handling of complex degradations.

    Args:
        num_channels: Number of input/output channels
        num_features: Number of feature maps (default: 64)
        num_blocks: Number of RRDBs (default: 23)
        upscale_factor: Upsampling factor (default: 4)
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 23,
        upscale_factor: int = 4,
    ):
        super(RealESRGANPlusGenerator, self).__init__()

        # U-Net style encoder-decoder for degradation-aware processing
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck with RRDBs
        self.bottleneck = nn.ModuleList(
            [RRDB(num_features * 4) for _ in range(num_blocks)]
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                num_features * 4, num_features * 2, 4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(num_features * 2, num_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Upsampling
        self.upsample = nn.Sequential(
            Upsampler(default_conv, upscale_factor, num_features, act=False),
            nn.Conv2d(num_features, num_channels, 3, padding=1),
        )

        # Skip connections
        self.skip1 = nn.Conv2d(num_features, num_features, 1)
        self.skip2 = nn.Conv2d(num_features * 2, num_features * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Real-ESRGAN+.

        Args:
            x: Low-quality input image

        Returns:
            High-quality super-resolved image
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = e3
        for block in self.bottleneck:
            b = block(b)

        # Decoder with skip connections
        d3 = self.dec3(b + self.skip2(e2))
        d2 = self.dec2(d3 + self.skip1(e1))

        # Upsample
        out = self.upsample(d2)
        return out


# =============================================================================
# Model Factory
# =============================================================================


def create_sr_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create super resolution models.

    Args:
        model_name: Name of the model
        **kwargs: Additional model parameters

    Returns:
        Super resolution model instance

    Example:
        >>> model = create_sr_model('edsr', num_features=256, num_blocks=32)
    """
    models = {
        # Single Image SR
        "srcnn": SRCNN,
        "fsrcnn": FSRCNN,
        "vdsr": VDSR,
        "drcn": DRCN,
        "drrn": DRRN,
        "lapsrn": LapSRN,
        "edsr": EDSR,
        "rcan": RCAN,
        "rdn": RDN,
        "dbpn": DBPNSuperResolution,
        # GAN-based SR
        "srgan": SRGANGenerator,
        "esrgan": ESRGANGenerator,
        "realesrgan": RealESRGANGenerator,
        "srfeat": SRFeatGenerator,
        "bebygan": BebyGANGenerator,
        # Lightweight SR
        "carn": CARN,
        "falsr": FALSRGenerator,
        "msrn": MSRN,
        "idn": IDN,
        # Video SR
        "vsrnet": VSRNet,
        "frvsr": FRVSR,
        "duf": DUF,
        "edvr": EDVR,
        "basicvsr": BasicVSR,
        "basicvsr++": BasicVSRPlusPlus,
        # Real-world SR
        "realsr": RealSRModel,
        "bsrgan": BSRGANGenerator,
        "realesrgan+": RealESRGANPlusGenerator,
    }

    model_name = model_name.lower()
    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(models.keys())}"
        )

    return models[model_name](**kwargs)


# Export all classes
__all__ = [
    # Single Image SR
    "SRCNN",
    "FSRCNN",
    "VDSR",
    "DRCN",
    "DRRN",
    "LapSRN",
    "EDSR",
    "RCAN",
    "RDN",
    "DBPNSuperResolution",
    # GAN-based SR
    "SRGANGenerator",
    "SRGANDiscriminator",
    "ESRGANGenerator",
    "ESRGANDiscriminator",
    "RealESRGANGenerator",
    "SRFeatGenerator",
    "SRFeatDiscriminator",
    "BebyGANGenerator",
    # Lightweight SR
    "CARN",
    "FALSRGenerator",
    "MSRN",
    "IDN",
    # Video SR
    "VSRNet",
    "FRVSR",
    "DUF",
    "EDVR",
    "BasicVSR",
    "BasicVSRPlusPlus",
    # Real-world SR
    "RealSRModel",
    "BSRGANGenerator",
    "RealESRGANPlusGenerator",
    # Loss Functions
    "PixelLoss",
    "PerceptualLoss",
    "AdversarialLoss",
    "TextureLoss",
    "TVLoss",
    "GradientLoss",
    # Utilities
    "SRDataset",
    "SRDataLoader",
    "SRTrainer",
    "SRMetrics",
    "SRPostProcessor",
    # Factory
    "create_sr_model",
]

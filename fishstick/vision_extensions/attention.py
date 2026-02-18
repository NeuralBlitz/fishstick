"""
Attention Mechanisms for Computer Vision

Novel attention modules including Squeeze-and-Excitation (SE),
Convolutional Block Attention Module (CBAM), Efficient Channel
Attention (ECA), Coordinate Attention, and SimAM.

References:
    - SE-Net: https://arxiv.org/abs/1709.01507
    - CBAM: https://arxiv.org/abs/1807.06521
    - ECA-Net: https://arxiv.org/abs/1910.03151
    - CoordAttention: https://arxiv.org/abs/2103.02907
    - SimAM: https://arxiv.org/abs/2202.13599
"""

from typing import Optional, Tuple
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention.

    Adaptively recalibrates channel-wise feature responses by modeling
    interdependencies between channels. Contains global average pooling,
    bottleneck FC layers, and sigmoid gating.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        reduced_channels = max(channels // reduction, 8)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
        Returns:
            [B, C, H, W] attention-weighted tensor
        """
        b, c, _, _ = x.size()

        squeeze_out = self.squeeze(x).view(b, c)
        excitation_out = self.excitation(squeeze_out).view(b, c, 1, 1)

        return x * excitation_out.expand_as(x)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.

    Uses both max-pooling and average-pooled features to compute
    channel attention weights.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio (default: 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
        Returns:
            [B, C, 1, 1] channel attention weights
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.

    Computes spatial attention using channel-wise pooling (both
    max and mean) followed by convolution.

    Args:
        kernel_size: Convolution kernel size (default: 7)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
        Returns:
            [B, 1, H, W] spatial attention weights
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        combined = torch.cat([avg_out, max_out], dim=1)
        spatial = self.sigmoid(self.conv(combined))

        return spatial


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Sequentially applies channel attention then spatial attention
    to refine feature maps. Both lightweight and effective.

    Args:
        channels: Number of input channels
        reduction: Channel attention reduction ratio (default: 16)
        kernel_size: Spatial attention kernel size (default: 7)
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
        Returns:
            [B, C, H, W] attention-refined tensor
        """
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ECAAttention(nn.Module):
    """
    Efficient Channel Attention Module.

    Uses 1D convolution for efficient cross-channel interaction
    without dimensionality reduction. Avoids SE bottleneck.

    Args:
        channels: Number of input channels
        kernel_size: 1D conv kernel size (default: 3)
        gamma: Adaptive kernel computation (default: 2)
        b: Adaptive kernel computation (default: 1)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        gamma: int = 2,
        b: int = 1,
    ):
        super().__init__()

        t = int(abs((math.log2(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
        Returns:
            [B, C, H, W] channel-refined tensor
        """
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-2, -1))
        y = y.transpose(-2, -1).unsqueeze(-1)

        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CoordAttention(nn.Module):
    """
    Coordinate Attention.

    Embeds location information into channel attention by encoding
    horizontal and vertical directions separately, then fusing.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio (default: 32)
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()

        reduced = max(channels // reduction, 1)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(reduced, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(reduced, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
        Returns:
            [B, C, H, W] coordinate-refined tensor
        """
        b, c, h, w = x.size()

        identity = x

        n_h = self.pool_h(x)
        n_w = self.pool_w(x)

        n_w = n_w.permute(0, 1, 3, 2)
        combined = torch.cat([n_h, n_w], dim=2)

        combined = self.conv1(combined)
        combined = self.bn1(combined)
        combined = self.act(combined)

        h_attn, w_attn = torch.split(combined, [h, w], dim=2)
        w_attn = w_attn.permute(0, 1, 3, 2)

        h_attn = self.conv_h(h_attn).sigmoid()
        w_attn = self.conv_w(w_attn).sigmoid()

        return identity * h_attn * w_attn


class SimAM(nn.Module):
    """
    Simple Attention Module (SimAM).

    Computers attention weights using a data-driven approach
    without extra parameters. Based on energy function minimization.

    Args:
        e_lambda: Epsilon for numerical stability (default: 1e-4)
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
        Returns:
            [B, C, H, W] attention-refined tensor
        """
        b, c, h, w = x.size()

        n = w * h - 1

        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        var = torch.var(x, dim=[2, 3], keepdim=True, unbiased=False)

        t = x - mean
        v = var + self.e_lambda

        t_sum = torch.sum(t**2, dim=[2, 3], keepdim=True)

        w = t / (torch.sqrt(v * n / t_sum + self.e_lambda))
        w = torch.exp(-(w**2) / (4 * (v + self.e_lambda)))

        return x * w


class MixedAttention(nn.Module):
    """
    Mixed Attention combining multiple attention mechanisms.

    Combines SE channel attention with spatial attention in
    a flexible manner.

    Args:
        channels: Number of input channels
        reduction: SE reduction ratio (default: 16)
        spatial_kernel: Spatial attention kernel (default: 7)
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        self.se = SqueezeExcitation(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
        Returns:
            [B, C, H, W] attention-refined tensor
        """
        x = self.se(x)
        x = self.spatial(x)
        return x


def create_attention(
    attention_type: str,
    channels: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create attention modules.

    Args:
        attention_type: Type of attention ('se', 'cbam', 'eca', 'coord', 'simam')
        channels: Number of input channels
        **kwargs: Additional arguments for specific attention modules

    Returns:
        Attention module

    Raises:
        ValueError: If attention_type is not recognized
    """
    attention_types = {
        "se": SqueezeExcitation,
        "cbam": CBAM,
        "eca": ECAAttention,
        "coord": CoordAttention,
        "simam": SimAM,
    }

    if attention_type.lower() not in attention_types:
        raise ValueError(
            f"Unknown attention type: {attention_type}. "
            f"Available: {list(attention_types.keys())}"
        )

    return attention_types[attention_type.lower()](channels, **kwargs)

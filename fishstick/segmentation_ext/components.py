"""
Common Components for Segmentation Models

Reusable building blocks for semantic and instance segmentation architectures
including upsampling, residual blocks, ASPP modules, and feature fusion layers.

Author: fishstick AI Framework
Version: 0.1.0
"""

from typing import List, Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    """
    Configurable upsampling block with optional fusion support.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        scale_factor: Upsampling scale factor
        mode: Upsampling mode ('bilinear', 'nearest', 'pixelshuffle')
        use_conv: Whether to apply conv after upsampling
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = "bilinear",
        use_conv: bool = True,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

        layers = []
        if use_conv:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )

        self.conv = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input tensor
            skip: [B, C_skip, H_skip, W_skip] Optional skip connection
        Returns:
            [B, out_channels, H*scale, W*scale] Upsampled tensor
        """
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=False if self.mode == "bilinear" else None,
        )

        if skip is not None:
            x = x + skip

        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    Residual block with optional dilation for segmentation.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        dilation: Dilation rate for atrous convolutions
        groups: Number of grouped convolutions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()

        mid_channels = out_channels // 2 if out_channels > 64 else out_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            mid_channels,
            out_channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input tensor
        Returns:
            [B, out_channels, H, W] Output tensor
        """
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for multi-scale feature extraction.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        atrous_rates: List of dilation rates
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: List[int] = (6, 12, 18),
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.atrous_rates = atrous_rates
        modules = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input features
        Returns:
            [B, out_channels, H, W] Multi-scale features
        """
        res = []
        for conv in self.convs:
            res.append(conv(x))

        res[-1] = F.interpolate(
            res[-1], size=x.shape[2:], mode="bilinear", align_corners=False
        )

        x = torch.cat(res, dim=1)
        return self.project(x)


class SpatialPyramidPool(nn.Module):
    """
    Spatial Pyramid Pooling for multi-scale context.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        pool_sizes: List of pooling window sizes
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_sizes: List[int] = (1, 2, 4, 8),
    ):
        super().__init__()

        self.pool_sizes = pool_sizes
        modules = []

        for pool_size in pool_sizes:
            modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(
                        in_channels, out_channels // len(pool_sizes), 1, bias=False
                    ),
                    nn.BatchNorm2d(out_channels // len(pool_sizes)),
                    nn.ReLU(inplace=True),
                )
            )

        self.pools = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input features
        Returns:
            [B, out_channels, H, W] Multi-scale pooled features
        """
        pool_outs = []
        for pool in self.pools:
            pool_out = pool(x)
            pool_outs.append(
                F.interpolate(
                    pool_out, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            )

        x = torch.cat([x] + pool_outs, dim=1)
        return self.project(x)


class FeatureFusionModule(nn.Module):
    """
    Feature fusion module for combining multi-scale features.

    Args:
        in_channels_low: Channel dimension of low-level features
        in_channels_high: Channel dimension of high-level features
        out_channels: Output channel dimension
    """

    def __init__(
        self,
        in_channels_low: int,
        in_channels_high: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv_low = nn.Sequential(
            nn.Conv2d(in_channels_low, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels_high, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_low: Tensor, x_high: Tensor) -> Tensor:
        """
        Args:
            x_low: [B, C_low, H, W] Low-level features
            x_high: [B, C_high, H*scale, W*scale] High-level features
        Returns:
            [B, out_channels, H, W] Fused features
        """
        x_high_up = F.interpolate(
            x_high, size=x_low.shape[2:], mode="bilinear", align_corners=False
        )

        x_low = self.conv_low(x_low)
        x_high_up = self.conv_high(x_high_up)

        x = x_low + x_high_up
        x = self.relu(x)

        return self.conv_fusion(x)


class ChannelAttention(nn.Module):
    """
    Channel attention module for feature recalibration.

    Args:
        channels: Number of channels
        reduction: Reduction ratio for bottleneck
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input features
        Returns:
            [B, C, H, W] Recalibrated features
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class DualAttentionBlock(nn.Module):
    """
    Dual attention block combining channel and spatial attention.

    Args:
        channels: Number of channels
        reduction: Channel attention reduction ratio
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input features
        Returns:
            [B, C, H, W] Attention-weighted features
        """
        x = self.channel_attention(x)

        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights

        return x


class ConvBNReLU(nn.Module):
    """
    Convolution + BatchNorm + ReLU block.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding size
        use_bn: Whether to use batch normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
    ):
        super().__init__()
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn
        )

        if use_bn:
            self.block = nn.Sequential(
                conv,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                conv,
                nn.ReLU(inplace=True),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Generic decoder block for segmentation models.

    Args:
        in_channels: Input channel dimension
        skip_channels: Skip connection channel dimension
        out_channels: Output channel dimension
        use_bn: Whether to use batch normalization
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_bn: bool = True,
    ):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.conv = ConvBNReLU(
            out_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn,
        )
        self.conv2 = ConvBNReLU(
            out_channels, out_channels, kernel_size=3, padding=1, use_bn=use_bn
        )

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, C, H, W] Input tensor
            skip: [B, C_skip, H_skip, W_skip] Skip connection
        Returns:
            [B, out_channels, H*2, W*2] Decoded features
        """
        x = self.upconv(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        x = self.conv2(x)

        return x


def create_upsample_block(
    in_channels: int,
    out_channels: int,
    scale_factor: int = 2,
    mode: str = "bilinear",
) -> UpsampleBlock:
    """
    Factory function to create an upsampling block.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        scale_factor: Upsampling scale factor
        mode: Upsampling interpolation mode

    Returns:
        UpsampleBlock instance
    """
    return UpsampleBlock(in_channels, out_channels, scale_factor, mode)


def create_aspp(
    in_channels: int,
    out_channels: int,
    atrous_rates: Tuple[int, ...] = (6, 12, 18),
    dropout_rate: float = 0.5,
) -> ASPPModule:
    """
    Factory function to create an ASPP module.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        atrous_rates: Tuple of dilation rates
        dropout_rate: Dropout probability

    Returns:
        ASPPModule instance
    """
    return ASPPModule(in_channels, out_channels, list(atrous_rates), dropout_rate)

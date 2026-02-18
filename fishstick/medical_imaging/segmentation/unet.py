"""
U-Net Architectures for Medical Image Segmentation

Implementation of 3D U-Net, Residual U-Net, and Attention U-Net
for volumetric medical image segmentation.
"""

from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBlock3D(nn.Module):
    """3D Convolutional block with batch normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_instance_norm: bool = False,
    ):
        super().__init__()
        norm = nn.InstanceNorm3d if use_instance_norm else nn.BatchNorm3d

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
            norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ResidualBlock3D(nn.Module):
    """3D Residual block for U-Net."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu(out)

        return out


class AttentionGate3D(nn.Module):
    """Attention gate for 3D U-Net."""

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = skip_channels // 2

        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm3d(inter_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: Tensor, skip: Tensor) -> Tensor:
        g1 = self.W_g(gate)
        x1 = self.W_x(skip)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return skip * psi


class DownBlock3D(nn.Module):
    """Downsampling block for U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_residual: bool = False,
    ):
        super().__init__()

        if use_residual:
            self.block = ResidualBlock3D(in_channels, out_channels)
        else:
            self.block = ConvBlock3D(in_channels, out_channels)

        self.pool = nn.MaxPool3d(2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        skip = self.block(x)
        x = self.pool(skip)
        return x, skip


class UpBlock3D(nn.Module):
    """Upsampling block for U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        mode: str = "trilinear",
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = (
            AttentionGate3D(out_channels, out_channels) if use_attention else None
        )
        self.block = ConvBlock3D(out_channels * 2, out_channels)
        self.mode = mode

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode=self.mode, align_corners=False
            )

        if self.attention is not None:
            skip = self.attention(x, skip)

        x = torch.cat([x, skip], dim=1)
        x = self.block(x)

        return x


class UNet3D(nn.Module):
    """3D U-Net for volumetric medical image segmentation.

    Architecture:
        - Encoder: 4 levels with increasing channels (32, 64, 128, 256)
        - Bottleneck: 512 channels
        - Decoder: 4 levels with attention gates (optional)

    Example:
        >>> model = UNet3D(in_channels=1, num_classes=3)
        >>> input = torch.randn(1, 1, 128, 128, 128)
        >>> output = model(input)
        >>> output.shape
        torch.Size([1, 3, 128, 128, 128])
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        depth: int = 4,
        use_attention: bool = False,
        use_residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth

        self.encoder = nn.ModuleList()
        channels = base_channels

        for i in range(depth):
            self.encoder.append(
                DownBlock3D(
                    in_channels if i == 0 else channels // 2,
                    channels,
                    use_residual=use_residual,
                )
            )
            channels *= 2

        self.bottleneck = ConvBlock3D(channels // 2, channels)

        self.decoder = nn.ModuleList()
        for i in range(depth):
            self.decoder.append(
                UpBlock3D(
                    channels,
                    channels // 2,
                    use_attention=use_attention,
                )
            )
            channels //= 2

        self.output = nn.Conv3d(base_channels, num_classes, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout3d(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        skip_connections = []

        for block in self.encoder:
            x, skip = block(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i, block in enumerate(self.decoder):
            x = block(x, skip_connections[i])

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.output(x)

        return x

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class ResidualUNet3D(UNet3D):
    """3D Residual U-Net.

    Uses residual connections in encoder blocks for better gradient flow.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
            use_residual=True,
            dropout=dropout,
        )


class AttentionUNet3D(UNet3D):
    """3D Attention U-Net.

    Uses attention gates in decoder for better feature selection.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
            use_attention=True,
            dropout=dropout,
        )


class UNet2D(nn.Module):
    """2D U-Net for slice-wise segmentation.

    Useful for processing 2D slices from 3D volumes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.encoder = nn.ModuleList()
        channels = base_channels

        for i in range(depth):
            self.encoder.append(
                ConvBlock2D(
                    in_channels if i == 0 else channels // 2,
                    channels,
                )
            )
            channels *= 2

        self.bottleneck = ConvBlock2D(channels // 2, channels)

        self.decoder = nn.ModuleList()
        for i in range(depth):
            self.decoder.append(UpBlock2D(channels, channels // 2))
            channels //= 2

        self.output = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skip_connections = []

        for block in self.encoder:
            x = block(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for i, block in enumerate(self.decoder[-len(skip_connections) :]):
            x = block(x, skip_connections[-(i + 1)])

        return self.output(x)


class ConvBlock2D(nn.Module):
    """2D Convolutional block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UpBlock2D(nn.Module):
    """2D Upsampling block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.block = ConvBlock2D(out_channels * 2, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        return self.block(x)

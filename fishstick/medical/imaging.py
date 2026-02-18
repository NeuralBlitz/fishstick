"""
Medical Imaging Module for Fishstick

Comprehensive deep learning tools for medical image analysis including:
- Segmentation (UNet, VNet, Attention UNet)
- Classification (CheXNet, Medical ResNet, Medical ViT)
- Registration (VoxelMorph, SynthMorph)
- Reconstruction (Denoising, Super-resolution)
- Anomaly Detection
- Datasets and Evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict, Callable, Union
import numpy as np
from scipy.spatial.distance import directed_hausdorff


# =============================================================================
# SEGMENTATION MODELS
# =============================================================================


class DoubleConv2D(nn.Module):
    """Double convolution block for U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock2D(nn.Module):
    """Downsampling block with maxpool and double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv2D(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock2D(nn.Module):
    """Upsampling block with upsample/conv transpose and double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv2D(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    2D U-Net for medical image segmentation.

    Reference: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512, 1024],
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv2D(in_channels, features[0])

        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(DownBlock2D(features[i], features[i + 1]))

        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(
                UpBlock2D(
                    features[i] * 2 if bilinear else features[i],
                    features[i - 1],
                    bilinear,
                )
            )

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)

        skip_connections = [x]
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        skip_connections = skip_connections[:-1][::-1]

        for up, skip in zip(self.ups, skip_connections):
            x = up(x, skip)

        if self.dropout is not None:
            x = self.dropout(x)

        logits = self.outc(x)
        return logits


class DoubleConv3D(nn.Module):
    """Double convolution block for 3D U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock3D(nn.Module):
    """3D Downsampling block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock3D(nn.Module):
    """3D Upsampling block."""

    def __init__(self, in_channels: int, out_channels: int, trilinear: bool = True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]
        x1 = F.pad(
            x1,
            [
                diffW // 2,
                diffW - diffW // 2,
                diffH // 2,
                diffH - diffH // 2,
                diffD // 2,
                diffD - diffD // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric medical image segmentation.

    Reference: Cicek et al. "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256, 512],
        trilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv3D(in_channels, features[0])

        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(DownBlock3D(features[i], features[i + 1]))

        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(
                UpBlock3D(
                    features[i] * 2 if trilinear else features[i],
                    features[i - 1],
                    trilinear,
                )
            )

        self.outc = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)

        skip_connections = [x]
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        skip_connections = skip_connections[:-1][::-1]

        for up, skip in zip(self.ups, skip_connections):
            x = up(x, skip)

        if self.dropout is not None:
            x = self.dropout(x)

        logits = self.outc(x)
        return logits


class VNetConvBlock(nn.Module):
    """VNet convolution block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()
        self.num_convs = num_convs

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            self.convs.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, out_channels, kernel_size=5, padding=2),
                    nn.BatchNorm3d(out_channels),
                    nn.PReLU(),
                )
            )

        self.skip_conv = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip_conv is None else self.skip_conv(x)

        for conv in self.convs:
            x = conv(x)

        return x + residual


class VNetDownBlock(nn.Module):
    """VNet downsampling block."""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )
        self.conv_block = VNetConvBlock(out_channels, out_channels, num_convs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        down = self.down(x)
        out = self.conv_block(down)
        return out, out - down


class VNetUpBlock(nn.Module):
    """VNet upsampling block."""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels // 2),
            nn.PReLU(),
        )
        self.conv_block = VNetConvBlock(out_channels, out_channels, num_convs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class VNet(nn.Module):
    """
    V-Net for volumetric medical image segmentation.

    Reference: Milletari et al. "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [16, 32, 64, 128, 256],
        num_convs: List[int] = [1, 2, 3, 3, 3],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_conv = VNetConvBlock(in_channels, features[0], num_convs[0])

        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(
                VNetDownBlock(features[i], features[i + 1], num_convs[i + 1])
            )

        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(
                VNetUpBlock(features[i], features[i - 1] * 2, num_convs[i - 1])
            )

        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)

        skips = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        skips = skips[:-1][::-1]

        for up, skip in zip(self.ups, skips):
            x = up(x, skip)

        return self.out_conv(x)


class AttentionGate(nn.Module):
    """Attention Gate for Attention U-Net."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net with attention gates for medical image segmentation.

    Reference: Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512, 1024],
    ):
        super().__init__()

        self.inc = DoubleConv2D(in_channels, features[0])

        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(DownBlock2D(features[i], features[i + 1]))

        self.attentions = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.attentions.append(
                AttentionGate(features[i], features[i - 1], features[i - 1] // 2)
            )

        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(UpBlock2D(features[i] * 2, features[i - 1]))

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)

        skip_connections = [x]
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        skip_connections = skip_connections[:-1][::-1]

        for i, (up, skip) in enumerate(zip(self.ups, skip_connections)):
            g = x
            skip = self.attentions[i](g, skip)
            x = up(x, skip)

        return self.outc(x)


class ResidualBlock(nn.Module):
    """Residual block for Residual U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ResidualUNet(nn.Module):
    """
    Residual U-Net with residual connections in encoder blocks.

    Reference: Alom et al. "Recurrent Residual Convolutional Neural Network based on U-Net"
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512, 1024],
    ):
        super().__init__()

        self.inc = ResidualBlock(in_channels, features[0])

        self.downs = nn.ModuleList()
        for i in range(len(features) - 1):
            self.downs.append(
                nn.Sequential(
                    nn.MaxPool2d(2), ResidualBlock(features[i], features[i + 1])
                )
            )

        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(UpBlock2D(features[i] * 2, features[i - 1]))

        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)

        skip_connections = [x]
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        skip_connections = skip_connections[:-1][::-1]

        for up, skip in zip(self.ups, skip_connections):
            x = up(x, skip)

        return self.outc(x)


# =============================================================================
# CLASSIFICATION MODELS
# =============================================================================


class DenseLayer(nn.Module):
    """Dense layer for DenseNet."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, bn_size * growth_rate, kernel_size=1, bias=False
        )

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.dropout is not None:
            new_features = self.dropout(new_features)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    """Dense block containing multiple dense layers."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                DenseLayer(
                    in_channels + i * growth_rate, growth_rate, bn_size, drop_rate
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Transition(nn.Module):
    """Transition layer between dense blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class CheXNet(nn.Module):
    """
    CheXNet for chest X-ray classification.
    DenseNet-121 based architecture for multi-label classification.

    Reference: Rajpurkar et al. "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays"
    """

    def __init__(
        self,
        num_classes: int = 14,
        in_channels: int = 1,
        growth_rate: int = 32,
        block_config: Tuple[int, ...] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        self.features = nn.Sequential()

        self.features.add_module(
            "conv0",
            nn.Conv2d(
                in_channels,
                num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
        )
        self.features.add_module("norm0", nn.BatchNorm2d(num_init_features))
        self.features.add_module("relu0", nn.ReLU(inplace=True))
        self.features.add_module(
            "pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_features, growth_rate, bn_size, drop_rate
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class BasicBlock(nn.Module):
    """Basic ResNet block."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckBlock(nn.Module):
    """Bottleneck ResNet block."""

    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetMedical(nn.Module):
    """
    ResNet for medical image classification.

    Adapted for medical imaging with single-channel input support.
    """

    def __init__(
        self,
        block: type,
        num_blocks: List[int],
        num_classes: int = 2,
        in_channels: int = 1,
        initial_filters: int = 64,
    ):
        super().__init__()
        self.in_planes = initial_filters

        self.conv1 = nn.Conv2d(
            in_channels, initial_filters, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(initial_filters)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, initial_filters, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, initial_filters * 2, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, initial_filters * 4, num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, initial_filters * 8, num_blocks[3], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(initial_filters * 8 * block.expansion, num_classes)

    def _make_layer(self, block: type, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet18_medical(num_classes: int = 2, in_channels: int = 1) -> ResNetMedical:
    """ResNet-18 for medical images."""
    return ResNetMedical(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


def resnet34_medical(num_classes: int = 2, in_channels: int = 1) -> ResNetMedical:
    """ResNet-34 for medical images."""
    return ResNetMedical(BasicBlock, [3, 4, 6, 3], num_classes, in_channels)


def resnet50_medical(num_classes: int = 2, in_channels: int = 1) -> ResNetMedical:
    """ResNet-50 for medical images."""
    return ResNetMedical(BottleneckBlock, [3, 4, 6, 3], num_classes, in_channels)


class DenseNetMedical(nn.Module):
    """
    DenseNet for medical image classification.

    Reference: Huang et al. "Densely Connected Convolutional Networks"
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        growth_rate: int = 32,
        block_config: Tuple[int, ...] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        self.features = nn.Sequential()
        self.features.add_module(
            "conv0",
            nn.Conv2d(
                in_channels,
                num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
        )
        self.features.add_module("norm0", nn.BatchNorm2d(num_init_features))
        self.features.add_module("relu0", nn.ReLU(inplace=True))
        self.features.add_module(
            "pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_features, growth_rate, bn_size, drop_rate
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class PatchEmbedding(nn.Module):
    """Patch embedding for Vision Transformer."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """MLP block for Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerMedical(nn.Module):
    """
    Vision Transformer for medical image classification.

    Reference: Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition"
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


# =============================================================================
# REGISTRATION MODELS
# =============================================================================


class VoxelMorphEncoder(nn.Module):
    """Encoder for VoxelMorph."""

    def __init__(self, in_channels: int = 2, features: List[int] = [16, 32, 32, 32]):
        super().__init__()
        self.layers = nn.ModuleList()

        prev_features = in_channels
        for i, num_features in enumerate(features):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        prev_features, num_features, kernel_size=3, stride=2, padding=1
                    ),
                    nn.LeakyReLU(0.2),
                )
            )
            prev_features = num_features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips = []
        for layer in self.layers[:-1]:
            x = layer(x)
            skips.append(x)
        x = self.layers[-1](x)
        return [x] + skips[::-1]


class VoxelMorphDecoder(nn.Module):
    """Decoder for VoxelMorph."""

    def __init__(
        self, out_channels: int = 3, features: List[int] = [32, 32, 32, 32, 32, 16]
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        prev_features = features[0]
        for i, num_features in enumerate(features[1:]):
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        prev_features + (features[0] if i < len(features) - 2 else 0),
                        num_features,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
                )
            )
            prev_features = num_features

        self.flow = nn.Conv3d(prev_features, out_channels, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(torch.zeros_like(self.flow.weight))
        self.flow.bias = nn.Parameter(torch.zeros_like(self.flow.bias))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[0]
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if i < len(features) - 1:
                x = torch.cat([x, features[i + 1]], dim=1)

        x = self.layers[-1](x) if len(self.layers) > 1 else x
        flow = self.flow(x)
        return flow


class SpatialTransformer(nn.Module):
    """Spatial transformer network for image warping."""

    def __init__(self, size: Tuple[int, ...]):
        super().__init__()
        self.size = size

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = grid.unsqueeze(0).float()
        self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode="bilinear")


class VoxelMorph(nn.Module):
    """
    VoxelMorph for deformable medical image registration.

    Reference: Balakrishnan et al. "An Unsupervised Learning Model for Deformable Medical Image Registration"
    """

    def __init__(
        self,
        in_channels: int = 1,
        vol_size: Tuple[int, int, int] = (160, 192, 224),
        enc_features: List[int] = [16, 32, 32, 32],
        dec_features: List[int] = [32, 32, 32, 32, 32, 16, 16],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.vol_size = vol_size

        self.encoder = VoxelMorphEncoder(in_channels * 2, enc_features)
        self.decoder = VoxelMorphDecoder(3, dec_features)
        self.transformer = SpatialTransformer(vol_size)

    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([moving, fixed], dim=1)
        features = self.encoder(x)
        flow = self.decoder(features)
        moved = self.transformer(moving, flow)
        return moved, flow


class DeeperReg(nn.Module):
    """
    Deep registration network with additional skip connections.

    Enhanced version of VoxelMorph with deeper architecture.
    """

    def __init__(
        self,
        in_channels: int = 1,
        vol_size: Tuple[int, int, int] = (160, 192, 224),
        enc_features: List[int] = [32, 64, 128, 256],
        dec_features: List[int] = [256, 128, 64, 32, 16],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.vol_size = vol_size

        self.encoder = nn.ModuleList()
        prev_features = in_channels * 2
        for num_features in enc_features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv3d(
                        prev_features, num_features, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm3d(num_features),
                    nn.LeakyReLU(0.2),
                    nn.Conv3d(num_features, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm3d(num_features),
                    nn.LeakyReLU(0.2),
                )
            )
            prev_features = num_features

        self.decoder = nn.ModuleList()
        for i, num_features in enumerate(dec_features):
            in_feat = enc_features[-(i + 1)] + (dec_features[i - 1] if i > 0 else 0)
            self.decoder.append(
                nn.Sequential(
                    nn.Conv3d(in_feat, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm3d(num_features),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
                )
            )

        self.flow = nn.Conv3d(dec_features[-1], 3, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(torch.zeros_like(self.flow.weight))
        self.flow.bias = nn.Parameter(torch.zeros_like(self.flow.bias))
        self.transformer = SpatialTransformer(vol_size)

    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([moving, fixed], dim=1)

        skips = []
        for enc in self.encoder[:-1]:
            x = enc(x)
            skips.append(x)
        x = self.encoder[-1](x)

        for i, dec in enumerate(self.decoder):
            if i < len(skips):
                x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = dec(x)

        flow = self.flow(x)
        moved = self.transformer(moving, flow)
        return moved, flow


class SynthMorph(nn.Module):
    """
    SynthMorph for registration with synthetic data training.

    Reference: Hoffmann et al. "Anatomy-Aware and Acquisition-Agnostic Image Registration"
    """

    def __init__(
        self,
        in_channels: int = 1,
        vol_size: Tuple[int, int, int] = (160, 192, 224),
        enc_features: List[int] = [32, 64, 128, 256],
        dec_features: List[int] = [256, 128, 64, 32],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.vol_size = vol_size

        self.unet = UNet3D(in_channels * 2, 3, enc_features + [512])
        self.transformer = SpatialTransformer(vol_size)

    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([moving, fixed], dim=1)
        flow = self.unet(x)
        moved = self.transformer(moving, flow)
        return moved, flow


# =============================================================================
# RECONSTRUCTION MODELS
# =============================================================================


class DeepDenoising(nn.Module):
    """
    Deep denoising autoencoder for medical images.

    Removes noise from medical images (CT, MRI, etc.)
    """

    def __init__(self, in_channels: int = 1, features: List[int] = [64, 128, 256, 512]):
        super().__init__()

        self.encoder = nn.ModuleList()
        prev_features = in_channels
        for num_features in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(prev_features, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            prev_features = num_features

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1]),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.ModuleList()
        for i in range(len(features) - 1, -1, -1):
            num_features = features[i]
            in_feat = num_features if i == len(features) - 1 else features[i + 1]
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_feat, num_features, kernel_size=2, stride=2),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                )
            )

        self.final = nn.Conv2d(features[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i, dec in enumerate(self.decoder):
            x = dec[0](x)
            x = torch.cat([x, skip_connections[i]], dim=1)
            for layer in dec[1:]:
                x = layer(x)

        return self.final(x)


class ResidualBlockSR(nn.Module):
    """Residual block for super-resolution."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out


class SuperResolutionMedical(nn.Module):
    """
    Super-resolution network for medical images.

    Reference: Ledig et al. "Photo-Realistic Single Image Super-Resolution Using a GAN"
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 16,
        scale_factor: int = 4,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv_input = nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4)

        self.residual_blocks = nn.ModuleList(
            [ResidualBlockSR(num_features) for _ in range(num_blocks)]
        )

        self.conv_mid = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_features)

        upsample_layers = []
        for _ in range(int(np.log2(scale_factor))):
            upsample_layers.extend(
                [
                    nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                ]
            )
        self.upsample = nn.Sequential(*upsample_layers)

        self.conv_output = nn.Conv2d(
            num_features, out_channels, kernel_size=9, padding=4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_input(x))
        residual = out

        for block in self.residual_blocks:
            out = block(out)

        out = self.bn_mid(self.conv_mid(out))
        out += residual

        out = self.upsample(out)
        out = self.conv_output(out)
        return out


class LimitedAngleCT(nn.Module):
    """
    Limited angle CT reconstruction network.

    Reconstructs full CT from limited angle projections.
    Reference: Jin et al. "Deep Convolutional Neural Network for Inverse Problems in Imaging"
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_angles: int = 90,
        img_size: int = 256,
        features: List[int] = [64, 128, 256, 512],
    ):
        super().__init__()
        self.num_angles = num_angles
        self.img_size = img_size

        self.sinogram_encoder = nn.Sequential(
            nn.Linear(num_angles * img_size, features[0] * 4),
            nn.ReLU(inplace=True),
            nn.Linear(features[0] * 4, features[0] * 16),
            nn.ReLU(inplace=True),
        )

        self.backbone = UNet2D(in_channels, in_channels, features)

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        B = sinogram.size(0)

        x = sinogram.view(B, -1)
        x = self.sinogram_encoder(x)
        x = x.view(B, -1, 4, 4)
        x = F.interpolate(
            x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True
        )
        x = x[:, :1, :, :]

        x = self.backbone(x)
        return x


# =============================================================================
# ANOMALY DETECTION
# =============================================================================


class MedicalAnomalyDetector(nn.Module):
    """
    Unsupervised anomaly detection for medical images.

    Uses reconstruction-based approach with autoencoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        features: List[int] = [32, 64, 128, 256],
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.ModuleList()
        prev_features = in_channels
        for num_features in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        prev_features, num_features, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(num_features),
                    nn.LeakyReLU(0.2),
                )
            )
            prev_features = num_features

        self.fc_mu = nn.Linear(features[-1] * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(features[-1] * 4 * 4, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, features[-1] * 4 * 4)

        self.decoder = nn.ModuleList()
        for i in range(len(features) - 1, -1, -1):
            num_features = features[i]
            in_feat = num_features if i == len(features) - 1 else features[i + 1]
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_feat, num_features, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(num_features),
                    nn.LeakyReLU(0.2),
                )
            )

        self.final = nn.Conv2d(features[0], in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for enc in self.encoder:
            x = enc(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decode(z)
        x = x.view(x.size(0), -1, 4, 4)

        for dec in self.decoder:
            x = dec(x)

        return torch.sigmoid(self.final(x))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score as reconstruction error."""
        recon, _, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2, 3])


class VAEAnomalyMedical(nn.Module):
    """
    VAE-based anomaly detection for medical images.

    Uses ELBO for anomaly scoring.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        img_size: int = 128,
        features: List[int] = [32, 64, 128, 256],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        self.encoder = nn.Sequential()
        prev_features = in_channels
        for num_features in features:
            self.encoder.append(nn.Conv2d(prev_features, num_features, 4, 2, 1))
            self.encoder.append(nn.BatchNorm2d(num_features))
            self.encoder.append(nn.LeakyReLU(0.2))
            prev_features = num_features

        self.fc_mu = nn.Linear(features[-1] * (img_size // 16) ** 2, latent_dim)
        self.fc_logvar = nn.Linear(features[-1] * (img_size // 16) ** 2, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, features[-1] * (img_size // 16) ** 2)

        self.decoder = nn.Sequential()
        for i in range(len(features) - 1, -1, -1):
            num_features = features[i]
            in_feat = num_features if i == len(features) - 1 else features[i + 1]
            self.decoder.append(nn.ConvTranspose2d(in_feat, num_features, 4, 2, 1))
            self.decoder.append(nn.BatchNorm2d(num_features))
            self.decoder.append(nn.ReLU(inplace=True))

        self.decoder.append(nn.ConvTranspose2d(features[0], in_channels, 4, 2, 1))
        self.decoder.append(nn.Sigmoid())

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decode(z)
        x = x.view(x.size(0), -1, self.img_size // 16, self.img_size // 16)
        return self.decoder(x)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        BCE = F.binary_cross_entropy(recon, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score using ELBO."""
        recon, mu, logvar = self.forward(x)
        recon_loss = F.binary_cross_entropy(recon, x, reduction="none").sum(
            dim=[1, 2, 3]
        )
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return recon_loss + kl_loss


class ContrastiveAnomalyMedical(nn.Module):
    """
    Contrastive learning-based anomaly detection.

    Uses contrastive loss to learn normal representations.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        features: List[int] = [32, 64, 128, 256],
        temperature: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature

        self.encoder = nn.ModuleList()
        prev_features = in_channels
        for num_features in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        prev_features, num_features, kernel_size=3, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                )
            )
            prev_features = num_features

        self.projector = nn.Sequential(
            nn.Linear(features[-1] * 4 * 4, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, features[-1] * 4 * 4),
            nn.Unflatten(1, (features[-1], 4, 4)),
            nn.ConvTranspose2d(features[-1], features[-2], 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features[-2], features[-3], 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features[-3], features[-4], 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features[-4], in_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for enc in self.encoder:
            x = enc(x)
        x = x.view(x.size(0), -1)
        return self.projector(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decoder(z)
        return recon, z

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss for contrastive learning."""
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)

        similarity_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        mask = torch.eye(2 * batch_size, device=z.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        pos_sim = torch.cat(
            [
                torch.diag(similarity_matrix, batch_size),
                torch.diag(similarity_matrix, -batch_size),
            ]
        )

        pos_sim = pos_sim / self.temperature
        neg_sim = similarity_matrix / self.temperature

        loss = -pos_sim + torch.logsumexp(neg_sim, dim=1)
        return loss.mean()

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score as reconstruction error."""
        recon, _ = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2, 3])


# =============================================================================
# DATASETS
# =============================================================================


class MedicalImageDataset(Dataset):
    """
    General medical image dataset.

    Supports NIfTI, DICOM, and standard image formats.
    """

    def __init__(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        load_fn: Optional[Callable] = None,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.target_transform = target_transform
        self.load_fn = load_fn or self._default_load

    def _default_load(self, path: str) -> np.ndarray:
        """Default loading function."""
        try:
            import nibabel as nib

            return nib.load(path).get_fdata()
        except:
            from PIL import Image

            return np.array(Image.open(path))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        image = self.load_fn(self.image_paths[idx])

        if self.transform:
            image = self.transform(image)

        if self.label_paths is not None:
            label = self.load_fn(self.label_paths[idx])
            if self.target_transform:
                label = self.target_transform(label)
            return image, label

        return image


def load_chexpert(
    root: str, split: str = "train", transform: Optional[Callable] = None
) -> MedicalImageDataset:
    """
    Load CheXpert dataset for chest X-ray classification.

    Args:
        root: Path to CheXpert dataset
        split: 'train', 'valid', or 'test'
        transform: Optional transform function

    Returns:
        MedicalImageDataset instance
    """
    import os
    import pandas as pd

    csv_path = os.path.join(root, f"{split}.csv")
    df = pd.read_csv(csv_path)

    image_paths = [os.path.join(root, path) for path in df["Path"].values]

    label_cols = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
        "Enlarged Cardiomediastinum",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pneumonia",
        "Pneumothorax",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    def load_label(idx: int) -> np.ndarray:
        labels = df.iloc[idx][label_cols].fillna(0).replace(-1, 0).values
        return labels.astype(np.float32)

    dataset = MedicalImageDataset(
        image_paths=image_paths,
        label_paths=[i for i in range(len(image_paths))],
        transform=transform,
        load_fn=lambda _: None,
    )
    dataset.df = df
    dataset.label_cols = label_cols

    return dataset


def load_brats(
    root: str,
    split: str = "train",
    modalities: List[str] = ["t1", "t1ce", "t2", "flair"],
    transform: Optional[Callable] = None,
) -> MedicalImageDataset:
    """
    Load BraTS dataset for brain tumor segmentation.

    Args:
        root: Path to BraTS dataset
        split: 'train' or 'test'
        modalities: List of modalities to load
        transform: Optional transform function

    Returns:
        MedicalImageDataset instance
    """
    import os
    import glob

    patient_dirs = sorted(glob.glob(os.path.join(root, split, "BraTS*")))

    image_paths = []
    label_paths = []

    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)

        mod_paths = []
        for mod in modalities:
            mod_file = glob.glob(os.path.join(patient_dir, f"*{mod}.nii.gz"))
            if mod_file:
                mod_paths.append(mod_file[0])

        if len(mod_paths) == len(modalities):
            image_paths.append(tuple(mod_paths))
            seg_file = glob.glob(os.path.join(patient_dir, "*seg.nii.gz"))
            if seg_file:
                label_paths.append(seg_file[0])

    def load_multimodal(paths: Tuple[str, ...]) -> np.ndarray:
        import nibabel as nib

        volumes = [nib.load(p).get_fdata() for p in paths]
        return np.stack(volumes, axis=0)

    dataset = MedicalImageDataset(
        image_paths=image_paths,
        label_paths=label_paths if len(label_paths) == len(image_paths) else None,
        transform=transform,
        load_fn=load_multimodal,
    )

    return dataset


def load_luna16(
    root: str, split: str = "train", transform: Optional[Callable] = None
) -> MedicalImageDataset:
    """
    Load LUNA16 dataset for lung nodule detection.

    Args:
        root: Path to LUNA16 dataset
        split: 'train' or 'test'
        transform: Optional transform function

    Returns:
        MedicalImageDataset instance
    """
    import os
    import glob
    import csv

    subset_dirs = sorted(glob.glob(os.path.join(root, "subset*")))

    image_paths = []
    annotations = []

    csv_path = os.path.join(root, "annotations.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations.append(
                    {
                        "seriesuid": row["seriesuid"],
                        "coordX": float(row["coordX"]),
                        "coordY": float(row["coordY"]),
                        "coordZ": float(row["coordZ"]),
                        "diameter_mm": float(row["diameter_mm"]),
                    }
                )

    for subset_dir in subset_dirs:
        mhd_files = glob.glob(os.path.join(subset_dir, "*.mhd"))
        image_paths.extend(mhd_files)

    def load_mhd(path: str) -> np.ndarray:
        import SimpleITK as sitk

        itk_img = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(itk_img)
        return img_array

    dataset = MedicalImageDataset(
        image_paths=image_paths, transform=transform, load_fn=load_mhd
    )
    dataset.annotations = annotations

    return dataset


# =============================================================================
# EVALUATION METRICS
# =============================================================================


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
    ignore_background: bool = False,
) -> torch.Tensor:
    """
    Compute Dice score for segmentation.

    Args:
        pred: Predicted segmentation (B, C, H, W) or (B, C, D, H, W)
        target: Ground truth segmentation
        smooth: Smoothing factor
        ignore_background: Whether to ignore background class

    Returns:
        Dice score per class
    """
    pred = pred.argmax(dim=1) if pred.dim() > target.dim() else pred

    if pred.dim() == 3:
        num_classes = int(target.max().item()) + 1
        pred = F.one_hot(pred.long(), num_classes).permute(0, 3, 1, 2).float()
        target = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    elif pred.dim() == 4:
        num_classes = int(target.max().item()) + 1
        pred = F.one_hot(pred.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        target = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()

    intersection = (pred * target).sum(dim=list(range(2, pred.dim())))
    union = pred.sum(dim=list(range(2, pred.dim()))) + target.sum(
        dim=list(range(2, pred.dim()))
    )

    dice = (2.0 * intersection + smooth) / (union + smooth)

    if ignore_background:
        dice = dice[:, 1:]

    return dice.mean(dim=0)


def hausdorff_distance(
    pred: torch.Tensor, target: torch.Tensor, percentile: float = 95.0
) -> torch.Tensor:
    """
    Compute Hausdorff distance for segmentation.

    Args:
        pred: Predicted segmentation (B, H, W) or (B, D, H, W)
        target: Ground truth segmentation
        percentile: Percentile for robust Hausdorff distance

    Returns:
        Hausdorff distance
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)

    distances = []

    for p, t in zip(pred, target):
        if p.dim() == 2:
            p_coords = torch.nonzero(p, as_tuple=False).cpu().numpy()
            t_coords = torch.nonzero(t, as_tuple=False).cpu().numpy()

            if len(p_coords) == 0 or len(t_coords) == 0:
                distances.append(float("inf"))
                continue

            d1 = directed_hausdorff(p_coords, t_coords)[0]
            d2 = directed_hausdorff(t_coords, p_coords)[0]
            distances.append(max(d1, d2))
        else:
            slice_distances = []
            for i in range(p.shape[0]):
                p_slice = p[i]
                t_slice = t[i]

                p_coords = torch.nonzero(p_slice, as_tuple=False).cpu().numpy()
                t_coords = torch.nonzero(t_slice, as_tuple=False).cpu().numpy()

                if len(p_coords) > 0 and len(t_coords) > 0:
                    d1 = directed_hausdorff(p_coords, t_coords)[0]
                    d2 = directed_hausdorff(t_coords, p_coords)[0]
                    slice_distances.append(max(d1, d2))

            if slice_distances:
                distances.append(np.percentile(slice_distances, percentile))
            else:
                distances.append(float("inf"))

    return torch.tensor(distances, dtype=torch.float32)


def sensitivity_specificity(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    ignore_background: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sensitivity (recall) and specificity for segmentation.

    Args:
        pred: Predicted segmentation (B, C, H, W) or (B, H, W)
        target: Ground truth segmentation
        num_classes: Number of classes
        ignore_background: Whether to ignore background class

    Returns:
        Tuple of (sensitivity, specificity) per class
    """
    if pred.dim() > target.dim():
        pred = pred.argmax(dim=1)

    if num_classes is None:
        num_classes = int(max(pred.max().item(), target.max().item())) + 1

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    sensitivities = []
    specificities = []

    start_class = 1 if ignore_background else 0

    for cls in range(start_class, num_classes):
        pred_cls = (pred_flat == cls).float()
        target_cls = (target_flat == cls).float()

        tp = (pred_cls * target_cls).sum()
        fn = ((1 - pred_cls) * target_cls).sum()
        tn = ((1 - pred_cls) * (1 - target_cls)).sum()
        fp = (pred_cls * (1 - target_cls)).sum()

        sensitivity = tp / (tp + fn + 1e-5)
        specificity = tn / (tn + fp + 1e-5)

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return torch.stack(sensitivities), torch.stack(specificities)


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
    ignore_background: bool = False,
) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) / Jaccard score.

    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        smooth: Smoothing factor
        ignore_background: Whether to ignore background class

    Returns:
        IoU score per class
    """
    pred = pred.argmax(dim=1) if pred.dim() > target.dim() else pred

    if pred.dim() == 3:
        num_classes = int(target.max().item()) + 1
        pred = F.one_hot(pred.long(), num_classes).permute(0, 3, 1, 2).float()
        target = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
    elif pred.dim() == 4:
        num_classes = int(target.max().item()) + 1
        pred = F.one_hot(pred.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        target = F.one_hot(target.long(), num_classes).permute(0, 4, 1, 2, 3).float()

    intersection = (pred * target).sum(dim=list(range(2, pred.dim())))
    union = (
        pred.sum(dim=list(range(2, pred.dim())))
        + target.sum(dim=list(range(2, pred.dim())))
        - intersection
    )

    iou = (intersection + smooth) / (union + smooth)

    if ignore_background:
        iou = iou[:, 1:]

    return iou.mean(dim=0)


def precision_recall_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    ignore_background: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute precision, recall, and F1 score.

    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        num_classes: Number of classes
        ignore_background: Whether to ignore background class

    Returns:
        Tuple of (precision, recall, f1) per class
    """
    if pred.dim() > target.dim():
        pred = pred.argmax(dim=1)

    if num_classes is None:
        num_classes = int(max(pred.max().item(), target.max().item())) + 1

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    precisions = []
    recalls = []
    f1s = []

    start_class = 1 if ignore_background else 0

    for cls in range(start_class, num_classes):
        pred_cls = (pred_flat == cls).float()
        target_cls = (target_flat == cls).float()

        tp = (pred_cls * target_cls).sum()
        fp = (pred_cls * (1 - target_cls)).sum()
        fn = ((1 - pred_cls) * target_cls).sum()

        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-5)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return torch.stack(precisions), torch.stack(recalls), torch.stack(f1s)


def surface_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    """
    Compute average surface distance between segmentation boundaries.

    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        spacing: Voxel spacing (default: 1.0 for all axes)

    Returns:
        Average surface distance
    """
    from scipy import ndimage

    if pred.dim() > target.dim():
        pred = pred.argmax(dim=1)

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    if spacing is None:
        spacing = (1.0,) * pred_np.ndim

    distances = []

    for p, t in zip(pred_np, target_np):
        p_surface = ndimage.binary_dilation(p != p) ^ (p != 0)
        t_surface = ndimage.binary_dilation(t != t) ^ (t != 0)

        if not p_surface.any() or not t_surface.any():
            distances.append(float("inf"))
            continue

        p_coords = np.argwhere(p_surface)
        t_coords = np.argwhere(t_surface)

        p_coords = p_coords * np.array(spacing)
        t_coords = t_coords * np.array(spacing)

        from scipy.spatial.distance import cdist

        dists = cdist(p_coords, t_coords, metric="euclidean")

        assd = (dists.min(axis=1).mean() + dists.min(axis=0).mean()) / 2
        distances.append(assd)

    return torch.tensor(distances, dtype=torch.float32)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0, ignore_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = dice_score(pred, target, self.smooth, self.ignore_background)
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    """Tversky loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)

        target_one_hot = F.one_hot(target, num_classes=pred.size(1))
        target_one_hot = target_one_hot.permute(
            0, -1, *range(1, target_one_hot.dim() - 1)
        ).float()

        dims = list(range(2, pred.dim()))

        tp = (pred * target_one_hot).sum(dim=dims)
        fp = (pred * (1 - target_one_hot)).sum(dim=dims)
        fn = ((1 - pred) * target_one_hot).sum(dim=dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1 - tversky.mean()


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined Dice + Cross-Entropy loss."""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        ce_loss = F.cross_entropy(pred, target)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def normalize_intensity(
    image: torch.Tensor,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> torch.Tensor:
    """
    Normalize image intensity.

    Args:
        image: Input image
        mean: Mean for z-score normalization
        std: Std for z-score normalization
        min_val: Min for min-max normalization
        max_val: Max for min-max normalization

    Returns:
        Normalized image
    """
    if mean is not None and std is not None:
        return (image - mean) / std
    elif min_val is not None and max_val is not None:
        return (image - min_val) / (max_val - min_val + 1e-8)
    else:
        return (image - image.mean()) / (image.std() + 1e-8)


def resample_image(
    image: torch.Tensor, target_size: Tuple[int, ...], mode: str = "trilinear"
) -> torch.Tensor:
    """
    Resample image to target size.

    Args:
        image: Input image (C, D, H, W) or (C, H, W)
        target_size: Target size
        mode: Interpolation mode

    Returns:
        Resampled image
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        output = F.interpolate(image, size=target_size, mode=mode, align_corners=True)
        return output.squeeze(0)
    else:
        return F.interpolate(image, size=target_size, mode=mode, align_corners=True)


def augment_medical(
    image: torch.Tensor,
    rotation_range: float = 10.0,
    scale_range: float = 0.1,
    flip_prob: float = 0.5,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    Apply medical image augmentation.

    Args:
        image: Input image
        rotation_range: Rotation range in degrees
        scale_range: Scale range
        flip_prob: Horizontal flip probability
        noise_std: Gaussian noise standard deviation

    Returns:
        Augmented image
    """
    B, C, H, W = image.shape

    if torch.rand(1) < flip_prob:
        image = torch.flip(image, dims=[-1])

    angle = torch.rand(1) * 2 * rotation_range - rotation_range
    theta = (
        torch.tensor(
            [
                [torch.cos(torch.deg2rad(angle)), -torch.sin(torch.deg2rad(angle)), 0],
                [torch.sin(torch.deg2rad(angle)), torch.cos(torch.deg2rad(angle)), 0],
            ],
            dtype=image.dtype,
            device=image.device,
        )
        .unsqueeze(0)
        .repeat(B, 1, 1)
    )

    grid = F.affine_grid(theta, image.size(), align_corners=True)
    image = F.grid_sample(image, grid, align_corners=True)

    scale = 1 + torch.rand(1) * 2 * scale_range - scale_range
    image = F.interpolate(
        image, scale_factor=float(scale), mode="bilinear", align_corners=True
    )
    image = F.interpolate(image, size=(H, W), mode="bilinear", align_corners=True)

    if noise_std > 0:
        noise = torch.randn_like(image) * noise_std
        image = image + noise

    return image


__all__ = [
    # Segmentation
    "UNet2D",
    "UNet3D",
    "VNet",
    "AttentionUNet",
    "ResidualUNet",
    # Classification
    "CheXNet",
    "ResNetMedical",
    "resnet18_medical",
    "resnet34_medical",
    "resnet50_medical",
    "DenseNetMedical",
    "VisionTransformerMedical",
    # Registration
    "VoxelMorph",
    "DeeperReg",
    "SynthMorph",
    "SpatialTransformer",
    # Reconstruction
    "DeepDenoising",
    "SuperResolutionMedical",
    "LimitedAngleCT",
    # Anomaly Detection
    "MedicalAnomalyDetector",
    "VAEAnomalyMedical",
    "ContrastiveAnomalyMedical",
    # Datasets
    "MedicalImageDataset",
    "load_chexpert",
    "load_brats",
    "load_luna16",
    # Evaluation
    "dice_score",
    "hausdorff_distance",
    "sensitivity_specificity",
    "iou_score",
    "precision_recall_f1",
    "surface_distance",
    # Losses
    "DiceLoss",
    "TverskyLoss",
    "FocalLoss",
    "CombinedLoss",
    # Utilities
    "normalize_intensity",
    "resample_image",
    "augment_medical",
]

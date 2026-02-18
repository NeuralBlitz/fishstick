"""
Encoder networks for few-shot learning.

Provides CNN and ResNet-based encoders optimized for few-shot learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CNNEncoder(nn.Module):
    """Simple CNN encoder for few-shot learning.

    Args:
        in_channels: Number of input channels
        hidden_dim: Hidden dimension size
        out_dim: Output embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 512,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNetEncoder(nn.Module):
    """ResNet encoder adapted for few-shot learning.

    Args:
        depth: ResNet depth (18, 34, 50)
        in_channels: Number of input channels
        out_dim: Output embedding dimension
        pretrained: Use pretrained weights
    """

    def __init__(
        self,
        depth: int = 18,
        in_channels: int = 3,
        out_dim: int = 512,
        pretrained: bool = False,
    ):
        super().__init__()

        from torchvision.models import (
            resnet18,
            resnet34,
            resnet50,
            ResNet18_Weights,
            ResNet34_Weights,
            ResNet50_Weights,
        )

        if depth == 18:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = resnet18(weights=weights)
        elif depth == 34:
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = resnet34(weights=weights)
        elif depth == 50:
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.resnet = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.fc = nn.Linear(in_features, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.fc(x)
        return x


class FourLayerCNN(nn.Module):
    """Four-layer CNN encoder commonly used in few-shot learning.

    Args:
        in_channels: Number of input channels
        out_dim: Output embedding dimension
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 1600,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def get_encoder(
    name: str = "conv4",
    in_channels: int = 3,
    out_dim: int = 1600,
) -> nn.Module:
    """Get an encoder by name.

    Args:
        name: Encoder name ('conv4', 'conv6', 'resnet18', 'resnet34')
        in_channels: Number of input channels
        out_dim: Output dimension

    Returns:
        Encoder module
    """
    if name == "conv4":
        return FourLayerCNN(in_channels=in_channels, out_dim=out_dim)
    elif name == "conv6":
        return SixLayerCNN(in_channels=in_channels, out_dim=out_dim)
    elif name == "resnet18":
        return ResNetEncoder(depth=18, in_channels=in_channels, out_dim=out_dim)
    elif name == "resnet34":
        return ResNetEncoder(depth=34, in_channels=in_channels, out_dim=out_dim)
    elif name == "cnn":
        return CNNEncoder(in_channels=in_channels, out_dim=out_dim)
    else:
        raise ValueError(f"Unknown encoder: {name}")


class SixLayerCNN(nn.Module):
    """Six-layer CNN encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 1600,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

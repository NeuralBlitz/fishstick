"""
Medical Image Classification Backbones

3D ResNet, DenseNet, and other architectures adapted for medical imaging.
"""

from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MedicalBackbone(nn.Module, ABC):
    """Abstract base class for medical image backbones."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        pass


class ResNet3DMedical(MedicalBackbone):
    """3D ResNet backbone adapted for medical imaging.
    
    Example:
        >>> backbone = ResNet3DMedical(in_channels=1, num_classes=5, pretrained=True)
        >>> features = backbone.extract_features(volume)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1000,
        block_depth: int = [3, 4, 6, 3],
        base_channels: int = 64,
        pretrained: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_channels, base_channels, block_depth[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, block_depth[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, block_depth[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, block_depth[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 1,
    ) -> nn.Sequential:
        layers = []
        
        layers.append(ResBlock3D(in_channels, out_channels, stride))
        
        for _ in range(1, depth):
            layers.append(ResBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        return x.flatten(1)

    def get_feature_dim(self) -> int:
        return self.fc.in_features


class ResBlock3D(nn.Module):
    """3D Residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class DenseNet3D(MedicalBackbone):
    """3D DenseNet for medical imaging.
    
    Dense connections for better feature reuse.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1000,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 64,
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, depth in enumerate(block_config):
            block = DenseBlock3D(depth, num_features, growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + depth * growth_rate
            
            if i != len(block_config) - 1:
                trans = Transition3D(num_features, num_features // 2)
                self.transitions.append(trans)
                num_features = num_features // 2
        
        self.bn_final = nn.BatchNorm3d(num_features)
        self.relu = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        x = self.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        return x

    def extract_features(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def get_feature_dim(self) -> int:
        return self.fc.in_features


class DenseBlock3D(nn.Module):
    """3D Dense block."""

    def __init__(
        self,
        depth: int,
        in_channels: int,
        growth_rate: int,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(depth):
            self.layers.append(
                DenseLayer3D(in_channels + i * growth_rate, growth_rate)
            )

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        
        return torch.cat(features, 1)


class DenseLayer3D(nn.Module):
    """3D Dense layer."""

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, 4 * growth_rate, 1, bias=False)
        
        self.bn2 = nn.BatchNorm3d(4 * growth_rate)
        self.conv2 = nn.Conv3d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        
        return out


class Transition3D(nn.Module):
    """3D Transition layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool3d(2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        
        return x

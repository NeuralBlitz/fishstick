"""
nnU-Net Implementation for Medical Image Segmentation

Based on the self-configuring nnU-Net approach for automatic
hyperparameter selection and preprocessing.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class nnUNetConfiguration:
    """Configuration for nnU-Net model.

    Attributes:
        patches_size: Network input patch size
        batch_size: Training batch size
        base_num_features: Base number of features
        num_pool: Number of pooling operations
        conv_kernel_sizes: Kernel sizes for each stage
        resampling_sizes: Resampling sizes for each stage
    """

    patches_size: Tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 2
    base_num_features: int = 32
    num_pool: int = 5
    conv_kernel_sizes: List[List[int]] = field(
        default_factory=lambda: [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ]
    )
    resampling_sizes: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    )


class nnUNetEncoder(nn.Module):
    """nnU-Net encoder with dynamic configuration."""

    def __init__(
        self,
        in_channels: int,
        base_num_features: int,
        num_pool: int,
        conv_kernel_sizes: List[List[int]],
    ):
        super().__init__()

        self.num_pool = num_pool
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        features = base_num_features

        for i in range(num_pool):
            kernel_size = conv_kernel_sizes[i]
            padding = [k // 2 for k in kernel_size]

            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels if i == 0 else features // 2,
                        features,
                        kernel_size,
                        padding=padding,
                    ),
                    nn.InstanceNorm3d(features),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv3d(features, features, kernel_size, padding=padding),
                    nn.InstanceNorm3d(features),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )

            if i < num_pool - 1:
                self.downsample.append(
                    nn.Conv3d(features, features, kernel_size=2, stride=2)
                )
                features *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv3d(features, features * 2, 3, padding=1),
            nn.InstanceNorm3d(features * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(features * 2, features * 2, 3, padding=1),
            nn.InstanceNorm3d(features * 2),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        skip_connections = []

        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_connections.append(x)

            if i < len(self.downsample):
                x = self.downsample[i](x)

        x = self.bottleneck(x)

        return x, skip_connections


class nnUNetDecoder(nn.Module):
    """nnU-Net decoder with dynamic configuration."""

    def __init__(
        self,
        base_num_features: int,
        num_pool: int,
        conv_kernel_sizes: List[List[int]],
    ):
        super().__init__()

        self.num_pool = num_pool
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        features = base_num_features * (2 ** (num_pool - 1))

        for i in range(num_pool - 1):
            kernel_size = conv_kernel_sizes[num_pool - 1 - i]
            padding = [k // 2 for k in kernel_size]

            self.upsample.append(
                nn.ConvTranspose3d(features, features // 2, kernel_size=2, stride=2)
            )

            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv3d(features, features // 2, kernel_size, padding=padding),
                    nn.InstanceNorm3d(features // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv3d(
                        features // 2, features // 2, kernel_size, padding=padding
                    ),
                    nn.InstanceNorm3d(features // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )

            features //= 2

    def forward(
        self,
        x: Tensor,
        skip_connections: List[Tensor],
    ) -> Tensor:
        skip_connections = skip_connections[::-1]

        for i, (up, block) in enumerate(zip(self.upsample, self.decoder_blocks)):
            x = up(x)

            skip = skip_connections[i]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x


class nnUNet(nn.Module):
    """nnU-Net: Self-adapting framework for U-Net-based medical image segmentation.

    nnU-Net automatically adapts to the dataset by configuring:
    - Network topology (patch size, number of pooling operations)
    - Preprocessing (normalization, resampling)
    - Training (batch size, learning rate)

    This implementation provides the dynamic architecture component.

    Example:
        >>> config = nnUNetConfiguration(patches_size=(128, 128, 128))
        >>> model = nnUNet(in_channels=1, num_classes=3, config=config)
        >>> input = torch.randn(1, 1, 128, 128, 128)
        >>> output = model(input)
        >>> output.shape
        torch.Size([1, 3, 128, 128, 128])
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        config: Optional[nnUNetConfiguration] = None,
    ):
        super().__init__()

        if config is None:
            config = nnUNetConfiguration()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.config = config

        self.encoder = nnUNetEncoder(
            in_channels=in_channels,
            base_num_features=config.base_num_features,
            num_pool=config.num_pool,
            conv_kernel_sizes=config.conv_kernel_sizes,
        )

        self.decoder = nnUNetDecoder(
            base_num_features=config.base_num_features,
            num_pool=config.num_pool,
            conv_kernel_sizes=config.conv_kernel_sizes,
        )

        self.output = nn.Conv3d(
            config.base_num_features,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        x = self.output(x)
        return x

    @staticmethod
    def plan_and_train(
        dataset_properties: Dict[str, Any],
    ) -> nnUNetConfiguration:
        """Plan nnU-Net configuration based on dataset properties.

        Args:
            dataset_properties: Dictionary with dataset statistics

        Returns:
            Optimized nnUNetConfiguration
        """
        median_shape = dataset_properties.get("median_shape", [128, 128, 128])

        patch_size = [min(128, s) for s in median_shape]

        patch_size[0] = patch_size[0] // 8 * 8
        patch_size[1] = patch_size[1] // 8 * 8
        patch_size[2] = patch_size[2] // 8 * 8

        num_pool = 5
        for i, s in enumerate(patch_size):
            if s < 64:
                num_pool = i + 1
                break

        config = nnUNetConfiguration(
            patches_size=tuple(patch_size),
            num_pool=num_pool,
        )

        return config


class TopKPathAggregation(nn.Module):
    """Top-k path aggregation for nnU-Net."""

    def __init__(self, in_channels: int, k: int = 3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv3d(in_channels * k, in_channels, 1)

    def forward(self, features: List[Tensor]) -> Tensor:
        if len(features) <= self.k:
            return torch.cat(features, dim=1)

        pooled = []
        for i in range(self.k):
            idx = i * len(features) // self.k
            pooled.append(features[idx])

        return torch.cat(pooled, dim=1)


class DeepSupervisionHead(nn.Module):
    """Deep supervision head for intermediate predictions."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        weight_factors: Optional[List[float]] = None,
    ):
        super().__init__()

        if weight_factors is None:
            weight_factors = [1.0, 0.5, 0.25, 0.125, 0.0625]

        self.weight_factors = weight_factors
        self.heads = nn.ModuleList()

        for i, channels in enumerate(in_channels):
            if i < len(weight_factors):
                self.heads.append(nn.Conv3d(channels, num_classes, kernel_size=1))

    def forward(
        self,
        features: List[Tensor],
        target_size: Tuple[int, int, int],
    ) -> List[Tensor]:
        outputs = []

        for i, head in enumerate(self.heads):
            if i < len(features):
                x = head(features[i])
                x = F.interpolate(
                    x, size=target_size, mode="trilinear", align_corners=False
                )
                outputs.append(x)

        return outputs

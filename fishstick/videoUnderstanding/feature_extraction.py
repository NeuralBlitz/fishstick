"""
Video Feature Extraction Module for fishstick

Comprehensive video feature extraction including:
- 3D CNN architectures (I3D, C3D, R3D)
- SlowFast dual-pathway feature extraction
- Video Transformers (TimeSformer, ViViT)
- Spatio-temporal attention mechanisms

Author: Fishstick Team
"""

from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class FeatureExtractionConfig:
    """Configuration for video feature extraction."""

    input_channels: int = 3
    feature_dim: int = 2048
    temporal_stride: int = 8
    spatial_size: int = 7
    dropout: float = 0.5


class I3DFeatureExtractor(nn.Module):
    """
    I3D (Inflated 3D ConvNet) Feature Extractor.

    Inflates 2D image classification models into 3D convolutions for
    spatio-temporal video feature extraction.

    Args:
        in_channels: Number of input channels (3 for RGB)
        base_channels: Base number of channels in first layer
        depths: Number of layers at each stage
        dropout: Dropout probability

    Example:
        >>> extractor = I3DFeatureExtractor(in_channels=3, base_channels=64)
        >>> video = torch.randn(1, 3, 16, 224, 224)
        >>> features = extractor(video)
        >>> print(features.shape)
        torch.Size([1, 1024, 8, 7, 7])
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=(5, 7, 7),
                stride=(2, 2, 2),
                padding=(2, 3, 3),
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        self.decoder1 = self._make_stage(
            base_channels, base_channels * 2, depths[0], stride=(2, 2, 2)
        )
        self.decoder2 = self._make_stage(
            base_channels * 2, base_channels * 4, depths[1], stride=(2, 2, 2)
        )
        self.decoder3 = self._make_stage(
            base_channels * 4, base_channels * 8, depths[2], stride=(2, 2, 2)
        )
        self.decoder4 = self._make_stage(
            base_channels * 8, base_channels * 16, depths[3], stride=(2, 2, 2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)

        self._out_channels = base_channels * 16

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        stride: Tuple[int, int, int] = (1, 2, 2),
    ) -> nn.Sequential:
        """Create a stage with residual 3D convolutions."""
        layers = []

        layers.append(BottleneckBlock(in_channels, out_channels, stride=stride))

        for _ in range(num_layers - 1):
            layers.append(BottleneckBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    @property
    def out_channels(self) -> int:
        """Return output channel dimension."""
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        """
        Extract features from video.

        Args:
            x: Input tensor of shape (B, C, T, H, W)

        Returns:
            Feature tensor of shape (B, D, T', H', W')
        """
        x = self.stem(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)

        return x

    def get_pooled_features(self, x: Tensor) -> Tensor:
        """Get globally pooled features."""
        x = self.forward(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x


class BottleneckBlock(nn.Module):
    """3D Bottleneck residual block for I3D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()

        mid_channels = out_channels // 4

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class C3DFeatureExtractor(nn.Module):
    """
    C3D (Convolutional 3D) Feature Extractor.

    Original 3D CNN architecture for video feature extraction.
    Uses repeated 3x3x3 convolutions with max pooling.

    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels
        dropout: Dropout probability

    Example:
        >>> extractor = C3DFeatureExtractor(in_channels=3)
        >>> video = torch.randn(1, 3, 16, 112, 112)
        >>> features = extractor(video)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(
                base_channels * 4,
                base_channels * 8,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)

        self._out_channels = base_channels * 8

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Extract features from video."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def get_pooled_features(self, x: Tensor) -> Tensor:
        """Get globally pooled features."""
        x = self.forward(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x


class R3DFeatureExtractor(nn.Module):
    """
    R3D (Residual 3D) Feature Extractor.

    3D ResNet architecture for video feature extraction.
    Uses 2D ResNet inflated to 3D with residual connections.

    Args:
        in_channels: Number of input channels
        depth: Depth variant (18, 34, 50, 101)
        dropout: Dropout probability
    """

    RESNET_DEPTHS = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
    }

    def __init__(
        self,
        in_channels: int = 3,
        depth: int = 50,
        dropout: float = 0.5,
    ):
        super().__init__()

        base_channels = 64
        self.depth = depth

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        stages = self.RESNET_DEPTHS[depth]

        self.layer1 = self._make_layer(
            base_channels, base_channels * 4, stages[0], stride=1
        )
        self.layer2 = self._make_layer(
            base_channels * 4, base_channels * 8, stages[1], stride=2
        )
        self.layer3 = self._make_layer(
            base_channels * 8, base_channels * 16, stages[2], stride=2
        )
        self.layer4 = self._make_layer(
            base_channels * 16, base_channels * 32, stages[3], stride=2
        )

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)

        self._out_channels = base_channels * 32

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(Residual3DBlock(in_channels, out_channels, stride=s))
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def get_pooled_features(self, x: Tensor) -> Tensor:
        x = self.forward(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x


class Residual3DBlock(nn.Module):
    """3D Residual block for R3D."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SlowFastFeatureExtractor(nn.Module):
    """
    SlowFast Dual-Pathway Feature Extractor.

    Captures both low frame rate semantic information (slow pathway)
    and high frame rate motion information (fast pathway).

    Args:
        in_channels: Number of input channels
        base_channels: Base channels for slow pathway
        fast_channels: Base channels for fast pathway
        alpha: Frame rate ratio between pathways
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        fast_channels: int = 8,
        alpha: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.alpha = alpha
        self.slow_channels = base_channels
        self.fast_channels = base_channels * fast_channels // 8

        self.slow_stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels * 2,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.fast_stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                self.fast_channels,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.fast_channels),
            nn.ReLU(inplace=True),
        )

        self.slow_path = self._make_slow_path(base_channels * 2)
        self.fast_path = self._make_fast_path(self.fast_channels)

        self.lateral_connection = LateralConnection(
            self.fast_channels * 8, base_channels * 2, alpha
        )

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)

        self._out_channels = base_channels * 2 + self.fast_channels * 8

    def _make_slow_path(self, in_channels: int) -> nn.Sequential:
        stages = [
            (in_channels, 64, 2),
            (64, 128, 2),
            (128, 256, 2),
            (256, 512, 2),
        ]

        layers = []
        for c1, c2, s in stages:
            layers.append(SlowFastStage(c1, c2, s))

        return nn.Sequential(*layers)

    def _make_fast_path(self, in_channels: int) -> nn.Sequential:
        stages = [
            (in_channels, 32, 2),
            (32, 64, 2),
            (64, 128, 2),
            (128, 256, 2),
        ]

        layers = []
        for c1, c2, s in stages:
            layers.append(SlowFastStage(c1, c2, s))

        return nn.Sequential(*layers)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract features from both pathways.

        Returns:
            Tuple of (slow_features, fast_features)
        """
        T = x.shape[2]
        slow_input = x[:, :, :: self.alpha, :, :]
        fast_input = x

        slow_x = self.slow_stem(slow_input)
        fast_x = self.fast_stem(fast_input)

        slow_x = self.slow_path(slow_x)
        fast_x = self.fast_path(fast_x)

        lateral_features = self.lateral_connection(fast_x)

        slow_x = torch.cat([slow_x, lateral_features], dim=1)

        return slow_x, fast_x

    def get_pooled_features(self, x: Tensor) -> Tensor:
        """Get combined pooled features from both pathways."""
        slow_features, fast_features = self.forward(x)

        slow_pooled = self.adaptive_pool(slow_features)
        fast_pooled = self.adaptive_pool(fast_features)

        combined = torch.cat([slow_pooled, fast_pooled], dim=1)
        combined = combined.view(combined.size(0), -1)

        return self.dropout(combined)


class SlowFastStage(nn.Module):
    """SlowFast pathway stage with lateral connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.conv3 = nn.Conv3d(
            out_channels, out_channels * 4, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels * 4),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LateralConnection(nn.Module):
    """Lateral connection from fast to slow pathway."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: int,
    ):
        super().__init__()

        self.conv_lateral = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

        self.pool = nn.MaxPool3d(kernel_size=(alpha, 1, 1), stride=(alpha, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_lateral(x)
        x = self.pool(x)
        return x


class TimeSformerEncoder(nn.Module):
    """
    TimeSformer Video Transformer Encoder.

    Pure transformer architecture for video understanding with
    divided space-time attention.

    Args:
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_frames: int = 8,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches * num_frames + 1, embed_dim)
        )
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                DividedSpaceTimeBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, self.num_patches, self.temporal_pos_embed)

        x = self.norm(x)

        return x[:, 0]

    def get_spatial_features(self, x: Tensor) -> Tensor:
        """Get spatial attention features."""
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block.get_spatial_attention(x, self.num_patches)

        x = self.norm(x)

        return x


class DividedSpaceTimeBlock(nn.Module):
    """Divided Space-Time Attention Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: Tensor, num_patches: int, temporal_pos_embed: Tensor
    ) -> Tensor:
        B, N, C = x.shape

        cls_token = x[:, 0:1]
        patch_tokens = x[:, 1:]

        patch_tokens = patch_tokens.reshape(B, -1, num_patches, C)
        patch_tokens = patch_tokens + temporal_pos_embed[:, :, :C].unsqueeze(2)
        patch_tokens = patch_tokens.reshape(B, -1, C)

        temporal_tokens = patch_tokens.reshape(B, num_patches, -1, C).transpose(1, 2)
        temporal_tokens = temporal_tokens.reshape(B * num_patches, -1, C)

        temporal_out = self.norm1(temporal_tokens)
        temporal_out, _ = self.attn(temporal_out, temporal_out, temporal_out)
        temporal_out = temporal_out + temporal_tokens
        temporal_out = temporal_out.reshape(B, num_patches, -1, C).transpose(1, 2)
        temporal_out = temporal_out.reshape(B, -1, C)

        spatial_out = self.norm1(temporal_out)
        spatial_out, _ = self.attn(spatial_out, spatial_out, spatial_out)
        spatial_out = spatial_out + temporal_out

        out = self.norm2(spatial_out)
        out = self.mlp(out) + spatial_out

        return torch.cat([cls_token, out], dim=1)

    def get_spatial_attention(self, x: Tensor, num_patches: int) -> Tensor:
        """Get spatial-only attention."""
        B, N, C = x.shape

        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + x

        x = self.norm2(x)
        x = self.mlp(x) + x

        return x


class ViViTEncoder(nn.Module):
    """
    ViViT (Video Vision Transformer) Encoder.

    Video transformer with factorized encoder approaches.

    Args:
        embed_dim: Embedding dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_frames: int = 8,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        factorized: bool = True,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.factorized = factorized

        self.patch_embed = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.dropout = nn.Dropout(dropout)

        if factorized:
            self.spatial_blocks = nn.ModuleList(
                [
                    TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                    for _ in range(depth // 2)
                ]
            )
            self.temporal_blocks = nn.ModuleList(
                [
                    TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                    for _ in range(depth // 2)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                    for _ in range(depth)
                ]
            )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)

        x = x.reshape(B, self.num_patches + 1, -1)

        if self.factorized:
            for i, block in enumerate(self.spatial_blocks):
                x = block(x)

            x = x.reshape(B, -1, self.num_patches + 1, x.shape[-1])
            x = x + self.temporal_pos_embed[:, :, : x.shape[-1]].unsqueeze(2)
            x = x.reshape(B, -1, x.shape[-1])

            for block in self.temporal_blocks:
                x = block(x)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.norm(x)

        return x[:, 0]

    def get_multimodal_features(self, x: Tensor) -> Tensor:
        """Get features with both spatial and temporal information."""
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)

        x = x.reshape(B, self.num_patches + 1, -1)

        if self.factorized:
            for block in self.spatial_blocks:
                x = block(x)

            x = x.reshape(B, -1, self.num_patches + 1, x.shape[-1])
            x = x + self.temporal_pos_embed[:, :, : x.shape[-1]].unsqueeze(2)
            x = x.reshape(B, -1, x.shape[-1])

            for block in self.temporal_blocks:
                x = block(x)
        else:
            for block in self.blocks:
                x = block(x)

        x = self.norm(x)

        return x


class TransformerBlock(nn.Module):
    """Standard Transformer Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SpatioTemporalAttention(nn.Module):
    """
    Spatio-Temporal Attention Module.

    Captures both spatial and temporal dependencies in video features.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.spatial_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        num_frames: int,
        num_patches: int,
    ) -> Tensor:
        """
        Apply spatio-temporal attention.

        Args:
            x: Input tensor (B, T*P+1, D)
            num_frames: Number of frames
            num_patches: Number of spatial patches per frame

        Returns:
            Attended features
        """
        B = x.shape[0]

        cls_token = x[:, 0:1]
        patch_tokens = x[:, 1:]

        patch_tokens = patch_tokens.reshape(B, num_frames, num_patches, -1)

        for i in range(num_frames):
            frame_tokens = patch_tokens[:, i]
            frame_tokens = (
                frame_tokens
                + self.spatial_attn(
                    self.norm1(frame_tokens),
                    self.norm1(frame_tokens),
                    self.norm1(frame_tokens),
                )[0]
            )
            patch_tokens[:, i] = frame_tokens

        patch_tokens = patch_tokens.reshape(B, -1, x.shape[-1])

        patch_tokens = patch_tokens.transpose(0, 1)
        patch_tokens = patch_tokens.reshape(num_frames, B * num_patches, -1)

        patch_tokens = (
            patch_tokens
            + self.temporal_attn(
                self.norm2(patch_tokens),
                self.norm2(patch_tokens),
                self.norm2(patch_tokens),
            )[0]
        )

        patch_tokens = patch_tokens.reshape(num_frames, B, num_patches, -1)
        patch_tokens = patch_tokens.permute(1, 0, 2, 3).reshape(B, -1, x.shape[-1])

        patch_tokens = patch_tokens + self.mlp(self.norm3(patch_tokens))

        return torch.cat([cls_token, patch_tokens], dim=1)


def create_i3d(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs,
) -> I3DFeatureExtractor:
    """
    Create I3D feature extractor.

    Args:
        pretrained: Whether to load pretrained weights
        progress: Show download progress
        **kwargs: Additional arguments for I3DFeatureExtractor

    Returns:
        I3D feature extractor model
    """
    model = I3DFeatureExtractor(**kwargs)
    return model


def create_c3d(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs,
) -> C3DFeatureExtractor:
    """
    Create C3D feature extractor.

    Args:
        pretrained: Whether to load pretrained weights
        progress: Show download progress
        **kwargs: Additional arguments for C3DFeatureExtractor

    Returns:
        C3D feature extractor model
    """
    model = C3DFeatureExtractor(**kwargs)
    return model


def create_r3d(
    depth: int = 50,
    pretrained: bool = False,
    progress: bool = True,
    **kwargs,
) -> R3DFeatureExtractor:
    """
    Create R3D (ResNet-3D) feature extractor.

    Args:
        depth: ResNet depth (18, 34, 50, 101)
        pretrained: Whether to load pretrained weights
        progress: Show download progress
        **kwargs: Additional arguments for R3DFeatureExtractor

    Returns:
        R3D feature extractor model
    """
    model = R3DFeatureExtractor(depth=depth, **kwargs)
    return model


def create_slowfast(
    pretrained: bool = False,
    progress: bool = True,
    **kwargs,
) -> SlowFastFeatureExtractor:
    """
    Create SlowFast feature extractor.

    Args:
        pretrained: Whether to load pretrained weights
        progress: Show download progress
        **kwargs: Additional arguments for SlowFastFeatureExtractor

    Returns:
        SlowFast feature extractor model
    """
    model = SlowFastFeatureExtractor(**kwargs)
    return model

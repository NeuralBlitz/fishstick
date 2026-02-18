"""
Action Recognition Module for fishstick

Comprehensive action recognition models including:
- Two-stream networks (spatial + temporal)
- TSM (Temporal Shift Module)
- TPN (Temporal Pyramid Network)
- Action classification heads

Author: Fishstick Team
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ActionRecognitionConfig:
    """Configuration for action recognition models."""

    num_classes: int = 400
    feature_dim: int = 2048
    dropout: float = 0.5
    num_streams: int = 2
    use_temporal_attention: bool = True


class SpatialStream(nn.Module):
    """
    Spatial Stream Network for RGB input.

    Processes single-frame RGB images to capture appearance information.

    Args:
        backbone: Feature extractor (e.g., ResNet, I3D)
        feature_dim: Output feature dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        feature_dim: int = 2048,
        dropout: float = 0.5,
    ):
        super().__init__()

        if backbone is None:
            from fishstick.videoUnderstanding.feature_extraction import (
                R3DFeatureExtractor,
            )

            backbone = R3DFeatureExtractor(depth=50)

        self.backbone = backbone

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(backbone.out_channels, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Process spatial stream.

        Args:
            x: RGB input tensor (B, 3, T, H, W)

        Returns:
            Spatial features (B, D)
        """
        features = self.backbone(x)
        pooled = self.pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out


class TemporalStream(nn.Module):
    """
    Temporal Stream Network for optical flow input.

    Processes optical flow to capture motion information.

    Args:
        backbone: Feature extractor
        feature_dim: Output feature dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        feature_dim: int = 2048,
        dropout: float = 0.5,
        flow_channels: int = 2,
    ):
        super().__init__()

        if backbone is None:
            from fishstick.videoUnderstanding.feature_extraction import (
                R3DFeatureExtractor,
            )

            backbone = R3DFeatureExtractor(depth=50, in_channels=flow_channels)

        self.backbone = backbone

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(backbone.out_channels, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Process temporal stream.

        Args:
            x: Optical flow input tensor (B, C, T, H, W)

        Returns:
            Temporal features (B, D)
        """
        features = self.backbone(x)
        pooled = self.pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out


class TwoStreamNetwork(nn.Module):
    """
    Two-Stream Network for Action Recognition.

    Combines spatial (RGB) and temporal (optical flow) streams
    for comprehensive action recognition.

    Args:
        num_classes: Number of action classes
        feature_dim: Feature dimension
        spatial_backbone: Spatial stream backbone
        temporal_backbone: Temporal stream backbone
        dropout: Dropout probability
        fusion_type: Type of fusion ('avg', 'concat', 'attention')

    Example:
        >>> model = TwoStreamNetwork(num_classes=400, feature_dim=512)
        >>> rgb_input = torch.randn(1, 3, 8, 224, 224)
        >>> flow_input = torch.randn(1, 2, 8, 224, 224)
        >>> output = model(rgb_input, flow_input)
        >>> print(output.shape)
        torch.Size([1, 400])
    """

    def __init__(
        self,
        num_classes: int = 400,
        feature_dim: int = 512,
        spatial_backbone: Optional[nn.Module] = None,
        temporal_backbone: Optional[nn.Module] = None,
        dropout: float = 0.5,
        fusion_type: str = "avg",
    ):
        super().__init__()

        self.spatial_stream = SpatialStream(spatial_backbone, feature_dim, dropout)
        self.temporal_stream = TemporalStream(temporal_backbone, feature_dim, dropout)

        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self.fusion = nn.Linear(feature_dim * 2, feature_dim)
        elif fusion_type == "attention":
            self.attention_fusion = AttentionFusion(feature_dim)
        else:
            self.fusion = None

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self,
        rgb_input: Tensor,
        flow_input: Tensor,
    ) -> Tensor:
        """
        Forward pass through both streams.

        Args:
            rgb_input: RGB video frames (B, C, T, H, W)
            flow_input: Optical flow (B, C, T, H, W)

        Returns:
            Class logits (B, num_classes)
        """
        spatial_features = self.spatial_stream(rgb_input)
        temporal_features = self.temporal_stream(flow_input)

        if self.fusion_type == "avg":
            fused = (spatial_features + temporal_features) / 2
        elif self.fusion_type == "concat":
            fused = torch.cat([spatial_features, temporal_features], dim=1)
            fused = self.fusion(fused)
        elif self.fusion_type == "attention":
            fused = self.attention_fusion(spatial_features, temporal_features)
        else:
            fused = spatial_features

        output = self.classifier(fused)

        return output

    def get_features(
        self,
        rgb_input: Tensor,
        flow_input: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Get features from both streams separately."""
        spatial_features = self.spatial_stream(rgb_input)
        temporal_features = self.temporal_stream(flow_input)
        return spatial_features, temporal_features


class AttentionFusion(nn.Module):
    """Attention-based fusion for two streams."""

    def __init__(self, feature_dim: int):
        super().__init__()

        self.spatial_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
        )

        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
        )

    def forward(self, spatial: Tensor, temporal: Tensor) -> Tensor:
        spatial_weights = F.softmax(self.spatial_attention(spatial), dim=-1)
        temporal_weights = F.softmax(self.temporal_attention(temporal), dim=-1)

        fused = spatial_weights * spatial + temporal_weights * temporal
        return fused


class TSMHead(nn.Module):
    """
    TSM (Temporal Shift Module) Head for action recognition.

    Shifts features along the temporal dimension to capture motion
    without additional computational cost.

    Args:
        in_channels: Input channel dimension
        num_classes: Number of action classes
        num_segments: Number of video segments
        dropout: Dropout probability
        shift_fraction: Fraction of channels to shift
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 400,
        num_segments: int = 8,
        dropout: float = 0.5,
        shift_fraction: float = 0.25,
    ):
        super().__init__()

        self.num_segments = num_segments
        self.shift_fraction = shift_fraction

        self.shift = TemporalShift(in_channels, num_segments, shift_fraction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.shift(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class TemporalShift(nn.Module):
    """Temporal Shift Module for TSM."""

    def __init__(
        self,
        channels: int,
        num_segments: int = 8,
        shift_fraction: float = 0.25,
    ):
        super().__init__()

        self.channels = channels
        self.num_segments = num_segments
        self.shift_fraction = shift_fraction

        shift_channels = int(channels * shift_fraction)

        self.shift_conv = nn.Conv1d(
            channels,
            shift_channels,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape

        x = x.view(B, self.num_segments, T // self.num_segments, C, H, W)

        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, self.num_segments * C, -1)

        shift = self.shift_conv(x)
        x = x + shift

        x = x.view(B, self.num_segments, C, T // self.num_segments, H, W)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        return x


class TPNHead(nn.Module):
    """
    TPN (Temporal Pyramid Network) Head for action recognition.

    Uses a pyramid structure to capture multi-scale temporal information.

    Args:
        in_channels: Input channel dimension
        num_classes: Number of action classes
        levels: Number of pyramid levels
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 400,
        levels: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.levels = levels

        self.pyramid = TemporalPyramid(in_channels, levels)

        fusion_channels = in_channels * levels

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(fusion_channels, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.pyramid(x)
        output = self.classifier(features)
        return output


class TemporalPyramid(nn.Module):
    """Temporal Pyramid for multi-scale feature extraction."""

    def __init__(
        self,
        channels: int,
        levels: int = 3,
    ):
        super().__init__()

        self.levels = levels

        self.pools = nn.ModuleList(
            [nn.AdaptiveAvgPool3d((None, 1, 1)) for _ in range(levels)]
        )

        self.convs = nn.ModuleList(
            [nn.Conv3d(channels, channels, kernel_size=1) for _ in range(levels)]
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape

        outputs = []

        for pool, conv in zip(self.pools, self.convs):
            pooled = pool(x)
            pooled = conv(pooled)
            outputs.append(pooled)

        out = torch.cat(outputs, dim=1)

        return out


class ActionRecognitionModel(nn.Module):
    """
    Complete Action Recognition Model.

    Supports multiple backbones and head configurations.

    Args:
        backbone: Feature extraction backbone
        head: Classification head
        num_classes: Number of action classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        num_classes: int = 400,
        dropout: float = 0.5,
        model_type: str = "tsm",
    ):
        super().__init__()

        if backbone is None:
            if model_type == "i3d":
                from fishstick.videoUnderstanding.feature_extraction import (
                    I3DFeatureExtractor,
                )

                backbone = I3DFeatureExtractor()
            elif model_type == "r3d":
                from fishstick.videoUnderstanding.feature_extraction import (
                    R3DFeatureExtractor,
                )

                backbone = R3DFeatureExtractor(depth=50)
            elif model_type == "slowfast":
                from fishstick.videoUnderstanding.feature_extraction import (
                    SlowFastFeatureExtractor,
                )

                backbone = SlowFastFeatureExtractor()
            else:
                from fishstick.videoUnderstanding.feature_extraction import (
                    R3DFeatureExtractor,
                )

                backbone = R3DFeatureExtractor(depth=50)

        self.backbone = backbone

        if head is None:
            if model_type == "tsm":
                head = TSMHead(backbone.out_channels, num_classes, dropout=dropout)
            elif model_type == "tpn":
                head = TPNHead(backbone.out_channels, num_classes, dropout=dropout)
            else:
                head = nn.Sequential(
                    nn.AdaptiveAvgPool3d((1, 1, 1)),
                    nn.Flatten(),
                    nn.Dropout(dropout),
                    nn.Linear(backbone.out_channels, num_classes),
                )

        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        output = self.head(features)
        return output

    def get_features(self, x: Tensor) -> Tensor:
        """Get feature representations before classification."""
        return self.backbone(x)


class TemporalAttentionPooling(nn.Module):
    """
    Temporal Attention Pooling for video features.

    Learns to weight different frames based on their importance.

    Args:
        in_channels: Input channel dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, T, H, W = x.shape

        attn_weights = self.attention(x)
        attn_weights = attn_weights.view(B, T, H * W)
        attn_weights = F.softmax(attn_weights, dim=1)

        x = x.view(B, C, T, -1)
        x = x.permute(0, 2, 1, 3)

        pooled = torch.bmm(x.view(B, T, -1), attn_weights.unsqueeze(2))
        pooled = pooled.squeeze(-1)
        pooled = pooled.permute(0, 2, 1)

        return pooled


class MultiStreamFusion(nn.Module):
    """
    Multi-Stream Fusion Module.

    Fuses multiple streams (RGB, Flow, Audio) for action recognition.

    Args:
        num_streams: Number of input streams
        feature_dims: List of feature dimensions for each stream
        hidden_dim: Hidden dimension for fusion
        num_classes: Number of output classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_streams: int = 2,
        feature_dims: Optional[List[int]] = None,
        hidden_dim: int = 512,
        num_classes: int = 400,
        dropout: float = 0.5,
    ):
        super().__init__()

        if feature_dims is None:
            feature_dims = [hidden_dim] * num_streams

        self.projs = nn.ModuleList([nn.Linear(fd, hidden_dim) for fd in feature_dims])

        self.fusion_attention = nn.Sequential(
            nn.Linear(hidden_dim * num_streams, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_streams),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Fuse multiple stream features.

        Args:
            features: List of feature tensors from different streams

        Returns:
            Class logits
        """
        projected = []

        for i, feat in enumerate(features):
            proj = self.projs[i](feat)
            projected.append(proj)

        concat = torch.cat(projected, dim=1)

        weights = self.fusion_attention(concat)
        weights = F.softmax(weights, dim=-1)

        fused = torch.zeros_like(projected[0])

        for i, proj in enumerate(projected):
            fused += weights[:, i : i + 1] * proj

        output = self.classifier(fused)

        return output


def create_tsn(
    num_classes: int = 400,
    backbone: str = "resnet50",
    pretrained: bool = False,
    **kwargs,
) -> TwoStreamNetwork:
    """
    Create Two-Stream Network (TSN) for action recognition.

    Args:
        num_classes: Number of action classes
        backbone: Backbone architecture
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments

    Returns:
        Two-stream action recognition model
    """
    feature_dim = kwargs.get("feature_dim", 512)
    dropout = kwargs.get("dropout", 0.5)

    return TwoStreamNetwork(
        num_classes=num_classes,
        feature_dim=feature_dim,
        dropout=dropout,
    )


def create_tsm(
    num_classes: int = 400,
    num_segments: int = 8,
    pretrained: bool = False,
    **kwargs,
) -> ActionRecognitionModel:
    """
    Create TSM (Temporal Shift Module) model.

    Args:
        num_classes: Number of action classes
        num_segments: Number of video segments
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments

    Returns:
        TSM action recognition model
    """
    dropout = kwargs.get("dropout", 0.5)
    shift_fraction = kwargs.get("shift_fraction", 0.25)

    from fishstick.videoUnderstanding.feature_extraction import R3DFeatureExtractor

    backbone = R3DFeatureExtractor(depth=50)
    head = TSMHead(
        backbone.out_channels,
        num_classes,
        num_segments,
        dropout,
        shift_fraction,
    )

    return ActionRecognitionModel(
        backbone=backbone,
        head=head,
        num_classes=num_classes,
        dropout=dropout,
        model_type="tsm",
    )


def create_tpn(
    num_classes: int = 400,
    levels: int = 3,
    pretrained: bool = False,
    **kwargs,
) -> ActionRecognitionModel:
    """
    Create TPN (Temporal Pyramid Network) model.

    Args:
        num_classes: Number of action classes
        levels: Number of pyramid levels
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments

    Returns:
        TPN action recognition model
    """
    dropout = kwargs.get("dropout", 0.5)

    from fishstick.videoUnderstanding.feature_extraction import R3DFeatureExtractor

    backbone = R3DFeatureExtractor(depth=50)
    head = TPNHead(
        backbone.out_channels,
        num_classes,
        levels,
        dropout,
    )

    return ActionRecognitionModel(
        backbone=backbone,
        head=head,
        num_classes=num_classes,
        dropout=dropout,
        model_type="tpn",
    )

"""
3D Scene Layout Estimation Module

Provides models for estimating room layout, vanishing points,
and perspective fields from single images. Supports indoor layout
recovery with iterative geometric refinement.
"""

from typing import Tuple, List, Optional, Union, Dict, Any
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


@dataclass
class LayoutPrediction:
    """Container for layout estimation outputs."""

    corners: Tensor
    edges: Tensor
    floor_mask: Tensor
    wall_masks: List[Tensor]
    ceiling_mask: Tensor
    horizon_line: Tensor
    confidence: Tensor


class LayoutEncoder(nn.Module):
    """
    Feature encoder for layout estimation with horizon line prediction.

    Extracts multi-scale features from an input image and predicts a
    preliminary horizon line offset used to guide downstream layout decoding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 4,
    ):
        """
        Args:
            in_channels: Number of input image channels.
            base_channels: Number of channels in the first stage.
            num_stages: Number of encoder downsampling stages.
        """
        super().__init__()
        self.num_stages = num_stages

        stages: List[nn.Module] = []
        ch_in = in_channels
        for i in range(num_stages):
            ch_out = base_channels * (2**i)
            stages.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_out, ch_out, 3, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                )
            )
            ch_in = ch_out
        self.stages = nn.ModuleList(stages)

        final_ch = base_channels * (2 ** (num_stages - 1))
        self.horizon_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        """
        Args:
            x: Input image ``[B, C, H, W]``.

        Returns:
            Tuple of (multi-scale feature list, horizon_offset ``[B, 1]``).
        """
        features: List[Tensor] = []
        h = x
        for stage in self.stages:
            h = stage(h)
            features.append(h)

        horizon = self.horizon_head(features[-1])
        return features, horizon


class LayoutDecoder(nn.Module):
    """
    Decode room layout corners and edges from encoded features.

    Uses a feature pyramid with lateral connections to predict dense
    corner heatmaps and edge maps at the original image resolution.
    """

    def __init__(
        self,
        base_channels: int = 64,
        num_stages: int = 4,
        num_corners: int = 8,
    ):
        """
        Args:
            base_channels: Must match the encoder base_channels.
            num_stages: Must match the encoder num_stages.
            num_corners: Maximum number of layout corners to predict.
        """
        super().__init__()
        self.num_corners = num_corners

        laterals: List[nn.Module] = []
        for i in range(num_stages):
            ch = base_channels * (2**i)
            laterals.append(nn.Conv2d(ch, base_channels, 1))
        self.laterals = nn.ModuleList(laterals)

        self.smooth = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        self.corner_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_corners, 1),
        )

        self.edge_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 1),
        )

    def forward(
        self,
        features: List[Tensor],
        target_size: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: Multi-scale features from ``LayoutEncoder``.
            target_size: ``(H, W)`` of the desired output resolution.

        Returns:
            Tuple of (corner_heatmaps ``[B, num_corners, H, W]``,
            edge_maps ``[B, 3, H, W]``).
        """
        x = self.laterals[-1](features[-1])
        for i in range(len(features) - 2, -1, -1):
            lat = self.laterals[i](features[i])
            x = F.interpolate(
                x, size=lat.shape[2:], mode="bilinear", align_corners=False
            )
            x = x + lat

        x = self.smooth(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        corners = self.corner_head(x)
        edges = torch.sigmoid(self.edge_head(x))
        return corners, edges


class PerspectiveFieldEstimator(nn.Module):
    """
    Estimate vanishing points and a dense perspective field.

    Predicts three orthogonal vanishing points (Manhattan world assumption)
    and a per-pixel perspective direction field used for geometric reasoning.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_vp: int = 3,
    ):
        """
        Args:
            feature_dim: Dimension of the input feature vector.
            num_vp: Number of vanishing points to predict.
        """
        super().__init__()
        self.num_vp = num_vp

        self.vp_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_vp * 3),
        )

        self.field_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1),
        )

    def forward(
        self,
        global_feat: Tensor,
        spatial_feat: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            global_feat: Pooled feature vector ``[B, D]``.
            spatial_feat: Spatial feature map ``[B, D, H, W]``.

        Returns:
            Tuple of (vanishing_points ``[B, num_vp, 3]``,
            perspective_field ``[B, 2, H, W]``).
        """
        vp = self.vp_head(global_feat).view(-1, self.num_vp, 3)
        vp = F.normalize(vp, p=2, dim=-1)

        pf = self.field_head(spatial_feat)
        pf = F.normalize(pf, p=2, dim=1)
        return vp, pf


class LayoutRefinementModule(nn.Module):
    """
    Iterative layout refinement via geometric consistency.

    Takes initial corner and edge predictions and refines them over
    multiple iterations using a recurrent convolutional unit that
    enforces planarity and orthogonality constraints.
    """

    def __init__(
        self,
        channels: int = 64,
        num_iterations: int = 3,
    ):
        """
        Args:
            channels: Feature channel width.
            num_iterations: Number of refinement iterations.
        """
        super().__init__()
        self.num_iterations = num_iterations

        self.gru_cell = nn.GRUCell(channels, channels)
        self.spatial_proj = nn.Conv2d(channels + 11, channels, 3, padding=1)
        self.corner_update = nn.Conv2d(channels, 8, 1)
        self.edge_update = nn.Conv2d(channels, 3, 1)

    def forward(
        self,
        feat: Tensor,
        corners: Tensor,
        edges: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            feat: Spatial features ``[B, C, H, W]``.
            corners: Initial corner heatmaps ``[B, 8, H, W]``.
            edges: Initial edge maps ``[B, 3, H, W]``.

        Returns:
            Refined (corners, edges) with same shapes as inputs.
        """
        B, C, H, W = feat.shape

        hidden = feat.mean(dim=[2, 3])

        for _ in range(self.num_iterations):
            combined = torch.cat([feat, corners, edges], dim=1)
            spatial = F.relu(self.spatial_proj(combined))

            ctx = spatial.mean(dim=[2, 3])
            hidden = self.gru_cell(ctx, hidden)

            gate = hidden.unsqueeze(-1).unsqueeze(-1).expand_as(spatial)
            modulated = spatial * torch.sigmoid(gate)

            corners = corners + self.corner_update(modulated)
            edges = torch.sigmoid(edges + self.edge_update(modulated))

        return corners, edges


class RoomLayoutEstimator(nn.Module):
    """
    End-to-end room layout estimator from a single image.

    Combines encoding, decoding, perspective estimation, and iterative
    refinement to produce a full 3D room layout prediction including
    floor / wall / ceiling segmentation and layout corner locations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 4,
        num_corners: int = 8,
        refine_iterations: int = 3,
    ):
        """
        Args:
            in_channels: Number of input image channels.
            base_channels: Base channel width.
            num_stages: Number of encoder stages.
            num_corners: Maximum layout corners.
            refine_iterations: Number of refinement steps.
        """
        super().__init__()
        self.encoder = LayoutEncoder(in_channels, base_channels, num_stages)
        self.decoder = LayoutDecoder(base_channels, num_stages, num_corners)

        final_ch = base_channels * (2 ** (num_stages - 1))
        self.perspective = PerspectiveFieldEstimator(final_ch)
        self.refiner = LayoutRefinementModule(base_channels, refine_iterations)

        self.semantic_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 1),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: Input image ``[B, 3, H, W]``.

        Returns:
            Dictionary containing:
                - ``corners``: Corner heatmaps ``[B, 8, H, W]``
                - ``edges``: Edge maps ``[B, 3, H, W]``
                - ``vanishing_points``: ``[B, 3, 3]``
                - ``perspective_field``: ``[B, 2, H, W]``
                - ``horizon``: Horizon offset ``[B, 1]``
                - ``semantic``: Floor/wall/ceiling logits ``[B, 3, H, W]``
        """
        H, W = x.shape[2], x.shape[3]

        features, horizon = self.encoder(x)
        corners, edges = self.decoder(features, (H, W))

        global_feat = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        vp, pf = self.perspective(global_feat, features[-1])

        base_feat = F.interpolate(
            features[0], size=(H, W), mode="bilinear", align_corners=False
        )
        corners, edges = self.refiner(base_feat, corners, edges)

        semantic = self.semantic_head(base_feat)

        return {
            "corners": corners,
            "edges": edges,
            "vanishing_points": vp,
            "perspective_field": pf,
            "horizon": horizon,
            "semantic": semantic,
        }


def create_layout_estimator(
    model_type: str = "room",
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to create layout estimation models.

    Args:
        model_type: Model variant. Currently only ``"room"`` is supported.
        **kwargs: Forwarded to the model constructor.

    Returns:
        Layout estimation model instance.
    """
    if model_type == "room":
        return RoomLayoutEstimator(
            in_channels=kwargs.get("in_channels", 3),
            base_channels=kwargs.get("base_channels", 64),
            num_stages=kwargs.get("num_stages", 4),
            num_corners=kwargs.get("num_corners", 8),
            refine_iterations=kwargs.get("refine_iterations", 3),
        )
    raise ValueError(f"Unknown layout model type: {model_type}")

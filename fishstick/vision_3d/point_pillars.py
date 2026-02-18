"""
PointPillars: Fast Encoders for 3D Object Detection

Implementation of PointPillars architecture:
- Pillar Feature Network
- Backbone (SSD-style)
- Detection Head
"""

from typing import Tuple, Optional, List
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from fishstick.vision_3d.voxel_grid import PointPillarsScatter


class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Network.

    Converts point clouds to pillar features using MLP.
    """

    def __init__(
        self,
        num_features: int = 4,
        pillar_dim: int = 64,
        max_points: int = 100,
        max_pillars: int = 12000,
    ):
        super().__init__()
        self.num_features = num_features
        self.pillar_dim = pillar_dim
        self.max_points = max_points
        self.max_pillars = max_pillars

        self.pillar_mlp = nn.Sequential(
            nn.Linear(num_features, pillar_dim),
            nn.BatchNorm1d(pillar_dim),
            nn.ReLU(),
            nn.Linear(pillar_dim, pillar_dim),
            nn.BatchNorm1d(pillar_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        points: Tensor,
        num_points_per_pillar: Tensor,
        pillar_indices: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            points: Points in pillars [P, max_points, num_features]
            num_points_per_pillar: Points per pillar [P]
            pillar_indices: Pillar indices [P, 4]

        Returns:
            pillar_features: [P, pillar_dim]
            pillar_coords: [P, 4]
        """
        P, K, C = points.shape

        points_mean = points.sum(dim=1, keepdim=True) / num_points_per_pillar.clamp(
            min=1
        ).unsqueeze(-1).unsqueeze(-1)
        points_mean = points_mean.expand(-1, K, -1)

        points_centered = points - points_mean

        points_cat = torch.cat([points, points_centered], dim=-1)

        points_flat = points_cat.view(-1, C * 2)
        pillar_features = self.pillar_mlp(points_flat)
        pillar_features = pillar_features.view(P, K, -1)

        pillar_features = pillar_features.max(dim=1)[0]

        return pillar_features, pillar_indices


class PointPillarsBackbone(nn.Module):
    """
    PointPillars Backbone.

    2D CNN backbone for feature extraction from pseudo-image.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: List[int] = [64, 128, 256],
        layer_strides: List[int] = [2, 2, 2],
        num_upsample_layers: int = 3,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        c_in = in_channels
        for i, (c_out, stride) in enumerate(zip(out_channels, layer_strides)):
            block = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
            )
            self.blocks.append(block)
            c_in = c_out

        for i in range(num_upsample_layers):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        out_channels[-(i + 1)],
                        out_channels[-(i + 1)] // 2,
                        2,
                        stride=2,
                    ),
                    nn.BatchNorm2d(out_channels[-(i + 1)] // 2),
                    nn.ReLU(),
                )
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            x: Pseudo-image [B, C, H, W]

        Returns:
            out: Fused features [B, C', H', W']
            features: Multi-scale features [B, C_i, H_i, W_i]
        """
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)

        outs = []
        for i, up in enumerate(self.up_blocks):
            idx = len(features) - 1 - i
            x = up(x)
            outs.append(x)

        return torch.cat(outs, dim=1), features


class PointPillarsHead(nn.Module):
    """
    PointPillars Detection Head.

    Single Shot Detection (SSD) style head for 3D boxes.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        num_anchors: int = 2,
        box_code_size: int = 7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv_cls = nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        self.conv_box = nn.Conv2d(in_channels, num_anchors * box_code_size, 1)
        self.conv_dir = nn.Conv2d(in_channels, num_anchors * 2, 1)

    def forward(
        self,
        features: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            features: Backbone features [B, C, H, W]

        Returns:
            cls_scores: [B, num_anchors * num_classes, H, W]
            box_preds: [B, num_anchors * box_code_size, H, W]
            dir_preds: [B, num_anchors * 2, H, W]
        """
        cls_scores = self.conv_cls(features)
        box_preds = self.conv_box(features)
        dir_preds = self.conv_dir(features)

        return cls_scores, box_preds, dir_preds


class PointPillars(nn.Module):
    """
    Complete PointPillars model.
    """

    def __init__(
        self,
        num_classes: int = 3,
        num_features: int = 4,
        pillar_dim: int = 64,
        max_points: int = 100,
        max_pillars: int = 12000,
        grid_size: Tuple[int, int] = (432, 496),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            0,
            -39.68,
            -3,
            69.12,
            39.68,
            1,
        ),
    ):
        super().__init__()

        self.pfn = PillarFeatureNet(num_features, pillar_dim, max_points, max_pillars)
        self.scatter = PointPillarsScatter(pillar_dim, grid_size, point_cloud_range)
        self.backbone = PointPillarsBackbone(in_channels=pillar_dim)
        self.head = PointPillarsHead(
            in_channels=256,
            num_classes=num_classes,
            num_anchors=2,
        )

    def forward(
        self,
        points: Tensor,
        num_points_per_pillar: Tensor,
        pillar_indices: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            points: [P, K, num_features]
            num_points_per_pillar: [P]
            pillar_indices: [P, 4]

        Returns:
            cls_scores, box_preds, dir_preds
        """
        pillar_features, pillar_coords = self.pfn(
            points, num_points_per_pillar, pillar_indices
        )

        pseudo_image = self.scatter(pillar_features, pillar_coords)

        features, _ = self.backbone(pseudo_image)

        cls_scores, box_preds, dir_preds = self.head(features)

        return cls_scores, box_preds, dir_preds

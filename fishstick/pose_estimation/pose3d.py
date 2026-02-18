"""
3D Human Pose Estimation Models

Implementation of 3D pose estimation architectures including VideoPose, GCN, and Transformer models.
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TemporalModel(nn.Module):
    """
    Temporal 1D Convolutional model for 3D pose estimation from 2D poses.

    Args:
        in_features: Number of input features (2 * num_joints for 2D coordinates)
        out_features: Number of output features (3 * num_joints for 3D coordinates)
        num_joints: Number of joints
        filter_widths: List of filter widths for temporal conv layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_joints: int,
        filter_widths: List[int] = [3, 3, 3],
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.in_features = in_features
        self.out_features = out_features

        self.layers = nn.ModuleList()
        for filter_width in filter_widths:
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_features,
                        in_features,
                        filter_width,
                        padding=filter_width // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(in_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )

        self.output = nn.Conv1d(in_features, out_features, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor (B, T, in_features)

        Returns:
            Output tensor (B, T, out_features)
        """
        x = x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x) + x

        x = self.output(x)

        return x.transpose(1, 2)


class SemigraphConv(nn.Module):
    """
    Semi-Graph Convolutional layer for pose estimation.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        adj: Adjacency matrix
    """

    def __init__(self, in_features: int, out_features: int, adj: Tensor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        self.b = nn.Parameter(torch.zeros(out_features))

        D = adj.sum(1, keepdim=True)
        D_hat = (D + 1e-5).pow(-0.5)
        A_hat = D_hat * adj * D_hat.transpose(0, 1)
        self.register_buffer("A_hat", A_hat)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input features (B, N, in_features)

        Returns:
            Output features (B, N, out_features)
        """
        support = torch.matmul(x, self.W)
        output = torch.matmul(self.A_hat, support) + self.b
        return output


class GCNPose3D(nn.Module):
    """
    Graph Convolutional Network for 3D Pose Estimation.

    Args:
        num_joints: Number of joints
        in_channels: Input channel dimension
        hidden_channels: Hidden layer dimensions
        out_channels: Output channel dimension
        adj: Adjacency matrix for the skeleton
        num_layers: Number of GCN layers
    """

    def __init__(
        self,
        num_joints: int = 17,
        in_channels: int = 3,
        hidden_channels: int = 256,
        out_channels: int = 3,
        adj: Optional[Tensor] = None,
        num_layers: int = 4,
    ):
        super().__init__()
        self.num_joints = num_joints

        if adj is None:
            adj = self._default_adjacency(num_joints)

        self.register_buffer("adj", adj)

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SemigraphConv(hidden_channels, hidden_channels, adj))

        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def _default_adjacency(self, num_joints: int) -> Tensor:
        """Create default adjacency matrix for human skeleton."""
        adj = torch.eye(num_joints)

        edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

        for i, j in edges:
            if i < num_joints and j < num_joints:
                adj[i, j] = 1
                adj[j, i] = 1

        return adj

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input (B, T, num_joints, 3) or (B, num_joints, 3)

        Returns:
            Output (B, T, num_joints, 3) or (B, num_joints, 3)
        """
        squeeze_time = False
        if x.dim() == 3:
            x = x.unsqueeze(1)
            squeeze_time = True

        B, T, J, C = x.shape
        x = x.reshape(B * T, J, C)

        x = self.input_proj(x)

        for layer in self.layers:
            x = F.relu(layer(x)) + x

        x = self.output_proj(x)

        x = x.reshape(B, T, J, C)

        if squeeze_time:
            x = x.squeeze(1)

        return x


class TransformerPose3D(nn.Module):
    """
    Transformer-based 3D Pose Estimation model.

    Args:
        num_joints: Number of joints
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_joints: int = 17,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_joints = num_joints

        self.input_proj = nn.Linear(3, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_joints, d_model))

        self.output_proj = nn.Linear(d_model, 3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input (B, T, num_joints, 3) or (B, num_joints, 3)

        Returns:
            Output (B, T, num_joints, 3) or (B, num_joints, 3)
        """
        squeeze_time = False
        if x.dim() == 3:
            x = x.unsqueeze(1)
            squeeze_time = True

        B, T, J, C = x.shape
        x = x.reshape(B * T, J, C)

        x = self.input_proj(x)
        x = x + self.pos_embedding

        x = self.transformer(x)

        x = self.output_proj(x)

        x = x.reshape(B, T, J, C)

        if squeeze_time:
            x = x.squeeze(1)

        return x


class VideoPose3D(nn.Module):
    """
    VideoPose3D: 3D human pose estimation from 2D poses.

    Based on "3D human pose estimation in video with temporal convolutions".

    Args:
        num_joints: Number of joints (default: 17 for COCO)
        in_channels: Number of input channels per joint (2 for 2D x,y)
        out_channels: Number of output channels per joint (3 for 3D x,y,z)
        filter_widths: List of temporal convolution filter widths
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_joints: int = 17,
        in_channels: int = 2,
        out_channels: int = 3,
        filter_widths: List[int] = [3, 3, 3, 3],
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_proj = nn.Linear(num_joints * in_channels, 1024)

        self.temporal_model = TemporalModel(
            in_features=1024,
            out_features=1024,
            num_joints=num_joints,
            filter_widths=filter_widths,
            dropout=dropout,
        )

        self.output_proj = nn.Linear(1024, num_joints * out_channels)

    def forward(self, keypoints_2d: Tensor) -> Tensor:
        """
        Args:
            keypoints_2d: 2D keypoints (B, T, num_joints, 2)

        Returns:
            3D keypoints (B, T, num_joints, 3)
        """
        B, T, J, C = keypoints_2d.shape
        assert J == self.num_joints and C == self.in_channels

        x = keypoints_2d.reshape(B, T, J * C)

        x = self.input_proj(x)
        x = self.temporal_model(x)
        x = self.output_proj(x)

        x = x.reshape(B, T, J, self.out_channels)

        return x


class ModelPose3D(nn.Module):
    """
    Model-based 3D pose estimation with SMPL-like parameters.

    Args:
        num_joints: Number of joints
        feature_dim: Feature dimension
    """

    def __init__(self, num_joints: int = 17, feature_dim: int = 256):
        super().__init__()
        self.num_joints = num_joints

        self.feature_extractor = nn.Sequential(
            nn.Linear(num_joints * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

        self.pose_predictor = nn.Linear(feature_dim, num_joints * 3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 2D keypoints (B, T, num_joints, 2)

        Returns:
            3D keypoints (B, T, num_joints, 3)
        """
        B, T, J, _ = x.shape

        x_flat = x.reshape(B, T, J * 2)

        features = self.feature_extractor(x_flat)

        pose = self.pose_predictor(features)

        pose_3d = pose.reshape(B, T, J, 3)

        return pose_3d


class PoseAugmenter3D(nn.Module):
    """
    Data augmentation for 3D poses.

    Args:
        scale_range: Range for random scaling
        rotation_range: Range for random rotation (degrees)
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_range: Tuple[float, float] = (-30, 30),
    ):
        super().__init__()
        self.scale_range = scale_range
        self.rotation_range = rotation_range

    def forward(self, pose: Tensor) -> Tensor:
        """
        Args:
            pose: 3D pose (B, J, 3) or (B, T, J, 3)

        Returns:
            Augmented pose
        """
        squeeze_time = False
        if pose.dim() == 3:
            pose = pose.unsqueeze(1)
            squeeze_time = True

        B, T, J, C = pose.shape

        if self.training:
            if self.scale_range is not None:
                scale = torch.empty(B, device=pose.device).uniform_(*self.scale_range)
                pose = pose * scale.view(B, 1, 1, 1)

            if self.rotation_range is not None:
                angle = torch.empty(B, device=pose.device).uniform_(
                    *self.rotation_range
                )
                angle_rad = angle * 3.14159 / 180

                cos_a = torch.cos(angle_rad)
                sin_a = torch.sin(angle_rad)

                R = torch.zeros(B, 3, 3, device=pose.device)
                R[:, 0, 0] = cos_a
                R[:, 0, 1] = -sin_a
                R[:, 1, 0] = sin_a
                R[:, 1, 1] = cos_a
                R[:, 2, 2] = 1

                pose_reshaped = pose.reshape(B, T * J, 3)
                pose_reshaped = torch.bmm(pose_reshaped, R.transpose(1, 2))
                pose = pose_reshaped.reshape(B, T, J, 3)

        if squeeze_time:
            pose = pose.squeeze(1)

        return pose


class HumanPose3DModel(nn.Module):
    """
    Generic 3D Human Pose Estimation Model.

    Supports multiple backbones.

    Args:
        backbone: Type of backbone ('videopose', 'gcn', 'transformer')
        num_joints: Number of joints
    """

    def __init__(self, backbone: str = "videopose", num_joints: int = 17, **kwargs):
        super().__init__()
        self.num_joints = num_joints
        self.backbone_type = backbone

        if backbone == "videopose":
            self.model = VideoPose3D(num_joints=num_joints, **kwargs)
        elif backbone == "gcn":
            self.model = GCNPose3D(num_joints=num_joints, **kwargs)
        elif backbone == "transformer":
            self.model = TransformerPose3D(num_joints=num_joints, **kwargs)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(x)


def create_videopose_spatial(num_joints: int = 17, num_frames: int = 81) -> VideoPose3D:
    """Create VideoPose model with spatial convolutions."""
    return VideoPose3D(num_joints=num_joints, filter_widths=[3, 3, 3, 3])


def create_videopose_temporal(
    num_joints: int = 17, num_frames: int = 243
) -> VideoPose3D:
    """Create VideoPose model with temporal convolutions."""
    return VideoPose3D(num_joints=num_joints, filter_widths=[3, 3, 3, 3, 3])

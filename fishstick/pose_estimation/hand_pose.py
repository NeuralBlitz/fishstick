"""
Hand Pose Estimation Models

Deep learning models for hand pose estimation including
MediaPipe-inspired networks, MANO-based models, and Interhand models.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HandPoseNet(nn.Module):
    """
    Convolutional neural network for 2D hand pose estimation.

    Architecture: Hourglass-style network with heatmap regression.

    Args:
        num_keypoints: Number of hand keypoints (default: 21)
        input_channels: Number of input image channels
        heatmap_size: Size of output heatmap
    """

    def __init__(
        self,
        num_keypoints: int = 21,
        input_channels: int = 3,
        heatmap_size: int = 64,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1),
            nn.Upsample(
                size=(heatmap_size, heatmap_size), mode="bilinear", align_corners=False
            ),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints * 2, 1),
            nn.Upsample(
                size=(heatmap_size, heatmap_size), mode="bilinear", align_corners=False
            ),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of (heatmap [B, K, H, W], offsets [B, K, 2, H, W])
        """
        features = self.backbone(x)

        heatmap = self.heatmap_head(features)
        offsets = self.offset_head(features)

        offsets = offsets.view(
            offsets.shape[0],
            self.num_keypoints,
            2,
            self.heatmap_size,
            self.heatmap_size,
        )

        return heatmap, offsets


class MediaPipeHand(nn.Module):
    """
    MediaPipe-style hand pose estimation model.

    Uses palm detection + hand landmark regression.

    Args:
        num_keypoints: Number of hand keypoints (default: 21)
        use_palm_detection: Whether to include palm detection branch
    """

    def __init__(
        self,
        num_keypoints: int = 21,
        use_palm_detection: bool = True,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.use_palm_detection = use_palm_detection

        self.palm_detector = (
            nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 4),
                nn.Sigmoid(),
            )
            if use_palm_detection
            else None
        )

        self.landmark_regressor = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_keypoints * 3),
        )

        self.visibility_head = nn.Sequential(
            nn.Linear(512, num_keypoints),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Dictionary with 'landmarks', 'visibility', and optionally 'palm_box'
        """
        landmarks = self.landmark_regressor(x)
        landmarks = landmarks.view(-1, self.num_keypoints, 3)

        visibility = self.visibility_head(
            self.landmark_regressor[:7](x).flatten(1)[:, :512]
        )

        outputs = {
            "landmarks": landmarks,
            "visibility": visibility,
        }

        if self.use_palm_detection and self.palm_detector is not None:
            palm_box = self.palm_detector(x)
            outputs["palm_box"] = palm_box

        return outputs


class ManoModel(nn.Module):
    """
    MANO (Minimal Hand) model for 3D hand mesh recovery.

    Implements the MANO layer for hand shape and pose parameters.

    Args:
        num_vertices: Number of mesh vertices (778 for MANO)
        num_shapes: Number of shape parameters (10)
        num_poses: Number of pose parameters (45 for 15 joints x 3)
        shape_mean: Mean shape parameters
        shape_pca: PCA components for shape
    """

    def __init__(
        self,
        num_vertices: int = 778,
        num_shapes: int = 10,
        num_poses: int = 45,
        shape_mean: Optional[Tensor] = None,
        shape_pca: Optional[Tensor] = None,
    ):
        super().__init__()

        self.num_vertices = num_vertices
        self.num_shapes = num_shapes
        self.num_poses = num_poses

        if shape_mean is None:
            shape_mean = torch.zeros(num_shapes)
        if shape_pca is None:
            shape_pca = torch.eye(num_shapes)

        self.register_buffer("shape_mean", shape_mean)
        self.register_buffer("shape_pca", shape_pca)

        self.shape_coeffs = nn.Parameter(torch.zeros(num_shapes))
        self.pose_coeffs = nn.Parameter(torch.zeros(num_poses))

        self.hand_pose_embed = nn.Linear(num_poses, 128)
        self.shape_embed = nn.Linear(num_shapes, 64)

        self.vertex_regressor = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_vertices * 3),
        )

        self.faces = self._get_mano_faces()

    def _get_mano_faces(self) -> Tensor:
        """Get MANO mesh faces."""
        faces = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],
                [4, 5, 6],
                [5, 7, 6],
                [8, 9, 10],
                [9, 11, 10],
                [12, 13, 14],
                [13, 15, 14],
                [16, 17, 18],
                [17, 19, 18],
                [20, 21, 22],
                [21, 23, 22],
                [24, 25, 26],
                [25, 27, 26],
                [28, 29, 30],
                [29, 31, 30],
                [32, 33, 34],
                [33, 35, 34],
                [36, 37, 38],
                [37, 39, 38],
                [40, 41, 42],
                [41, 43, 42],
                [44, 45, 46],
                [45, 47, 46],
            ]
        )
        return faces

    def forward(
        self,
        pose: Optional[Tensor] = None,
        shape: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass to generate hand mesh.

        Args:
            pose: Optional pose coefficients [B, 45]
            shape: Optional shape coefficients [B, 10]

        Returns:
            Dictionary with 'vertices', 'joints', 'faces'
        """
        if pose is None:
            pose = self.pose_coeffs.unsqueeze(0)
        if shape is None:
            shape = self.shape_coeffs.unsqueeze(0)

        pose_embed = self.hand_pose_embed(pose)
        shape_embed = self.shape_embed(shape)

        combined = torch.cat([pose_embed, shape_embed], dim=-1)

        vertices = self.vertex_regressor(combined)
        vertices = vertices.view(-1, self.num_vertices, 3)

        joints = vertices[:, :21, :]

        return {
            "vertices": vertices,
            "joints": joints,
            "faces": self.faces,
            "pose": pose,
            "shape": shape,
        }


class InterhandModel(nn.Module):
    """
    InterHand model for simultaneous left/right hand pose estimation.

    Args:
        num_keypoints: Keypoints per hand (21)
        use_interaction: Whether to model hand-hand interactions
    """

    def __init__(
        self,
        num_keypoints: int = 21,
        use_interaction: bool = True,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.use_interaction = use_interaction

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.left_hand_head = nn.Conv2d(256, num_keypoints, 1)
        self.right_hand_head = nn.Conv2d(256, num_keypoints, 1)

        if use_interaction:
            self.interaction_module = nn.Sequential(
                nn.Linear(num_keypoints * 2 * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )

            self.left_refine = nn.Linear(128 + num_keypoints * 2, num_keypoints * 2)
            self.right_refine = nn.Linear(128 + num_keypoints * 2, num_keypoints * 2)

        self.confidence_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Dictionary with left/right hand keypoints and confidence
        """
        features = self.encoder(x)

        left_heatmap = self.left_hand_head(features)
        right_heatmap = self.right_hand_head(features)

        confidence = self.confidence_head(features)

        B = x.shape[0]

        left_kp = left_heatmap.flatten(2).max(dim=2)[0]
        right_kp = right_heatmap.flatten(2).max(dim=2)[0]

        if self.use_interaction:
            left_flat = left_heatmap.flatten(1)
            right_flat = right_heatmap.flatten(1)

            interaction = torch.cat([left_flat, right_flat], dim=1)
            interaction_feat = self.interaction_module(interaction)

            left_refined = self.left_refine(
                torch.cat([interaction_feat, left_flat], dim=1)
            ).view(B, self.num_keypoints, 2)

            right_refined = self.right_refine(
                torch.cat([interaction_feat, right_flat], dim=1)
            ).view(B, self.num_keypoints, 2)

            return {
                "left_hand": left_refined,
                "right_hand": right_refined,
                "confidence": confidence,
            }

        return {
            "left_hand": left_kp,
            "right_hand": right_kp,
            "confidence": confidence,
        }


class HandPoseModel(nn.Module):
    """
    Unified hand pose estimation model supporting multiple modes.

    Args:
        model_type: Type of model ("heatmap", "regression", "mano", "interhand")
        num_keypoints: Number of keypoints
    """

    def __init__(
        self,
        model_type: str = "heatmap",
        num_keypoints: int = 21,
    ):
        super().__init__()

        self.model_type = model_type
        self.num_keypoints = num_keypoints

        if model_type == "heatmap":
            self.model = HandPoseNet(num_keypoints=num_keypoints)
        elif model_type == "regression":
            self.model = MediaPipeHand(num_keypoints=num_keypoints)
        elif model_type == "mano":
            self.model = ManoModel()
        elif model_type == "interhand":
            self.model = InterhandModel(num_keypoints=num_keypoints)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass."""
        return self.model(x)


def create_mano_layer(
    model_path: Optional[str] = None,
    num_shapes: int = 10,
) -> ManoModel:
    """
    Create MANO layer with optional pretrained weights.

    Args:
        model_path: Path to MANO model weights
        num_shapes: Number of shape parameters

    Returns:
        MANO model
    """
    model = ManoModel(num_shapes=num_shapes)

    if model_path is not None:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

    return model


__all__ = [
    "HandPoseNet",
    "MediaPipeHand",
    "ManoModel",
    "InterhandModel",
    "HandPoseModel",
    "create_mano_layer",
]

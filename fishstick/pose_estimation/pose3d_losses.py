"""
Loss Functions for 3D Pose Estimation

Contains loss functions for training 3D pose estimation models.
"""

from typing import List, Tuple, Optional, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


class MPJPELoss(nn.Module):
    """
    Mean Per Joint Position Error (MPJPE) loss.

    The primary metric for 3D pose estimation.

    Args:
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            pred: Predicted 3D keypoints (B, J, 3) or (B, T, J, 3)
            target: Target 3D keypoints (B, J, 3) or (B, T, J, 3)
            weight: Optional per-joint weights
            valid_mask: Optional validity mask

        Returns:
            MPJPE loss value
        """
        diff = pred - target
        dist = torch.norm(diff, dim=-1)

        if weight is not None:
            dist = dist * weight

        if valid_mask is not None:
            dist = dist * valid_mask
            if self.reduction == "mean":
                return dist.sum() / (valid_mask.sum() + 1e-8)

        if self.reduction == "mean":
            return dist.mean()
        elif self.reduction == "sum":
            return dist.sum()
        return dist


class PCKLoss3D(nn.Module):
    """
    Percentage of Correct Keypoints (PCK) loss for 3D poses.

    Args:
        threshold: Distance threshold for correct keypoint
    """

    def __init__(self, threshold: float = 0.15):
        super().__init__()
        self.threshold = threshold

    def forward(
        self, pred: Tensor, target: Tensor, valid_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            pred: Predicted 3D keypoints (B, J, 3)
            target: Target 3D keypoints (B, J, 3)
            valid_mask: Optional validity mask

        Returns:
            1 - PCK (to use as loss)
        """
        diff = pred - target
        dist = torch.norm(diff, dim=-1)

        correct = dist < self.threshold

        if valid_mask is not None:
            correct = correct & valid_mask
            pck = correct.float().sum() / (valid_mask.float().sum() + 1e-8)
        else:
            pck = correct.float().mean()

        return 1 - pck


class ProcrustesLoss(nn.Module):
    """
    Procrustes loss for 3D pose alignment.

    Aligns poses using similarity transformation before computing loss.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted 3D keypoints (B, J, 3)
            target: Target 3D keypoints (B, J, 3)

        Returns:
            Procrustes-aligned error
        """
        pred_aligned = compute_similarity_transform(pred, target)

        diff = pred_aligned - target
        dist = torch.norm(diff, dim=-1)

        if self.reduction == "mean":
            return dist.mean()
        elif self.reduction == "sum":
            return dist.sum()
        return dist


class BoneLengthLoss(nn.Module):
    """
    Bone length consistency loss.

    Encourages correct bone lengths in the predicted pose.

    Args:
        skeleton: List of (parent, child) joint pairs
    """

    def __init__(self, skeleton: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        if skeleton is None:
            self.skeleton = [
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
        else:
            self.skeleton = skeleton

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted 3D keypoints (B, J, 3)
            target: Target 3D keypoints (B, J, 3)

        Returns:
            Bone length error
        """
        losses = []

        for parent, child in self.skeleton:
            pred_bone = pred[:, child] - pred[:, parent]
            target_bone = target[:, child] - target[:, parent]

            pred_len = torch.norm(pred_bone, dim=-1)
            target_len = torch.norm(target_bone, dim=-1)

            loss = torch.abs(pred_len - target_len)
            losses.append(loss)

        bone_losses = torch.stack(losses, dim=1)
        return bone_losses.mean()


class VelocityLoss(nn.Module):
    """
    Velocity consistency loss for temporal pose sequences.

    Args:
        reduction: Reduction method
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted 3D keypoints (B, T, J, 3)
            target: Target 3D keypoints (B, T, J, 3)

        Returns:
            Velocity error
        """
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]

        diff = pred_vel - target_vel
        dist = torch.norm(diff, dim=-1)

        if self.reduction == "mean":
            return dist.mean()
        elif self.reduction == "sum":
            return dist.sum()
        return dist


class CombinedPose3DLoss(nn.Module):
    """
    Combined loss for 3D pose estimation.

    Args:
        mpjpe_weight: Weight for MPJPE loss
        velocity_weight: Weight for velocity loss
        bone_weight: Weight for bone length loss
        procrustes_weight: Weight for Procrustes loss
    """

    def __init__(
        self,
        mpjpe_weight: float = 1.0,
        velocity_weight: float = 0.0,
        bone_weight: float = 0.0,
        procrustes_weight: float = 0.0,
    ):
        super().__init__()
        self.mpjpe_weight = mpjpe_weight
        self.velocity_weight = velocity_weight
        self.bone_weight = bone_weight
        self.procrustes_weight = procrustes_weight

        self.mpjpe = MPJPELoss()
        self.velocity = VelocityLoss()
        self.bone = BoneLengthLoss()
        self.procrustes = ProcrustesLoss()

    def forward(
        self, pred: Tensor, target: Tensor, valid_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Args:
            pred: Predicted 3D keypoints (B, T, J, 3) or (B, J, 3)
            target: Target 3D keypoints
            valid_mask: Optional validity mask

        Returns:
            Total loss and dictionary of individual losses
        """
        losses = {}
        total_loss = 0.0

        mpjpe_loss = self.mpjpe(pred, target, valid_mask=valid_mask)
        losses["mpjpe"] = mpjpe_loss
        total_loss += self.mpjpe_weight * mpjpe_loss

        if self.bone_weight > 0:
            bone_loss = self.bone(pred, target)
            losses["bone"] = bone_loss
            total_loss += self.bone_weight * bone_loss

        if self.procrustes_weight > 0:
            proc_loss = self.procrustes(pred, target)
            losses["procrustes"] = proc_loss
            total_loss += self.procrustes_weight * proc_loss

        if self.velocity_weight > 0 and pred.dim() == 4:
            vel_loss = self.velocity(pred, target)
            losses["velocity"] = vel_loss
            total_loss += self.velocity_weight * vel_loss

        losses["total"] = total_loss
        return total_loss, losses


class Pose3DLoss(nn.Module):
    """
    Main 3D pose estimation loss.

    Unified interface for various 3D pose estimation loss functions.
    """

    def __init__(self, loss_config: Optional[Dict[str, float]] = None):
        super().__init__()

        if loss_config is None:
            loss_config = {"mpjpe": 1.0}

        self.loss = CombinedPose3DLoss(
            mpjpe_weight=loss_config.get("mpjpe", 1.0),
            velocity_weight=loss_config.get("velocity", 0.0),
            bone_weight=loss_config.get("bone", 0.0),
            procrustes_weight=loss_config.get("procrustes", 0.0),
        )

    def forward(
        self, pred: Tensor, target: Tensor, valid_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute total loss."""
        return self.loss(pred, target, valid_mask)


def compute_similarity_transform(
    pred: Tensor, target: Tensor, use_optimal_scale: bool = True
) -> Tensor:
    """
    Compute similarity transformation to align predicted pose to target.

    Implements the Procrustes analysis to find optimal rotation, scale, and translation.

    Args:
        pred: Predicted pose (B, J, 3)
        target: Target pose (B, J, 3)
        use_optimal_scale: Whether to compute optimal scale

    Returns:
        Aligned predicted pose
    """
    B, J, _ = pred.shape

    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)

    pred_norm = torch.norm(pred_centered, dim=-1).sum(dim=1, keepdim=True)
    target_norm = torch.norm(target_centered, dim=-1).sum(dim=1, keepdim=True)

    scale = (target_norm / (pred_norm + 1e-8)).view(B, 1, 1)

    if use_optimal_scale:
        pred_scaled = pred_centered * scale
    else:
        pred_scaled = pred_centered

    H = torch.bmm(pred_scaled.transpose(1, 2), target_centered)
    U, S, Vt = torch.svd(H)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    det = torch.det(R)
    Vt_neg = Vt.clone()
    Vt_neg[:, 2, :] = -Vt_neg[:, 2, :]
    R_adj = torch.bmm(Vt_neg.transpose(1, 2), U.transpose(1, 2))
    R = torch.where(det.view(B, 1, 1) > 0, R, R_adj)

    aligned = torch.bmm(pred_scaled, R.transpose(1, 2)) + target.mean(
        dim=1, keepdim=True
    )

    return aligned

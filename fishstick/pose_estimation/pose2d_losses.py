"""
Loss Functions for 2D Pose Estimation

Contains various losses for training 2D pose estimation models including
heatmap loss, keypoint loss, OKS loss, and combined losses.
"""

from typing import List, Tuple, Optional, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Mean Squared Error loss for keypoint coordinates."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(
        self, pred: Tensor, target: Tensor, weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            pred: Predicted coordinates (B, N, 2)
            target: Target coordinates (B, N, 2)
            weight: Optional weights for each keypoint (B, N)

        Returns:
            Loss value
        """
        if weight is not None:
            weight = weight.unsqueeze(-1)
            return (self.loss(pred, target, reduction="none") * weight).sum() / (
                weight.sum() + 1e-8
            )
        return self.loss(pred, target)


class SmoothL1Loss(nn.Module):
    """Smooth L1 (Huber) loss for keypoint coordinates."""

    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self, pred: Tensor, target: Tensor, weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            pred: Predicted coordinates (B, N, 2)
            target: Target coordinates (B, N, 2)
            weight: Optional weights for each keypoint

        Returns:
            Loss value
        """
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta, 0.5 * diff**2 / self.beta, diff - 0.5 * self.beta
        )

        if weight is not None:
            weight = weight.unsqueeze(-1)
            return (loss * weight).sum() / (weight.sum() + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class OKSLoss(nn.Module):
    """
    Object Keypoint Similarity (OKS) loss.

    OKS is the standard metric for COCO keypoint detection.

    Args:
        num_keypoints: Number of keypoints
        in_vis_oks: Initial OKS value for invisible keypoints
        focus_on_visible: Whether to only penalize visible keypoints
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        in_vis_oks: float = 0.0,
        focus_on_visible: bool = False,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.in_vis_oks = in_vis_oks
        self.focus_on_visible = focus_on_visible

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        target_vis: Optional[Tensor] = None,
        areas: Optional[Tensor] = None,
        sigmas: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            pred: Predicted keypoints (B, N, 2) or (N, 2)
            target: Target keypoints (B, N, 2) or (N, 2)
            target_vis: Visibility flags (B, N) or (N,)
            areas: Area scales for each person (B,) or scalar
            sigmas: Per-keypoint standard deviations (N,)

        Returns:
            OKS loss (1 - OKS)
        """
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0)

        B, N, _ = pred.shape

        if sigmas is None:
            sigmas = torch.ones(N, device=pred.device) * 0.1
        if areas is None:
            areas = torch.ones(B, device=pred.device) * 100.0

        if target_vis is None:
            target_vis = torch.ones(B, N, device=pred.device)

        dist = torch.norm(pred - target, dim=-1)

        k = (sigmas * 2) ** 2
        area = areas.view(B, 1)
        oks = torch.exp(-(dist**2) / (2 * k * area + 1e-8))

        if self.focus_on_visible:
            valid = target_vis > 0
            if valid.any():
                return 1 - (oks * valid).sum() / valid.sum()
            return torch.tensor(0.0, device=pred.device)

        valid = target_vis > 0
        invisible = target_vis == 0

        if valid.any():
            loss_vis = 1 - (oks * valid.float()).sum() / valid.float().sum()
        else:
            loss_vis = torch.tensor(0.0, device=pred.device)

        if invisible.any() and self.in_vis_oks > 0:
            loss_invis = (
                oks * invisible.float() * self.in_vis_oks
            ).sum() / invisible.float().sum()
            return loss_vis + loss_invis

        return loss_vis


class WingLoss(nn.Module):
    """
    Wing loss for facial landmark or pose estimation.

    Provides more focused optimization for smaller errors.

    Args:
        omega: Upper bound for linear regime
        epsilon: Small value to avoid log(0)
    """

    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(1 + self.omega / epsilon)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted coordinates (B, N, 2)
            target: Target coordinates (B, N, 2)

        Returns:
            Wing loss value
        """
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C,
        )
        return loss.mean()


class HeatmapLoss(nn.Module):
    """Loss for heatmap-based pose estimation."""

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        elif loss_type == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        pred_heatmaps: Tensor,
        target_heatmaps: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            pred_heatmaps: Predicted heatmaps (B, K, H, W)
            target_heatmaps: Target heatmaps (B, K, H, W)
            mask: Optional mask for valid regions (B, H, W)

        Returns:
            Heatmap loss value
        """
        if self.loss_type == "bce":
            pred_heatmaps = pred_heatmaps.float()
            target_heatmaps = target_heatmaps.float()

        loss = self.loss(pred_heatmaps, target_heatmaps)

        if mask is not None:
            mask = mask.unsqueeze(1).float()
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        return loss


class PAFLoss(nn.Module):
    """
    Part Affinity Fields loss for OpenPose-style models.

    Args:
        loss_type: Type of loss ('l1', 'l2')
    """

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self, pred_pafs: Tensor, target_pafs: Tensor, paf_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            pred_pafs: Predicted PAF maps (B, 2*L, H, W)
            target_pafs: Target PAF maps (B, 2*L, H, W)
            paf_mask: Optional mask for valid regions (B, H, W)

        Returns:
            PAF loss value
        """
        if self.loss_type == "l1":
            loss = F.l1_loss(pred_pafs, target_pafs, reduction="none")
        else:
            loss = F.mse_loss(pred_pafs, target_pafs, reduction="none")

        if paf_mask is not None:
            paf_mask = paf_mask.unsqueeze(1).float()
            loss = (loss * paf_mask).sum() / (paf_mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


class CombinedPose2DLoss(nn.Module):
    """
    Combined loss for 2D pose estimation with multiple components.

    Supports heatmap loss, keypoint coordinate loss, PAF loss, and wing loss.

    Args:
        heatmap_weight: Weight for heatmap loss
        coord_weight: Weight for coordinate loss
        paf_weight: Weight for PAF loss
        wing_weight: Weight for wing loss
        loss_type: Type of coordinate loss ('mse', 'l1', 'smoothl1', 'wing')
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        coord_weight: float = 1.0,
        paf_weight: float = 0.0,
        wing_weight: float = 0.0,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.paf_weight = paf_weight
        self.wing_weight = wing_weight

        if loss_type == "mse":
            self.coord_loss = MSELoss()
        elif loss_type == "l1":
            self.coord_loss = nn.L1Loss()
        elif loss_type == "smoothl1":
            self.coord_loss = SmoothL1Loss()
        elif loss_type == "wing":
            self.coord_loss = WingLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.heatmap_loss = HeatmapLoss()
        self.paf_loss = PAFLoss()
        self.wing_loss = WingLoss()

    def forward(
        self, pred: Dict[str, Tensor], target: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Args:
            pred: Dictionary with 'heatmap', 'coords', 'paf' predictions
            target: Dictionary with corresponding targets

        Returns:
            Total loss and dictionary of individual losses
        """
        losses = {}
        total_loss = 0.0

        if "heatmap" in pred and "heatmap" in target:
            hm_loss = self.heatmap_loss(
                pred["heatmap"], target["heatmap"], target.get("heatmap_mask")
            )
            losses["heatmap"] = hm_loss
            total_loss += self.heatmap_weight * hm_loss

        if "coords" in pred and "coords" in target:
            coord_loss = self.coord_loss(
                pred["coords"], target["coords"], target.get("coord_weight")
            )
            losses["coord"] = coord_loss
            total_loss += self.coord_weight * coord_loss

        if "paf" in pred and "paf" in target:
            paf_loss = self.paf_loss(pred["paf"], target["paf"], target.get("paf_mask"))
            losses["paf"] = paf_loss
            total_loss += self.paf_weight * paf_loss

        if "wing_coords" in pred and "coords" in target:
            wing_loss = self.wing_loss(pred["wing_coords"], target["coords"])
            losses["wing"] = wing_loss
            total_loss += self.wing_weight * wing_loss

        losses["total"] = total_loss
        return total_loss, losses


class Pose2DLoss(nn.Module):
    """
    Main 2D pose estimation loss.

    Unified interface for various pose estimation loss functions.

    Args:
        loss_config: Dictionary specifying loss components and weights
    """

    def __init__(self, loss_config: Optional[Dict[str, float]] = None):
        super().__init__()

        if loss_config is None:
            loss_config = {
                "heatmap": 1.0,
                "coords": 1.0,
            }

        self.loss_config = loss_config
        self.combined_loss = CombinedPose2DLoss(
            heatmap_weight=loss_config.get("heatmap", 1.0),
            coord_weight=loss_config.get("coords", 1.0),
            paf_weight=loss_config.get("paf", 0.0),
            wing_weight=loss_config.get("wing", 0.0),
            loss_type=loss_config.get("loss_type", "mse"),
        )

    def forward(
        self, pred: Dict[str, Tensor], target: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute total loss."""
        return self.combined_loss(pred, target)

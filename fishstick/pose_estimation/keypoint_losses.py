"""
Keypoint Detection Loss Functions

Loss functions for keypoint detection including heatmap loss,
offset loss, and combined losses.
"""

from typing import List, Tuple, Optional, Dict, Any, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class KeypointHeatmapLoss(nn.Module):
    """
    Keypoint heatmap regression loss using focal loss.

    Args:
        alpha: Focal loss alpha parameter
        beta: Focal loss beta parameter
        use_focal: Whether to use focal loss (else MSE)
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        use_focal: bool = True,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.use_focal = use_focal

    def forward(
        self,
        pred_heatmap: Tensor,
        gt_heatmap: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute heatmap loss.

        Args:
            pred_heatmap: Predicted heatmap (B, K, H, W)
            gt_heatmap: Ground truth heatmap (B, K, H, W)
            mask: Optional mask for valid regions

        Returns:
            Loss value
        """
        pred = torch.sigmoid(pred_heatmap)

        if self.use_focal:
            pos_mask = gt_heatmap
            neg_mask = 1 - gt_heatmap

            pos_loss = (
                -pos_mask * torch.pow(1 - pred, self.alpha) * torch.log(pred + 1e-8)
            )
            neg_loss = (
                -neg_mask * torch.pow(pred, self.alpha) * torch.log(1 - pred + 1e-8)
            )

            loss = pos_loss + neg_loss
        else:
            loss = F.mse_loss(pred, gt_heatmap, reduction="none")

        if mask is not None:
            loss = loss * mask.unsqueeze(1)

        return loss.mean()


class KeypointOffsetLoss(nn.Module):
    """
    Keypoint offset regression loss.

    Args:
        loss_type: Type of loss ("l1", "l2", "smoothl1")
    """

    def __init__(
        self,
        loss_type: str = "l1",
    ):
        super().__init__()

        self.loss_type = loss_type

        if loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type == "l2":
            self.loss_fn = F.mse_loss
        elif loss_type == "smoothl1":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        pred_offset: Tensor,
        gt_offset: Tensor,
        gt_heatmap: Tensor,
    ) -> Tensor:
        """
        Compute offset loss.

        Args:
            pred_offset: Predicted offset (B, K, 2, H, W)
            gt_offset: Ground truth offset (B, K, 2, H, W)
            gt_heatmap: Ground truth heatmap for masking (B, K, H, W)

        Returns:
            Loss value
        """
        mask = (gt_heatmap.sum(dim=1, keepdim=True) > 0).float()

        loss = self.loss_fn(pred_offset, gt_offset, reduction="none")

        loss = loss * mask

        return loss.mean()


class KeypointLoss(nn.Module):
    """
    Combined keypoint detection loss.

    Args:
        heatmap_weight: Weight for heatmap loss
        offset_weight: Weight for offset loss
        visibility_weight: Weight for visibility loss
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        offset_weight: float = 0.1,
        visibility_weight: float = 0.5,
    ):
        super().__init__()

        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.visibility_weight = visibility_weight

        self.heatmap_loss = KeypointHeatmapLoss()
        self.offset_loss = KeypointOffsetLoss()
        self.visibility_loss = nn.BCELoss()

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute combined keypoint loss.

        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with ground truth targets

        Returns:
            Dictionary with loss values
        """
        losses = {}

        if "heatmap" in predictions and "heatmap" in targets:
            heatmap_loss = self.heatmap_loss(
                predictions["heatmap"],
                targets["heatmap"],
                targets.get("mask"),
            )
            losses["heatmap_loss"] = heatmap_loss

        if "offset" in predictions and "offset" in targets:
            offset_loss = self.offset_loss(
                predictions["offset"],
                targets["offset"],
                targets.get("heatmap"),
            )
            losses["offset_loss"] = offset_loss

        if "visibility" in predictions and "visibility" in targets:
            vis_loss = self.visibility_loss(
                predictions["visibility"],
                targets["visibility"],
            )
            losses["visibility_loss"] = vis_loss

        total_loss = (
            self.heatmap_weight * losses.get("heatmap_loss", 0.0)
            + self.offset_weight * losses.get("offset_loss", 0.0)
            + self.visibility_weight * losses.get("visibility_loss", 0.0)
        )

        losses["total_loss"] = total_loss

        return losses


class KeypointLossCombined(nn.Module):
    """
    Combined loss for multi-stack hourglass networks.

    Args:
        num_stacks: Number of hourglass stacks
        heatmap_weight: Weight for heatmap loss
        offset_weight: Weight for offset loss
    """

    def __init__(
        self,
        num_stacks: int = 2,
        heatmap_weight: float = 1.0,
        offset_weight: float = 0.1,
    ):
        super().__init__()

        self.num_stacks = num_stacks
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight

        self.heatmap_loss = KeypointHeatmapLoss()
        self.offset_loss = KeypointOffsetLoss()

    def forward(
        self,
        predictions: Dict[str, List[Tensor]],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute combined loss for all stacks.

        Args:
            predictions: Dictionary with lists of predictions per stack
            targets: Dictionary with ground truth targets

        Returns:
            Dictionary with loss values per stack and total
        """
        losses = {}

        heatmap_losses = []
        offset_losses = []

        for i in range(self.num_stacks):
            if "heatmaps" in predictions:
                hm_loss = self.heatmap_loss(
                    predictions["heatmaps"][i],
                    targets["heatmap"],
                    targets.get("mask"),
                )
                heatmap_losses.append(hm_loss)

            if "offsets" in predictions:
                off_loss = self.offset_loss(
                    predictions["offsets"][i],
                    targets["offset"],
                    targets.get("heatmap"),
                )
                offset_losses.append(off_loss)

        if heatmap_losses:
            losses["heatmap_loss"] = torch.stack(heatmap_losses).mean()

        if offset_losses:
            losses["offset_loss"] = torch.stack(offset_losses).mean()

        total_loss = self.heatmap_weight * losses.get(
            "heatmap_loss", 0.0
        ) + self.offset_weight * losses.get("offset_loss", 0.0)

        losses["total_loss"] = total_loss

        return losses


__all__ = [
    "KeypointHeatmapLoss",
    "KeypointOffsetLoss",
    "KeypointLoss",
    "KeypointLossCombined",
]

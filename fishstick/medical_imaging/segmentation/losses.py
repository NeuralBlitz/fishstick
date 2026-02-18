"""
Segmentation Loss Functions

Dice, Tversky, Focal, and boundary losses for medical image segmentation.
"""

from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiceLoss(nn.Module):
    """Dice Loss for segmentation.

    Computes the Dice coefficient between prediction and ground truth,
    which is more robust to class imbalance than cross-entropy.

    Example:
        >>> criterion = DiceLoss(num_classes=3, smooth=1.0)
        >>> pred = torch.randn(2, 3, 32, 32, 32)
        >>> target = torch.randint(0, 3, (2, 32, 32, 32))
        >>> loss = criterion(pred, target)
    """

    def __init__(
        self,
        num_classes: int = 2,
        smooth: float = 1.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            pred: Prediction logits (B, C, D, H, W) or (B, C, H, W)
            target: Ground truth labels (B, D, H, W) or (B, H, W)
            weight: Optional per-class weights

        Returns:
            Dice loss value
        """
        pred = F.softmax(pred, dim=1)

        target_one_hot = (
            F.one_hot(
                target.long(),
                num_classes=self.num_classes,
            )
            .permute(0, -1, *range(1, target.ndim))
            .float()
        )

        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).float()
            target_one_hot = target_one_hot * mask.unsqueeze(1)
            pred = pred * mask.unsqueeze(1)

        dims = tuple(range(1, target_one_hot.ndim))

        intersection = (pred * target_one_hot).sum(dim=dims)
        cardinality = pred.sum(dim=dims) + target_one_hot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if weight is not None:
            dice_score = dice_score * weight

        dice_loss = 1.0 - dice_score.mean()

        if self.reduction == "mean":
            return dice_loss
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice with controllable false positives/negatives.

    The alpha and beta parameters control the trade-off between:
    - alpha: penalizing false negatives
    - beta: penalizing false positives

    Example:
        >>> criterion = TverskyLoss(num_classes=3, alpha=0.7, beta=0.3)
    """

    def __init__(
        self,
        num_classes: int = 2,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = F.softmax(pred, dim=1)

        target_one_hot = (
            F.one_hot(
                target.long(),
                num_classes=self.num_classes,
            )
            .permute(0, -1, *range(1, target.ndim))
            .float()
        )

        dims = tuple(range(1, target_one_hot.ndim))

        true_pos = (pred * target_one_hot).sum(dim=dims)
        false_neg = ((1 - pred) * target_one_hot).sum(dim=dims)
        false_pos = (pred * (1 - target_one_hot)).sum(dim=dims)

        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth
        )

        loss = 1.0 - tversky

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalDiceLoss(nn.Module):
    """Combined Focal and Dice loss for handling class imbalance.

    Combines the benefits of focal loss (hard example mining) with
    dice loss (robust to imbalanced data).
    """

    def __init__(
        self,
        num_classes: int = 2,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.dice_loss = DiceLoss(num_classes, reduction="none")
        self.focal_loss = FocalLoss(num_classes, gamma, reduction="none")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        loss = self.alpha * dice + (1 - self.alpha) * focal

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Focuses training on hard examples by down-weighting easy examples.
    """

    def __init__(
        self,
        num_classes: int = 2,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float]]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, float):
                self.alpha = torch.tensor([alpha] * num_classes)
            else:
                self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none")

        p_t = torch.exp(-ce_loss)

        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(pred.device)[target]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):
    """Boundary loss for boundary-aware segmentation.

    Penalizes distance between predicted and ground truth boundaries.
    """

    def __init__(
        self,
        theta0: float = 3.0,
        theta: float = 5.0,
    ):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = F.softmax(pred, dim=1)

        target_one_hot = (
            F.one_hot(
                target.long(),
                num_classes=pred.shape[1],
            )
            .permute(0, -1, *range(1, target.ndim))
            .float()
        )

        target_dist = self._compute_distance_transform(target_one_hot)

        pred_boundary = self._compute_boundary(pred)
        target_boundary = self._compute_boundary(target_one_hot)

        boundary_loss = (pred_boundary * target_dist).sum() / pred_boundary.sum()

        return boundary_loss

    def _compute_distance_transform(self, mask: Tensor) -> Tensor:
        """Compute approximate distance transform."""
        import numpy as np

        dist = torch.zeros_like(mask)

        for b in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                m = mask[b, c].cpu().numpy()

                if m.sum() > 0:
                    from scipy.ndimage import distance_transform_edt

                    d = distance_transform_edt(1 - m)
                    dist[b, c] = torch.from_numpy(d).to(mask.device)

        return dist

    def _compute_boundary(self, mask: Tensor) -> Tensor:
        """Compute boundary from mask."""
        laplacian_kernel = torch.tensor(
            [
                [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]],
            ]
        ).float()

        if mask.ndim == 5:
            laplacian_kernel = laplacian_kernel.unsqueeze(0)

        boundary = F.conv3d(
            mask,
            laplacian_kernel.to(mask.device),
            padding=1,
        )

        boundary = torch.abs(boundary)

        return boundary


class DiceScore(nn.Module):
    """Dice score metric for evaluation (not a loss)."""

    def __init__(self, num_classes: int, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Union[Tensor, float]:
        pred = pred.argmax(dim=1)

        dice_scores = []

        for c in range(self.num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > 0:
                dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
                dice_scores.append(dice.item())

        if self.reduction == "mean":
            return sum(dice_scores) / len(dice_scores)
        else:
            return dice_scores


class IoUScore(nn.Module):
    """Intersection over Union (IoU) metric."""

    def __init__(self, num_classes: int, reduction: str = "mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Union[Tensor, float]:
        pred = pred.argmax(dim=1)

        iou_scores = []

        for c in range(self.num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection

            if union > 0:
                iou = (intersection + 1e-8) / (union + 1e-8)
                iou_scores.append(iou.item())

        if self.reduction == "mean":
            return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        else:
            return iou_scores


class CompoundLoss(nn.Module):
    """Compound loss combining multiple segmentation losses."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        if weights is None:
            weights = {"dice": 1.0, "ce": 1.0}

        self.weights = weights
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = 0.0

        if "dice" in self.weights:
            loss += self.weights["dice"] * self.dice(pred, target)

        if "ce" in self.weights:
            loss += self.weights["ce"] * self.ce(pred, target)

        return loss

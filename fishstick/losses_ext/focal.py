"""
Focal Loss Variants

Implementation of focal loss variants for handling class imbalance
and hard example mining in classification tasks.
"""

from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss applies a modulating term to the cross entropy loss to
    focus learning on hard misclassified examples.

    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples.
            Can be a scalar or tensor of shape (num_classes,).
        gamma: Focusing parameter for modulating loss. Higher values
            focus more on hard examples. Typical values: 0.5-2.0.
        reduction: Specifies the reduction to apply: 'none', 'mean', 'sum'.

    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(8, 10)
        >>> targets = torch.randint(0, 10, (8,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (int, float)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Args:
        alpha: Weighting factor for the positive class.
        gamma: Focusing parameter.
        reduction: Specifies the reduction to apply.

    Example:
        >>> loss_fn = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(8)
        >>> targets = torch.randint(0, 2, (8,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that automatically adjusts focusing parameter.

    The gamma parameter is dynamically adjusted based on the difficulty
    of the sample, estimated from the prediction confidence.

    Args:
        base_gamma: Base focusing parameter.
        gamma_range: Tuple of (min, max) gamma values.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        base_gamma: float = 2.0,
        gamma_range: tuple = (0.5, 4.0),
        reduction: str = "mean",
    ):
        super().__init__()
        self.base_gamma = base_gamma
        self.gamma_min, self.gamma_max = gamma_range
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        probs = F.softmax(logits, dim=-1)
        confidence = probs.gather(1, targets.unsqueeze(1)).squeeze()

        difficulty = 1 - confidence
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * difficulty
        gamma = gamma.clamp(self.gamma_min, self.gamma_max)

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-balanced Focal Loss with per-class weights.

    Automatically computes effective number of samples for each class
    and uses them to balance the loss.

    Args:
        samples_per_class: Number of samples per class for computing weights.
        beta: Hyperparameter for computing effective number.
        gamma: Focusing parameter.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        samples_per_class: torch.Tensor,
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        effective_num = 1.0 - torch.pow(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)
        self.register_buffer("weights", weights)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        weights = self.weights[targets]
        focal_loss = weights * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class PolynomialFocalLoss(nn.Module):
    """
    Polynomial Focal Loss with configurable polynomial order.

    Extends focal loss with polynomial term for better gradient behavior.

    Args:
        order: Polynomial order (1 = standard focal, 2+ = polynomial extension).
        gamma: Focusing parameter.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        order: int = 2,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.order = order
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        poly_term = 1.0
        for i in range(1, self.order + 1):
            poly_term += torch.pow(1 - pt, i) / (i + 1)

        focal_loss = (1 - pt) ** self.gamma * ce_loss * poly_term

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


__all__ = [
    "FocalLoss",
    "BinaryFocalLoss",
    "AdaptiveFocalLoss",
    "ClassBalancedFocalLoss",
    "PolynomialFocalLoss",
]

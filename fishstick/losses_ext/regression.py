"""
Custom Regression Loss Functions

Advanced regression loss implementations with robust variants,
including Huber, Logcosh, Quantile, and specialized losses.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HuberLoss(nn.Module):
    """
    Huber Loss with adaptive delta.

    Combines L1 and L2 loss, robust to outliers while providing smooth gradients.

    Args:
        delta: Threshold at which to switch between L1 and L2 behavior.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        residual = torch.abs(predictions - targets)
        quadratic = torch.clamp(residual, max=self.delta)
        linear = residual - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AdaptiveHuberLoss(nn.Module):
    """
    Adaptive Huber Loss with learned delta.

    Delta parameter is learned or adapted based on data distribution.

    Args:
        initial_delta: Initial value for delta.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        initial_delta: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(initial_delta))
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        residual = torch.abs(predictions - targets)
        delta = torch.abs(self.delta) + 1e-6

        quadratic = torch.clamp(residual, max=delta)
        linear = residual - quadratic
        loss = 0.5 * quadratic**2 + delta * linear

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss function.

    Smooth approximation to L1 loss with logarithmic correction.

    Args:
        reduction: Specifies the reduction to apply.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        residual = predictions - targets
        loss = torch.log(torch.cosh(residual + 1e-6))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class QuantileLoss(nn.Module):
    """
    Quantile Loss for probabilistic regression.

    Predicts conditional quantiles rather than mean values.

    Args:
        quantile: Quantile to predict in [0, 1].
        reduction: Specifies the reduction to apply.

    Example:
        >>> loss_fn = QuantileLoss(quantile=0.5)
        >>> preds = torch.randn(8, 1)
        >>> targets = torch.randn(8, 1)
        >>> loss = loss_fn(preds, targets)
    """

    def __init__(
        self,
        quantile: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.quantile = quantile
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        residual = targets - predictions
        loss = torch.max(self.quantile * residual, (self.quantile - 1) * residual)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiQuantileLoss(nn.Module):
    """
    Multi-Quantile Loss for predicting multiple quantiles.

    Args:
        quantiles: List of quantiles to predict.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.quantiles = quantiles if quantiles is not None else [0.1, 0.5, 0.9]
        self.reduction = reduction

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        batch_size = predictions.size(0)
        num_quantiles = len(self.quantiles)

        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        predictions = predictions.view(batch_size, num_quantiles, -1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(1)

        total_loss = 0.0
        for i, q in enumerate(self.quantiles):
            residual = targets - predictions[:, i]
            loss = torch.max(q * residual, (q - 1) * residual)
            total_loss += loss

        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        return total_loss


class TweedieLoss(nn.Module):
    """
    Tweedie Loss for count data and positive continuous targets.

    Generalizes Poisson and Gamma losses.

    Args:
        power: Tweedie power parameter (1=Poisson, 2=Gamma, 3=Compound Poisson).
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        power: float = 1.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.power = power
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if self.power == 1:
            loss = targets * torch.exp(-predictions) + predictions
        elif self.power == 2:
            loss = targets * torch.exp(-predictions) + predictions - 1
        else:
            pred_pos = torch.exp(predictions)
            if self.power < 1 or self.power > 2:
                loss = (
                    torch.pow(targets, 2 - self.power)
                    / ((2 - self.power) * (1 - self.power))
                    + targets * torch.pow(pred_pos, self.power - 1) / (1 - self.power)
                    + torch.pow(pred_pos, self.power - 2) / (2 - self.power)
                )
            else:
                loss = targets * torch.exp(-predictions) + predictions - 1

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PseudoHuberLoss(nn.Module):
    """
    Pseudo-Huber Loss.

    Differentiable approximation to Huber loss using smooth function.

    Args:
        delta: Threshold for switching between L1 and L2 behavior.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        residual = predictions - targets
        loss = self.delta**2 * (torch.sqrt(1 + (residual / self.delta) ** 2) - 1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CauchyLoss(nn.Module):
    """
    Cauchy (Lorentzian) Loss.

    Heavy-tailed loss function highly robust to outliers.

    Args:
        scale: Scale parameter controlling the spread.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        scale: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.scale = scale
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        residual = predictions - targets
        loss = 0.5 * torch.log(1 + (residual / self.scale) ** 2)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss with configurable beta.

    Similar to Huber but with different parameterization.

    Args:
        beta: Threshold for switching between L1 and L2 regions.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        diff = torch.abs(predictions - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta,
            diff - 0.5 * self.beta,
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss.

    Full probabilistic loss with learned variance.

    Args:
        reduction: Specifies the reduction to apply.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: Tensor,
        log_variance: Tensor,
        targets: Tensor,
    ) -> Tensor:
        variance = torch.exp(log_variance) + 1e-6
        loss = 0.5 * (torch.log(variance) + (targets - predictions) ** 2 / variance)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LaplaceNLLLoss(nn.Module):
    """
    Laplace Negative Log-Likelihood Loss.

    Full probabilistic loss with Laplace distribution.

    Args:
        reduction: Specifies the reduction to apply.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: Tensor,
        log_scale: Tensor,
        targets: Tensor,
    ) -> Tensor:
        scale = torch.exp(log_scale) + 1e-6
        loss = torch.log(scale) + torch.abs(targets - predictions) / scale

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CompositeLoss(nn.Module):
    """
    Composite Loss combining multiple regression losses.

    Allows combining different loss functions with learnable weights.

    Args:
        loss_factories: List of tuples (loss_name, loss_fn, initial_weight).
    """

    def __init__(
        self,
        loss_factories: Optional[List[Tuple[str, nn.Module, float]]] = None,
    ):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = nn.Parameter(
            torch.ones(len(loss_factories)) if loss_factories else torch.tensor([])
        )

        if loss_factories:
            for name, loss_fn, _ in loss_factories:
                self.losses[name] = loss_fn

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        if len(self.losses) == 0:
            return F.mse_loss(predictions, targets)

        weights = F.softmax(self.weights, dim=0)
        total_loss = 0.0

        for i, (name, loss_fn) in enumerate(self.losses.items()):
            loss_val = loss_fn(predictions, targets)
            total_loss += weights[i] * loss_val

        return total_loss


class TrimmedLoss(nn.Module):
    """
    Trimmed Loss ignoring outliers.

    Computes loss on the alpha fraction of samples with smallest residuals.

    Args:
        trim_fraction: Fraction of samples to ignore (0-1).
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        trim_fraction: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.trim_fraction = trim_fraction
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        residuals = (predictions - targets).abs()

        if predictions.dim() > 1:
            residuals = residuals.flatten()
            predictions_flat = predictions.flatten()
            targets_flat = targets.flatten()

            k = int(self.trim_fraction * residuals.numel())
            if k > 0:
                sorted_residuals, indices = torch.topk(
                    residuals, residuals.size(0) - k, largest=False
                )
                loss = sorted_residuals
            else:
                loss = residuals
        else:
            k = int(self.trim_fraction * residuals.size(0))
            if k > 0:
                sorted_residuals, _ = torch.sort(residuals)
                loss = sorted_residuals[k:]
            else:
                loss = residuals

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


__all__ = [
    "HuberLoss",
    "AdaptiveHuberLoss",
    "LogCoshLoss",
    "QuantileLoss",
    "MultiQuantileLoss",
    "TweedieLoss",
    "PseudoHuberLoss",
    "CauchyLoss",
    "SmoothL1Loss",
    "GaussianNLLLoss",
    "LaplaceNLLLoss",
    "CompositeLoss",
    "TrimmedLoss",
]

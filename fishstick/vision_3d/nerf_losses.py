"""
NeRF-specific Loss Functions

RGB, MSE, PSNR losses for NeRF training.
"""

from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RGBLoss(nn.Module):
    """
    RGB reconstruction loss.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted RGB [B, 3] or [B, H, W, 3]
            target: Target RGB [B, 3] or [B, H, W, 3]

        Returns:
            Loss value
        """
        loss = torch.abs(pred - target)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MSE(nn.Module):
    """
    Mean Squared Error loss.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted values
            target: Target values

        Returns:
            MSE loss
        """
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss


class PSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio metric (negative = loss).
    """

    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute PSNR.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            PSNR value (higher is better)
        """
        mse = F.mse_loss(pred, target)
        psnr = 10 * torch.log10(self.max_val**2 / mse)
        return psnr

    def psnr_to_loss(self, psnr: Tensor) -> Tensor:
        """Convert PSNR to loss (lower is better)."""
        return -psnr


class NerfLoss(nn.Module):
    """
    Combined loss for NeRF training.
    """

    def __init__(
        self,
        rgb_weight: float = 1.0,
        mse_weight: float = 0.0,
        use_ssim: bool = False,
    ):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.mse_weight = mse_weight

        self.rgb_loss = RGBLoss()
        self.mse_loss = MSE()

        self.use_ssim = use_ssim

    def forward(
        self,
        pred_rgb: Tensor,
        target_rgb: Tensor,
        pred_depth: Optional[Tensor] = None,
        target_depth: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute NeRF loss.

        Args:
            pred_rgb: Predicted RGB [B, 3]
            target_rgb: Target RGB [B, 3]
            pred_depth: Predicted depth (optional)
            target_depth: Target depth (optional)

        Returns:
            Total loss
        """
        loss = 0.0

        if self.rgb_weight > 0:
            loss += self.rgb_weight * self.rgb_loss(pred_rgb, target_rgb)

        if self.mse_weight > 0 and pred_depth is not None and target_depth is not None:
            loss += self.mse_weight * self.mse_loss(pred_depth, target_depth)

        return loss


class DistortionLoss(nn.Module):
    """
    Distortion loss for efficient NeRF training.
    """

    def __init__(self):
        super().__init__()

    def forward(self, weights: Tensor, t_vals: Tensor) -> Tensor:
        """
        Compute distortion regularization.

        Args:
            weights: Sample weights [B, N]
            t_vals: Distance values [B, N]

        Returns:
            Distortion loss
        """
        B, N = weights.shape

        w = weights.unsqueeze(-1)
        t = t_vals.unsqueeze(-1)

        dt = (t[:, 1:] - t[:, :-1]).unsqueeze(-1)

        loss = w[:, :-1] * w[:, 1:] * dt

        return loss.sum()

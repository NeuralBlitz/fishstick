"""
Switch Normalization layer.

SwitchNorm normalizes activations by dynamically switching between different
normalization methods (batch, instance, layer) based on learned gating weights.
This allows the network to adaptively choose the most effective normalization
strategy for each layer.

Reference: https://arxiv.org/abs/1911.08613
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwitchNorm(nn.Module):
    """
    Switch Normalization layer.

    Combines batch normalization, instance normalization, and layer normalization
    with learnable gating weights to adaptively select the best normalization
    strategy for each input.

    Args:
        num_features: Number of channels in the input
        eps: Small constant for numerical stability (default: 1e-5)
        momentum: Momentum for running statistics (default: 0.1)
        use_bias: Whether to include learnable bias (default: True)
        track_running_stats: Whether to track running statistics (default: True)

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) same shape

    Example:
        >>> layer = SwitchNorm(num_features=64)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = layer(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        use_bias: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.use_bias = use_bias
        self.track_running_stats = track_running_stats

        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.w_bn = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.w_in = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.w_ln = nn.Parameter(torch.ones(1, 1, 1, 1))

        if track_running_stats:
            self.register_buffer("running_mean_bn", torch.zeros(1, num_features, 1, 1))
            self.register_buffer("running_var_bn", torch.ones(1, num_features, 1, 1))
            self.running_mean_bn: Optional[Tensor]
            self.running_var_bn: Optional[Tensor]
        else:
            self.running_mean_bn = None
            self.running_var_bn = None

    def _compute_bn_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute batch normalization statistics."""
        if self.training or not self.track_running_stats:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            if self.track_running_stats and self.training:
                self.running_mean_bn = (
                    self.momentum * mean.detach()
                    + (1 - self.momentum) * self.running_mean_bn
                )
                self.running_var_bn = (
                    self.momentum * var.detach()
                    + (1 - self.momentum) * self.running_var_bn
                )
            return mean, var
        else:
            return self.running_mean_bn, self.running_var_bn

    def _compute_in_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute instance normalization statistics."""
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        return mean, var

    def _compute_ln_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute layer normalization statistics."""
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        return mean, var

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with adaptive normalization switching.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Normalized tensor of same shape
        """
        bn_mean, bn_var = self._compute_bn_stats(x)
        in_mean, in_var = self._compute_in_stats(x)
        ln_mean, ln_var = self._compute_ln_stats(x)

        w_total = self.w_bn + self.w_in + self.w_ln
        w_bn = self.w_bn / w_total
        w_in = self.w_in / w_total
        w_ln = self.w_ln / w_total

        bn_norm = (x - bn_mean) / torch.sqrt(bn_var + self.eps)
        in_norm = (x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_norm = (x - ln_mean) / torch.sqrt(ln_var + self.eps)

        output = w_bn * bn_norm + w_in * in_norm + w_ln * ln_norm

        if self.use_bias:
            return self.weight * output + self.bias
        return self.weight * output

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}"
        )


class SwitchNorm1d(SwitchNorm):
    """SwitchNorm for 1D inputs (sequences)."""

    def _compute_bn_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training or not self.track_running_stats:
            mean = x.mean(dim=(0, 2), keepdim=True)
            var = x.var(dim=(0, 2), keepdim=True, unbiased=False)
            if self.track_running_stats and self.training:
                self.running_mean_bn = (
                    self.momentum * mean.detach()
                    + (1 - self.momentum) * self.running_mean_bn
                )
                self.running_var_bn = (
                    self.momentum * var.detach()
                    + (1 - self.momentum) * self.running_var_bn
                )
            return mean, var
        else:
            return self.running_mean_bn, self.running_var_bn

    def _compute_in_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        return mean, var

    def _compute_ln_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        return mean, var


class SwitchNorm3d(SwitchNorm):
    """SwitchNorm for 3D inputs (volumes)."""

    def _compute_bn_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.training or not self.track_running_stats:
            mean = x.mean(dim=(0, 2, 3, 4), keepdim=True)
            var = x.var(dim=(0, 2, 3, 4), keepdim=True, unbiased=False)
            if self.track_running_stats and self.training:
                self.running_mean_bn = (
                    self.momentum * mean.detach()
                    + (1 - self.momentum) * self.running_mean_bn
                )
                self.running_var_bn = (
                    self.momentum * var.detach()
                    + (1 - self.momentum) * self.running_var_bn
                )
            return mean, var
        else:
            return self.running_mean_bn, self.running_var_bn

    def _compute_in_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        return mean, var

    def _compute_ln_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        var = x.var(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
        return mean, var


def create_switch_norm(num_features: int, dim: int = 2, **kwargs) -> SwitchNorm:
    """
    Factory function to create SwitchNorm layers.

    Args:
        num_features: Number of channels
        dim: Dimension of input (1, 2, or 3)
        **kwargs: Additional arguments passed to SwitchNorm

    Returns:
        SwitchNorm layer

    Example:
        >>> layer = create_switch_norm(64, dim=2)
    """
    if dim == 1:
        return SwitchNorm1d(num_features, **kwargs)
    elif dim == 2:
        return SwitchNorm(num_features, **kwargs)
    elif dim == 3:
        return SwitchNorm3d(num_features, **kwargs)
    else:
        raise ValueError(f"Invalid dimension: {dim}. Choose from 1, 2, 3.")


from typing import Tuple

"""
Filter Response Synchronization (FilterResponseSync) normalization layer.

FilterResponseSync normalizes activations across spatial dimensions but synchronizes
statistics across channels/filter dimension to prevent collapse in depthwise separable
convolutions. Based on "Filter Response Normalization Layer" by Singh et al.

Reference: https://arxiv.org/abs/1911.09737
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FilterResponseSync(nn.Module):
    """
    Filter Response Synchronization Normalization.

    Normalizes activations across spatial dimensions (H, W) while synchronizing
    channel-wise statistics to prevent filter collapse in depthwise separable convolutions.

    Args:
        num_features: Number of channels in the input
        eps: Small constant for numerical stability (default: 1e-3)
        learnable: Whether to use learnable beta/gamma (default: True)
        sync_stats: Whether to synchronize statistics across channels (default: True)

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) same shape

    Example:
        >>> layer = FilterResponseSync(num_features=64)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = layer(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        learnable: bool = True,
        sync_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.learnable = learnable
        self.sync_stats = sync_stats

        if learnable:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_buffer("weight", torch.ones(1, num_features, 1, 1))
            self.register_buffer("bias", torch.zeros(1, num_features, 1, 1))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        if self.learnable:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Normalized tensor of same shape
        """
        if self.sync_stats:
            x_mean = x.mean(dim=(2, 3), keepdim=True)
            x_var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        else:
            x_mean = x.mean(dim=(2, 3), keepdim=True)
            x_var = x.var(dim=(2, 3), keepdim=True, unbiased=False)

        x_normalized = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return self.weight * x_normalized + self.bias

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"eps={self.eps}, "
            f"learnable={self.learnable}, "
            f"sync_stats={self.sync_stats}"
        )

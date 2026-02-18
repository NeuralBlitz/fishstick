"""
EvoNorm normalization layers.

Evolutionary Normalization (EvoNorm) combines activation and normalization in a single
learnable operation. Based on "Evolving Normalization-Activation Layers" by Liu et al.

Supports multiple variants:
- EvoNormB0: Batch-independent version with Swish activation
- EvoNormS0: Sample-independent version with Swish activation
- EvoNormB: Batch-dependent version
- EvoNormS: Sample-dependent version

Reference: https://arxiv.org/abs/2004.02967
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EvoNormB0(nn.Module):
    """
    EvoNorm-B0: Batch-independent version with Swish activation.

    Combines batch normalization with Swish activation in a single learnable layer
    that prevents activation blow-up while maintaining representational power.

    Args:
        num_features: Number of channels
        eps: Small constant for numerical stability (default: 1e-3)
        momentum: Momentum for running statistics (default: 0.1)
        affine: Whether to use learnable affine parameters (default: True)

    Shape:
        - Input: (N, C, H, W) or (N, C)
        - Output: Same shape as input
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_buffer("weight", torch.ones(1, num_features, 1, 1))
            self.register_buffer("bias", torch.zeros(1, num_features, 1, 1))

        self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with Swish activation and batch normalization."""
        if self.training:
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            running_var = var.detach()
            self.running_var = (
                self.momentum * running_var + (1 - self.momentum) * self.running_var
            )
        else:
            var = self.running_var

        swish_x = x * torch.sigmoid(x)
        normalized = (swish_x - x.mean(dim=(0, 2, 3), keepdim=True)) / torch.sqrt(
            var + self.eps
        )

        if self.affine:
            return self.weight * normalized + self.bias
        return normalized

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


class EvoNormS0(nn.Module):
    """
    EvoNorm-S0: Sample-independent version with Swish activation.

    Applies instance-level normalization with Swish activation.
    More stable than batch normalization for small batch sizes.

    Args:
        num_features: Number of channels
        eps: Small constant for numerical stability (default: 1e-3)
        affine: Whether to use learnable affine parameters (default: True)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_buffer("weight", torch.ones(1, num_features, 1, 1))
            self.register_buffer("bias", torch.zeros(1, num_features, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with Swish activation and instance normalization."""
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        swish_x = x * torch.sigmoid(x)
        normalized = (swish_x - x.mean(dim=(2, 3), keepdim=True)) / torch.sqrt(
            var + self.eps
        )

        if self.affine:
            return self.weight * normalized + self.bias
        return normalized

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


class EvoNormB(nn.Module):
    """
    EvoNorm-B: Batch-dependent version with learnable activation.

    A more expressive variant that depends on batch statistics.

    Args:
        num_features: Number of channels
        eps: Small constant for numerical stability (default: 1e-3)
        momentum: Momentum for running statistics (default: 0.1)
        affine: Whether to use learnable affine parameters (default: True)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
            self.v = nn.Parameter(torch.ones(1, num_features, 1, 1))
        else:
            self.register_buffer("weight", torch.ones(1, num_features, 1, 1))
            self.register_buffer("bias", torch.zeros(1, num_features, 1, 1))
            self.register_buffer("v", torch.ones(1, num_features, 1, 1))

        self.register_buffer("running_var", torch.ones(1, num_features, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with learnable activation combination."""
        if self.training:
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            running_var = var.detach()
            self.running_var = (
                self.momentum * running_var + (1 - self.momentum) * self.running_var
            )
        else:
            var = self.running_var

        v = torch.sigmoid(self.v) if hasattr(self, "v") else 1.0
        act = x * torch.sigmoid(x)

        normalized = (act - x.mean(dim=(0, 2, 3), keepdim=True)) / torch.sqrt(
            var + self.eps
        )

        if self.affine:
            return self.weight * normalized + self.bias
        return normalized

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


class EvoNormS(nn.Module):
    """
    EvoNorm-S: Sample-dependent version with learnable activation.

    Uses instance statistics with a learnable activation combination.

    Args:
        num_features: Number of channels
        eps: Small constant for numerical stability (default: 1e-3)
        affine: Whether to use learnable affine parameters (default: True)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
            self.v = nn.Parameter(torch.ones(1, num_features, 1, 1))
        else:
            self.register_buffer("weight", torch.ones(1, num_features, 1, 1))
            self.register_buffer("bias", torch.zeros(1, num_features, 1, 1))
            self.register_buffer("v", torch.ones(1, num_features, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with instance normalization and learnable activation."""
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        act = x * torch.sigmoid(x)

        normalized = (act - x.mean(dim=(2, 3), keepdim=True)) / torch.sqrt(
            var + self.eps
        )

        if self.affine:
            return self.weight * normalized + self.bias
        return normalized

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


def create_evo_norm(variant: str, num_features: int, **kwargs) -> nn.Module:
    """
    Factory function to create EvoNorm layers.

    Args:
        variant: One of 'b0', 's0', 'b', 's'
        num_features: Number of channels
        **kwargs: Additional arguments passed to the layer

    Returns:
        EvoNorm layer

    Example:
        >>> layer = create_evo_norm('b0', num_features=64)
    """
    variants = {
        "b0": EvoNormB0,
        "s0": EvoNormS0,
        "b": EvoNormB,
        "s": EvoNormS,
    }

    if variant.lower() not in variants:
        raise ValueError(
            f"Unknown variant: {variant}. Choose from {list(variants.keys())}"
        )

    return variants[variant.lower()](num_features, **kwargs)

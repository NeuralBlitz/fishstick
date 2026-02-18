"""
Advanced activation functions for neural networks.

This module provides implementations of modern activation functions:
- SiLU (Sigmoid Linear Unit) / Swish
- GELU (Gaussian Error Linear Unit)
- Snake (Periodic activation function)

These activations have been shown to improve performance in modern architectures
like Transformers and CNNs.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (SiLU) / Swish activation.

    Computes x * sigmoid(x). Named Swish by Ramachandran et al. and
    SiLU by Hendrycks and Gimpel.

    Properties:
        - Non-monotonic (has a bell-shaped region)
        - Smooth everywhere
        - Self-gated (uses x itself as the gate)

    Reference: https://arxiv.org/abs/1710.05941

    Args:
        inplace: Whether to perform operation in-place (default: False)

    Example:
        >>> act = SiLU()
        >>> x = torch.randn(1, 10)
        >>> output = act(x)
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return torch.sigmoid_(x) * x
        return torch.sigmoid(x) * x

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"


class Swish(nn.Module):
    """
    Swish activation function (alias for SiLU).

    Named and popularized by Ramachandran et al. (2017).
    See SiLU for implementation details.

    Args:
        inplace: Whether to perform operation in-place (default: False)
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return torch.sigmoid_(x) * x
        return torch.sigmoid(x) * x

    def extra_repr(self) -> str:
        return f"inplace={self.inplace}"


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation.

    Computes x * Φ(x) where Φ is the cumulative distribution function
    of the standard normal distribution. This is the default activation
    in GPT, BERT, and other transformer models.

    Properties:
        - Probabilistic interpretation
        - Smooth and non-monotonic in small region
        - Computationally more expensive than ReLU

    Reference: https://arxiv.org/abs/1606.08415

    Args:
        approximate: Which approximation to use ('none', 'tanh', 'exact')
                    - 'none': Use exact GELU (requires erf)
                    - 'tanh': Use tanh approximation (faster, default)

    Example:
        >>> act = GELU()
        >>> x = torch.randn(1, 10)
        >>> output = act(x)
    """

    def __init__(self, approximate: str = "tanh") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        if self.approximate == "none":
            return x * 0.5 * (1.0 + torch.erf(x / (2**0.5)))
        elif self.approximate == "tanh":
            return (
                x
                * 0.5
                * (1.0 + torch.tanh((2**0.5 / (torch.pi**0.5)) * (x + 0.044715 * x**3)))
            )
        else:
            raise ValueError(f"Unknown approximation: {self.approximate}")

    def extra_repr(self) -> str:
        return f"approximate={self.approximate}"


class Snake(nn.Module):
    """
    Snake activation function with learnable frequency parameter.

    Computes x + (1/β) * sin²(βx) where β is a learnable frequency parameter.
    This periodic activation can capture periodic patterns in data.

    Properties:
        - Periodic (inductive bias for periodic patterns)
        - Learns frequency from data when β is learnable
        - Includes residual connection (x + ...)

    Reference: https://arxiv.org/abs/2006.08195

    Args:
        beta: Initial frequency parameter (default: 1.0)
        learnable: Whether β is learnable (default: True)
        periodic: Whether to include periodic term only (no residual, default: False)

    Example:
        >>> act = Snake(beta=1.0, learnable=True)
        >>> x = torch.randn(1, 10)
        >>> output = act(x)
    """

    def __init__(
        self,
        beta: float = 1.0,
        learnable: bool = True,
        periodic: bool = False,
    ) -> None:
        super().__init__()
        self.periodic = periodic

        if learnable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(beta))

    def forward(self, x: Tensor) -> Tensor:
        if self.periodic:
            return torch.sin(self.beta * x)
        return x + (1.0 / self.beta) * torch.sin(self.beta * x) ** 2

    def extra_repr(self) -> str:
        return f"beta={self.beta.item():.4f}, periodic={self.periodic}"


class SnakeBeta(nn.Module):
    """
    Snake-Beta: Alternative formulation with learnable α and β.

    Computes: x + (1/β) * sin²(α + βx)

    Provides more flexibility than standard Snake by including
    both a phase shift (α) and frequency (β) parameter.

    Reference: https://arxiv.org/abs/2202.08714

    Args:
        alpha: Initial phase parameter (default: 0.0)
        beta: Initial frequency parameter (default: 1.0)
        learnable: Whether parameters are learnable (default: True)
    """

    def __init__(
        self,
        alpha: float = 0.0,
        beta: float = 1.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()

        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
            self.register_buffer("beta", torch.tensor(beta))

    def forward(self, x: Tensor) -> Tensor:
        return x + (1.0 / self.beta) * torch.sin(self.alpha + self.beta * x) ** 2


def silu(x: Tensor, inplace: bool = False) -> Tensor:
    """
    Functional SiLU/Swish activation.

    Args:
        x: Input tensor
        inplace: Whether to modify in-place

    Returns:
        Activated tensor
    """
    if inplace:
        return torch.sigmoid_(x) * x
    return torch.sigmoid(x) * x


def gelu(x: Tensor, approximate: str = "tanh") -> Tensor:
    """
    Functional GELU activation.

    Args:
        x: Input tensor
        approximate: Approximation type ('none', 'tanh', 'exact')

    Returns:
        Activated tensor
    """
    if approximate == "none":
        return x * 0.5 * (1.0 + torch.erf(x / (2**0.5)))
    elif approximate == "tanh":
        return (
            x
            * 0.5
            * (1.0 + torch.tanh((2**0.5 / (torch.pi**0.5)) * (x + 0.044715 * x**3)))
        )
    else:
        raise ValueError(f"Unknown approximation: {approximate}")


def snake(
    x: Tensor,
    beta: float = 1.0,
    periodic: bool = False,
) -> Tensor:
    """
    Functional Snake activation.

    Args:
        x: Input tensor
        beta: Frequency parameter
        periodic: Whether to use periodic-only form

    Returns:
        Activated tensor
    """
    if periodic:
        return torch.sin(beta * x)
    return x + (1.0 / beta) * torch.sin(beta * x) ** 2


def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Factory function to get activation by name.

    Args:
        name: Activation name ('silu', 'swish', 'gelu', 'snake', 'relu', 'leaky_relu', etc.)
        **kwargs: Arguments passed to activation constructor

    Returns:
        Activation module

    Example:
        >>> act = get_activation('gelu', approximate='tanh')
    """
    activations = {
        "silu": SiLU,
        "swish": Swish,
        "gelu": GELU,
        "snake": Snake,
        "snake_beta": SnakeBeta,
    }

    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Choose from {list(activations.keys())}"
        )

    return activations[name_lower](**kwargs)

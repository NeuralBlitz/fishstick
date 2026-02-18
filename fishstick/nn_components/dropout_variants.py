"""
Dropout variants for neural networks.

This module provides advanced dropout techniques:
- SpatialDropout: Drops entire channels
- DropBlock: Drops contiguous regions of features
- ConcreteDropout: Learnable dropout with continuous relaxation

These variants address limitations of standard dropout in convolutional
and recurrent networks.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialDropout(nn.Module):
    """
    Spatial Dropout.

    Drops entire feature maps (channels) instead of individual elements.
    More effective than standard dropout for convolutional networks as
    it forces the network to learn redundant representations across channels.

    Reference: https://arxiv.org/abs/1411.4280

    Args:
        p: Probability of dropping a channel (default: 0.5)
        inplace: Whether to modify in-place (default: False)

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) same shape

    Example:
        >>> dropout = SpatialDropout(p=0.2)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = dropout(x)
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        if self.inplace:
            mask = (torch.rand(x.size(1), device=x.device) > self.p).view(1, -1, 1, 1)
            x.mul_(mask)
            x.div_(1 - self.p)
        else:
            mask = (torch.rand(x.size(1), device=x.device) > self.p).view(1, -1, 1, 1)
            x = x * mask.float() / (1 - self.p)

        return x

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


class SpatialDropout1d(SpatialDropout):
    """Spatial Dropout for 1D inputs (sequences)."""

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        mask = (torch.rand(x.size(1), device=x.device) > self.p).view(1, -1, 1)
        x = x * mask.float() / (1 - self.p)
        return x


class SpatialDropout3d(SpatialDropout):
    """Spatial Dropout for 3D inputs (volumes)."""

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        mask = (torch.rand(x.size(1), device=x.device) > self.p).view(1, -1, 1, 1, 1)
        x = x * mask.float() / (1 - self.p)
        return x


class DropBlock(nn.Module):
    """
    DropBlock regularization.

    Drops contiguous regions of feature maps rather than individual elements.
    More effective than SpatialDropout for convolutional networks as it
    forces the network to learn more distributed representations.

    Reference: https://arxiv.org/abs/1710.05234

    Args:
        p: Probability of dropping a region (default: 0.1)
        block_size: Size of the block to drop (default: 7)
        gamma: Probability of dropping each position (computed if None)
        inplace: Whether to modify in-place (default: False)

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) same shape

    Example:
        >>> dropout = DropBlock(p=0.1, block_size=7)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = dropout(x)
    """

    def __init__(
        self,
        p: float = 0.1,
        block_size: int = 7,
        gamma: Optional[float] = None,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

        if gamma is None:
            self.gamma = p * (block_size**2) / ((32 - block_size + 1) ** 2)
        else:
            self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

        _, _, H, W = x.shape

        gamma = self.gamma

        mask = (torch.rand(x.size(0), *x.size()[2:], device=x.device) >= gamma).float()
        mask = mask.unsqueeze(1)

        if self.block_size > 1:
            block_mask = self._compute_block_mask(mask, self.block_size)
        else:
            block_mask = mask

        mask = mask.float() * block_mask

        if self.inplace:
            x.mul_(mask)
            x.div_(1 - self.p)
        else:
            x = x * mask / (1 - self.p)

        return x

    def _compute_block_mask(self, mask: Tensor, block_size: int) -> Tensor:
        """Compute block mask using max pooling."""
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        mask = mask.float()

        block_mask = F.max_pool2d(
            1 - mask,
            kernel_size=(block_size, block_size),
            stride=(1, 1),
            padding=block_size // 2,
        )

        block_mask = 1 - block_mask
        block_mask = block_mask.repeat_interleave(x.size(1), dim=1)

        return block_mask

    def extra_repr(self) -> str:
        return f"p={self.p}, block_size={self.block_size}, gamma={self.gamma:.4f}"


class DropBlock2d(DropBlock):
    """DropBlock for 2D inputs (alias for DropBlock)."""

    pass


class DropBlock1d(nn.Module):
    """DropBlock for 1D inputs (sequences)."""

    def __init__(
        self,
        p: float = 0.1,
        block_size: int = 7,
        gamma: Optional[float] = None,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

        if gamma is None:
            self.gamma = p * block_size / (32 - block_size + 1)
        else:
            self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")

        gamma = self.gamma
        mask = (torch.rand(x.size(0), x.size(2), device=x.device) >= gamma).float()
        mask = mask.unsqueeze(1)

        mask = 1 - F.max_pool1d(
            1 - mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )

        mask = mask.unsqueeze(1).repeat_interleave(x.size(1), dim=1)
        x = x * mask / (1 - self.p)

        return x


class DropBlock3d(nn.Module):
    """DropBlock for 3D inputs (volumes)."""

    def __init__(
        self,
        p: float = 0.1,
        block_size: int = 5,
        gamma: Optional[float] = None,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.p = p
        self.block_size = block_size
        self.inplace = inplace

        if gamma is None:
            self.gamma = p * (block_size**3) / ((32 - block_size + 1) ** 3)
        else:
            self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        if x.dim() != 5:
            raise ValueError(f"Expected 5D input, got {x.dim()}D")

        gamma = self.gamma
        mask = (torch.rand(x.size(0), *x.size()[2:], device=x.device) >= gamma).float()
        mask = mask.unsqueeze(1)

        mask = 1 - F.max_pool3d(
            1 - mask,
            kernel_size=(self.block_size,) * 3,
            stride=(1, 1, 1),
            padding=self.block_size // 2,
        )

        mask = 1 - mask
        mask = mask.repeat_interleave(x.size(1), dim=1)
        x = x * mask / (1 - self.p)

        return x


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout (Continuous Dropout).

    Implements a continuous relaxation of the dropout mask, allowing
    the dropout probability to be learned via gradient descent.
    The mask is sampled from a concrete (Gumbel-Softmax) distribution.

    Reference: https://arxiv.org/abs/1705.07832

    Args:
        p: Initial dropout probability (can be learned)
        weight: Regularization weight for dropout (default: 1e-6)
        from_list: Whether to store p in a learnable parameter (default: True)

    Shape:
        - Input: Any shape
        - Output: Same shape

    Example:
        >>> dropout = ConcreteDropout(p=0.1, weight=1e-6)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = dropout(x)
    """

    def __init__(
        self,
        p: float = 0.1,
        weight: float = 1e-6,
        from_list: bool = True,
    ) -> None:
        super().__init__()

        if from_list:
            self.p = nn.Parameter(torch.tensor(p))
        else:
            self.register_buffer("p", torch.tensor(p))

        self.weight = weight
        self.from_list = from_list

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x

        if self.p.item() <= 0.0 or self.p.item() >= 1.0:
            return x

        p = torch.sigmoid(self.p)

        temp = 0.1

        log_prob = torch.log(p + 1e-8)
        log_one_minus_prob = torch.log(1 - p + 1e-8)

        logits = log_prob - log_one_minus_prob

        random_noise = torch.rand_like(x)

        logistic_noise = torch.log(random_noise + 1e-8) - torch.log(
            1 - random_noise + 1e-8
        )

        drop_prob = logits + logistic_noise

        sample = torch.sigmoid(drop_prob / temp)

        ones = torch.ones_like(x)

        masked = sample * x + (ones - sample) * x.detach()

        expected = p * x + (1 - p) * x.detach()
        diff = (masked - expected) ** 2
        regularization = diff.mean() * self.weight

        if not hasattr(self, "reg"):
            self.reg = regularization
        else:
            self.reg = regularization

        return masked / (p + 1e-8) * x

    def extra_repr(self) -> str:
        return (
            f"p={self.p.item():.4f}, weight={self.weight}, from_list={self.from_list}"
        )


class Dropout2dVariant(nn.Module):
    """
    Dropout2d variant with learned mask patterns.

    Alternative implementation of channel dropout using learned
    dropping patterns.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x

        channels = x.size(1)

        mask = (torch.rand(channels, device=x.device) > self.p).float()
        mask = mask.view(1, -1, 1, 1)

        return x * mask / (1 - self.p)


class AutoDropout(nn.Module):
    """
    AutoDropout: Learnable dropout scheduling.

    Automatically learns the optimal dropout rate for each layer
    using a shared learnable parameter.

    Reference: https://arxiv.org/abs/1908.01878
    """

    def __init__(
        self,
        p_init: float = 0.1,
        learnable: bool = True,
    ) -> None:
        super().__init__()

        if learnable:
            logit_p = torch.logit(torch.tensor(p_init))
            self.logit_p = nn.Parameter(logit_p)
        else:
            self.register_buffer("logit_p", torch.logit(torch.tensor(p_init)))

    @property
    def p(self) -> float:
        return torch.sigmoid(self.logit_p).item()

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0:
            return x

        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)


def create_dropout(dropout_type: str, p: float = 0.5, **kwargs) -> nn.Module:
    """
    Factory function to create dropout layers.

    Args:
        dropout_type: Type of dropout ('spatial', 'dropblock', 'concrete', 'auto')
        p: Dropout probability
        **kwargs: Additional arguments

    Returns:
        Dropout module

    Example:
        >>> dropout = create_dropout('spatial', p=0.2)
    """
    if dropout_type == "spatial":
        return SpatialDropout(p=p, **kwargs)
    elif dropout_type == "dropblock":
        return DropBlock(p=p, **kwargs)
    elif dropout_type == "concrete":
        return ConcreteDropout(p=p, **kwargs)
    elif dropout_type == "auto":
        return AutoDropout(p_init=p, **kwargs)
    else:
        raise ValueError(f"Unknown dropout type: {dropout_type}")

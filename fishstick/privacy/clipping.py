"""
Gradient Clipping Mechanisms for Differential Privacy.

This module provides various gradient clipping strategies used in DP-SGD
to bound the sensitivity of the gradient updates.

Example:
    >>> from fishstick.privacy import GradientClipper, PerLayerClipper
    >>>
    >>> clipper = GradientClipper(max_norm=1.0)
    >>> clipped_grads = clipper.clip(grads)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

Tensor = torch.Tensor
Module = nn.Module


class GradientClipper(ABC):
    """Abstract base class for gradient clipping.

    Example:
        >>> class MyClipper(GradientClipper):
        ...     def clip(self, gradients: Tensor) -> Tuple[Tensor, float]:
        ...         pass
    """

    @abstractmethod
    def clip(
        self,
        gradients: Union[Tensor, List[Tensor], Dict[str, Tensor]],
    ) -> Tuple[Union[Tensor, List[Tensor], Dict[str, Tensor]], float]:
        """Clip gradients to bounded norm.

        Args:
            gradients: Input gradients to clip.

        Returns:
            Tuple of (clipped gradients, total norm before clipping).
        """
        pass

    @abstractmethod
    def clip_per_layer(
        self,
        gradients: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
        """Clip gradients per layer.

        Args:
            gradients: Dictionary of layer gradients.

        Returns:
            Tuple of (clipped gradients, norms per layer).
        """
        pass


class StaticClipper(GradientClipper):
    """Static gradient clipping with fixed norm bound.

    Clips all gradients to have maximum L2 norm of max_norm.
    This is the standard approach used in DP-SGD.

    Reference:
        Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.

    Args:
        max_norm: Maximum L2 norm for gradients.
        norm_type: Type of norm (default: L2).

    Example:
        >>> clipper = StaticClipper(max_norm=1.0)
        >>> clipped, norm = clipper.clip(gradient)
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
    ):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(
        self,
        gradients: Union[Tensor, List[Tensor], Dict[str, Tensor]],
    ) -> Tuple[Union[Tensor, List[Tensor], Dict[str, Tensor]], float]:
        """Clip gradients to bounded norm.

        Args:
            gradients: Input gradients.

        Returns:
            Tuple of (clipped gradients, total norm).
        """
        if isinstance(gradients, Tensor):
            return self._clip_single(gradients)
        elif isinstance(gradients, list):
            return self._clip_list(gradients)
        elif isinstance(gradients, dict):
            return self._clip_dict(gradients)
        else:
            raise TypeError(f"Unsupported gradient type: {type(gradients)}")

    def _clip_single(self, grad: Tensor) -> Tuple[Tensor, float]:
        """Clip a single gradient tensor.

        Args:
            grad: Gradient tensor.

        Returns:
            Tuple of (clipped gradient, original norm).
        """
        grad_flat = grad.view(-1)
        norm = grad_flat.norm(p=self.norm_type)

        if norm > self.max_norm:
            scale = self.max_norm / norm
            grad = grad * scale

        return grad, norm.item()

    def _clip_list(
        self,
        grads: List[Tensor],
    ) -> Tuple[List[Tensor], float]:
        """Clip a list of gradients.

        Args:
            grads: List of gradient tensors.

        Returns:
            Tuple of (clipped gradients, total norm).
        """
        norms = [g.view(-1).norm(p=self.norm_type) for g in grads]
        total_norm = torch.stack(norms).norm(p=self.norm_type)

        if total_norm > self.max_norm:
            scale = self.max_norm / total_norm
            grads = [g * scale for g in grads]

        return grads, total_norm.item()

    def _clip_dict(
        self,
        grads: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], float]:
        """Clip a dictionary of gradients.

        Args:
            grads: Dictionary of layer gradients.

        Returns:
            Tuple of (clipped gradients, total norm).
        """
        norms = [g.view(-1).norm(p=self.norm_type) for g in grads.values()]
        total_norm = torch.stack(norms).norm(p=self.norm_type)

        clipped = {}
        for name, grad in grads.items():
            if total_norm > self.max_norm:
                scale = self.max_norm / total_norm
                clipped[name] = grad * scale
            else:
                clipped[name] = grad

        return clipped, total_norm.item()

    def clip_per_layer(
        self,
        gradients: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
        """Clip each layer's gradient independently.

        Args:
            gradients: Dictionary of layer gradients.

        Returns:
            Tuple of (clipped gradients, norms per layer).
        """
        clipped = {}
        norms = {}

        for name, grad in gradients.items():
            grad_flat = grad.view(-1)
            norm = grad_flat.norm(p=self.norm_type)
            norms[name] = norm.item()

            if norm > self.max_norm:
                scale = self.max_norm / norm
                clipped[name] = grad * scale
            else:
                clipped[name] = grad

        return clipped, norms

    def __repr__(self) -> str:
        return f"StaticClipper(max_norm={self.max_norm}, norm_type={self.norm_type})"


class AdaptiveClipper(GradientClipper):
    """Adaptive gradient clipping that adjusts bounds based on statistics.

    Uses running statistics of gradient norms to adaptively adjust
    the clipping threshold for better utility.

    Reference:
        Andrew et al., "Differentially Private Deep Learning", JMLR 2021.

    Args:
        initial_norm: Initial clipping norm.
        adaptation_rate: Rate of adaptation for moving average.
        norm_type: Type of norm.
        warmup_steps: Number of warmup steps before adaptation.

    Example:
        >>> clipper = AdaptiveClipper(initial_norm=1.0, adaptation_rate=0.1)
        >>> clipped, norm = clipper.clip(gradient)
    """

    def __init__(
        self,
        initial_norm: float = 1.0,
        adaptation_rate: float = 0.1,
        norm_type: float = 2.0,
        warmup_steps: int = 100,
    ):
        self.max_norm = initial_norm
        self.adaptation_rate = adaptation_rate
        self.norm_type = norm_type
        self.warmup_steps = warmup_steps

        self._step = 0
        self._moving_avg_norm = None

    def clip(
        self,
        gradients: Union[Tensor, List[Tensor], Dict[str, Tensor]],
    ) -> Tuple[Union[Tensor, List[Tensor], Dict[str, Tensor]], float]:
        """Clip gradients with adaptive norm.

        Args:
            gradients: Input gradients.

        Returns:
            Tuple of (clipped gradients, total norm).
        """
        if isinstance(gradients, Tensor):
            clipped, norm = self._clip_single(gradients)
        elif isinstance(gradients, list):
            clipped, norm = self._clip_list(gradients)
        elif isinstance(gradients, dict):
            clipped, norm = self._clip_dict(gradients)
        else:
            raise TypeError(f"Unsupported gradient type: {type(gradients)}")

        self._update_adaptive_norm(norm)
        self._step += 1

        return clipped, norm

    def _clip_single(self, grad: Tensor) -> Tuple[Tensor, float]:
        """Clip a single gradient tensor."""
        grad_flat = grad.view(-1)
        norm = grad_flat.norm(p=self.norm_type)

        if norm > self.max_norm:
            scale = self.max_norm / norm
            grad = grad * scale

        return grad, norm.item()

    def _clip_list(self, grads: List[Tensor]) -> Tuple[List[Tensor], float]:
        """Clip a list of gradients."""
        norms = [g.view(-1).norm(p=self.norm_type) for g in grads]
        total_norm = torch.stack(norms).norm(p=self.norm_type)

        if total_norm > self.max_norm:
            scale = self.max_norm / total_norm
            grads = [g * scale for g in grads]

        return grads, total_norm.item()

    def _clip_dict(self, grads: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], float]:
        """Clip a dictionary of gradients."""
        norms = [g.view(-1).norm(p=self.norm_type) for g in grads.values()]
        total_norm = torch.stack(norms).norm(p=self.norm_type)

        clipped = {}
        for name, grad in grads.items():
            if total_norm > self.max_norm:
                scale = self.max_norm / total_norm
                clipped[name] = grad * scale
            else:
                clipped[name] = grad

        return clipped, total_norm.item()

    def _update_adaptive_norm(self, norm: float) -> None:
        """Update moving average of gradient norm."""
        if self._step < self.warmup_steps:
            return

        if self._moving_avg_norm is None:
            self._moving_avg_norm = norm
        else:
            self._moving_avg_norm = (
                1 - self.adaptation_rate
            ) * self._moving_avg_norm + self.adaptation_rate * norm

        self.max_norm = self._moving_avg_norm

    def clip_per_layer(
        self,
        gradients: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
        """Clip each layer's gradient independently.

        Args:
            gradients: Dictionary of layer gradients.

        Returns:
            Tuple of (clipped gradients, norms per layer).
        """
        clipped = {}
        norms = {}

        for name, grad in gradients.items():
            grad_flat = grad.view(-1)
            norm = grad_flat.norm(p=self.norm_type)
            norms[name] = norm.item()

            if norm > self.max_norm:
                scale = self.max_norm / norm
                clipped[name] = grad * scale
            else:
                clipped[name] = grad

        total_norm = torch.stack(torch.tensor(list(norms.values()))).norm(
            p=self.norm_type
        )
        self._update_adaptive_norm(total_norm.item())
        self._step += 1

        return clipped, norms

    def __repr__(self) -> str:
        return f"AdaptiveClipper(initial_norm={self.max_norm}, step={self._step})"


class PerLayerClipper(GradientClipper):
    """Per-layer gradient clipping with individual norms.

    Clips each layer's gradient to its own norm bound, which can
    improve utility when different layers have different gradient scales.

    Args:
        max_norms: Dictionary of max norms per layer, or float for all layers.
        norm_type: Type of norm.

    Example:
        >>> clipper = PerLayerClipper({'layer1': 1.0, 'layer2': 2.0})
        >>> clipped, norms = clipper.clip_per_layer(gradients)
    """

    def __init__(
        self,
        max_norms: Union[float, Dict[str, float]],
        norm_type: float = 2.0,
    ):
        self.norm_type = norm_type

        if isinstance(max_norms, float):
            self.max_norms: Dict[str, float] = {}
            self._default_norm = max_norms
        else:
            self.max_norms = max_norms
            self._default_norm = 1.0

    def set_layer_norm(self, name: str, max_norm: float) -> None:
        """Set clipping norm for a specific layer.

        Args:
            name: Layer name.
            max_norm: Maximum norm for this layer.
        """
        self.max_norms[name] = max_norm

    def get_layer_norm(self, name: str) -> float:
        """Get clipping norm for a layer.

        Args:
            name: Layer name.

        Returns:
            Maximum norm for this layer.
        """
        return self.max_norms.get(name, self._default_norm)

    def clip(
        self,
        gradients: Union[Tensor, List[Tensor], Dict[str, Tensor]],
    ) -> Tuple[Union[Tensor, List[Tensor], Dict[str, Tensor]], float]:
        """Clip gradients (applies same norm to all).

        Args:
            gradients: Input gradients.

        Returns:
            Tuple of (clipped gradients, total norm).
        """
        if isinstance(gradients, dict):
            return self._clip_dict(gradients)
        else:
            raise TypeError("PerLayerClipper requires dict input")

    def _clip_dict(
        self,
        grads: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], float]:
        """Clip a dictionary of gradients with per-layer norms.

        Args:
            grads: Dictionary of layer gradients.

        Returns:
            Tuple of (clipped gradients, total norm).
        """
        clipped = {}
        total_norm_sq = 0.0

        for name, grad in grads.items():
            grad_flat = grad.view(-1)
            norm = grad_flat.norm(p=self.norm_type)
            max_norm = self.get_layer_norm(name)

            if norm > max_norm:
                scale = max_norm / norm
                clipped[name] = grad * scale
                total_norm_sq += max_norm**2
            else:
                clipped[name] = grad
                total_norm_sq += norm.item() ** 2

        total_norm = total_norm_sq**0.5
        return clipped, total_norm

    def clip_per_layer(
        self,
        gradients: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
        """Clip each layer's gradient to its own norm.

        Args:
            gradients: Dictionary of layer gradients.

        Returns:
            Tuple of (clipped gradients, norms per layer).
        """
        clipped = {}
        norms = {}

        for name, grad in gradients.items():
            grad_flat = grad.view(-1)
            norm = grad_flat.norm(p=self.norm_type)
            norms[name] = norm.item()
            max_norm = self.get_layer_norm(name)

            if norm > max_norm:
                scale = max_norm / norm
                clipped[name] = grad * scale
            else:
                clipped[name] = grad

        return clipped, norms

    def __repr__(self) -> str:
        return f"PerLayerClipper(norms={self.max_norms}, default={self._default_norm})"


class NormAccountingClipper(StaticClipper):
    """Gradient clipper with per-sample norm accounting.

    Computes norms per sample in a batch for better privacy accounting.

    Args:
        max_norm: Maximum L2 norm per sample.
        norm_type: Type of norm.
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
    ):
        super().__init__(max_norm, norm_type)
        self._sample_norms: List[float] = []

    def clip(
        self,
        gradients: Union[Tensor, List[Tensor], Dict[str, Tensor]],
    ) -> Tuple[Union[Tensor, List[Tensor], Dict[str, Tensor]], float]:
        """Clip gradients with per-sample accounting.

        Args:
            gradients: Input gradients (first dim is sample batch).

        Returns:
            Tuple of (clipped gradients, max sample norm).
        """
        if isinstance(gradients, Tensor):
            return self._clip_per_sample(gradients)
        else:
            raise TypeError("NormAccountingClipper requires tensor input")

    def _clip_per_sample(self, grad: Tensor) -> Tuple[Tensor, float]:
        """Clip gradients per sample.

        Args:
            grad: Gradient tensor with first dim being batch.

        Returns:
            Tuple of (clipped gradient, max sample norm).
        """
        batch_size = grad.shape[0]
        flat = grad.view(batch_size, -1)
        norms = flat.norm(p=self.norm_type, dim=1)

        self._sample_norms = norms.tolist()

        max_norm = norms.max()
        if max_norm > self.max_norm:
            scale = self.max_norm / max_norm
            grad = grad * scale.view(-1, *([1] * (grad.dim() - 1)))

        return grad, max_norm.item()

    def get_sample_norms(self) -> List[float]:
        """Get norms for each sample in last batch.

        Returns:
            List of sample norms.
        """
        return self._sample_norms.copy()

    def get_avg_norm(self) -> float:
        """Get average sample norm.

        Returns:
            Average norm across samples.
        """
        if not self._sample_norms:
            return 0.0
        return sum(self._sample_norms) / len(self._sample_norms)


def create_clipper(
    clipper_type: str = "static",
    max_norm: float = 1.0,
    **kwargs,
) -> GradientClipper:
    """Factory function to create gradient clippers.

    Args:
        clipper_type: Type of clipper ('static', 'adaptive', 'per_layer').
        max_norm: Maximum gradient norm.
        **kwargs: Additional arguments.

    Returns:
        Configured gradient clipper.

    Example:
        >>> clipper = create_clipper('static', max_norm=1.0)
    """
    clipper_type = clipper_type.lower()

    if clipper_type == "static":
        return StaticClipper(max_norm, **kwargs)
    elif clipper_type == "adaptive":
        return AdaptiveClipper(max_norm, **kwargs)
    elif clipper_type == "per_layer":
        return PerLayerClipper(max_norm, **kwargs)
    elif clipper_type == "accounting":
        return NormAccountingClipper(max_norm, **kwargs)
    else:
        raise ValueError(f"Unknown clipper type: {clipper_type}")

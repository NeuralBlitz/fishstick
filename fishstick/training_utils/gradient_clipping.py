"""
Gradient Clipping Strategies

Comprehensive gradient clipping utilities including norm clipping, value clipping,
adaptive clipping, and gradient accumulation-aware clipping strategies.
"""

from typing import Optional, Callable, Dict, Any, List, Union, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


class ClipStrategy(Enum):
    """Gradient clipping strategy types."""

    NORM = "norm"
    VALUE = "value"
    ADAPTIVE = "adaptive"
    GLOBAL_NORM = "global_norm"
    PER_LAYER_NORM = "per_layer_norm"


class GradientClipper(ABC):
    """
    Abstract base class for gradient clipping strategies.

    All gradient clippers should inherit from this class and implement
    the clip method.
    """

    @abstractmethod
    def clip(
        self,
        parameters: Union[Iterator[nn.Parameter], List[Tensor]],
    ) -> float:
        """
        Clip gradients and return the gradient norm.

        Args:
            parameters: Model parameters or gradient tensors

        Returns:
            Gradient norm after clipping
        """
        pass

    def __call__(
        self,
        parameters: Union[Iterator[nn.Parameter], List[Tensor]],
    ) -> float:
        """Allow clipper to be called as a function."""
        return self.clip(parameters)


class NormClipper(GradientClipper):
    """
    Gradient norm clipping.

    Clips gradients by their global L2 norm, scaling all gradients
    proportionally when the total norm exceeds the threshold.

    Example:
        >>> clipper = NormClipper(max_norm=1.0, norm_type=2.0)
        >>> grad_norm = clipper(model.parameters())
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        accumulate_over_loss_reduction: bool = True,
    ):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite
        self.accumulate_over_loss_reduction = accumulate_over_loss_reduction

    def clip(
        self,
        parameters: Union[Iterator[nn.Parameter], List[Tensor]],
    ) -> float:
        """
        Clip gradients by global norm.

        Args:
            parameters: Model parameters

        Returns:
            Total gradient norm before clipping
        """
        if isinstance(parameters, list):
            params = parameters
        else:
            params = list(parameters)

        grads = [p.grad for p in params if p.grad is not None]

        if not grads:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), self.norm_type) for g in grads]),
            self.norm_type,
        )

        if self.error_if_nonfinite and not torch.isfinite(total_norm):
            raise ValueError(f"Non-finite gradient norm: {total_norm}")

        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grads:
                g.mul_(clip_coef)

        return total_norm.item()


class ValueClipper(GradientClipper):
    """
    Gradient value clipping.

    Clips gradients by clamping individual gradient values to a specified range.

    Example:
        >>> clipper = ValueClipper(clip_value_min=-1.0, clip_value_max=1.0)
        >>> grad_norm = clipper(model.parameters())
    """

    def __init__(
        self,
        clip_value_min: Optional[float] = None,
        clip_value_max: Optional[float] = None,
        clip_value: Optional[float] = None,
    ):
        if clip_value is not None:
            self.clip_value_min = -clip_value
            self.clip_value_max = clip_value
        else:
            self.clip_value_min = clip_value_min
            self.clip_value_max = clip_value_max

    def clip(
        self,
        parameters: Union[Iterator[nn.Parameter], List[Tensor]],
    ) -> float:
        """
        Clip gradients by value.

        Args:
            parameters: Model parameters

        Returns:
            Maximum absolute gradient value
        """
        if isinstance(parameters, list):
            params = parameters
        else:
            params = list(parameters)

        grads = [p.grad for p in params if p.grad is not None]

        if not grads:
            return 0.0

        max_abs = 0.0
        for g in grads:
            g.clamp_(min=self.clip_value_min, max=self.clip_value_max)
            max_abs = max(max_abs, g.abs().max().item())

        return max_abs


class AdaptiveClipper(GradientClipper):
    """
    Adaptive gradient clipping based on running statistics.

    Automatically adjusts clipping threshold based on gradient magnitude
    statistics, useful for training that spans multiple phases or has
    varying gradient scales.

    Example:
        >>> clipper = AdaptiveClipper(initial_clip_value=1.0, adaptation_rate=0.99)
        >>> grad_norm = clipper(model.parameters())
    """

    def __init__(
        self,
        initial_clip_value: float = 1.0,
        adaptation_rate: float = 0.99,
        clip_value_min: float = 0.1,
        clip_value_max: float = 10.0,
        window_size: int = 100,
    ):
        self.current_clip_value = initial_clip_value
        self.adaptation_rate = adaptation_rate
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max
        self.window_size = window_size

        self.norm_history: List[float] = []
        self.running_mean = 0.0

    def clip(
        self,
        parameters: Union[Iterator[nn.Parameter], List[Tensor]],
    ) -> float:
        """
        Clip gradients adaptively.

        Args:
            parameters: Model parameters

        Returns:
            Gradient norm after clipping
        """
        if isinstance(parameters, list):
            params = parameters
        else:
            params = list(parameters)

        grads = [p.grad for p in params if p.grad is not None]

        if not grads:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(g, 2) for g in grads]),
            2,
        ).item()

        self.norm_history.append(total_norm)
        if len(self.norm_history) > self.window_size:
            self.norm_history.pop(0)

        mean_norm = sum(self.norm_history) / len(self.norm_history)

        target_clip = mean_norm * 1.5
        self.current_clip_value = (
            self.adaptation_rate * self.current_clip_value
            + (1 - self.adaptation_rate) * target_clip
        )
        self.current_clip_value = max(
            self.clip_value_min, min(self.current_clip_value, self.clip_value_max)
        )

        clip_coef = self.current_clip_value / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grads:
                g.mul_(clip_coef)

        return total_norm


class GlobalNormClipper(GradientClipper):
    """
    Global gradient norm clipping with per-parameter type awareness.

    Computes separate norms for different parameter types (weights, biases, etc.)
    and applies appropriate clipping to each group.
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        clip_embed: bool = True,
    ):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.clip_embed = clip_embed

    def clip(
        self,
        parameters: Union[Iterator[nn.Parameter], List[Tensor]],
    ) -> float:
        """
        Clip gradients globally.

        Args:
            parameters: Model parameters

        Returns:
            Total gradient norm
        """
        if isinstance(parameters, list):
            params = parameters
        else:
            params = list(parameters)

        grads = [p.grad for p in params if p.grad is not None]

        if not grads:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(g, self.norm_type) for g in grads]),
            self.norm_type,
        )

        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grads:
                g.mul_(clip_coef)

        return total_norm.item()


class PerLayerNormClipper(GradientClipper):
    """
    Per-layer gradient norm clipping.

    Clips gradients layer-by-layer, allowing different clipping thresholds
    for different layers based on their size or sensitivity.

    Example:
        >>> clipper = PerLayerNormClipper(max_norm=1.0, scale_by_layer_size=True)
        >>> grad_norm = clipper(model.parameters())
    """

    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        scale_by_layer_size: bool = False,
        layer_scale_factors: Optional[Dict[str, float]] = None,
    ):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_by_layer_size = scale_by_layer_size
        self.layer_scale_factors = layer_scale_factors or {}

    def clip(
        self,
        parameters: Union[Iterator[nn.Parameter], List[Tensor]],
    ) -> float:
        """
        Clip gradients per layer.

        Args:
            parameters: Model parameters

        Returns:
            Maximum layer gradient norm
        """
        if isinstance(parameters, list):
            params = parameters
        else:
            params = list(parameters)

        max_norm = 0.0

        current_layer_params = []
        layer_idx = 0

        for p in params:
            if p.grad is None:
                continue

            current_layer_params.append(p.grad)

            if hasattr(p, "shape") and len(p.shape) > 0:
                layer_name = f"layer_{layer_idx // 2}"
                scale = self.layer_scale_factors.get(layer_name, 1.0)

                if self.scale_by_layer_size:
                    numel = p.numel()
                    scale = 1.0 / (numel**0.5)

                layer_norm = torch.norm(p.grad, self.norm_type).item()
                layer_max = self.max_norm * scale

                if layer_norm > layer_max:
                    p.grad.mul_(layer_max / (layer_norm + 1e-6))

                max_norm = max(max_norm, layer_norm)

            layer_idx += 1

        return max_norm


class GradientClipperFactory:
    """
    Factory for creating gradient clippers.

    Provides a convenient interface for creating different types of
    gradient clippers based on configuration.
    """

    _clippers = {
        "norm": NormClipper,
        "value": ValueClipper,
        "adaptive": AdaptiveClipper,
        "global_norm": GlobalNormClipper,
        "per_layer_norm": PerLayerNormClipper,
    }

    @classmethod
    def create(
        cls,
        strategy: Union[str, ClipStrategy],
        **kwargs,
    ) -> GradientClipper:
        """
        Create a gradient clipper.

        Args:
            strategy: Clipping strategy name or enum
            **kwargs: Arguments for the clipper

        Returns:
            Configured gradient clipper

        Example:
            >>> clipper = GradientClipperFactory.create("norm", max_norm=1.0)
            >>> clipper = GradientClipperFactory.create("adaptive", initial_clip_value=2.0)
        """
        if isinstance(strategy, ClipStrategy):
            strategy = strategy.value

        if strategy not in cls._clippers:
            raise ValueError(
                f"Unknown clipping strategy: {strategy}. "
                f"Available: {list(cls._clippers.keys())}"
            )

        return cls._clippers[strategy](**kwargs)

    @classmethod
    def register(cls, name: str, clipper_class: type) -> None:
        """
        Register a custom clipper.

        Args:
            name: Name for the clipper
            clipper_class: Clipper class
        """
        cls._clippers[name] = clipper_class


def clip_gradients(
    parameters: Union[Iterator[nn.Parameter], List[Tensor]],
    clip_type: str = "norm",
    max_norm: float = 1.0,
    norm_type: float = 2.0,
    clip_value_min: Optional[float] = None,
    clip_value_max: Optional[float] = None,
    adaptive: bool = False,
    **kwargs,
) -> float:
    """
    Convenience function for gradient clipping.

    Args:
        parameters: Model parameters
        clip_type: Type of clipping ('norm', 'value', 'adaptive', 'global_norm')
        max_norm: Maximum gradient norm for norm clipping
        norm_type: Norm type (1, 2, or 'inf')
        clip_value_min: Minimum value for value clipping
        clip_value_max: Maximum value for value clipping
        adaptive: Use adaptive clipping
        **kwargs: Additional arguments

    Returns:
        Gradient norm before clipping

    Example:
        >>> grad_norm = clip_gradients(model.parameters(), clip_type="norm", max_norm=1.0)
    """
    if adaptive:
        clipper = AdaptiveClipper(**kwargs)
    elif clip_type == "norm":
        clipper = NormClipper(max_norm=max_norm, norm_type=norm_type)
    elif clip_type == "value":
        clipper = ValueClipper(
            clip_value_min=clip_value_min,
            clip_value_max=clip_value_max,
        )
    elif clip_type == "global_norm":
        clipper = GlobalNormClipper(max_norm=max_norm, norm_type=norm_type)
    else:
        clipper = NormClipper(max_norm=max_norm, norm_type=norm_type)

    return clipper(parameters)


@dataclass
class GradientStats:
    """Container for gradient statistics."""

    total_norm: float = 0.0
    per_layer_norms: Optional[List[float]] = None
    max_norm: float = 0.0
    min_norm: float = 0.0
    mean_norm: float = 0.0
    num_params_with_grad: int = 0
    num_nan_grads: int = 0
    num_inf_grads: int = 0


def compute_gradient_stats(
    parameters: Union[Iterator[nn.Parameter], List[Tensor]],
    norm_type: float = 2.0,
    per_layer: bool = True,
) -> GradientStats:
    """
    Compute comprehensive gradient statistics.

    Args:
        parameters: Model parameters
        norm_type: Norm type for computation
        per_layer: Compute per-layer statistics

    Returns:
        GradientStats object with computed statistics
    """
    if isinstance(parameters, list):
        params = list(parameters)
    else:
        params = list(parameters)

    grads = [p.grad for p in params if p.grad is not None]

    stats = GradientStats()
    stats.num_params_with_grad = len(grads)

    if not grads:
        return stats

    layer_norms = []
    nan_count = 0
    inf_count = 0

    for g in grads:
        layer_norm = torch.norm(g.detach(), norm_type).item()
        layer_norms.append(layer_norm)

        if not torch.isfinite(g).all():
            nan_count += torch.isnan(g).sum().item()
            inf_count += torch.isinf(g).sum().item()

    stats.per_layer_norms = layer_norms if per_layer else None
    stats.total_norm = torch.norm(
        torch.tensor(layer_norms),
        norm_type,
    ).item()
    stats.max_norm = max(layer_norms) if layer_norms else 0.0
    stats.min_norm = min(layer_norms) if layer_norms else 0.0
    stats.mean_norm = sum(layer_norms) / len(layer_norms) if layer_norms else 0.0
    stats.num_nan_grads = nan_count
    stats.num_inf_grads = inf_count

    return stats


class GradientClipperWithStats(GradientClipper):
    """
    Gradient clipper that tracks statistics.

    Combines gradient clipping with comprehensive statistics tracking
    for monitoring and debugging.
    """

    def __init__(
        self,
        clipper: GradientClipper,
        track_stats: bool = True,
        stats_window: int = 100,
    ):
        self.clipper = clipper
        self.track_stats = track_stats
        self.stats_window = stats_window

        self.history: List[GradientStats] = []

    def clip(
        self,
        parameters: Union[Iterator[nn.Parameter], List[Tensor]],
    ) -> float:
        """
        Clip gradients and optionally track statistics.

        Args:
            parameters: Model parameters

        Returns:
            Gradient norm
        """
        if self.track_stats:
            stats = compute_gradient_stats(parameters)
            self.history.append(stats)
            if len(self.history) > self.stats_window:
                self.history.pop(0)

        return self.clipper.clip(parameters)

    def get_average_stats(self) -> Dict[str, float]:
        """Get average statistics over tracked history."""
        if not self.history:
            return {}

        return {
            "avg_total_norm": sum(s.total_norm for s in self.history)
            / len(self.history),
            "avg_max_norm": sum(s.max_norm for s in self.history) / len(self.history),
            "avg_mean_norm": sum(s.mean_norm for s in self.history) / len(self.history),
            "total_clips": sum(1 for s in self.history if s.total_norm > 1.0),
        }

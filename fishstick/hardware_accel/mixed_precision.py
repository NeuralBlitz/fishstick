"""
Mixed Precision Training Utilities for fishstick.

Provides automatic mixed precision (AMP) training, gradient scaling,
and precision management for efficient GPU computation.

Based on:
- NVIDIA Apex AMP
- PyTorch Native AMP
- FP16/BF16 mixed precision training (Micikevicius et al., 2018)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any, List, Callable, Union
from contextlib import contextmanager
import threading

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class Precision(Enum):
    """Precision types for mixed precision training."""

    FP32 = auto()  # Full precision (32-bit float)
    FP16 = auto()  # Half precision (16-bit float)
    BF16 = auto()  # Brain float (16-bit float, more range)
    TF32 = auto()  # Tensor float (19-bit, on Ampere+)
    AUTO = auto()  # Auto-select based on hardware


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""

    precision: Precision = Precision.AUTO
    loss_scale: Optional[float] = None
    initial_scale: float = 2.0**16
    min_scale: float = 1.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled: bool = True
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None


class AMPGradScaler(GradScaler):
    """
    Enhanced Gradient Scaler for mixed precision training.

    Extends PyTorch's GradScaler with additional features like
    dynamic loss scaling and gradient clipping.

    Attributes:
        config: Mixed precision configuration
    """

    def __init__(
        self,
        config: Optional[MixedPrecisionConfig] = None,
        **kwargs: Any,
    ):
        config = config or MixedPrecisionConfig()

        super().__init__(
            init_scale=config.initial_scale,
            growth_factor=config.growth_factor,
            backoff_factor=config.backoff_factor,
            growth_interval=config.growth_interval,
        )

        self.config = config
        self._last_scale = config.initial_scale

    def scale(self, loss: Tensor) -> Tensor:
        """Scale loss for gradient computation."""
        return super().scale(loss)

    def step(
        self,
        optimizer: Optimizer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Perform optimizer step with unscale."""
        # Apply gradient clipping if configured
        if self.config.gradient_clip_val is not None:
            self.unscale_(optimizer)

            if self.config.gradient_clip_algorithm == "norm":
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]["params"],
                    self.config.gradient_clip_val,
                )
            elif self.config.gradient_clip_algorithm == "value":
                torch.nn.utils.clip_grad_value_(
                    optimizer.param_groups[0]["params"],
                    self.config.gradient_clip_val,
                )

        super().step(optimizer, *args, **kwargs)
        super().update()

    def update(self, new_scale: Optional[float] = None) -> None:
        """Update scale factor dynamically."""
        if new_scale is not None:
            self._last_scale = new_scale
            self.load_state_dict({"scale": new_scale})
        super().update()


class MixedPrecisionTrainer:
    """
    Complete mixed precision training wrapper.

    Provides a simple interface for mixed precision training
    with automatic precision detection and optimization.

    Attributes:
        config: Mixed precision configuration
        device: Training device
    """

    def __init__(
        self,
        config: Optional[MixedPrecisionConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or MixedPrecisionConfig()
        self.device = device or torch.device("cuda:0")

        self._scaler: Optional[AMPGradScaler] = None
        self._precision = self._detect_precision()

        if self._precision != Precision.FP32:
            self._scaler = AMPGradScaler(self.config)

    def _detect_precision(self) -> Precision:
        """Detect available precision based on hardware."""
        if self.config.precision != Precision.AUTO:
            return self.config.precision

        if not torch.cuda.is_available():
            return Precision.FP32

        # Check compute capability
        cc = torch.cuda.get_device_capability()

        # BF16 supported on Ampere (8.0+) and newer
        if cc >= (8, 0):
            return Precision.BF16

        # TF32 supported on Ampere (8.0+)
        if cc >= (8, 0):
            return Precision.TF32

        # FP16 supported on Maxwell (5.0+) and newer
        if cc >= (5, 0):
            return Precision.FP16

        return Precision.FP32

    @property
    def precision(self) -> Precision:
        """Get current precision."""
        return self._precision

    @property
    def is_enabled(self) -> bool:
        """Check if mixed precision is enabled."""
        return self._scaler is not None and self.config.enabled

    @contextmanager
    def train_context(self) -> None:
        """Context manager for training mode."""
        try:
            yield
        finally:
            pass

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale loss for gradient computation."""
        if self.is_enabled:
            return self._scaler.scale(loss)
        return loss

    def step(
        self,
        optimizer: Optimizer,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        """
        Perform optimizer step with gradient scaling.

        Args:
            optimizer: Optimizer to step
            closure: Loss closure for optimizer

        Returns:
            Loss value if closure provided
        """
        if self.is_enabled and self._scaler is not None:
            return self._scaler.step(optimizer, closure)
        elif closure is not None:
            return closure()
        return None

    def update(self) -> None:
        """Update the gradient scaler."""
        if self._scaler is not None:
            self._scaler.update()

    def backward(
        self,
        loss: Tensor,
        retain_graph: bool = False,
    ) -> None:
        """
        Perform backward pass with gradient scaling.

        Args:
            loss: Loss tensor
            retain_graph: Whether to retain computation graph
        """
        if self.is_enabled:
            loss.backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def unscale_gradients(self, optimizer: Optimizer) -> None:
        """Unscale gradients for gradient manipulation."""
        if self.is_enabled and self._scaler is not None:
            self._scaler.unscale_(optimizer)

    def get_state(self) -> Dict[str, Any]:
        """Get trainer state."""
        return {
            "precision": self._precision.name,
            "enabled": self.is_enabled,
            "scale": self._scaler.get_scale() if self._scaler else None,
        }


@contextmanager
def create_amp_context(
    enabled: bool = True,
    dtype: torch.dtype = torch.float16,
) -> None:
    """
    Context manager for automatic mixed precision.

    Args:
        enabled: Whether to enable AMP
        dtype: Target dtype for computations

    Yields:
        None
    """
    if enabled and torch.cuda.is_available():
        with autocast(dtype=dtype, enabled=enabled):
            yield
    else:
        yield


def convert_to_fp16(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert model to FP16.

    Args:
        model: Model to convert
        inplace: Whether to modify in place

    Returns:
        Converted model
    """
    if not inplace:
        model = model.copy()

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            module.half()
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm doesn't support fp16 well, keep in fp32
            pass
        elif isinstance(module, nn.Embedding):
            # Embedding usually kept in fp32
            pass

    return model


def convert_to_bf16(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert model to BF16.

    Args:
        model: Model to convert
        inplace: Whether to modify in place

    Returns:
        Converted model
    """
    if not inplace:
        model = model.copy()

    for module in model.modules():
        if isinstance(module, nn.Parameter):
            module.data = module.data.to(dtype=torch.bfloat16)
        elif hasattr(module, "bias") and module.bias is not None:
            module.bias.data = module.bias.data.to(dtype=torch.bfloat16)

    return model


class DynamicLossScaler:
    """
    Dynamic loss scaling for mixed precision training.

    Automatically adjusts loss scale based on gradient overflow
    to maximize training stability and performance.
    """

    def __init__(
        self,
        init_scale: float = 2.0**16,
        scale_factor: float = 2.0,
        scale_window: int = 1000,
        min_scale: float = 1.0,
    ):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale

        self._iter = 0
        self._overflow_count = 0
        self._last_overflow_iter = -1

    def update_scale(self, overflow: bool) -> None:
        """
        Update scale based on overflow status.

        Args:
            overflow: Whether gradient overflow occurred
        """
        if overflow:
            self.scale = max(self.scale / self.scale_factor, self.min_scale)
            self._overflow_count += 1
            self._last_overflow_iter = self._iter
        else:
            if self._iter - self._last_overflow_iter >= self.scale_window:
                self.scale = min(self.scale * self.scale_factor, 2.0**20)

        self._iter += 1

    def scale_value(self, value: Tensor) -> Tensor:
        """Scale a tensor value."""
        return value * self.scale

    def unscale_value(self, value: Tensor) -> Tensor:
        """Unscale a tensor value."""
        return value / self.scale


class FP16OptimizerWrapper:
    """
    Wrapper to make any optimizer work with FP16.

    Maintains FP32 master weights while training in FP16.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        loss_scale: float = 2.0**16,
        clip_grad: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.loss_scaler = DynamicLossScaler(init_scale=loss_scale)
        self.clip_grad = clip_grad

        # Create FP32 master weights
        self._master_params: List[Tensor] = []
        self._fp16_params: Dict[int, Tensor] = {}

        self._init_master_params()

    def _init_master_params(self) -> None:
        """Initialize FP32 master parameters."""
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.dtype == torch.float16:
                    # Store FP32 copy
                    master = param.float().clone()
                    master.requires_grad = True
                    self._master_params.append(master)
                    self._fp16_params[id(param)] = master

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimizer step."""
        # Unscale gradients
        for master in self._master_params:
            if master.grad is not None:
                master.grad.data = master.grad.data / self.loss_scaler.scale

        # Clip gradients if needed
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self._master_params, self.clip_grad)

        # Step with FP32 weights
        self.optimizer.step()

        # Copy FP32 weights back to FP16
        for master in self._master_params:
            for param in self.optimizer.param_groups[0]["params"]:
                if id(param) in self._fp16_params:
                    param.data = self._fp16_params[id(param)].half()

        return closure() if closure is not None else None

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "loss_scaler": self.loss_scaler.scale,
        }


__all__ = [
    "Precision",
    "MixedPrecisionConfig",
    "AMPGradScaler",
    "MixedPrecisionTrainer",
    "create_amp_context",
    "convert_to_fp16",
    "convert_to_bf16",
    "DynamicLossScaler",
    "FP16OptimizerWrapper",
]

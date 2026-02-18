"""
Mixed Precision Training Utilities

Comprehensive mixed precision training utilities including AMP (Automatic Mixed Precision),
FP16, and BF16 training support with dynamic loss scaling and precision management.
"""

from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
from torch.cuda.amp import (
    autocast,
    GradScaler,
    custom_fwd,
    custom_bwd,
)
import torch.distributed as dist
from pathlib import Path


class PrecisionType(Enum):
    """Precision type enumeration."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class MixedPrecisionManager:
    """
    Central manager for mixed precision training.

    Handles precision switching, gradient scaling, and memory optimization
    for mixed precision training with automatic fallback to FP32 when needed.

    Example:
        >>> manager = MixedPrecisionManager(precision="fp16")
        >>> with manager.forward():
        ...     output = model(input)
        >>> with manager.backward(loss):
        ...     loss.backward()
    """

    def __init__(
        self,
        precision: Union[str, PrecisionType] = "fp16",
        loss_scale: Optional[float] = None,
        dynamic_scaling: bool = True,
        initial_scale: float = 2**16,
        min_scale: float = 1.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        if isinstance(precision, str):
            precision = PrecisionType(precision)

        self.precision = precision
        self.enabled = enabled and self._check_support()
        self.dynamic_scaling = dynamic_scaling

        if self.enabled and self.precision == PrecisionType.FP16:
            self.scaler = (
                GradScaler(
                    init_scale=initial_scale,
                    growth_factor=growth_factor,
                    backoff_factor=backoff_factor,
                    growth_interval=growth_interval,
                )
                if dynamic_scaling
                else None
            )
            self.fixed_scale = loss_scale or initial_scale
        else:
            self.scaler = None
            self.fixed_scale = 1.0

        self._forward_context = None
        self._setup_contexts()

    def _check_support(self) -> bool:
        """Check if mixed precision is supported on this device."""
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    def _setup_contexts(self):
        """Setup appropriate context managers based on precision type."""
        if not self.enabled:
            self._forward_context = torch.autocast(
                device_type="cuda",
                dtype=torch.float32,
                enabled=False,
            )
        elif self.precision == PrecisionType.FP16:
            self._forward_context = torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            )
        elif self.precision == PrecisionType.BF16:
            self._forward_context = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            )
        else:
            self._forward_context = torch.autocast(
                device_type="cuda",
                dtype=torch.float32,
                enabled=True,
            )

    def forward(self):
        """Context manager for forward pass with mixed precision."""
        return self._forward_context

    def backward(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Context manager for backward pass with gradient scaling.

        Args:
            loss: Loss tensor to backward

        Returns:
            Unscaled loss for logging
        """
        if self.scaler is not None:
            scaled_loss = loss * self.scaler.get_scale()
            return scaled_loss
        return loss

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for gradient computation.

        Args:
            loss: Input loss tensor

        Returns:
            Scaled loss tensor
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Unscale gradients for gradient clipping or other operations.

        Args:
            optimizer: Optimizer with gradients to unscale
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        clip_grad: Optional[Callable] = None,
        **clip_kwargs,
    ) -> None:
        """
        Step optimizer with gradient scaling.

        Args:
            optimizer: Optimizer to step
            clip_grad: Optional gradient clipping function
            **clip_kwargs: Arguments for gradient clipping
        """
        if self.scaler is not None:
            if clip_grad is not None:
                self.unscale_(optimizer)
                clip_grad(optimizer, **clip_kwargs)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def update(self) -> None:
        """Update the loss scale for dynamic scaling."""
        if self.scaler is not None:
            self.scaler.update()

    @property
    def get_scale(self) -> float:
        """Get current gradient scale factor."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        state = {
            "enabled": self.enabled,
            "precision": self.precision.value,
            "dynamic_scaling": self.dynamic_scaling,
        }
        if self.scaler is not None:
            state["scaler_state"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.enabled = state.get("enabled", self.enabled)
        self.dynamic_scaling = state.get("dynamic_scaling", self.dynamic_scaling)
        if self.scaler is not None and "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])


class AMPTrainer:
    """
    Automatic Mixed Precision Trainer wrapper.

    Simplified interface for training with AMP, handling all aspects
    of mixed precision training including forward, backward, and optimizer steps.

    Example:
        >>> trainer = AMPTrainer(model, optimizer, device='cuda')
        >>> for batch in dataloader:
        ...     metrics = trainer.train_step(batch)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Union[str, torch.device] = "cuda",
        precision: Union[str, PrecisionType] = "fp16",
        loss_fn: Optional[Callable] = None,
        clip_grad: Optional[Callable] = None,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.clip_grad = clip_grad
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.accumulation_counter = 0

        self.mixed_precision = MixedPrecisionManager(
            precision=precision,
            enabled=True,
        )

        self.training_stats: Dict[str, List[float]] = {
            "loss": [],
            "grad_norm": [],
            "scale": [],
        }

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        compute_loss: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Execute a single training step with mixed precision.

        Args:
            batch: Input batch dictionary
            compute_loss: Optional custom loss computation function

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        inputs = {
            k: v.to(self.device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }

        with self.mixed_precision.forward():
            if compute_loss is not None:
                loss = compute_loss(self.model, inputs)
            else:
                outputs = self.model(**inputs)
                loss = self.loss_fn(
                    outputs, inputs.get("labels", torch.zeros(1).long().to(self.device))
                )

        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps

        scaled_loss = self.mixed_precision.scale(loss)
        scaled_loss.backward()

        self.accumulation_counter += 1

        metrics = {"loss": loss.item()}

        if self.accumulation_counter >= self.gradient_accumulation_steps:
            grad_norm = 0.0

            if self.clip_grad is not None:
                self.mixed_precision.unscale_(self.optimizer)
                if hasattr(self.model, "parameters"):
                    grad_norm = self.clip_grad(self.model.parameters())

            self.mixed_precision.step(self.optimizer)
            self.optimizer.zero_grad()
            self.accumulation_counter = 0

            metrics["grad_norm"] = grad_norm
            metrics["scale"] = self.mixed_precision.get_scale

        self.training_stats["loss"].append(metrics["loss"])

        return metrics

    def validate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute validation step with mixed precision.

        Args:
            batch: Validation batch

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        with torch.no_grad():
            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            with self.mixed_precision.forward():
                outputs = self.model(**inputs)
                loss = self.loss_fn(
                    outputs, inputs.get("labels", torch.zeros(1).long().to(self.device))
                )

        return {"val_loss": loss.item()}

    def get_stats(self) -> Dict[str, float]:
        """Get aggregated training statistics."""
        return {
            "avg_loss": sum(self.training_stats["loss"])
            / len(self.training_stats["loss"])
            if self.training_stats["loss"]
            else 0.0,
            "total_steps": len(self.training_stats["loss"]),
        }

    def reset_stats(self) -> None:
        """Reset training statistics."""
        for key in self.training_stats:
            self.training_stats[key] = []


class FP16Manager:
    """
    FP16-specific precision manager.

    Optimized for FP16 training with additional memory optimization
    techniques specific to half-precision floating point.
    """

    def __init__(
        self,
        loss_scale: Optional[float] = None,
        dynamic_scaling: bool = True,
        initial_scale: float = 2**16,
        find_unused_parameters: bool = False,
    ):
        self.dynamic_scaling = dynamic_scaling
        self.find_unused_parameters = find_unused_parameters

        if dynamic_scaling:
            self.scaler = GradScaler(init_scale=initial_scale)
        else:
            self.scaler = None
            self.loss_scale = loss_scale or initial_scale

        self._enabled = torch.cuda.is_available()

    def __enter__(self):
        if self._enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False)

    def __exit__(self, *args):
        pass

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss * self.loss_scale

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer with gradient unscaling."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        state = {"enabled": self._enabled}
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state


class BF16Manager:
    """
    BF16-specific precision manager.

    Brain floating point (BF16) offers wider dynamic range than FP16
    with similar memory savings but less precision.
    """

    def __init__(
        self,
        loss_scale: Optional[float] = None,
        use_fp32_gradients: bool = True,
        find_unused_parameters: bool = False,
    ):
        self.use_fp32_gradients = use_fp32_gradients
        self.find_unused_parameters = find_unused_parameters
        self.loss_scale = loss_scale or 1.0
        self._enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    def __enter__(self):
        if self._enabled:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", dtype=torch.float32, enabled=False)

    def __exit__(self, *args):
        pass

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        return loss * self.loss_scale

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer."""
        optimizer.step()

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "enabled": self._enabled,
            "loss_scale": self.loss_scale,
        }


@dataclass
class DynamicLossScaler:
    """
    Dynamic loss scaler with automatic scale adjustment.

    Monitors for NaN/Inf in gradients and adjusts the loss scale
    accordingly to maintain training stability.

    Example:
        >>> scaler = DynamicLossScaler(initial_scale=1024.0)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     scaled_loss = scaler.scale(loss)
        ...     scaled_loss.backward()
        ...     if scaler.check_isfinite():
        ...         optimizer.step()
        ...         scaler.update()
    """

    initial_scale: float = 2**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    min_scale: float = 1.0
    max_scale: float = float(2**24)

    def __post_init__(self):
        self.scale = self.initial_scale
        self.step_count = 0
        self._last_overflow = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss by current scale factor."""
        return loss * self.scale

    def update(self, overflow: bool) -> None:
        """
        Update scale based on overflow status.

        Args:
            overflow: Whether overflow occurred in previous step
        """
        self.step_count += 1

        if overflow:
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self._last_overflow = True
        elif self.step_count % self.growth_interval == 0:
            self.scale = min(self.scale * self.growth_factor, self.max_scale)
            self._last_overflow = False
        else:
            self._last_overflow = False

    def check_isfinite(self, grads: List[torch.Tensor]) -> bool:
        """
        Check if gradients are finite (no NaN or Inf).

        Args:
            grads: List of gradient tensors

        Returns:
            True if all gradients are finite
        """
        for grad in grads:
            if grad is not None:
                if not torch.isfinite(grad).all():
                    return False
        return True

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "scale": self.scale,
            "step_count": self.step_count,
            "last_overflow": self._last_overflow,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.scale = state.get("scale", self.initial_scale)
        self.step_count = state.get("step_count", 0)
        self._last_overflow = state.get("last_overflow", False)

"""
Gradient Accumulation Strategies

Advanced gradient accumulation techniques for efficient training:
- Dynamic gradient accumulation with adaptive thresholds
- Variance-based accumulation
- Gradient scaling with automatic mixed precision support
- Multi-scale gradient accumulation

Reference:
- Chen et al. (2020). Gradient Centralization.
- Lin et al. (2020). Gradient Centralization: A New Optimization Technique.
"""

from typing import Optional, Dict, List, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
import numpy as np
from collections import deque


class GradientAccumulator:
    """
    Gradient Accumulation Wrapper.

    Accumulates gradients over multiple micro-batches before performing
    an optimizer step, effectively increasing the batch size without
    additional memory.

    Args:
        optimizer: The underlying optimizer
        accumulation_steps: Number of steps to accumulate gradients
        scale_grad: Whether to scale gradients by accumulation_steps
        clip_grad: Maximum norm for gradient clipping (None to disable)

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> accumulator = GradientAccumulator(optimizer, accumulation_steps=4)
        >>> for data, target in dataloader:
        ...     output = model(data)
        ...     loss = loss_fn(output, target)
        ...     loss.backward()
        ...     accumulator.step()  # Accumulates, steps every 4th call
    """

    def __init__(
        self,
        optimizer: Optimizer,
        accumulation_steps: int = 4,
        scale_grad: bool = True,
        clip_grad: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scale_grad = scale_grad
        self.clip_grad = clip_grad
        self.step_count = 0
        self._accumulated = False

    def zero_grad(self):
        """Clear accumulated gradients."""
        self.optimizer.zero_grad()
        self._accumulated = False

    def step(self):
        """
        Perform an optimizer step, either accumulating or updating weights.
        """
        if self.step_count % self.accumulation_steps == 0:
            if self._accumulated:
                if self.scale_grad:
                    self._scale_gradients()

                if self.clip_grad is not None:
                    self._clip_gradients()

                self.optimizer.step()

            self.optimizer.zero_grad()
            self._accumulated = False
        else:
            self._accumulated = True

        self.step_count += 1

    def _scale_gradients(self):
        """Scale accumulated gradients."""
        scale_factor = self.accumulation_steps
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.div_(scale_factor)

    def _clip_gradients(self):
        """Clip gradients by norm."""
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.optimizer.param_groups[0]["params"] if p.grad is not None],
            self.clip_grad,
        )

    def state_dict(self) -> Dict:
        """Return state dict for checkpointing."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "accumulation_steps": self.accumulation_steps,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.step_count = state_dict["step_count"]
        self.accumulation_steps = state_dict["accumulation_steps"]


class AdaptiveGradientAccumulator:
    """
    Adaptive Gradient Accumulation.

    Dynamically adjusts the number of accumulation steps based on
    gradient variance and loss behavior.

    Args:
        optimizer: The underlying optimizer
        min_accumulation_steps: Minimum accumulation steps
        max_accumulation_steps: Maximum accumulation steps
        variance_window: Window size for variance estimation
        increase_threshold: Increase accumulation when variance < threshold
        decrease_threshold: Decrease accumulation when variance > threshold

    Reference:
        - Liu et al. (2021). Adaptive Gradient Methods with Dynamic Bound.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        min_accumulation_steps: int = 1,
        max_accumulation_steps: int = 16,
        variance_window: int = 10,
        increase_threshold: float = 0.01,
        decrease_threshold: float = 0.1,
        history_size: int = 100,
    ):
        self.optimizer = optimizer
        self.min_accumulation_steps = min_accumulation_steps
        self.max_accumulation_steps = max_accumulation_steps
        self.variance_window = variance_window
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.history_size = history_size

        self.current_accumulation = min_accumulation_steps
        self.step_count = 0
        self.accumulated = False

        self.loss_history = deque(maxlen=history_size)
        self.grad_norm_history = deque(maxlen=history_size)

    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()

    def step(self):
        """Perform optimization step with adaptive accumulation."""
        self._update_accumulation_schedule()

        if self.step_count % self.current_accumulation == 0:
            if self.accumulated:
                self._scale_gradients()
                if self.optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        self.optimizer.param_groups[0]["params"], 1.0
                    )
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.accumulated = False
        else:
            self.accumulated = True

        self.step_count += 1

    def _update_accumulation_schedule(self):
        """Dynamically update accumulation steps based on loss variance."""
        if len(self.loss_history) < self.variance_window:
            return

        recent_losses = list(self.loss_history)[-self.variance_window :]
        recent_grad_norms = list(self.grad_norm_history)[-self.variance_window :]

        loss_variance = np.var(recent_losses) if len(recent_losses) > 1 else 0
        grad_variance = np.var(recent_grad_norms) if len(recent_grad_norms) > 1 else 0

        mean_grad_norm = np.mean(recent_grad_norms)
        relative_variance = grad_variance / (mean_grad_norm**2 + 1e-8)

        if relative_variance < self.increase_threshold:
            new_accumulation = min(
                self.current_accumulation + 1, self.max_accumulation_steps
            )
            if new_accumulation != self.current_accumulation:
                self.current_accumulation = new_accumulation

        elif relative_variance > self.decrease_threshold:
            new_accumulation = max(
                self.current_accumulation - 1, self.min_accumulation_steps
            )
            if new_accumulation != self.current_accumulation:
                self.current_accumulation = new_accumulation

    def _scale_gradients(self):
        """Scale gradients by accumulation factor."""
        scale_factor = self.current_accumulation
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.div_(scale_factor)

    def record_step(self, loss: float, grad_norm: float):
        """Record metrics for adaptive scheduling."""
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)

    @property
    def accumulation_steps(self) -> int:
        """Current accumulation steps."""
        return self.current_accumulation


class GradientAccumulatorWithScaling(GradientAccumulator):
    """
    Gradient Accumulation with Automatic Mixed Precision (AMP) Support.

    Supports gradient scaling for mixed precision training while maintaining
    gradient accumulation.

    Args:
        optimizer: The underlying optimizer
        accumulation_steps: Number of steps to accumulate
        loss_scale: Initial loss scale for AMP (None for dynamic)

    Example:
        >>> scaler = torch.cuda.amp.GradScaler()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> accumulator = GradientAccumulatorWithScaling(optimizer, accumulation_steps=4)
        >>> for data, target in dataloader:
        ...     with torch.cuda.amp.autocast():
        ...         output = model(data)
        ...         loss = loss_fn(output, target)
        ...     scaler.scale(loss).backward()
        ...     accumulator.step(scaler)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        accumulation_steps: int = 4,
        loss_scale: Optional[float] = None,
    ):
        super().__init__(optimizer, accumulation_steps, scale_grad=False)
        self.loss_scale = loss_scale or "dynamic"
        self._current_scale = 65536.0 if self.loss_scale == "dynamic" else loss_scale
        self._scale_changed = False

    def step(self, scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """Perform step with gradient scaling."""
        if self.step_count % self.accumulation_steps == 0:
            if self._accumulated and scaler is not None:
                self._scale_changed = scaler.scale(self._accumulated_grad())
                scaler.step(self.optimizer)
                scaler.update()

            self.optimizer.zero_grad()
            self._accumulated = False
        else:
            self._accumulated = True

        self.step_count += 1

    def _accumulated_grad(self) -> Tensor:
        """Get accumulated gradients as a single tensor."""
        total_loss = None
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    scaled = param.grad.data / self.accumulation_steps
                    param.grad.data = scaled
        return total_loss


class MultiScaleGradientAccumulator:
    """
    Multi-Scale Gradient Accumulation.

    Maintains multiple gradient accumulators at different accumulation levels
    and dynamically selects the optimal one based on training dynamics.

    Args:
        optimizer: The underlying optimizer
        accumulation_levels: List of accumulation step options
        selection_metric: Metric for level selection ('loss_variance', 'grad_norm', 'combined')
    """

    def __init__(
        self,
        optimizer: Optimizer,
        accumulation_levels: Optional[List[int]] = None,
        selection_metric: str = "combined",
    ):
        self.optimizer = optimizer
        self.accumulation_levels = accumulation_levels or [1, 2, 4, 8]
        self.selection_metric = selection_metric

        self.current_level = 0
        self.step_count = 0
        self._accumulated = False

        self.loss_history = {
            level: deque(maxlen=50) for level in self.accumulation_levels
        }
        self.grad_history = {
            level: deque(maxlen=50) for level in self.accumulation_levels
        }

    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()

    def step(self):
        """Perform step with multi-scale accumulation."""
        if self.step_count % self.accumulation_levels[self.current_level] == 0:
            if self._accumulated:
                self._scale_gradients()
                self.optimizer.step()

            self.optimizer.zero_grad()
            self._accumulated = False
        else:
            self._accumulated = True

        self.step_count += 1

    def select_level(self, loss: float, grad_norm: float):
        """Select optimal accumulation level based on training metrics."""
        for level in self.accumulation_levels:
            self.loss_history[level].append(loss)
            self.grad_history[level].append(grad_norm)

        if len(self.loss_history[self.accumulation_levels[0]]) < 10:
            return

        scores = {}

        for level in self.accumulation_levels:
            losses = list(self.loss_history[level])
            grads = list(self.grad_history[level])

            if len(losses) < 2:
                scores[level] = 0
                continue

            if self.selection_metric == "loss_variance":
                scores[level] = -np.var(losses)
            elif self.selection_metric == "grad_norm":
                scores[level] = -np.mean(grads)
            elif self.selection_metric == "combined":
                loss_var = np.var(losses) if len(losses) > 1 else 0
                grad_mean = np.mean(grads)
                scores[level] = -(loss_var + 0.1 * grad_mean)

        best_level = max(scores, key=scores.get)

        if best_level != self.current_level:
            self.current_level = best_level

    def _scale_gradients(self):
        """Scale gradients."""
        scale_factor = self.accumulation_levels[self.current_level]
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.div_(scale_factor)

    @property
    def current_accumulation(self) -> int:
        """Current accumulation steps."""
        return self.accumulation_levels[self.current_level]


class GradientCentralization:
    """
    Gradient Centralization Utility.

    Centers gradients to the origin before updating parameters.
    This technique has been shown to improve convergence speed and
    solution quality.

    Reference:
        - Lin et al. (2020). Gradient Centralization: A New Optimization Technique.

    Args:
        optimizer: The underlying optimizer
        centralize: Whether to apply gradient centralization

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> gc_optimizer = GradientCentralization(optimizer, centralize=True)
        >>> for data, target in dataloader:
        ...     output = model(data)
        ...     loss = loss_fn(output, target)
        ...     loss.backward()
        ...     gc_optimizer.step()
    """

    def __init__(self, optimizer: Optimizer, centralize: bool = True):
        self.optimizer = optimizer
        self.centralize = centralize

    def zero_grad(self):
        """Clear gradients."""
        self.optimizer.zero_grad()

    def step(self):
        """Perform optimization step with optional centralization."""
        if self.centralize:
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        if param.grad.ndim > 1:
                            mean = param.grad.data.mean(
                                dim=list(range(1, param.grad.ndim)), keepdim=True
                            )
                            param.grad.data.sub_(mean)

        self.optimizer.step()

    def state_dict(self):
        """Return state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.optimizer.load_state_dict(state_dict)

    def __getattr__(self, name):
        """Forward attribute access to underlying optimizer."""
        if name in ["optimizer"]:
            return object.__getattribute__(self, name)
        return getattr(self.optimizer, name)


__all__ = [
    "GradientAccumulator",
    "AdaptiveGradientAccumulator",
    "GradientAccumulatorWithScaling",
    "MultiScaleGradientAccumulator",
    "GradientCentralization",
]

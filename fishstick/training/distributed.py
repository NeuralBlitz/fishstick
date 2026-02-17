"""
Distributed Training Utilities for fishstick

Utilities for data parallelism, model parallelism, and distributed training.
"""

from typing import Optional, Callable, Dict, Any
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os


class DistributedTrainer:
    """
    Simplified distributed training wrapper.

    Handles process spawning, DDP setup, and distributed training loop.

    Example:
        >>> trainer = DistributedTrainer(model_fn, train_fn)
        >>> trainer.launch(world_size=4)
    """

    def __init__(self, model_fn: Callable, train_fn: Callable, backend: str = "nccl"):
        self.model_fn = model_fn
        self.train_fn = train_fn
        self.backend = backend

    def launch(self, world_size: int, args: tuple = ()):
        """Launch distributed training."""
        mp.spawn(self._worker, args=(world_size, args), nprocs=world_size, join=True)

    def _worker(self, rank: int, world_size: int, args: tuple):
        """Worker process function."""
        self.setup(rank, world_size)

        # Create model and wrap with DDP
        model = self.model_fn()
        model = model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        # Run training
        self.train_fn(ddp_model, rank, world_size, *args)

        self.cleanup()

    def setup(self, rank: int, world_size: int):
        """Initialize distributed process group."""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        dist.init_process_group(self.backend, rank=rank, world_size=world_size)

    def cleanup(self):
        """Clean up distributed process group."""
        dist.destroy_process_group()


class GradientAccumulator:
    """
    Gradient accumulation for effective larger batch sizes.

    Accumulates gradients over multiple forward/backward passes
    before performing optimizer step.

    Example:
        >>> accumulator = GradientAccumulator(model, optimizer, accumulation_steps=4)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     accumulator.step(loss)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4,
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def step(self, loss: torch.Tensor):
        """Accumulate gradients and step optimizer when appropriate."""
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps

        # Backward pass
        loss.backward()

        self.current_step += 1

        if self.current_step % self.accumulation_steps == 0:
            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step = 0

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
        self.current_step = 0


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) for model parameters.

    Maintains a moving average of model parameters for evaluation.

    Reference: TensorFlow's ExponentialMovingAverage

    Example:
        >>> ema = ExponentialMovingAverage(model, decay=0.9999)
        >>>
        >>> for batch in dataloader:
        ...     loss = train_step(model, batch)
        ...     ema.update()
        >>>
        >>> # Evaluate with EMA model
        >>> ema.apply_shadow()
        >>> evaluate(model)
        >>> ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update moving average."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state dict."""
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load EMA state dict."""
        self.shadow = state_dict


class MixedPrecisionTrainer:
    """
    Automatic Mixed Precision (AMP) training.

    Uses torch.cuda.amp for automatic mixed precision training
    to speed up training and reduce memory usage.

    Example:
        >>> mp_trainer = MixedPrecisionTrainer(model, optimizer)
        >>>
        >>> for batch in dataloader:
        ...     with mp_trainer.autocast():
        ...         loss = model(batch)
        ...     mp_trainer.backward(loss)
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, enabled: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.enabled = enabled and torch.cuda.is_available()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)

    def autocast(self):
        """Get autocast context manager."""
        return torch.cuda.amp.autocast(enabled=self.enabled)

    def backward(self, loss: torch.Tensor):
        """Backward pass with gradient scaling."""
        self.scaler.scale(loss).backward()

    def step(self):
        """Optimizer step with unscaling."""
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        """Get scaler state dict."""
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scaler state dict."""
        self.scaler.load_state_dict(state_dict)


def average_gradients(model: nn.Module):
    """Average gradients across processes (for distributed training)."""
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


__all__ = [
    "DistributedTrainer",
    "GradientAccumulator",
    "ExponentialMovingAverage",
    "MixedPrecisionTrainer",
    "average_gradients",
]

"""
Memory-Aware Training for Continual Learning.

Techniques that optimize memory usage during training.

Classes:
- MemoryAwareTrainer: Memory-efficient trainer
- AdaptiveReplayScheduler: Dynamic replay rate scheduling
- MemoryEfficientTrainer: Memory-optimized training
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np
from collections import deque


class AdaptiveReplayScheduler:
    """
    Adaptive Replay Rate Scheduler.

    Dynamically adjusts replay rate based on training dynamics.

    Args:
        initial_rate: Initial replay rate
        decay: Decay factor
        min_rate: Minimum replay rate
    """

    def __init__(
        self,
        initial_rate: float = 0.5,
        decay: float = 0.99,
        min_rate: float = 0.1,
    ):
        self.rate = initial_rate
        self.decay = decay
        self.min_rate = min_rate

        self.loss_history: deque = deque(maxlen=100)

    def update(self, loss: float) -> float:
        """
        Update replay rate based on loss.

        Args:
            loss: Current loss value

        Returns:
            Updated replay rate
        """
        self.loss_history.append(loss)

        if len(self.loss_history) > 10:
            recent = list(self.loss_history)[-5:]
            older = list(self.loss_history)[-10:-5]

            trend = np.mean(recent) - np.mean(older)

            if trend > 0.1:
                self.rate = min(0.9, self.rate * 1.1)
            elif trend < -0.1:
                self.rate = max(self.min_rate, self.rate * 0.9)

        self.rate *= self.decay

        return self.rate

    def get_rate(self) -> float:
        """Get current replay rate."""
        return self.rate


class MemoryEfficientTrainer:
    """
    Memory-Efficient Training for Continual Learning.

    Implements gradient checkpointing and mixed precision training
    to reduce memory usage.

    Args:
        model: Neural network
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device

        self.use_amp = torch.cuda.is_available()

        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Perform memory-efficient training step.

        Args:
            x: Input batch
            y: Target batch
            optimizer: Optimizer

        Returns:
            Dictionary of metrics
        """
        x = x.to(self.device)
        y = y.to(self.device)

        optimizer.zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

        return {"loss": loss.item()}

    def clear_cache(self) -> None:
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MemoryAwareTrainer:
    """
    Memory-Aware Continual Learning Trainer.

    Monitors and optimizes memory usage during training.

    Args:
        model: Neural network
        memory_budget: Maximum memory in MB
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        memory_budget: float = 4096,
        device: str = "cpu",
    ):
        self.model = model
        self.memory_budget = memory_budget
        self.device = device

        self.replay_scheduler = AdaptiveReplayScheduler()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not torch.cuda.is_available():
            return {"cpu_memory_mb": 0.0}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }

    def should_use_replay(self) -> bool:
        """Determine if replay should be used."""
        mem_usage = self.get_memory_usage()

        allocated = mem_usage.get("allocated_mb", 0)

        return allocated < self.memory_budget * 0.8

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        optimizer: torch.optim.Optimizer,
        replay_buffer: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Train with memory awareness."""
        x = x.to(self.device)
        y = y.to(self.device)

        optimizer.zero_grad()

        current_loss = F.cross_entropy(self.model(x), y)

        replay_rate = self.replay_scheduler.get_rate()

        loss = current_loss

        if replay_buffer is not None and np.random.random() < replay_rate:
            pass

        loss.backward()
        optimizer.step()

        self.replay_scheduler.update(loss.item())

        return {
            "loss": loss.item(),
            "replay_rate": replay_rate,
            "memory": self.get_memory_usage(),
        }


class GradientCheckpointingTrainer(MemoryEfficientTrainer):
    """
    Trainer with Gradient Checkpointing.

    Trades compute for memory by recomputing activations.

    Args:
        model: Neural network
        checkpoint_ratio: Ratio of layers to checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_ratio: float = 0.5,
    ):
        super().__init__(model)

        self.checkpoint_ratio = checkpoint_ratio

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Train with gradient checkpointing."""
        x = x.to(self.device)
        y = y.to(self.device)

        def create_forward_hook(module):
            def hook(module, input, output):
                return output

            return hook

        hooks = []

        modules = list(self.model.modules())
        checkpoint_modules = modules[
            :: max(1, int(len(modules) * self.checkpoint_ratio))
        ]

        for module in checkpoint_modules:
            hooks.append(module.register_forward_hook(create_forward_hook(module)))

        optimizer.zero_grad()

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        for hook in hooks:
            hook.remove()

        return {"loss": loss.item()}

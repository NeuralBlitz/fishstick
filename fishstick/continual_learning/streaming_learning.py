"""
Streaming Learning Methods for Continual Learning.

Methods designed for learning from data streams with bounded memory.

Classes:
- StreamingMethod: Base streaming method
- ReservoirSampling: Reservoir sampling implementation
- BoundedMemoryLearner: Bounded memory continual learner
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np
from collections import deque
import random


class StreamingMethod:
    """
    Base class for streaming continual learning methods.

    Args:
        model: Neural network
        buffer_size: Maximum memory size
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1000,
    ):
        self.model = model
        self.buffer_size = buffer_size

    def update(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Update on stream sample."""
        raise NotImplementedError


class ReservoirSampling:
    """
    Reservoir Sampling for Streaming Data.

    Maintains uniform sample from potentially infinite stream.

    Args:
        capacity: Maximum reservoir size
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: List[Tuple[Tensor, Tensor]] = []
        self.count = 0

    def add(self, x: Tensor, y: Tensor) -> None:
        """Add sample using reservoir sampling."""
        if len(self.buffer) < self.capacity:
            self.buffer.append((x, y))
        else:
            idx = random.randint(0, self.count)
            if idx < self.capacity:
                self.buffer[idx] = (x, y)
        self.count += 1

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from reservoir."""
        if len(self.buffer) == 0:
            raise ValueError("Empty reservoir")

        indices = random.sample(
            range(len(self.buffer)), min(batch_size, len(self.buffer))
        )

        x = torch.stack([self.buffer[i][0] for i in indices])
        y = torch.stack([self.buffer[i][1] for i in indices])

        return x, y

    def __len__(self) -> int:
        return len(self.buffer)


class BoundedMemoryLearner(StreamingMethod):
    """
    Bounded Memory Learner for Streaming.

    Uses fixed memory budget for continual learning from streams.

    Args:
        model: Neural network
        buffer_size: Maximum buffer size
        lr: Learning rate
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1000,
        lr: float = 1e-3,
    ):
        super().__init__(model, buffer_size)

        self.reservoir = ReservoirSampling(buffer_size)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Update with streaming sample."""
        x = x.cuda() if x.is_cuda else x
        y = y.cuda() if y.is_cuda else y

        self.reservoir.add(x, y)

        self.optimizer.zero_grad()

        current_loss = F.cross_entropy(self.model(x), y)

        if len(self.reservoir) > 0:
            replay_x, replay_y = self.reservoir.sample(min(32, len(self.reservoir)))

            replay_loss = F.cross_entropy(self.model(replay_x), replay_y)

            loss = current_loss + 0.5 * replay_loss
        else:
            loss = current_loss

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}


class ExponentialWeightedLearner(StreamingMethod):
    """
    Exponential Weighting for Streaming.

    Uses exponential weighting of samples based on recency.

    Args:
        model: Neural network
        decay: Exponential decay factor
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.99,
    ):
        super().__init__(model)

        self.decay = decay
        self.weighted_buffer: List[Tuple[Tensor, Tensor, float]] = []

    def update(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Update with exponential weighting."""
        self.weighted_buffer.append((x, y, 1.0))

        for i in range(len(self.weighted_buffer)):
            self.weighted_buffer[i] = (
                self.weighted_buffer[i][0],
                self.weighted_buffer[i][1],
                self.weighted_buffer[i][2] * self.decay,
            )

        self.weighted_buffer = self.weighted_buffer[-self.buffer_size :]

        return {"buffer_size": len(self.weighted_buffer)}


class MinibatchStreamLearner(StreamingMethod):
    """
    Minibatch Streaming Learner.

    Accumulates samples into minibatches for efficient training.

    Args:
        model: Neural network
        minibatch_size: Size of accumulated minibatch
        update_freq: Update frequency
    """

    def __init__(
        self,
        model: nn.Module,
        minibatch_size: int = 32,
        update_freq: int = 10,
    ):
        super().__init__(model)

        self.minibatch_size = minibatch_size
        self.update_freq = update_freq

        self.buffer_x: List[Tensor] = []
        self.buffer_y: List[Tensor] = []

        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def update(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Accumulate and update."""
        self.buffer_x.append(x)
        self.buffer_y.append(y)

        if len(self.buffer_x) >= self.minibatch_size:
            batch_x = torch.cat(self.buffer_x[: self.minibatch_size])
            batch_y = torch.cat(self.buffer_y[: self.minibatch_size])

            self.optimizer.zero_grad()
            loss = F.cross_entropy(self.model(batch_x), batch_y)
            loss.backward()
            self.optimizer.step()

            self.buffer_x = self.buffer_x[self.minibatch_size :]
            self.buffer_y = self.buffer_y[self.minibatch_size :]

            return {"loss": loss.item()}

        return {"buffer": len(self.buffer_x)}

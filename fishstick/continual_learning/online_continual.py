"""
Online Continual Learning Methods.

Methods for learning continuously without explicit task boundaries.

Classes:
- OnlineContinualLearner: Base online continual learner
- StreamingLearner: Streaming learning variant
- SlidingWindowLearner: Sliding window approach
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from collections import deque
import copy


class OnlineContinualLearner(nn.Module):
    """
    Base class for Online Continual Learning.

    Learns from a stream of data without explicit task boundaries.

    Args:
        model: Neural network
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        super().__init__()

        self.model = model
        self.device = device

        self.seen_samples = 0

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(x)

    def update(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """
        Update on a single batch.

        Args:
            x: Input batch
            y: Target batch

        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError

    def observe(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """
        Main observation method for online learning.

        Args:
            batch: (x, y) tuple

        Returns:
            Dictionary of metrics
        """
        x, y = batch
        return self.update(x, y)


class StreamingLearner(OnlineContinualLearner):
    """
    Streaming Learning with Experience Replay.

    Continuously learns from data stream with replay buffer.

    Args:
        model: Neural network
        buffer_size: Replay buffer size
        batch_size: Training batch size
        lr: Learning rate
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1000,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__(model, device)

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.buffer_x: deque = deque(maxlen=buffer_size)
        self.buffer_y: deque = deque(maxlen=buffer_size)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Update on batch with replay."""
        x = x.to(self.device)
        y = y.to(self.device)

        self.buffer_x.append(x.detach().cpu())
        self.buffer_y.append(y.detach().cpu())

        self.optimizer.zero_grad()

        current_loss = F.cross_entropy(self.model(x), y)

        replay_loss = torch.tensor(0.0, device=self.device)

        if len(self.buffer_x) >= self.batch_size // 2:
            replay_indices = np.random.choice(
                len(self.buffer_x),
                self.batch_size // 2,
                replace=False,
            )

            replay_x = torch.stack([self.buffer_x[i] for i in replay_indices]).to(
                self.device
            )
            replay_y = torch.stack([self.buffer_y[i] for i in replay_indices]).to(
                self.device
            )

            replay_loss = F.cross_entropy(self.model(replay_x), replay_y)

        loss = current_loss + 0.5 * replay_loss

        loss.backward()
        self.optimizer.step()

        self.seen_samples += x.size(0)

        return {
            "current_loss": current_loss.item(),
            "replay_loss": replay_loss.item()
            if isinstance(replay_loss, Tensor)
            else 0.0,
            "total_loss": loss.item(),
        }


class SlidingWindowLearner(OnlineContinualLearner):
    """
    Sliding Window Continual Learning.

    Uses a sliding window over recent samples for training.

    Args:
        model: Neural network
        window_size: Size of sliding window
        batch_size: Training batch size
        lr: Learning rate
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 500,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__(model, device)

        self.window_size = window_size
        self.batch_size = batch_size

        self.window_x: deque = deque(maxlen=window_size)
        self.window_y: deque = deque(maxlen=window_size)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Update with sliding window."""
        x = x.to(self.device)
        y = y.to(self.device)

        self.window_x.append(x)
        self.window_y.append(y)

        if len(self.window_x) < self.batch_size:
            return {"loss": 0.0}

        self.optimizer.zero_grad()

        indices = np.random.choice(len(self.window_x), self.batch_size, replace=False)

        batch_x = torch.stack([self.window_x[i] for i in indices])
        batch_y = torch.stack([self.window_y[i] for i in indices])

        loss = F.cross_entropy(self.model(batch_x), batch_y)

        loss.backward()
        self.optimizer.step()

        self.seen_samples += x.size(0)

        return {"loss": loss.item()}


class AdaptiveReplayLearner(OnlineContinualLearner):
    """
    Adaptive Replay Rate Learner.

    Adjusts replay rate based on loss trends to balance
    plasticity and stability.

    Args:
        model: Neural network
        buffer_size: Replay buffer size
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1000,
        device: str = "cpu",
    ):
        super().__init__(model, device)

        self.buffer_size = buffer_size

        self.buffer_x: deque = deque(maxlen=buffer_size)
        self.buffer_y: deque = deque(maxlen=buffer_size)

        self.replay_rate = 0.5
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        self.loss_history: deque = deque(maxlen=100)

    def update(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Update with adaptive replay rate."""
        x = x.to(self.device)
        y = y.to(self.device)

        self.buffer_x.append(x.detach().cpu())
        self.buffer_y.append(y.detach().cpu())

        self.optimizer.zero_grad()

        current_loss = F.cross_entropy(self.model(x), y)

        self.loss_history.append(current_loss.item())

        if len(self.buffer_x) > 0:
            if len(self.loss_history) > 10:
                recent_trend = np.mean(list(self.loss_history)[-5:]) - np.mean(
                    list(self.loss_history)[-10:-5]
                )

                if recent_trend > 0.1:
                    self.replay_rate = min(0.9, self.replay_rate + 0.05)
                elif recent_trend < -0.1:
                    self.replay_rate = max(0.1, self.replay_rate - 0.05)

        if np.random.random() < self.replay_rate and len(self.buffer_x) > 0:
            replay_size = min(len(self.buffer_x), x.size(0))
            replay_indices = np.random.choice(
                len(self.buffer_x), replay_size, replace=False
            )

            replay_x = torch.stack([self.buffer_x[i] for i in replay_indices]).to(
                self.device
            )
            replay_y = torch.stack([self.buffer_y[i] for i in replay_indices]).to(
                self.device
            )

            replay_loss = F.cross_entropy(self.model(replay_x), replay_y)

            loss = current_loss + replay_loss
        else:
            loss = current_loss

        loss.backward()
        self.optimizer.step()

        self.seen_samples += x.size(0)

        return {
            "loss": loss.item(),
            "replay_rate": self.replay_rate,
        }


class ER_Adam(OnlineContinualLearner):
    """
    Experience Replay with Adam Optimizer.

    Simple but effective online continual learning with
    experience replay.

    Args:
        model: Neural network
        buffer_size: Replay buffer size
        lr: Learning rate
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 10000,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__(model, device)

        self.buffer_size = buffer_size

        self.buffer_x: List[Tensor] = []
        self.buffer_y: List[Tensor] = []

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Update with experience replay."""
        x = x.to(self.device)
        y = y.to(self.device)

        self.buffer_x.append(x.detach().cpu())
        self.buffer_y.append(y.detach().cpu())

        if len(self.buffer_x) > self.buffer_size:
            self.buffer_x = self.buffer_x[-self.buffer_size :]
            self.buffer_y = self.buffer_y[-self.buffer_size :]

        self.optimizer.zero_grad()

        batch_size = min(x.size(0), len(self.buffer_x))

        replay_idx = np.random.choice(len(self.buffer_x), batch_size, replace=False)

        combined_x = torch.cat(
            [x, torch.stack([self.buffer_x[i] for i in replay_idx]).to(self.device)]
        )

        combined_y = torch.cat(
            [y, torch.stack([self.buffer_y[i] for i in replay_idx]).to(self.device)]
        )

        loss = F.cross_entropy(self.model(combined_x), combined_y)

        loss.backward()
        self.optimizer.step()

        self.seen_samples += x.size(0)

        return {"loss": loss.item()}

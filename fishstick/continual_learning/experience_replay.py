"""
Experience Replay Implementations for Continual Learning.

Provides various experience replay strategies for storing and sampling
past experiences to prevent catastrophic forgetting.

Classes:
- ExperienceReplay: Base experience replay buffer
- ReservoirBuffer: Reservoir sampling for fixed-size buffer
- WeightedReplayBuffer: Replay buffer with sample weighting
"""

from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F

import numpy as np
from collections import deque
import random
import copy


@dataclass
class ReplaySample:
    """Single sample stored in replay buffer."""

    state: Tensor
    action: Tensor
    reward: float
    next_state: Optional[Tensor] = None
    done: bool = False
    task_id: int = 0
    priority: float = 1.0


class ExperienceReplay:
    """
    Experience Replay Buffer for Continual Learning.

    Standard replay buffer that stores transitions from past experiences
    and samples uniformly for training.

    Args:
        capacity: Maximum number of transitions to store
        device: Device to store tensors on
        sample_strategy: Strategy for sampling ('uniform', 'balanced', 'recent')
    """

    def __init__(
        self,
        capacity: int = 10000,
        device: str = "cpu",
        sample_strategy: str = "uniform",
    ):
        self.capacity = capacity
        self.device = device
        self.sample_strategy = sample_strategy

        self.buffer: deque[ReplaySample] = deque(maxlen=capacity)
        self.task_buffers: Dict[int, deque[ReplaySample]] = {}

    def add(
        self,
        state: Tensor,
        action: Tensor,
        reward: float,
        next_state: Optional[Tensor] = None,
        done: bool = False,
        task_id: int = 0,
    ) -> None:
        """Add a transition to the replay buffer."""
        sample = ReplaySample(
            state=state.detach().cpu(),
            action=action.detach().cpu()
            if isinstance(action, Tensor)
            else torch.tensor(action),
            reward=reward,
            next_state=next_state.detach().cpu() if next_state is not None else None,
            done=done,
            task_id=task_id,
        )

        self.buffer.append(sample)

        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = deque(maxlen=self.capacity)
        self.task_buffers[task_id].append(sample)

    def sample(
        self,
        batch_size: int,
        task_id: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample a batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if task_id is not None and task_id in self.task_buffers:
            source_buffer = self.task_buffers[task_id]
        else:
            source_buffer = self.buffer

        samples = random.sample(list(source_buffer), batch_size)

        states = torch.stack([s.state for s in samples]).to(self.device)
        actions = torch.stack([s.action for s in samples]).to(self.device)
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32).to(
            self.device
        )

        next_states = []
        for s in samples:
            if s.next_state is not None:
                next_states.append(s.next_state)
            else:
                next_states.append(torch.zeros_like(s.state))
        next_states = torch.stack(next_states).to(self.device)

        dones = torch.tensor([s.done for s in samples], dtype=torch.float32).to(
            self.device
        )

        return states, actions, rewards, next_states, dones

    def sample_for_continual(
        self,
        batch_size: int,
        current_task_id: int,
        ratio: float = 0.5,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample for continual learning with balanced current vs past data.

        Args:
            batch_size: Total batch size
            current_task_id: ID of current task
            ratio: Ratio of current task data (1-ratio for past tasks)

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, task_ids)
        """
        current_batch_size = int(batch_size * ratio)
        past_batch_size = batch_size - current_batch_size

        current_samples = []
        past_samples = []

        if (
            current_task_id in self.task_buffers
            and len(self.task_buffers[current_task_id]) > 0
        ):
            current_pool = list(self.task_buffers[current_task_id])
            if len(current_pool) >= current_batch_size:
                current_samples = random.sample(current_pool, current_batch_size)
            else:
                current_samples = current_pool

        all_past = []
        for tid, buf in self.task_buffers.items():
            if tid != current_task_id:
                all_past.extend(list(buf))

        if len(all_past) >= past_batch_size:
            past_samples = random.sample(all_past, past_batch_size)
        else:
            past_samples = all_past

        all_samples = current_samples + past_samples

        if len(all_samples) == 0:
            raise ValueError("No samples in replay buffer")

        states = torch.stack([s.state for s in all_samples]).to(self.device)
        actions = torch.stack([s.action for s in all_samples]).to(self.device)
        rewards = torch.tensor([s.reward for s in all_samples], dtype=torch.float32).to(
            self.device
        )

        next_states = []
        for s in all_samples:
            if s.next_state is not None:
                next_states.append(s.next_state)
            else:
                next_states.append(torch.zeros_like(s.state))
        next_states = torch.stack(next_states).to(self.device)

        dones = torch.tensor([s.done for s in all_samples], dtype=torch.float32).to(
            self.device
        )
        task_ids = torch.tensor([s.task_id for s in all_samples], dtype=torch.long).to(
            self.device
        )

        return states, actions, rewards, next_states, dones, task_ids

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, min_samples: int = 1) -> bool:
        """Check if buffer has enough samples."""
        return len(self.buffer) >= min_samples

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "task_counts": {tid: len(buf) for tid, buf in self.task_buffers.items()},
        }


class ReservoirBuffer:
    """
    Reservoir Sampling Buffer.

    Maintains a fixed-size buffer that guarantees uniform sampling
    from all seen data regardless of stream length.

    Args:
        capacity: Maximum buffer size
        device: Device for tensor storage
    """

    def __init__(self, capacity: int = 10000, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.buffer: List[ReplaySample] = []
        self.count = 0

    def add(
        self,
        state: Tensor,
        action: Tensor,
        reward: float,
        next_state: Optional[Tensor] = None,
        done: bool = False,
        task_id: int = 0,
    ) -> None:
        """Add sample using reservoir sampling."""
        sample = ReplaySample(
            state=state.detach().cpu(),
            action=action.detach().cpu()
            if isinstance(action, Tensor)
            else torch.tensor(action),
            reward=reward,
            next_state=next_state.detach().cpu() if next_state is not None else None,
            done=done,
            task_id=task_id,
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            idx = random.randint(0, self.count)
            if idx < self.capacity:
                self.buffer[idx] = sample

        self.count += 1

    def sample(self, batch_size: int) -> List[ReplaySample]:
        """Sample uniformly from buffer."""
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class WeightedReplayBuffer:
    """
    Weighted Experience Replay Buffer.

    Replay buffer with sample weighting based on recency,
    task difficulty, or learned importance.

    Args:
        capacity: Maximum buffer size
        device: Device for tensor storage
        weight_strategy: Strategy for computing weights ('recency', 'loss', 'uncertainty')
    """

    def __init__(
        self,
        capacity: int = 10000,
        device: str = "cpu",
        weight_strategy: str = "recency",
    ):
        self.capacity = capacity
        self.device = device
        self.weight_strategy = weight_strategy
        self.buffer: List[ReplaySample] = []
        self.weights: List[float] = []

    def add(
        self,
        state: Tensor,
        action: Tensor,
        reward: float,
        next_state: Optional[Tensor] = None,
        done: bool = False,
        task_id: int = 0,
        loss: Optional[float] = None,
    ) -> None:
        """Add sample with computed weight."""
        sample = ReplaySample(
            state=state.detach().cpu(),
            action=action.detach().cpu()
            if isinstance(action, Tensor)
            else torch.tensor(action),
            reward=reward,
            next_state=next_state.detach().cpu() if next_state is not None else None,
            done=done,
            task_id=task_id,
        )

        weight = self._compute_weight(reward, loss)

        if len(self.buffer) >= self.capacity:
            min_idx = self.weights.index(min(self.weights))
            self.buffer[min_idx] = sample
            self.weights[min_idx] = weight
        else:
            self.buffer.append(sample)
            self.weights.append(weight)

    def _compute_weight(self, reward: float, loss: Optional[float]) -> float:
        """Compute sample weight based on strategy."""
        if self.weight_strategy == "recency":
            return 1.0
        elif self.weight_strategy == "loss" and loss is not None:
            return max(loss, 0.01)
        elif self.weight_strategy == "reward":
            return abs(reward) + 0.1
        return 1.0

    def update_weights(self, losses: List[float]) -> None:
        """Update sample weights based on losses."""
        for i, loss in enumerate(losses):
            if i < len(self.weights):
                self.weights[i] = max(loss, 0.01)

    def sample(self, batch_size: int) -> Tuple[List[ReplaySample], Tensor]:
        """Sample with probability proportional to weight."""
        if len(self.buffer) == 0:
            return [], torch.tensor([])

        weights = torch.tensor(self.weights, dtype=torch.float32)
        probs = weights / weights.sum()

        indices = torch.multinomial(probs, batch_size, replacement=True)
        samples = [self.buffer[i] for i in indices.tolist()]

        return samples, indices


class BalancedReplayBuffer(ExperienceReplay):
    """
    Balanced Experience Replay for Multi-Task Continual Learning.

    Ensures equal representation from all tasks in each batch.

    Args:
        capacity: Maximum buffer size per task
        device: Device for tensor storage
        num_tasks: Number of tasks expected
    """

    def __init__(
        self,
        capacity: int = 1000,
        device: str = "cpu",
        num_tasks: int = 10,
    ):
        super().__init__(capacity, device)
        self.num_tasks = num_tasks
        self.task_buffers = {i: deque(maxlen=capacity) for i in range(num_tasks)}

    def sample_balanced(self, batch_size: int) -> Dict[int, List[ReplaySample]]:
        """Sample balanced batch from all tasks."""
        samples_per_task = batch_size // self.num_tasks
        result = {}

        for task_id in range(self.num_tasks):
            if len(self.task_buffers[task_id]) > 0:
                task_samples = list(self.task_buffers[task_id])
                n_samples = min(samples_per_task, len(task_samples))
                result[task_id] = random.sample(task_samples, n_samples)

        return result

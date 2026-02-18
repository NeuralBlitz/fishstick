"""
Prioritized Experience Replay for Continual Learning.

Implements prioritized replay buffers using SumTrees for efficient
sampling based on TD-error or other priority measures.

Classes:
- SumTree: Efficient binary tree for priority sampling
- PrioritizedReplayBuffer: Replay buffer with prioritization
- ProportionalPriority: Priority computation strategy
- RankBasedPriority: Rank-based priority computation
"""

from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import numpy as np
import random
import math


@dataclass
class PrioritizedSample:
    """Sample with priority information."""

    state: Tensor
    action: Tensor
    reward: float
    next_state: Optional[Tensor]
    done: bool
    task_id: int
    priority: float = 1.0
    td_error: float = 0.0


class SumTree:
    """
    Binary Tree for Efficient Priority Sampling.

    Stores priorities in a complete binary tree for O(log n)
    sampling and update operations.

    Args:
        capacity: Number of leaf nodes (must be power of 2)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree: np.ndarray = np.zeros(2 * capacity - 1)
        self.data_pointer: int = 0
        self.size: int = 0

    def add(self, priority: float, data_idx: int) -> None:
        """Add priority at specific index."""
        tree_idx = data_idx + self.capacity - 1
        self._update(tree_idx, priority)

        if self.size < self.capacity:
            self.size += 1

    def update(self, data_idx: int, priority: float) -> None:
        """Update priority at index."""
        tree_idx = data_idx + self.capacity - 1
        self._update(tree_idx, priority)

    def _update(self, tree_idx: int, priority: float) -> None:
        """Update tree node and propagate changes."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def _propagate(self, tree_idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def get(self, value: float) -> Tuple[int, float]:
        """
        Sample priority based on value.

        Args:
            value: Random value in [0, total_priority]

        Returns:
            Tuple of (data_index, priority)
        """
        node_idx = 0

        while node_idx < self.capacity - 1:
            left = 2 * node_idx + 1
            right = left + 1

            if value <= self.tree[left]:
                node_idx = left
            else:
                value -= self.tree[left]
                node_idx = right

        data_idx = node_idx - self.capacity + 1
        return data_idx, self.tree[node_idx]

    def total_priority(self) -> float:
        """Get total priority sum."""
        return self.tree[0]

    def __len__(self) -> int:
        return self.size


class ProportionalPriority:
    """
    Proportional Priority Computation.

    Computes priorities proportional to TD-error using
    exponentiated values.

    Args:
        alpha: Priority exponent (higher = more prioritization)
        beta: Importance sampling exponent
        epsilon: Small constant for numerical stability
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-6):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def compute_priority(self, td_error: float) -> float:
        """Compute priority from TD-error."""
        return (abs(td_error) + self.epsilon) ** self.alpha

    def compute_is_weight(self, priority: float, min_priority: float) -> float:
        """Compute importance sampling weight."""
        p_min = min_priority / (self.epsilon**self.alpha)
        return (self.size * p_min) ** (-self.beta)

    def update(self, td_errors: List[float], capacities: List[int]) -> List[float]:
        """Update priorities from TD-errors."""
        priorities = []
        for td in td_errors:
            p = (abs(td) + self.epsilon) ** self.alpha
            priorities.append(p)
        return priorities


class RankBasedPriority:
    """
    Rank-Based Priority Computation.

    Computes priorities based on rank in priority distribution
    rather than proportional to TD-error.

    Args:
        alpha: Priority exponent
        beta: Importance sampling exponent
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta

    def compute_priority(self, rank: int, total: int) -> float:
        """Compute priority from rank."""
        return (1.0 / (rank + 1)) ** self.alpha

    def compute_is_weight(self, rank: int, total: int) -> float:
        """Compute importance sampling weight."""
        p = (total * (rank + 1)) ** (-self.beta)
        return p


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.

    Samples experiences based on TD-error priority rather than uniformly.

    Args:
        capacity: Maximum buffer size
        alpha: Priority exponent (0 = uniform, 1 = full priority)
        beta: Importance sampling exponent for bias correction
        device: Device for tensor storage
        priority_type: 'proportional' or 'rank_based'
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        device: str = "cpu",
        priority_type: str = "proportional",
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.priority_type = priority_type

        self.tree = SumTree(capacity)
        self.buffer: List[PrioritizedSample] = []
        self.max_priority = 1.0

        if priority_type == "proportional":
            self.priority_strategy = ProportionalPriority(alpha, beta)
        else:
            self.priority_strategy = RankBasedPriority(alpha, beta)

    def add(
        self,
        state: Tensor,
        action: Tensor,
        reward: float,
        next_state: Optional[Tensor] = None,
        done: bool = False,
        task_id: int = 0,
        td_error: Optional[float] = None,
    ) -> None:
        """Add sample with initial priority."""
        if td_error is None:
            priority = self.max_priority
        else:
            priority = self.priority_strategy.compute_priority(td_error)

        sample = PrioritizedSample(
            state=state.detach().cpu(),
            action=action.detach().cpu()
            if isinstance(action, Tensor)
            else torch.tensor(action),
            reward=reward,
            next_state=next_state.detach().cpu() if next_state is not None else None,
            done=done,
            task_id=task_id,
            priority=priority,
            td_error=td_error if td_error is not None else 0.0,
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
            self.tree.add(priority, len(self.buffer) - 1)
        else:
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = sample
            self.tree.update(idx, priority)

    def sample(
        self, batch_size: int
    ) -> Tuple[List[PrioritizedSample], List[int], Tensor]:
        """
        Sample batch based on priorities.

        Returns:
            Tuple of (samples, indices, weights)
        """
        batch = []
        indices = []
        weights = []

        segment = self.tree.total_priority() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)

            idx, priority = self.tree.get(value)

            if idx >= len(self.buffer):
                continue

            batch.append(self.buffer[idx])
            indices.append(idx)

            weight = self._compute_weight(priority)
            weights.append(weight)

        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def _compute_weight(self, priority: float) -> float:
        """Compute importance sampling weight."""
        min_priority = self.tree.tree[-self.tree.capacity :].min()
        p = (priority / min_priority) ** (-self.beta)
        return p / max(self.buffer).priority if len(self.buffer) > 0 else 1.0

    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        """Update priorities based on new TD-errors."""
        for idx, td in zip(indices, td_errors):
            priority = self.priority_strategy.compute_priority(td)
            self.tree.update(idx, priority)
            self.buffer[idx].td_error = td
            self.buffer[idx].priority = priority

            if priority > self.max_priority:
                self.max_priority = priority

    def __len__(self) -> int:
        return len(self.buffer)


class HierarchicalPriorityBuffer:
    """
    Hierarchical Priority Buffer with Multiple Levels.

    Maintains separate priority queues for different task types
    or experience categories.

    Args:
        capacity: Capacity per level
        num_levels: Number of priority levels
        device: Device for storage
    """

    def __init__(
        self,
        capacity: int = 5000,
        num_levels: int = 3,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.num_levels = num_levels
        self.device = device

        self.level_buffers: List[PrioritizedReplayBuffer] = [
            PrioritizedReplayBuffer(capacity, device=device) for _ in range(num_levels)
        ]

    def add(
        self,
        state: Tensor,
        action: Tensor,
        reward: float,
        next_state: Optional[Tensor] = None,
        done: bool = False,
        task_id: int = 0,
        level: int = 0,
        td_error: Optional[float] = None,
    ) -> None:
        """Add sample to specified level."""
        level = max(0, min(level, self.num_levels - 1))
        self.level_buffers[level].add(
            state, action, reward, next_state, done, task_id, td_error
        )

    def sample_from_level(
        self, level: int, batch_size: int
    ) -> Tuple[List, List, Tensor]:
        """Sample from specific level."""
        return self.level_buffers[level].sample(batch_size)

    def sample_balanced(self, batch_size: int) -> Tuple[List, List, Tensor]:
        """Sample equally from all levels."""
        samples_per_level = batch_size // self.num_levels
        all_samples = []
        all_indices = []
        all_weights = []

        for level in range(self.num_levels):
            samples, indices, weights = self.sample_from_level(level, samples_per_level)
            all_samples.extend(samples)
            all_indices.extend(indices)
            all_weights.append(weights)

        return all_samples, all_indices, torch.cat(all_weights)

    def __len__(self) -> int:
        return sum(len(buf) for buf in self.level_buffers)

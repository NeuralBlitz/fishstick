"""
Advanced Memory and Attention Architectures for fishstick.

Implements:
- Differentiable memory stacks with push/pop operations
- Neural Turing Machine (NTM) with read/write heads
- Memory attention mechanisms
- Content-based memory addressing
- Hierarchical memory systems (working, episodic, semantic)

Based on:
- Neural Turing Machines (Graves et al., 2014)
- Differentiable Neural Computers (Graves et al., 2016)
- Memory Networks (Weston et al., 2014)
- Differentiable Data Structures (Jang et al., 2016)
"""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.distributions import Categorical


class DifferentiableStack(nn.Module):
    """
    Differentiable stack memory with push and pop operations.

    Uses soft (differentiable) attention for stack operations,
    enabling gradient-based learning of memory access patterns.

    Attributes:
        capacity: Maximum number of memory locations
        item_size: Dimensionality of each memory item
    """

    def __init__(
        self,
        capacity: int = 10,
        item_size: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.capacity = capacity
        self.item_size = item_size
        self.temperature = temperature

        self.memory = nn.Parameter(torch.zeros(capacity, item_size))
        self.stack_pointer = nn.Parameter(torch.zeros(capacity))

        nn.init.xavier_uniform_(self.memory)
        nn.init.zeros_(self.stack_pointer)
        self.stack_pointer.data[0] = 1.0

    def forward(
        self,
        item: Tensor,
        operation: str = "push",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Execute stack operation.

        Args:
            item: Item to push [batch, item_size]
            operation: "push", "pop", or "peek"

        Returns:
            output: Top of stack [batch, item_size]
            info: Dictionary with attention weights and debug info
        """
        batch_size = item.size(0)

        if operation == "push":
            return self._push(item)
        elif operation == "pop":
            return self._pop()
        elif operation == "peek":
            return self._peek()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _push(self, item: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Push item onto stack."""
        batch_size = item.size(0)

        pointer_weights = F.softmax(self.stack_pointer / self.temperature, dim=0)

        read_vector = pointer_weights @ self.memory

        new_memory = self.memory.clone()
        shifted_weights = torch.zeros_like(pointer_weights)
        shifted_weights[1:] = pointer_weights[:-1]

        item_avg = item.mean(dim=0, keepdim=True)
        new_memory = (
            shifted_weights.view(-1, 1) * item_avg
            + (1 - shifted_weights.view(-1, 1)) * self.memory
        )

        self.memory.data = new_memory

        updated_weights = torch.zeros_like(pointer_weights)
        updated_weights[0] = 1.0

        self.stack_pointer.data = updated_weights

        return read_vector, {"pointer_weights": pointer_weights}

    def _pop(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Pop item from stack."""
        pointer_weights = F.softmax(self.stack_pointer / self.temperature, dim=0)
        read_vector = pointer_weights @ self.memory

        updated_weights = torch.zeros_like(pointer_weights)
        updated_weights[:-1] = pointer_weights[1:]

        self.stack_pointer.data = updated_weights

        return read_vector, {"pointer_weights": pointer_weights}

    def _peek(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Peek at top of stack without modifying."""
        pointer_weights = F.softmax(self.stack_pointer / self.temperature, dim=0)
        read_vector = pointer_weights @ self.memory

        return read_vector, {"pointer_weights": pointer_weights}

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Reset stack state for new batch."""
        self.stack_pointer.data = torch.zeros(self.capacity, device=device)
        self.stack_pointer.data[0] = 1.0


class DifferentiableQueue(nn.Module):
    """
    Differentiable queue (FIFO) memory structure.

    Supports soft (differentiable) enqueue and dequeue operations.
    """

    def __init__(
        self,
        capacity: int = 10,
        item_size: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.capacity = capacity
        self.item_size = item_size
        self.temperature = temperature

        self.memory = nn.Parameter(torch.zeros(capacity, item_size))
        self.queue_weights = nn.Parameter(torch.zeros(capacity))

        nn.init.xavier_uniform_(self.memory)
        nn.init.zeros_(self.queue_weights)
        self.queue_weights.data[0] = 1.0

    def forward(
        self,
        item: Tensor,
        operation: str = "enqueue",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Execute queue operation."""
        if operation == "enqueue":
            return self._enqueue(item)
        elif operation == "dequeue":
            return self._dequeue()
        elif operation == "peek":
            return self._peek()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _enqueue(self, item: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Enqueue item."""
        queue_weights = F.softmax(self.queue_weights / self.temperature, dim=0)
        read_vector = queue_weights @ self.memory

        new_memory = self.memory.clone()
        shifted_weights = torch.zeros_like(queue_weights)
        shifted_weights[:-1] = queue_weights[1:]

        new_memory = (
            shifted_weights.view(-1, 1) * item.unsqueeze(0)
            + (1 - shifted_weights.view(-1, 1)) * self.memory
        )

        self.memory.data = new_memory

        updated_weights = torch.zeros_like(queue_weights)
        updated_weights[0] = 1.0
        self.queue_weights.data = updated_weights

        return read_vector, {"queue_weights": queue_weights}

    def _dequeue(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Dequeue item (FIFO)."""
        queue_weights = F.softmax(self.queue_weights / self.temperature, dim=0)
        read_vector = queue_weights @ self.memory

        updated_weights = torch.zeros_like(queue_weights)
        updated_weights[:-1] = queue_weights[1:]
        self.queue_weights.data = updated_weights

        return read_vector, {"queue_weights": queue_weights}

    def _peek(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Peek at front of queue."""
        queue_weights = F.softmax(self.queue_weights / self.temperature, dim=0)
        read_vector = queue_weights @ self.memory

        return read_vector, {"queue_weights": queue_weights}


class DifferentiableDeque(nn.Module):
    """
    Differentiable double-ended queue (deque).

    Supports push_front, push_back, pop_front, pop_back operations.
    """

    def __init__(
        self,
        capacity: int = 10,
        item_size: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.capacity = capacity
        self.item_size = item_size
        self.temperature = temperature

        self.memory = nn.Parameter(torch.zeros(capacity, item_size))
        self.front_pointer = nn.Parameter(torch.zeros(capacity))
        self.back_pointer = nn.Parameter(torch.zeros(capacity))

        nn.init.xavier_uniform_(self.memory)
        nn.init.zeros_(self.front_pointer)
        nn.init.zeros_(self.back_pointer)
        self.front_pointer.data[0] = 1.0
        self.back_pointer.data[-1] = 1.0

    def forward(
        self,
        item: Tensor,
        operation: str = "push_back",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Execute deque operation."""
        if operation == "push_front":
            return self._push_front(item)
        elif operation == "push_back":
            return self._push_back(item)
        elif operation == "pop_front":
            return self._pop_front()
        elif operation == "pop_back":
            return self._pop_back()
        elif operation == "peek_front":
            return self._peek_front()
        elif operation == "peek_back":
            return self._peek_back()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _push_front(self, item: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Push to front of deque."""
        front_weights = F.softmax(self.front_pointer / self.temperature, dim=0)
        read_vector = front_weights @ self.memory

        new_memory = self.memory.clone()
        shifted = torch.zeros_like(front_weights)
        shifted[1:] = front_weights[:-1]

        new_memory = (
            shifted.view(-1, 1) * item.unsqueeze(0)
            + (1 - shifted.view(-1, 1)) * self.memory
        )

        self.memory.data = new_memory

        updated = torch.zeros_like(front_weights)
        updated[0] = 1.0
        self.front_pointer.data = updated
        self.back_pointer.data = updated

        return read_vector, {"weights": front_weights}

    def _push_back(self, item: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Push to back of deque."""
        back_weights = F.softmax(self.back_pointer / self.temperature, dim=0)
        read_vector = back_weights @ self.memory

        new_memory = self.memory.clone()
        shifted = torch.zeros_like(back_weights)
        shifted[:-1] = back_weights[1:]

        new_memory = (
            shifted.view(-1, 1) * item.unsqueeze(0)
            + (1 - shifted.view(-1, 1)) * self.memory
        )

        self.memory.data = new_memory

        updated = torch.zeros_like(back_weights)
        updated[-1] = 1.0
        self.front_pointer.data = updated
        self.back_pointer.data = updated

        return read_vector, {"weights": back_weights}

    def _pop_front(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Pop from front."""
        front_weights = F.softmax(self.front_pointer / self.temperature, dim=0)
        read_vector = front_weights @ self.memory

        updated = torch.zeros_like(front_weights)
        updated[:-1] = front_weights[1:]
        self.front_pointer.data = updated
        self.back_pointer.data = updated

        return read_vector, {"weights": front_weights}

    def _pop_back(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Pop from back."""
        back_weights = F.softmax(self.back_pointer / self.temperature, dim=0)
        read_vector = back_weights @ self.memory

        updated = torch.zeros_like(back_weights)
        updated[1:] = back_weights[:-1]
        self.front_pointer.data = updated
        self.back_pointer.data = updated

        return read_vector, {"weights": back_weights}

    def _peek_front(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Peek at front."""
        front_weights = F.softmax(self.front_pointer / self.temperature, dim=0)
        read_vector = front_weights @ self.memory
        return read_vector, {"weights": front_weights}

    def _peek_back(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Peek at back."""
        back_weights = F.softmax(self.back_pointer / self.temperature, dim=0)
        read_vector = back_weights @ self.memory
        return read_vector, {"weights": back_weights}


class PriorityQueue(nn.Module):
    """
    Differentiable priority queue.

    Items are retrieved based on learned priority scores.
    """

    def __init__(
        self,
        capacity: int = 10,
        item_size: int = 64,
        priority_dim: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.capacity = capacity
        self.item_size = item_size
        self.priority_dim = priority_dim
        self.temperature = temperature

        self.memory = nn.Parameter(torch.zeros(capacity, item_size))
        self.priorities = nn.Parameter(torch.zeros(capacity, priority_dim))

        nn.init.xavier_uniform_(self.memory)

    def forward(
        self,
        item: Optional[Tensor] = None,
        operation: str = "enqueue",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Execute priority queue operation."""
        if operation == "enqueue" and item is not None:
            return self._enqueue(item)
        elif operation == "dequeue":
            return self._dequeue()
        elif operation == "peek":
            return self._peek()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _enqueue(self, item: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Enqueue with priority (highest priority first)."""
        priority_scores = self.priorities.mean(dim=-1)
        attention = F.softmax(priority_scores / self.temperature, dim=0)

        read_vector = attention @ self.memory

        min_idx = torch.argmin(priority_scores)
        self.memory.data[min_idx] = item.squeeze(0)
        self.priorities.data[min_idx] = torch.randn_like(self.priorities[0]) * 0.1

        return read_vector, {"attention": attention, "priority": priority_scores}

    def _dequeue(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Dequeue highest priority item."""
        priority_scores = self.priorities.mean(dim=-1)
        attention = F.softmax(priority_scores / self.temperature, dim=0)

        read_vector = attention @ self.memory

        max_idx = torch.argmax(priority_scores)
        self.priorities.data[max_idx] = float("-inf")

        return read_vector, {"attention": attention, "priority": priority_scores}

    def _peek(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Peek at highest priority item."""
        priority_scores = self.priorities.mean(dim=-1)
        attention = F.softmax(priority_scores / self.temperature, dim=0)

        read_vector = attention @ self.memory
        return read_vector, {"attention": attention, "priority": priority_scores}

    def update_priorities(self, indices: Tensor, priorities: Tensor) -> None:
        """Update priorities for specific items."""
        self.priorities.data[indices] = priorities


class DifferentiableStackEnsemble(nn.Module):
    """
    Ensemble of differentiable data structures.

    Combines stack, queue, and priority queue for diverse memory operations.
    """

    def __init__(
        self,
        capacity: int = 10,
        item_size: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.capacity = capacity
        self.item_size = item_size

        self.stack = DifferentiableStack(capacity, item_size, temperature)
        self.queue = DifferentiableQueue(capacity, item_size, temperature)
        self.priority_queue = PriorityQueue(
            capacity, item_size, temperature=temperature
        )

        self.selector = nn.Linear(item_size * 3, 3)

    def forward(
        self,
        item: Tensor,
        operation: str = "push",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Select and execute appropriate memory operation.

        Returns:
            output: Blended output from all structures
            info: Dictionary with individual operation outputs
        """
        stack_out, stack_info = self.stack(item, operation)
        queue_out, queue_info = self.queue(item, operation)
        pq_out, pq_info = self.priority_queue(item, operation)

        combined = torch.cat([stack_out, queue_out, pq_out], dim=-1)
        weights = F.softmax(self.selector(combined.mean(dim=0, keepdim=True)), dim=-1)

        output = (
            weights[0, 0] * stack_out
            + weights[0, 1] * queue_out
            + weights[0, 2] * pq_out
        )

        info = {
            "stack": stack_out,
            "queue": queue_out,
            "priority_queue": pq_out,
            "selector_weights": weights,
            "stack_info": stack_info,
            "queue_info": queue_info,
            "pq_info": pq_info,
        }

        return output, info

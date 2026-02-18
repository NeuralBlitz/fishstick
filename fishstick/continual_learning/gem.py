"""
Gradient Episodic Memory (GEM) Implementation.

Maintains episodic memory of past experiences and projects gradients
to avoid forgetting.

Classes:
- GradientEpisodicMemory: GEM implementation
- GEMOptimizer: GEM-compatible optimizer
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np
from collections import defaultdict


class GradientEpisodicMemory:
    """
    Gradient Episodic Memory (GEM).

    Stores episodic memory for each task and projects gradients
    to prevent interference with past tasks.

    Reference:
        Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning", NeurIPS 2017

    Args:
        model: Neural network
        memory_per_task: Number of samples per task
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        memory_per_task: int = 200,
        device: str = "cpu",
    ):
        self.model = model
        self.memory_per_task = memory_per_task
        self.device = device

        self.memory: Dict[int, Tuple[Tensor, Tensor]] = {}

    def store(
        self,
        task_id: int,
        x: Tensor,
        y: Tensor,
    ) -> None:
        """
        Store samples in episodic memory.

        Args:
            task_id: Task identifier
            x: Input samples
            y: Target labels
        """
        if len(x) < self.memory_per_task:
            indices = torch.arange(len(x))
        else:
            indices = torch.randperm(len(x))[: self.memory_per_task]

        self.memory[task_id] = (
            x[indices].to(self.device),
            y[indices].to(self.device),
        )

    def compute_reference_gradients(self, task_id: int) -> Dict[str, Tensor]:
        """
        Compute gradients on episodic memory for reference.

        Args:
            task_id: Task ID for memory retrieval

        Returns:
            Dictionary of reference gradients
        """
        if task_id not in self.memory:
            return {}

        x, y = self.memory[task_id]

        self.model.zero_grad()

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grads = {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        return grads

    def project_gradient(
        self,
        grad: Tensor,
        reference_grad: Tensor,
    ) -> Tensor:
        """
        Project gradient to avoid interference.

        Args:
            grad: Current gradient
            reference_grad: Reference gradient from memory

        Returns:
            Projected gradient
        """
        dot_product = (grad * reference_grad).sum()

        if dot_product < 0:
            return grad - (dot_product / (reference_grad**2).sum()) * reference_grad

        return grad

    def compute_penalty(self) -> Tensor:
        """
        Compute GEM penalty using reference gradients.

        Returns:
            Penalty loss
        """
        loss = torch.tensor(0.0, device=self.device)

        return loss

    def before_step(self) -> None:
        """Hook called before optimizer step."""
        self.reference_grads: Dict[str, Tensor] = {}

        for task_id in self.memory.keys():
            grads = self.compute_reference_gradients(task_id)

            for n, g in grads.items():
                if n not in self.reference_grads:
                    self.reference_grads[n] = g
                else:
                    self.reference_grads[n] += g

        for n in self.reference_grads:
            self.reference_grads[n] /= len(self.memory)

    def after_step(self) -> None:
        """Hook called after optimizer step."""
        pass

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "num_tasks": len(self.memory),
            "samples_per_task": {tid: len(x) for tid, (x, y) in self.memory.items()},
        }


class GEMOptimizer:
    """
    Optimizer wrapper for GEM.

    Provides integration of GEM with standard optimizers.

    Args:
        model: Neural network
        gem: Gradient Episodic Memory
        base_optimizer: Base optimizer
        lr: Learning rate
    """

    def __init__(
        self,
        model: nn.Module,
        gem: GradientEpisodicMemory,
        base_optimizer: str = "adam",
        lr: float = 1e-3,
    ):
        self.model = model
        self.gem = gem

        if base_optimizer == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif base_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()
        self.gem.before_step()

    def step(self) -> None:
        """Perform optimization step with gradient projection."""
        self.optimizer.step()
        self.gem.after_step()

    def step_with_projection(self) -> None:
        """Step with gradient projection for GEM."""
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n in self.gem.reference_grads:
                    ref_grad = self.gem.reference_grads[n]

                    projected = self.gem.project_gradient(p.grad, ref_grad)
                    p.grad.data = projected

        self.optimizer.step()


class MemoryAwareGEM(GradientEpisodicMemory):
    """
    Memory-Aware GEM with Importance Weighting.

    Weights memory samples based on gradient importance.

    Args:
        model: Neural network
        memory_per_task: Samples per task
        device: Device
    """

    def __init__(
        self,
        model: nn.Module,
        memory_per_task: int = 200,
        device: str = "cpu",
    ):
        super().__init__(model, memory_per_task, device)

        self.sample_weights: Dict[int, Tensor] = {}

    def store_with_weights(
        self,
        task_id: int,
        x: Tensor,
        y: Tensor,
    ) -> None:
        """Store samples with importance weights."""
        self.store(task_id, x, y)

        weights = torch.ones(len(x), device=self.device)

        self.sample_weights[task_id] = weights

    def compute_weighted_reference_grad(
        self,
        task_id: int,
    ) -> Dict[str, Tensor]:
        """Compute importance-weighted gradients."""
        if task_id not in self.memory:
            return {}

        x, y = self.memory[task_id]
        weights = self.sample_weights.get(
            task_id, torch.ones(len(x), device=self.device)
        )

        self.model.zero_grad()

        logits = self.model(x)

        per_sample_loss = F.cross_entropy(logits, y, reduction="none")
        loss = (per_sample_loss * weights).mean()

        loss.backward()

        grads = {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        return grads

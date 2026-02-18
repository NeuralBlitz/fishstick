"""
PackNet Implementation for Continual Learning.

Prunes and freezes weights after each task to prevent forgetting.

Classes:
- PackNetPruner: Weight pruning utility
- PackNetMethod: Complete PackNet method
- MultiTaskPackNet: Multi-task PackNet variant
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np


class PackNetPruner:
    """
    Weight Pruning Utility for PackNet.

    Provides utilities for pruning and managing network weights.

    Args:
        model: Neural network to prune
        sparsity: Fraction of weights to prune
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
    ):
        self.model = model
        self.sparsity = sparsity

        self.masks: Dict[str, Tensor] = {}
        self.task_masks: Dict[int, Dict[str, Tensor]] = {}

    def compute_importance(
        self,
        dataloader: Optional[Any] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute parameter importance based on weight magnitudes.

        Args:
            dataloader: Optional dataloader for importance computation

        Returns:
            Dictionary of importance scores per parameter
        """
        importance = {}

        for name, param in self.model.named_parameters():
            if "weight" in name:
                importance[name] = param.data.abs()
            elif "bias" in name:
                importance[name] = param.data.abs()

        return importance

    def compute_gradient_importance(
        self,
        dataloader: Any,
    ) -> Dict[str, Tensor]:
        """Compute importance based on gradient magnitudes."""
        importance = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance[name] = torch.zeros_like(param)

        self.model.eval()

        for inputs, _ in dataloader:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = outputs.sum()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    importance[name] += param.grad.abs()

        return importance

    def prune(
        self,
        task_id: int,
        sparsity: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        """
        Prune weights for a task.

        Args:
            task_id: Task identifier
            sparsity: Optional override for sparsity

        Returns:
            Dictionary of masks for the task
        """
        if sparsity is None:
            sparsity = self.sparsity

        masks = {}
        importance = self.compute_importance()

        for name, param in self.model.named_parameters():
            if "weight" not in name and "bias" not in name:
                continue

            imp = importance.get(name, param.data.abs())

            threshold = torch.quantile(imp.flatten(), sparsity)

            mask = (imp > threshold).float()

            masks[name] = mask

            param.data *= mask

            if task_id not in self.task_masks:
                self.task_masks[task_id] = {}
            self.task_masks[task_id][name] = mask

        return masks

    def apply_masks(self, task_id: int) -> None:
        """Apply masks for a specific task."""
        if task_id not in self.task_masks:
            return

        for name, mask in self.task_masks[task_id].items():
            for n, param in self.model.named_parameters():
                if n == name:
                    param.data *= mask

    def get_active_params(self, task_id: int) -> List[str]:
        """Get list of active parameter names for task."""
        active = set()

        for tid in range(task_id + 1):
            if tid in self.task_masks:
                for name in self.task_masks[tid]:
                    active.add(name)

        return list(active)

    def freeze_task(self, task_id: int) -> None:
        """Freeze parameters used by task."""
        active = self.get_active_params(task_id)

        for name, param in self.model.named_parameters():
            if name in active:
                param.requires_grad = False

    def unfreeze_task(self, task_id: int) -> None:
        """Unfreeze parameters for a task."""
        if task_id in self.task_masks:
            for name, param in self.model.named_parameters():
                if name in self.task_masks[task_id]:
                    param.requires_grad = True


class PackNetMethod(nn.Module):
    """
    PackNet: Adding Multiple Tasks to a Single Network.

    Prunes and freezes weights after each task to prevent forgetting
    while leaving remaining weights free for new tasks.

    Reference:
        Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network", CVPR 2018

    Args:
        base_model: Base neural network
        num_tasks: Maximum number of tasks
        sparsity_per_task: Sparsity level per task
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_tasks: int = 10,
        sparsity_per_task: float = 0.5,
    ):
        super().__init__()

        self.base_model = base_model
        self.num_tasks = num_tasks
        self.sparsity = sparsity_per_task

        self.pruner = PackNetPruner(base_model, sparsity_per_task)
        self.current_task = 0
        self.task_masks: Dict[int, Dict[str, Tensor]] = {}

    def forward(self, x: Tensor, task_id: Optional[int] = None) -> Tensor:
        """Forward pass with task-specific masking."""
        return self.base_model(x)

    def after_task(self, task_id: int, dataloader: Optional[Any] = None) -> None:
        """
        Prune and freeze weights after task completion.

        Args:
            task_id: Completed task ID
            dataloader: Optional dataloader for importance computation
        """
        masks = self.pruner.prune(task_id)
        self.task_masks[task_id] = masks

        self.pruner.freeze_task(task_id)
        self.current_task = task_id + 1

    def get_task_performance(self, task_id: int) -> Dict[str, float]:
        """Get performance metrics for a task."""
        return {
            "task_id": task_id,
            "num_masks": len(self.task_masks.get(task_id, {})),
        }

    def get_sparsity(self) -> Dict[str, float]:
        """Get overall network sparsity."""
        total = 0
        zero = 0

        for task_id, masks in self.task_masks.items():
            for name, mask in masks.items():
                total += mask.numel()
                zero += (mask == 0).sum().item()

        if total == 0:
            return {"total": 0.0}

        return {
            "total": total,
            "zero": zero,
            "sparsity": zero / total,
        }


class MultiTaskPackNet(nn.Module):
    """
    Multi-Task PackNet with Multiple Output Heads.

    Combines PackNet pruning with multiple output heads
    for different tasks.

    Args:
        base_model: Base feature extractor
        num_tasks: Number of tasks
        hidden_dim: Hidden dimension
        num_classes_per_task: Classes per task
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_tasks: int = 10,
        hidden_dim: int = 512,
        num_classes_per_task: int = 10,
    ):
        super().__init__()

        self.base_model = base_model
        self.num_tasks = num_tasks

        self.task_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes_per_task) for _ in range(num_tasks)]
        )

        self.pruner = PackNetPruner(base_model)
        self.current_task = 0

    def forward(
        self,
        x: Tensor,
        task_id: int,
    ) -> Tensor:
        """
        Forward pass for specific task.

        Args:
            x: Input tensor
            task_id: Task identifier

        Returns:
            Task-specific logits
        """
        features = self.base_model(x)

        if task_id < len(self.task_heads):
            return self.task_heads[task_id](features)
        else:
            return self.task_heads[0](features)

    def after_task(self, task_id: int) -> None:
        """Prune and freeze after task."""
        self.pruner.prune(task_id)
        self.pruner.freeze_task(task_id)
        self.current_task = task_id + 1


class IterativePackNet:
    """
    Iterative PackNet with Multiple Pruning Cycles.

    Performs iterative pruning within each task for
    finer-grained weight allocation.

    Args:
        model: Neural network
        num_iterations: Pruning iterations per task
        initial_sparsity: Initial sparsity per iteration
    """

    def __init__(
        self,
        model: nn.Module,
        num_iterations: int = 3,
        initial_sparsity: float = 0.1,
    ):
        self.model = model
        self.num_iterations = num_iterations
        self.initial_sparsity = initial_sparsity

        self.task_masks: Dict[int, Dict[str, Tensor]] = {}

    def prune_iteration(
        self,
        task_id: int,
        iteration: int,
    ) -> None:
        """Perform one pruning iteration."""
        sparsity = 1 - ((1 - self.initial_sparsity) ** (iteration + 1))

        importance = {}

        for name, param in self.model.named_parameters():
            if "weight" in name:
                importance[name] = param.data.abs()

        masks = {}

        for name, param in self.model.named_parameters():
            if name in importance:
                imp = importance[name]

                threshold = torch.quantile(imp.flatten(), sparsity)
                mask = (imp > threshold).float()

                masks[name] = mask
                param.data *= mask

        if task_id not in self.task_masks:
            self.task_masks[task_id] = masks
        else:
            for name in masks:
                self.task_masks[task_id][name] *= masks[name]

    def prune_task(self, task_id: int) -> None:
        """Prune task across multiple iterations."""
        for i in range(self.num_iterations):
            self.prune_iteration(task_id, i)

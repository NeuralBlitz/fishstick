"""
Reptile Meta-Learning Algorithm Implementation.

Provides Reptile and related meta-learning algorithms:
- Reptile (first-order meta-learning)
- Reptile with weight decay
- Reptile with support for various architectures
- Efficient implementation with gradient accumulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np

from .maml import Task, MetaLearner, create_few_shot_task


class Reptile(nn.Module):
    """
    Reptile: First-order Meta-Learning.

    Reptile learns a parameter initialization that can quickly adapt to new tasks
    using stochastic gradient descent. Unlike MAML, Reptile only uses first-order
    gradients, making it more computationally efficient.

    Reference: Nichol et al., "On First-Order Meta-Learning Algorithms", 2018

    Args:
        model: Base model to meta-learn
        inner_lr: Learning rate for task adaptation
        inner_steps: Number of SGD steps in inner loop
        epsilon: Small constant for numerical stability

    Example:
        >>> model = SimpleCNN()
        >>> reptile = Reptile(model, inner_lr=0.01, inner_steps=5)
        >>> adapted_params = reptile.adapt(support_x, support_y)
        >>> predictions = reptile.forward(query_x, adapted_params)
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        epsilon: float = 1e-8,
        grad_clip: Optional[float] = None,
    ):
        super().__init__()

        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.epsilon = epsilon
        self.grad_clip = grad_clip

    def forward(self, x: Tensor, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """
        Forward pass with optional parameter override.

        Args:
            x: Input tensor
            params: Optional parameter dict

        Returns:
            Model output
        """
        if params is None:
            return self.model(x)
        return self._forward_with_params(x, params)

    def _forward_with_params(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """Forward with specific parameters."""
        # Create temporary model
        temp_model = copy.deepcopy(self.model)
        temp_model.load_state_dict({k: v for k, v in params.items()}, strict=False)

        with torch.no_grad():
            return temp_model(x)

    def clone_params(self) -> Dict[str, Tensor]:
        """Get a copy of current parameters."""
        return {n: p.clone() for n, p in self.model.named_parameters()}

    def adapt(
        self,
        support_x: Tensor,
        support_y: Tensor,
        inner_lr: Optional[float] = None,
        inner_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]]:
        """
        Adapt to a task using SGD (inner loop).

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            inner_lr: Learning rate for inner loop
            inner_steps: Number of gradient steps
            return_trajectory: Whether to return parameter trajectory

        Returns:
            Adapted parameters, or (adapted_params, trajectory) if return_trajectory=True
        """
        lr = inner_lr if inner_lr is not None else self.inner_lr
        steps = inner_steps if inner_steps is not None else self.inner_steps

        # Initialize from current model
        params = self.clone_params()

        trajectory = [] if return_trajectory else None

        for step in range(steps):
            # Forward pass
            outputs = self._forward_with_params(support_x, params)
            loss = F.cross_entropy(outputs, support_y)

            # Compute gradients manually (for efficiency)
            grads = self._compute_gradients(support_x, support_y, params)

            # Update parameters
            with torch.no_grad():
                for name in params:
                    if name in grads:
                        params[name] = params[name] - lr * grads[name]

            if return_trajectory:
                trajectory.append({n: p.clone() for n, p in params.items()})

        if return_trajectory:
            return params, trajectory
        return params

    def _compute_gradients(
        self,
        x: Tensor,
        y: Tensor,
        params: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute gradients for parameters."""
        # Get model output
        temp_model = copy.deepcopy(self.model)
        temp_model.load_state_dict({k: v for k, v in params.items()}, strict=False)

        output = temp_model(x)
        loss = F.cross_entropy(output, y)

        # Compute gradients
        temp_model.zero_grad()
        loss.backward()

        grads = {
            n: p.grad.clone()
            for n, p in temp_model.named_parameters()
            if p.grad is not None
        }

        return grads

    def meta_update(
        self,
        tasks: List[Task],
        optimizer: torch.optim.Optimizer,
        inner_lr: Optional[float] = None,
        inner_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Perform meta-update using Reptile.

        Args:
            tasks: List of tasks for meta-update
            optimizer: Optimizer for outer loop
            inner_lr: Learning rate for inner loop
            inner_steps: Number of inner loop steps

        Returns:
            Dictionary of metrics
        """
        optimizer.zero_grad()

        # Get initial parameters
        initial_params = self.clone_params()

        total_task_loss = 0.0

        for task in tasks:
            # Clone initial params for this task
            task_params = {k: v.clone() for k, v in initial_params.items()}

            # Adapt to task
            adapted_params = self._adapt_task(
                task.train_support[0],
                task.train_support[1],
                task_params,
                inner_lr,
                inner_steps,
            )

            # Evaluate on query set
            query_output = self._forward_with_params(
                task.train_query[0], adapted_params
            )
            task_loss = F.cross_entropy(query_output, task.train_query[1])
            total_task_loss += task_loss.item()

            # Compute Reptile gradient: (theta - theta_initial)
            with torch.no_grad():
                for name in initial_params:
                    if name in adapted_params:
                        # Reptile uses: grad = (theta_adapted - theta_initial) / inner_lr
                        reptile_grad = (adapted_params[name] - initial_params[name]) / (
                            inner_lr if inner_lr is not None else self.inner_lr
                        )

                        # Set gradient for parameter
                        if name in dict(self.model.named_parameters()):
                            p = dict(self.model.named_parameters())[name]
                            if p.grad is None:
                                p.grad = torch.zeros_like(p)
                            p.grad.add_(reptile_grad)

        # Average loss
        meta_loss = total_task_loss / len(tasks)

        # Gradient clipping
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        # Update
        optimizer.step()

        return {
            "meta_loss": meta_loss,
            "task_loss": total_task_loss / len(tasks),
        }

    def _adapt_task(
        self,
        support_x: Tensor,
        support_y: Tensor,
        params: Dict[str, Tensor],
        inner_lr: Optional[float] = None,
        inner_steps: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Adapt to a single task."""
        lr = inner_lr if inner_lr is not None else self.inner_lr
        steps = inner_steps if inner_steps is not None else self.inner_steps

        for step in range(steps):
            output = self._forward_with_params(support_x, params)
            loss = F.cross_entropy(output, support_y)

            grads = self._compute_gradients(support_x, support_y, params)

            with torch.no_grad():
                for name in params:
                    if name in grads:
                        params[name] = params[name] - lr * grads[name]

        return params

    def evaluate_task(
        self,
        task: Task,
        inner_lr: Optional[float] = None,
        inner_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on a task after adaptation.

        Args:
            task: Task to evaluate
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner loop steps

        Returns:
            Dictionary with accuracies
        """
        # Adapt on support
        adapted_params = self.adapt(
            task.train_support[0],
            task.train_support[1],
            inner_lr,
            inner_steps,
        )

        self.model.eval()

        with torch.no_grad():
            # Support accuracy
            support_output = self._forward_with_params(
                task.train_support[0], adapted_params
            )
            _, support_pred = support_output.max(1)
            support_acc = support_pred.eq(task.train_support[1]).float().mean().item()

            # Query accuracy
            query_output = self._forward_with_params(
                task.train_query[0], adapted_params
            )
            _, query_pred = query_output.max(1)
            query_acc = query_pred.eq(task.train_query[1]).float().mean().item()

        return {
            "support_accuracy": support_acc,
            "query_accuracy": query_acc,
        }


class ReptileWithWeightDecay(Reptile):
    """
    Reptile with weight decay regularization.

    Adds L2 regularization to prevent overfitting to specific tasks.
    """

    def __init__(self, *args, weight_decay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_decay = weight_decay

    def meta_update(
        self,
        tasks: List[Task],
        optimizer: torch.optim.Optimizer,
        inner_lr: Optional[float] = None,
        inner_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Meta-update with weight decay."""
        # First apply Reptile update
        result = super().meta_update(tasks, optimizer, inner_lr, inner_steps)

        # Apply weight decay
        if self.weight_decay > 0:
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data.mul_(1 - self.weight_decay)

        return result


class FOMAML(Reptile):
    """
    First-Order MAML (FOMAML) implementation using Reptile-style update.

    Combines aspects of both MAML and Reptile for efficient first-order
    meta-learning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def meta_update(
        self,
        tasks: List[Task],
        optimizer: torch.optim.Optimizer,
        inner_lr: Optional[float] = None,
        inner_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """FOMAML-style meta-update."""
        optimizer.zero_grad()

        initial_params = self.clone_params()

        total_task_loss = 0.0

        for task in tasks:
            # Clone initial params
            task_params = {k: v.clone() for k, v in initial_params.items()}

            # Adapt (only compute final params, not trajectory)
            adapted_params = self._adapt_task(
                task.train_support[0],
                task.train_support[1],
                task_params,
                inner_lr,
                inner_steps,
            )

            # Evaluate on query set
            query_output = self._forward_with_params(
                task.train_query[0], adapted_params
            )
            task_loss = F.cross_entropy(query_output, task.train_query[1])
            total_task_loss += task_loss.item()

            # Compute gradient of query loss w.r.t. adapted params
            # Then backprop through adaptation
            temp_model = copy.deepcopy(self.model)
            temp_model.load_state_dict(
                {k: v for k, v in adapted_params.items()}, strict=False
            )

            query_output = temp_model(task.train_query[0])
            task_loss = F.cross_entropy(query_output, task.train_query[1])

            task_loss.backward()

            # FOMAML: use the gradients on adapted params directly
            with torch.no_grad():
                for (name, param), adapted_param in zip(
                    self.model.named_parameters(), adapted_params.values()
                ):
                    if param.grad is not None:
                        # Compute gradient from adapted to initial
                        grad = param.grad
                        param.data.sub_(optimizer.param_groups[0]["lr"] * grad)

        meta_loss = total_task_loss / len(tasks)
        optimizer.step()

        return {
            "meta_loss": meta_loss,
            "task_loss": total_task_loss / len(tasks),
        }


@dataclass
class ReptileConfig:
    """Configuration for Reptile."""

    inner_lr: float = 0.01
    inner_steps: int = 5
    epsilon: float = 1e-8
    grad_clip: Optional[float] = None
    weight_decay: float = 0.0


class ReptileTrainer:
    """
    Trainer for Reptile meta-learning.

    Handles task sampling, training, and evaluation.
    """

    def __init__(
        self,
        reptile: Reptile,
        outer_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        meta_batch_size: int = 4,
    ):
        self.reptile = reptile
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size

        self.optimizer = torch.optim.Adam(reptile.parameters(), lr=outer_lr)

    def sample_tasks(
        self,
        task_sampler: Callable[[], Task],
        num_tasks: int,
    ) -> List[Task]:
        """Sample tasks from task distribution."""
        return [task_sampler() for _ in range(num_tasks)]

    def train_step(self, tasks: List[Task]) -> Dict[str, float]:
        """Perform one training step."""
        return self.reptile.meta_update(
            tasks,
            self.optimizer,
            self.inner_lr,
            self.inner_steps,
        )

    def evaluate(self, tasks: List[Task]) -> Dict[str, float]:
        """Evaluate on tasks."""
        support_accs = []
        query_accs = []

        for task in tasks:
            result = self.reptile.evaluate_task(task, self.inner_lr, self.inner_steps)
            support_accs.append(result["support_accuracy"])
            query_accs.append(result["query_accuracy"])

        return {
            "mean_support_accuracy": np.mean(support_accs),
            "mean_query_accuracy": np.mean(query_accs),
            "std_query_accuracy": np.std(query_accs),
        }

    def meta_train(
        self,
        task_sampler: Callable[[], Task],
        num_iterations: int,
        eval_every: int = 100,
    ) -> List[Dict[str, float]]:
        """Meta-training loop."""
        history = []

        for iteration in range(num_iterations):
            # Sample tasks
            tasks = self.sample_tasks(task_sampler, self.meta_batch_size)

            # Train
            metrics = self.train_step(tasks)
            metrics["iteration"] = iteration

            history.append(metrics)

            if iteration % eval_every == 0:
                # Evaluate
                eval_tasks = self.sample_tasks(task_sampler, self.meta_batch_size)
                eval_metrics = self.evaluate(eval_tasks)
                print(
                    f"Iteration {iteration}: query_acc={eval_metrics['mean_query_accuracy']:.4f}"
                )

        return history


def reptile_outer_step(
    model: nn.Module,
    tasks: List[Task],
    inner_lr: float,
    inner_steps: int,
) -> Tensor:
    """
    Compute Reptile outer step loss.

    Args:
        model: Model to meta-learn
        tasks: List of tasks
        inner_lr: Inner loop learning rate
        inner_steps: Number of inner loop steps

    Returns:
        Meta-loss
    """
    # Get initial parameters
    initial_params = {n: p.clone() for n, p in model.named_parameters()}

    total_loss = 0.0

    for task in tasks:
        # Clone params for adaptation
        params = {k: v.clone() for k, v in initial_params.items()}

        # Inner loop: adapt to task
        for _ in range(inner_steps):
            temp_model = copy.deepcopy(model)
            temp_model.load_state_dict({k: v for k, v in params.items()}, strict=False)

            output = temp_model(task.train_support[0])
            loss = F.cross_entropy(output, task.train_support[1])

            temp_model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name in params:
                    if name in dict(temp_model.named_parameters()):
                        grad = dict(temp_model.named_parameters())[name].grad
                        if grad is not None:
                            params[name] = params[name] - inner_lr * grad

        # Evaluate on query
        temp_model = copy.deepcopy(model)
        temp_model.load_state_dict({k: v for k, v in params.items()}, strict=False)

        query_output = temp_model(task.train_query[0])
        query_loss = F.cross_entropy(query_output, task.train_query[1])

        total_loss = total_loss + query_loss

    return total_loss / len(tasks)


def simple_task_sampler(
    x: Tensor,
    y: Tensor,
    num_ways: int = 5,
    num_shots: int = 5,
    num_queries: int = 15,
) -> Callable[[], Task]:
    """Create a task sampler from data."""

    def sampler() -> Task:
        return create_few_shot_task(x, y, num_shots, num_queries, num_ways)

    return sampler

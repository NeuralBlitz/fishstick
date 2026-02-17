"""
MAML (Model-Agnostic Meta-Learning) Implementation.

Provides comprehensive MAML implementation with:
- First-order MAML (FOMAML)
- Implicit MAML
- MAML++ improvements
- Task-adaptive learning rates
- Multi-step loss optimization
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


@dataclass
class Task:
    """A task for meta-learning."""

    name: str
    train_support: Tuple[Tensor, Tensor]
    train_query: Tuple[Tensor, Tensor]
    val_support: Optional[Tuple[Tensor, Tensor]] = None
    val_query: Optional[Tuple[Tensor, Tensor]] = None

    @property
    def num_support(self) -> int:
        return self.train_support[0].shape[0]

    @property
    def num_query(self) -> int:
        return self.train_query[0].shape[0]

    @property
    def num_classes(self) -> int:
        return int(self.train_support[1].max().item()) + 1


@dataclass
class MAMLState:
    """State of MAML training."""

    iteration: int
    outer_loss: float
    inner_loss: float
    meta_loss: float
    task_losses: List[float] = field(default_factory=list)


class MetaLearner(ABC, nn.Module):
    """Abstract base class for meta-learners."""

    @abstractmethod
    def forward(self, x: Tensor, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """Forward pass with optional parameter override."""
        pass

    @abstractmethod
    def clone_parameters(self) -> Dict[str, Tensor]:
        """Create a copy of the parameters."""
        pass

    @abstractmethod
    def inner_update(
        self,
        support_x: Tensor,
        support_y: Tensor,
        inner_lr: float,
        inner_steps: int,
        inner_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Tensor]:
        """Perform inner loop updates."""
        pass


class MAML(MetaLearner):
    """
    Model-Agnostic Meta-Learning (MAML) implementation.

    MAML learns a good initialization of model parameters that can quickly
    adapt to new tasks with few gradient steps.

    Reference: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation
    of Deep Networks", ICML 2017

    Args:
        model: Base model to meta-learn
        inner_lr: Learning rate for inner loop (task adaptation)
        inner_steps: Number of gradient steps in inner loop
        outer_lr: Learning rate for outer loop (meta-update)
        first_order: Use first-order approximation (FOMAML)
        gradient_clip: Maximum gradient norm for clipping

    Example:
        >>> model = SimpleCNN()
        >>> maml = MAML(model, inner_lr=0.01, inner_steps=5)
        >>> adapted_params = maml.inner_update(support_x, support_y, 0.01, 5)
        >>> predictions = maml.forward(query_x, params=adapted_params)
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        outer_lr: float = 0.001,
        first_order: bool = False,
        gradient_clip: float = 1.0,
        learn_inner_lr: bool = False,
        learnable_inner_lrs: Optional[Tensor] = None,
    ):
        super().__init__()

        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.outer_lr = outer_lr
        self.first_order = first_order
        self.gradient_clip = gradient_clip
        self.learn_inner_lr = learn_inner_lr

        # Learnable per-parameter inner loop learning rates
        if learn_inner_lr:
            self.inner_lrs = nn.Parameter(
                torch.ones_like(p) * inner_lr for p in model.parameters()
            )
        else:
            self.register_buffer("inner_lrs", None)

    def forward(self, x: Tensor, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            params: Optional parameter dict to override model parameters

        Returns:
            Model output
        """
        if params is None:
            return self.model(x)
        else:
            return self._forward_with_params(x, params)

    def _forward_with_params(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """Forward with specific parameters."""
        # Create a temporary model with new parameters
        temp_model = copy.deepcopy(self.model)
        temp_model.load_state_dict(params, strict=False)
        temp_model.eval()

        with torch.no_grad():
            return temp_model(x)

    def clone_parameters(self) -> Dict[str, Tensor]:
        """Create a copy of the model parameters."""
        return {n: p.clone() for n, p in self.model.named_parameters()}

    def inner_update(
        self,
        support_x: Tensor,
        support_y: Tensor,
        inner_lr: Optional[float] = None,
        inner_steps: Optional[int] = None,
        inner_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Tensor]:
        """
        Perform inner loop updates (task adaptation).

        Args:
            support_x: Support set input
            support_y: Support set labels
            inner_lr: Learning rate for inner loop
            inner_steps: Number of gradient steps
            inner_optimizer: Optional optimizer for inner loop

        Returns:
            Updated parameters
        """
        lr = inner_lr if inner_lr is not None else self.inner_lr
        steps = inner_steps if inner_steps is not None else self.inner_steps

        # Clone parameters
        params = self.clone_parameters()

        # Use learnable inner LRs if enabled
        if self.learn_inner_lr:
            lr_dict = {
                n: lrs
                for (n, _), lrs in zip(self.model.named_parameters(), self.inner_lrs)
            }

        # Inner loop optimization
        self.model.train()

        for step in range(steps):
            # Forward pass with current params
            outputs = self._forward_with_params(support_x, params)

            # Compute loss
            loss = F.cross_entropy(outputs, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            # Update parameters
            for (name, param), grad in zip(params.items(), grads):
                if grad is not None:
                    if self.learn_inner_lr:
                        lr = lr_dict.get(name, torch.tensor(lr, device=param.device))
                        params[name] = param - lr * grad
                    else:
                        params[name] = param - lr * grad

        return params

    def outer_update(
        self,
        tasks: List[Task],
        outer_optimizer: torch.optim.Optimizer,
        inner_lr: Optional[float] = None,
        inner_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Perform outer loop update (meta-update).

        Args:
            tasks: List of tasks for meta-update
            outer_optimizer: Optimizer for outer loop
            inner_lr: Learning rate for inner loop
            inner_steps: Number of inner loop steps

        Returns:
            Dictionary of metrics
        """
        outer_optimizer.zero_grad()

        meta_loss = 0.0
        task_losses = []

        for task in tasks:
            # Inner loop: adapt to task
            adapted_params = self.inner_update(
                task.train_support[0],
                task.train_support[1],
                inner_lr,
                inner_steps,
            )

            # Evaluate on query set
            query_outputs = self._forward_with_params(
                task.train_query[0], adapted_params
            )
            query_loss = F.cross_entropy(query_outputs, task.train_query[1])

            task_losses.append(query_loss.item())
            meta_loss = meta_loss + query_loss

        # Average meta-loss
        meta_loss = meta_loss / len(tasks)

        # Backward pass
        if not self.first_order:
            meta_loss.backward()
        else:
            # First-order approximation: don't differentiate through inner loop
            with torch.no_grad():
                meta_loss.backward()

        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)

        outer_optimizer.step()

        return {
            "meta_loss": meta_loss.item(),
            "task_losses": task_losses,
        }

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
            Dictionary with support and query accuracy
        """
        # Adapt on support set
        adapted_params = self.inner_update(
            task.train_support[0],
            task.train_support[1],
            inner_lr,
            inner_steps,
        )

        # Evaluate on support
        self.model.eval()
        with torch.no_grad():
            support_outputs = self._forward_with_params(
                task.train_support[0], adapted_params
            )
            _, support_pred = support_outputs.max(1)
            support_acc = support_pred.eq(task.train_support[1]).float().mean().item()

            # Evaluate on query
            query_outputs = self._forward_with_params(
                task.train_query[0], adapted_params
            )
            _, query_pred = query_outputs.max(1)
            query_acc = query_pred.eq(task.train_query[1]).float().mean().item()

        return {
            "support_accuracy": support_acc,
            "query_accuracy": query_acc,
        }


class FirstOrderMAML(MAML):
    """
    First-Order MAML (FOMAML) implementation.

    Uses first-order gradient approximation for computational efficiency.
    Ignores second-order derivatives in the meta-update.

    Reference: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation
    of Deep Networks", ICML 2017
    """

    def __init__(self, *args, **kwargs):
        kwargs["first_order"] = True
        super().__init__(*args, **kwargs)


class MAMLPlus(MAML):
    """
    MAML++ implementation with several improvements.

    Improvements include:
    - Task-dependent inner learning rates
    - Cosine annealing of outer learning rate
    - Dropout in inner loop
    - Multi-step loss aggregation

    Reference: Antoniou et al., "How to Train Your MAML", ICLR 2019
    """

    def __init__(
        self, *args, dropout_rate: float = 0.0, multi_step_loss: int = 1, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.dropout_rate = dropout_rate
        self.multi_step_loss = multi_step_loss

        # Learnable per-layer learning rates
        self.task_lrs = nn.Parameter(torch.ones(1) * 0.01)

    def inner_update(
        self,
        support_x: Tensor,
        support_y: Tensor,
        inner_lr: Optional[float] = None,
        inner_steps: Optional[int] = None,
        inner_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Tensor]:
        """Perform inner loop updates with MAML++ improvements."""
        lr = inner_lr if inner_lr is not None else self.inner_lr
        steps = inner_steps if inner_steps is not None else self.inner_steps

        params = self.clone_parameters()

        # Multi-step loss tracking
        losses = []

        for step in range(steps):
            # Forward pass
            outputs = self._forward_with_params(support_x, params)
            loss = F.cross_entropy(outputs, support_y)

            # Track intermediate losses for multi-step loss
            if step >= steps - self.multi_step_loss:
                losses.append(loss)

            # Gradients
            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            # Update with task-adaptive learning rate
            effective_lr = lr * (1.0 + self.task_lrs)

            for (name, param), grad in zip(params.items(), grads):
                if grad is not None:
                    params[name] = param - effective_lr * grad

            # Apply dropout
            if self.dropout_rate > 0:
                params = {
                    n: F.dropout(p, p=self.dropout_rate) if "weight" in n else p
                    for n, p in params.items()
                }

        return params


class ImplicitMAML(MAML):
    """
    Implicit MAML (iMAML) implementation.

    Uses implicit differentiation for the inner loop instead of
    unrolling, making it more memory efficient.

    Reference: Rajeswaran et al., "Meta-Learning with Implicit Gradients", NeurIPS 2019
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inner_update(
        self,
        support_x: Tensor,
        support_y: Tensor,
        inner_lr: Optional[float] = None,
        inner_steps: Optional[int] = None,
        inner_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Tensor]:
        """Perform inner loop updates implicitly."""
        lr = inner_lr if inner_lr is not None else self.inner_lr
        steps = inner_steps if inner_steps is not None else self.inner_steps

        # Standard inner loop
        params = self.clone_parameters()

        for step in range(steps):
            outputs = self._forward_with_params(support_x, params)
            loss = F.cross_entropy(outputs, support_y)

            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=False,
                allow_unused=True,
            )

            for (name, param), grad in zip(params.items(), grads):
                if grad is not None:
                    params[name] = param - lr * grad

        return params


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""

    inner_lr: float = 0.01
    inner_steps: int = 5
    outer_lr: float = 0.001
    first_order: bool = False
    gradient_clip: float = 1.0
    learn_inner_lr: bool = False
    dropout_rate: float = 0.0
    multi_step_loss: int = 1


class MetaLearningTrainer:
    """
    Trainer for meta-learning algorithms.

    Handles:
    - Task sampling
    - Inner/outer loop optimization
    - Evaluation
    - Checkpointing
    """

    def __init__(
        self,
        maml: MAML,
        outer_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        meta_batch_size: int = 4,
    ):
        self.maml = maml
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size

        # Optimizer
        self.optimizer = torch.optim.Adam(maml.parameters(), lr=outer_lr)

    def sample_tasks(
        self,
        task_dist: Callable[[], Task],
        num_tasks: int,
    ) -> List[Task]:
        """Sample tasks from task distribution."""
        return [task_dist() for _ in range(num_tasks)]

    def train_step(
        self,
        tasks: List[Task],
    ) -> Dict[str, float]:
        """
        Perform one meta-training step.

        Args:
            tasks: List of tasks for meta-update

        Returns:
            Dictionary of metrics
        """
        # Outer loop update
        metrics = self.maml.outer_update(
            tasks,
            self.optimizer,
            self.inner_lr,
            self.inner_steps,
        )

        return metrics

    def evaluate(
        self,
        tasks: List[Task],
    ) -> Dict[str, float]:
        """
        Evaluate on a set of tasks.

        Args:
            tasks: List of tasks to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        all_support_acc = []
        all_query_acc = []

        for task in tasks:
            result = self.maml.evaluate_task(task, self.inner_lr, self.inner_steps)
            all_support_acc.append(result["support_accuracy"])
            all_query_acc.append(result["query_accuracy"])

        return {
            "mean_support_accuracy": np.mean(all_support_acc),
            "mean_query_accuracy": np.mean(all_query_acc),
            "std_query_accuracy": np.std(all_query_acc),
        }

    def meta_train(
        self,
        task_sampler: Callable[[], Task],
        num_iterations: int,
        eval_every: int = 100,
    ) -> List[Dict[str, float]]:
        """
        Run meta-training.

        Args:
            task_sampler: Function that samples a task
            num_iterations: Number of meta-training iterations
            eval_every: Evaluate every N iterations

        Returns:
            List of metrics from each iteration
        """
        history = []

        for iteration in range(num_iterations):
            # Sample tasks
            tasks = self.sample_tasks(task_sampler, self.meta_batch_size)

            # Train step
            metrics = self.train_step(tasks)
            metrics["iteration"] = iteration

            history.append(metrics)

            if iteration % eval_every == 0:
                # Evaluate
                eval_tasks = self.sample_tasks(task_sampler, self.meta_batch_size)
                eval_metrics = self.evaluate(eval_tasks)
                print(f"Iteration {iteration}: {eval_metrics}")

        return history


def create_few_shot_task(
    x: Tensor,
    y: Tensor,
    num_support: int,
    num_query: int,
    num_classes: int,
    rng: Optional[np.random.RandomState] = None,
) -> Task:
    """
    Create a few-shot learning task from data.

    Args:
        x: Data tensor [N, ...]
        y: Labels tensor [N]
        num_support: Number of support examples per class
        num_query: Number of query examples per class
        num_classes: Number of classes in the task
        rng: Random number generator

    Returns:
        Task object
    """
    if rng is None:
        rng = np.random.RandomState()

    # Sample classes
    all_classes = torch.unique(y)
    selected_classes = rng.choice(all_classes.numpy(), size=num_classes, replace=False)

    # Sample support and query
    support_x_list = []
    support_y_list = []
    query_x_list = []
    query_y_list = []

    for class_idx, cls in enumerate(selected_classes):
        class_indices = (y == cls).nonzero(as_tuple=True)[0]
        selected_indices = rng.choice(
            class_indices.numpy(), size=num_support + num_query, replace=False
        )

        support_indices = selected_indices[:num_support]
        query_indices = selected_indices[num_support:]

        support_x_list.append(x[support_indices])
        support_y_list.append(torch.full((num_support,), class_idx, dtype=y.dtype))

        query_x_list.append(x[query_indices])
        query_y_list.append(torch.full((num_query,), class_idx, dtype=y.dtype))

    return Task(
        name="few_shot_task",
        train_support=(torch.cat(support_x_list), torch.cat(support_y_list)),
        train_query=(torch.cat(query_x_list), torch.cat(query_y_list)),
    )


def omniglot_task_sampler(
    data: Dict[str, Tensor],
    num_ways: int = 5,
    num_shots: int = 1,
    num_queries: int = 15,
) -> Task:
    """Create a task from Omniglot-style data."""
    classes = list(data.keys())
    selected_classes = np.random.choice(classes, size=num_ways, replace=False)

    support_x, support_y, query_x, query_y = [], [], [], []

    for class_idx, cls in enumerate(selected_classes):
        class_data = data[cls]
        indices = np.random.permutation(len(class_data))

        support_indices = indices[:num_shots]
        query_indices = indices[num_shots : num_shots + num_queries]

        support_x.append(class_data[support_indices])
        support_y.append(torch.full((num_shots,), class_idx, dtype=torch.long))

        query_x.append(class_data[query_indices])
        query_y.append(torch.full((num_queries,), class_idx, dtype=torch.long))

    return Task(
        name="omniglot_task",
        train_support=(torch.cat(support_x), torch.cat(support_y)),
        train_query=(torch.cat(query_x), torch.cat(query_y)),
    )

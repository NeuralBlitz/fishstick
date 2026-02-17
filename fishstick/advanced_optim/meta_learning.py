"""
Meta-Learning Optimization Primitives

Advanced meta-learning based optimization techniques:
- Learned optimizer (hypernetwork-based)
- Meta-update rules for few-shot learning
- Gradient-based meta-learning with Reptile
- Neural optimizer layers

Reference:
- Andrychowicz et al. (2016). Learning to Learn by Gradient Descent by Gradient Descent.
- Finn et al. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.
- Nichol et al. (2018). On First-Order Meta-Learning Algorithms.
"""

import math
from typing import Optional, Dict, List, Callable, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F
from collections import defaultdict


class LearnedOptimizer(nn.Module):
    """
    Learned Optimizer using a Hypernetwork.

    A neural network that learns to produce optimization updates,
    replacing hand-crafted optimization rules.

    Args:
        hidden_size: Size of hidden layer in hypernetwork
        num_layers: Number of layers in hypernetwork
        activation: Activation function

    Reference:
        - Andrychowicz et al. (2016). Learning to Learn by Gradient Descent by Gradient Descent.

    Example:
        >>> learned_opt = LearnedOptimizer(param_dim=1000, hidden_size=64)
        >>> optimizer = torch.optim.Adam(learned_opt.parameters(), lr=1e-3)
        >>> for data, target in task:
        ...     loss = task_loss(model, data, target)
        ...     grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        ...     update = learned_opt(grads)
        ...     # Apply update to model parameters
    """

    def __init__(
        self,
        param_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.param_dim = param_dim
        self.hidden_size = hidden_size

        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        act = activations.get(activation, nn.ReLU())

        layers = []
        input_dim = param_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(act)
            input_dim = hidden_size

        layers.append(nn.Linear(hidden_size, param_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, grads: List[Tensor]) -> List[Tensor]:
        """
        Generate parameter updates from gradients.

        Args:
            grads: List of gradient tensors

        Returns:
            List of update tensors
        """
        flat_grads = torch.cat([g.flatten() for g in grads])

        if flat_grads.shape[0] != self.param_dim:
            flat_grads = F.pad(flat_grads, (0, self.param_dim - flat_grads.shape[0]))

        updates = self.network(flat_grads)

        result = []
        idx = 0

        for g in grads:
            numel = g.numel()
            update = updates[idx : idx + numel].reshape_as(g)
            result.append(update)
            idx += numel

        return result


class MetaLearnedOptimizer:
    """
    Meta-Learned Optimizer Wrapper.

    Wraps a base optimizer with a learned update rule learned via meta-learning.

    Args:
        model: Model to optimize
        learned_opt_module
        base_lr: Base learning: Learned optimizer module rate for meta-update
    """

    def __init__(
        self,
        model: nn.Module,
        learned_opt_module: nn.Module,
        base_lr: float = 0.01,
    ):
        self.model = model
        self.learned_opt = learned_opt_module
        self.base_lr = base_lr

    def step(self):
        """Perform meta-learned optimization step."""
        params = list(self.model.parameters())
        grads = [p.grad for p in params if p.grad is not None]

        if len(grads) > 0:
            updates = self.learned_opt(grads)

            param_idx = 0
            for p in params:
                if p.grad is not None:
                    p.data.add_(updates[param_idx], alpha=-self.base_lr)
                    param_idx += 1

    def zero_grad(self):
        """Zero gradients."""
        self.model.zero_grad()


class MAMLOptimizerStep(nn.Module):
    """
    MAML-style Optimizer Step for Meta-Learning.

    Implements the gradient-based meta-learning update rule used in MAML.

    Args:
        inner_lr: Learning rate for inner loop adaptation
        num_inner_steps: Number of inner loop gradient steps

    Reference:
        - Finn et al. (2017). Model-Agnostic Meta-Learning for Fast Adaptation.
    """

    def __init__(
        self,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
    ):
        super().__init__()
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

    def forward(
        self,
        support_data: Tuple[Tensor, Tensor],
        query_data: Tuple[Tensor, Tensor],
        model: nn.Module,
    ) -> Tuple[nn.Module, Tensor]:
        """
        Perform MAML-style optimization.

        Args:
            support_data: Support set (x, y) for inner loop
            query_data: Query set (x, y) for outer loop
            model: Model to adapt

        Returns:
            Adapted model and query loss
        """
        support_x, support_y = support_data
        query_x, query_y = query_data

        adapted_params = {name: p.clone() for name, p in model.named_parameters()}

        for _ in range(self.num_inner_steps):
            support_logits = model.forward_with_params(support_x, adapted_params)
            support_loss = F.cross_entropy(support_logits, support_y)

            grads = torch.autograd.grad(
                support_loss,
                adapted_params.values(),
                create_graph=True,
            )

            adapted_params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(adapted_params.items(), grads)
            }

        query_logits = model.forward_with_params(query_x, adapted_params)
        query_loss = F.cross_entropy(query_logits, query_y)

        return model, query_loss

    def meta_update(
        self,
        model: nn.Module,
        query_loss: Tensor,
        outer_lr: float = 0.001,
    ) -> nn.Module:
        """
        Perform meta-update using query loss.

        Args:
            model: Model to update
            query_loss: Loss on query set
            outer_lr: Learning rate for outer loop update

        Returns:
            Updated model
        """
        grads = torch.autograd.grad(
            query_loss,
            model.parameters(),
            create_graph=True,
        )

        with torch.no_grad():
            for (name, param), grad in zip(model.named_parameters(), grads):
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad.copy_(grad)

        return model


class ReptileOptimizer:
    """
    Reptile: A Simple Meta-Learning Algorithm.

    Implements the Reptile algorithm for gradient-based meta-learning.

    Args:
        model: Model to optimize
        inner_lr: Learning rate for inner loop
        num_inner_steps: Number of gradient steps in inner loop

    Reference:
        - Nichol et al. (2018). On First-Order Meta-Learning Algorithms.

    Example:
        >>> reptile = ReptileOptimizer(model, inner_lr=0.01, num_inner_steps=5)
        >>> for task in tasks:
        ...     # Inner loop: adapt to task
        ...     adapted_model = reptile.inner_step(task)
        ...     # Outer loop: update meta-weights
        ...     reptile.outer_step(task)
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

    def inner_step(self, task_data: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """
        Perform inner loop adaptation (task-specific update).

        Args:
            task_data: Task data (x, y)

        Returns:
            Adapted parameters
        """
        x, y = task_data

        adapted_params = {n: p.clone() for n, p in self.model.named_parameters()}

        for _ in range(self.num_inner_steps):
            outputs = self.model.forward_with_params(x, adapted_params)
            loss = F.cross_entropy(outputs, y)

            grads = torch.autograd.grad(loss, adapted_params.values())

            adapted_params = {
                n: p - self.inner_lr * g
                for (n, p), g in zip(adapted_params.items(), grads)
            }

        return adapted_params

    def outer_step(
        self,
        original_params: Dict[str, Tensor],
        adapted_params: Dict[str, Tensor],
        meta_lr: float = 0.001,
    ):
        """
        Perform outer loop update (meta-update).

        Args:
            original_params: Original parameters before inner loop
            adapted_params: Adapted parameters after inner loop
            meta_lr: Meta learning rate
        """
        with torch.no_grad():
            for name in original_params:
                orig = original_params[name]
                adapted = adapted_params[name]

                update = (adapted - orig) * meta_lr

                if self.model.named_parameters()[name].grad is None:
                    self.model.named_parameters()[name].grad = torch.zeros_like(orig)

                self.model.named_parameters()[name].grad.copy_(update)

        self.model.parameters()


class MetaLearningRateScheduler:
    """
    Meta-Learning Rate Scheduler.

    Learns an optimal per-parameter learning rate through meta-learning.

    Args:
        model: Model to optimize
        num_meta_steps: Number of meta-update steps per epoch
    """

    def __init__(
        self,
        model: nn.Module,
        num_meta_steps: int = 100,
    ):
        self.model = model
        self.num_meta_steps = num_meta_steps

        self.lr_module = nn.ModuleDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.lr_module[name] = nn.Parameter(torch.tensor(0.1))

    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates."""
        return {name: torch.sigmoid(lr).item() for name, lr in self.lr_module.items()}

    def step(self, loss: Tensor):
        """
        Perform meta-learning rate update.

        Args:
            loss: Current loss value
        """
        loss.backward()

        for name, param in self.model.named_parameters():
            if name in self.lr_module:
                lr = torch.sigmoid(self.lr_module[name])

                if param.grad is not None:
                    param.data.add_(param.grad, alpha=-lr)

        self.model.zero_grad()


class NeuralOptimizerLayer(nn.Module):
    """
    Neural Optimizer Layer.

    A differentiable layer that can be inserted into models to
    learn task-specific optimization behavior.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_steps: Number of optimization steps to simulate

    Reference:
        - Metz et al. (2019). Training Deep Neural Networks with Implicit Optimization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_steps: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        self.update_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        params: Tensor,
        grads: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply learned optimization to parameters.

        Args:
            params: Current parameters
            grads: Optional gradients (if None, computed from loss)

        Returns:
            Updated parameters
        """
        if grads is None:
            raise ValueError("Gradients must be provided")

        for _ in range(self.num_steps):
            update = self.update_net(grads)
            params = params - update

        return params


class MetaGradientAccumulator:
    """
    Meta-Learning based Gradient Accumulation.

    Learns when to accumulate gradients vs when to step,
    optimizing the trade-off between gradient accuracy and step frequency.

    Args:
        model: Model to optimize
        max_accumulation: Maximum accumulation steps
    """

    def __init__(
        self,
        model: nn.Module,
        max_accumulation: int = 8,
    ):
        self.model = model
        self.max_accumulation = max_accumulation

        self.step_policy = nn.Sequential(
            nn.Linear(1, max_accumulation),
            nn.Softmax(dim=-1),
        )

        self.loss_history = []

    def should_step(self, loss: float) -> bool:
        """
        Decide whether to perform an optimization step.

        Args:
            loss: Current loss value

        Returns:
            Whether to step
        """
        self.loss_history.append(loss)

        if len(self.loss_history) < 2:
            return False

        loss_change = abs(self.loss_history[-1] - self.loss_history[-2])
        loss_input = torch.tensor([[loss_change]], dtype=torch.float32)

        with torch.no_grad():
            probs = self.step_policy(loss_input)
            decision = torch.multinomial(probs, 1).item()

        return decision == 0 or len(self.loss_history) >= self.max_accumulation


class FastGradientOptimizer:
    """
    Fast Gradient Optimization for Few-Shot Learning.

    Implements a fast gradient-based optimizer specifically designed
    for rapid adaptation in few-shot scenarios.

    Args:
        model: Model to optimize
        fast_lr: Learning rate for fast adaptation
        num_fast_steps: Number of fast gradient steps

    Reference:
        - Finn et al. (2017). Model-Agnostic Meta-Learning for Fast Adaptation.
    """

    def __init__(
        self,
        model: nn.Module,
        fast_lr: float = 0.1,
        num_fast_steps: int = 5,
    ):
        self.model = model
        self.fast_lr = fast_lr
        self.num_fast_steps = num_fast_steps

    def fast_adapt(
        self,
        data: Tuple[Tensor, Tensor],
        max_steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Perform fast adaptation on a task.

        Args:
            data: Task data (x, y)
            max_steps: Maximum adaptation steps (default: self.num_fast_steps)

        Returns:
            Adapted model (cloned)
        """
        x, y = data
        max_steps = max_steps or self.num_fast_steps

        adapted_model = type(self.model)(*self.model.__dict__.get("args", ()))
        adapted_model.load_state_dict(self.model.state_dict())
        adapted_model.train()

        for _ in range(max_steps):
            output = adapted_model(x)
            loss = F.cross_entropy(output, y)

            grads = torch.autograd.grad(loss, adapted_model.parameters())

            with torch.no_grad():
                for (name, param), grad in zip(adapted_model.named_parameters(), grads):
                    if grad is not None:
                        param.copy_(param - self.fast_lr * grad)

        return adapted_model


class MetaSGDOptimizer(nn.Module):
    """
    Meta-SGD: Learning to Learn the Learning Rate.

    Learns per-parameter learning rates for faster adaptation.

    Args:
        model: Model to optimize
        initial_lr: Initial learning rate

    Reference:
        - Li et al. (2017). Meta-SGD: Learning to Learn Fast.
    """

    def __init__(
        self,
        model: nn.Module,
        initial_lr: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.initial_lr = initial_lr

        self.lrs = nn.ParameterDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.lrs[name] = nn.Parameter(torch.ones_like(param) * initial_lr)

    def forward(self, loss: Tensor, inner_steps: int = 5) -> Tensor:
        """
        Perform meta-SGD optimization step.

        Args:
            loss: Task loss
            inner_steps: Number of inner loop steps

        Returns:
            Final loss after adaptation
        """
        adapted_params = {n: p.clone() for n, p in self.model.named_parameters()}

        for _ in range(inner_steps):
            output = self._forward_with_params(adapted_params, requires_grad=True)

        return loss

    def _forward_with_params(
        self,
        params: Dict[str, Tensor],
        requires_grad: bool = False,
    ) -> Tensor:
        """Forward pass with given parameters."""
        # Placeholder - implement based on specific model
        pass


__all__ = [
    "LearnedOptimizer",
    "MetaLearnedOptimizer",
    "MAMLOptimizerStep",
    "ReptileOptimizer",
    "MetaLearningRateScheduler",
    "NeuralOptimizerLayer",
    "MetaGradientAccumulator",
    "FastGradientOptimizer",
    "MetaSGDOptimizer",
]

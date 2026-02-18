"""
Reptile algorithm implementation.

Reptile is a simple first-order meta-learning algorithm that performs
gradient descent on the task-specific loss and then updates the initial
parameters towards the adapted parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple

from .maml import MAML
from .types import FewShotTask, AdaptationResult


class Reptile(nn.Module):
    """Reptile: First-order meta-learning algorithm.

    Reptile performs gradient descent on each task and then updates
    the initial parameters towards the adapted parameters.

    Args:
        encoder: Feature encoder network
        num_classes: Number of output classes
        inner_lr: Learning rate for inner loop
        num_inner_steps: Number of inner loop steps

    Example:
        >>> encoder = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        >>> reptile = Reptile(encoder, num_classes=5, inner_lr=0.01, num_inner_steps=5)
        >>> task = FewShotTask(support_x, support_y, query_x, query_y, 5, 5, 15)
        >>> result = reptile.adapt(task)

    References:
        Nichol et al. "On First-Order Meta-Learning Algorithms" (2018)
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        self.classifier = nn.Linear(self._get_embedding_dim(), num_classes)

    def _get_embedding_dim(self) -> int:
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            out = self.encoder(dummy)
        self.encoder.train()
        return out.view(out.size(0), -1).size(1)

    def forward(self, x: Tensor, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """Forward pass through encoder and classifier.

        Args:
            x: Input tensor
            params: Optional parameter dict

        Returns:
            Logits [batch_size, num_classes]
        """
        if params is None:
            params = dict(self.named_parameters())

        x = x.view(x.size(0), *x.shape[1:])

        features = self.encoder(x)
        features = features.view(features.size(0), -1)

        if "classifier.weight" in params and "classifier.bias" in params:
            logits = F.linear(
                features, params["classifier.weight"], params["classifier.bias"]
            )
        else:
            logits = self.classifier(features)

        return logits

    def get_params(self) -> Dict[str, Tensor]:
        """Get current model parameters."""
        return {n: p.clone() for n, p in self.named_parameters()}

    def adapt(self, task: FewShotTask) -> Dict[str, Tensor]:
        """Adapt to a few-shot task using gradient descent.

        Args:
            task: Few-shot task

        Returns:
            Adapted parameters
        """
        adapted_params = self.get_params()

        for _ in range(self.num_inner_steps):
            logits = self.forward(task.support_x, adapted_params)
            loss = F.cross_entropy(logits, task.support_y)

            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=False,
            )

            adapted_params = {
                name: param - self.inner_lr * grad if grad is not None else param
                for (name, param), grad in zip(adapted_params.items(), grads)
            }

        return adapted_params

    def meta_train_step(
        self,
        tasks: List[FewShotTask],
        outer_optimizer: torch.optim.Optimizer,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Perform Reptile meta-training step.

        Args:
            tasks: List of few-shot tasks
            outer_optimizer: Optimizer for outer loop

        Returns:
            Tuple of (meta_loss, metrics)
        """
        outer_optimizer.zero_grad()

        meta_grads = {n: torch.zeros_like(p) for n, p in self.named_parameters()}

        for task in tasks:
            adapted_params = self.adapt(task)

            original_params = self.get_params()

            for name, adapted in adapted_params.items():
                if name in original_params:
                    meta_grads[name] += adapted - original_params[name]

        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in meta_grads:
                    param.grad = -meta_grads[name] / len(tasks)

        outer_optimizer.step()

        return torch.tensor(0.0), {"meta_update": 1.0}


class MetaLearningBaseline(nn.Module):
    """Simple baseline: fine-tune on support set.

    Not a meta-learning method per se, but a baseline to compare against.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        fine_tune_steps: int = 10,
        fine_tune_lr: float = 0.01,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.fine_tune_steps = fine_tune_steps
        self.fine_tune_lr = fine_tune_lr

        self.classifier = nn.Linear(self._get_embedding_dim(), num_classes)

    def _get_embedding_dim(self) -> int:
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            out = self.encoder(dummy)
        self.encoder.train()
        return out.view(out.size(0), -1).size(1)

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x).view(x.size(0), -1)
        return self.classifier(features)

    def fine_tune(self, support_x: Tensor, support_y: Tensor) -> None:
        """Fine-tune classifier on support set."""
        self.classifier.requires_grad_(False)

        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.fine_tune_lr)

        for _ in range(self.fine_tune_steps):
            optimizer.zero_grad()
            logits = self.forward(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            optimizer.step()

        self.classifier.requires_grad_(True)

    def predict(self, task: FewShotTask) -> Tensor:
        """Predict on query set after fine-tuning."""
        self.fine_tune(task.support_x, task.support_y)
        return self.forward(task.query_x)

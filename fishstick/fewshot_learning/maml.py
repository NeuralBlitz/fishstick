"""
MAML (Model-Agnostic Meta-Learning) implementation for few-shot learning.

This module provides a comprehensive MAML implementation with support for
first-order approximation, learnable inner learning rates, and various
inner loop configurations.
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MAMLConfig
from .types import AdaptationResult, FewShotTask


class MAML(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) for few-shot learning.

    MAML learns a good initialization of model parameters that can be quickly
    adapted to new tasks with a few gradient steps.

    Args:
        encoder: Feature encoder network
        num_classes: Number of output classes
        config: MAML configuration

    Example:
        >>> encoder = nn.Sequential(nn.Conv2d(1, 64, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        >>> maml = MAML(encoder, num_classes=5, config=MAMLConfig(inner_lr=0.01, num_inner_steps=5))
        >>> task = FewShotTask(support_x, support_y, query_x, query_y, 5, 5, 15)
        >>> result = maml.adapt(task)

    References:
        Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (ICML 2017)
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        config: Optional[MAMLConfig] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.config = config or MAMLConfig()

        self.classifier = nn.Linear(self._get_embedding_dim(), num_classes)

        if self.config.learn_inner_lr:
            self._init_learnable_inner_lrs()

    def _get_embedding_dim(self) -> int:
        """Get the embedding dimension by doing a forward pass."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 84, 84)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            dummy_output = self.encoder(dummy_input)
        return dummy_output.view(dummy_output.size(0), -1).size(1)

    def _init_learnable_inner_lrs(self) -> None:
        """Initialize per-parameter learnable inner loop learning rates."""
        self.inner_lrs = nn.ParameterDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.inner_lrs[name] = nn.Parameter(
                    torch.ones_like(param) * self.config.inner_lr_init
                )

    def forward(self, x: Tensor, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """Forward pass through encoder and classifier.

        Args:
            x: Input tensor [batch_size, ...]
            params: Optional parameter dict for custom forward

        Returns:
            Logits [batch_size, num_classes]
        """
        if params is None:
            params = dict(self.named_parameters())

        features = self._forward_encoder(x, params)
        logits = self.classifier(features)
        return logits

    def _forward_encoder(
        self, x: Tensor, params: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """Forward pass through encoder only.

        Args:
            x: Input tensor
            params: Optional parameter dict

        Returns:
            Feature tensor
        """
        if params is None:
            params = dict(self.named_parameters())

        x = x.view(x.size(0), *x.shape[1:])

        for name, module in self.encoder._modules.items():
            if isinstance(module, nn.Module):
                x = module(x)
            else:
                x = module.forward(x)

        return x.view(x.size(0), -1)

    def get_params(self) -> Dict[str, Tensor]:
        """Get current model parameters."""
        return {n: p.clone() for n, p in self.named_parameters() if p.requires_grad}

    def adapt(
        self,
        task: FewShotTask,
        return_adapted_logits: bool = False,
    ) -> AdaptationResult:
        """Adapt the model to a few-shot task.

        Args:
            task: Few-shot task with support and query sets
            return_adapted_logits: Whether to return adapted support logits

        Returns:
            AdaptationResult with predictions and losses
        """
        adapted_params = self._inner_loop(task.support_x, task.support_y)

        query_logits = self.forward(task.query_x, adapted_params)
        query_loss = F.cross_entropy(query_logits, task.query_y)

        adapted_logits = None
        if return_adapted_logits:
            adapted_logits = self.forward(task.support_x, adapted_params)

        return AdaptationResult(
            adapted_model=adapted_params,
            adapted_logits=adapted_logits,
            query_logits=query_logits,
            query_loss=query_loss,
        )

    def _inner_loop(
        self,
        support_x: Tensor,
        support_y: Tensor,
    ) -> Dict[str, Tensor]:
        """Perform inner loop adaptation on support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels

        Returns:
            Adapted parameters
        """
        adapted_params = self.get_params()

        inner_lr = self._get_inner_lr()

        for step in range(self.config.num_inner_steps):
            logits = self.forward(support_x, adapted_params)
            loss = F.cross_entropy(logits, support_y)

            create_graph = not self.config.first_order
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=create_graph,
                allow_unused=True,
            )

            adapted_params = {
                name: param - inner_lr[name] * grad if grad is not None else param
                for (name, param), grad in zip(adapted_params.items(), grads)
            }

            if self.config.grad_clip is not None:
                adapted_params = self._clip_grads(adapted_params)

        return adapted_params

    def _get_inner_lr(self) -> Dict[str, Tensor]:
        """Get inner loop learning rates."""
        if self.config.learn_inner_lr:
            return self.inner_lrs
        else:
            return {
                n: self.config.inner_lr
                for n, _ in self.named_parameters()
                if _.requires_grad
            }

    def _clip_grads(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Clip gradients of adapted parameters."""
        total_norm = 0.0
        for p in params.values():
            if p is not None and p.grad is not None:
                total_norm += p.grad.data.norm(2) ** 2
        total_norm = total_norm**0.5

        clip_coef = self.config.grad_clip / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in params.values():
                if p is not None and p.grad is not None:
                    p.grad.data.mul_(clip_coef)

        return params

    def meta_train_step(
        self,
        tasks: List[FewShotTask],
        outer_optimizer: torch.optim.Optimizer,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Perform a meta-training step.

        Args:
            tasks: List of few-shot tasks
            outer_optimizer: Optimizer for outer loop

        Returns:
            Tuple of (meta_loss, metrics_dict)
        """
        outer_optimizer.zero_grad()

        meta_loss = 0.0
        metrics = {"query_loss": 0.0, "inner_loss": 0.0}

        for task in tasks:
            result = self.adapt(task)
            meta_loss = meta_loss + result.query_loss
            metrics["query_loss"] += result.query_loss.item()

        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()

        outer_optimizer.step()

        metrics["query_loss"] /= len(tasks)

        return meta_loss, metrics


class MetaSGD(MAML):
    """Meta-SGD: Learn the inner loop learning rates.

    Meta-SGD extends MAML by learning the inner loop learning rates
    as part of meta-learning.

    References:
        Li et al. "Meta-SGD: Learning to Learn Quickly" (ICLR 2017)
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        inner_lr_init: float = 0.1,
        config: Optional[MAMLConfig] = None,
    ):
        if config is None:
            config = MAMLConfig(learn_inner_lr=True, inner_lr_init=inner_lr_init)
        else:
            config.learn_inner_lr = True
            config.inner_lr_init = inner_lr_init

        super().__init__(encoder, num_classes, config)

    def _get_inner_lr(self) -> Dict[str, Tensor]:
        """Get learned inner loop learning rates with sign preservation."""
        lrs = {}
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.inner_lrs:
                lrs[name] = (
                    torch.sigmoid(self.inner_lrs[name]) * self.config.inner_lr_init
                )
        return lrs

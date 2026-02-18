"""
Synaptic Intelligence (SI) Implementation.

Regularization-based continual learning that tracks parameter
contributions to loss reduction during training.

Classes:
- SynapticIntelligence: SI regularizer
- SIOptimizer: Optimizer wrapper with SI updates
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from collections import defaultdict


@dataclass
class SITracking:
    """State tracking for SI."""

    omega: Tensor
    W: Tensor
    prev_params: Tensor
    delta_params: Tensor


class SynapticIntelligence:
    """
    Synaptic Intelligence (SI) Regularizer.

    Estimates parameter importance online during training by tracking
    how much each parameter contributes to loss reduction.

    Reference:
        Zenke et al., "Continual Learning Through Synaptic Intelligence", ICML 2017

    Args:
        model: Neural network to regularize
        si_lambda: Regularization strength
        xi: Damping parameter for numerical stability
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        si_lambda: float = 1.0,
        xi: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model
        self.si_lambda = si_lambda
        self.xi = xi
        self.device = device

        self.omega: Dict[str, Tensor] = {}
        self.W: Dict[str, Tensor] = {}
        self.params_prev: Dict[str, Tensor] = {}
        self.params_current: Dict[str, Tensor] = {}
        self.delta_params: Dict[str, Tensor] = {}

        self._initialize_tracking()

    def _initialize_tracking(self) -> None:
        """Initialize SI tracking variables."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.omega[n] = torch.zeros_like(p, device=self.device)
                self.W[n] = torch.zeros_like(p, device=self.device)
                self.params_prev[n] = p.detach().clone().to(self.device)
                self.params_current[n] = p.detach().clone().to(self.device)
                self.delta_params[n] = torch.zeros_like(p, device=self.device)

    def update_tracking(self) -> None:
        """Update parameter tracking before training step."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params_prev[n] = p.detach().clone()
                self.delta_params[n].zero_()

    def compute_weight_contributions(self, loss: Tensor) -> None:
        """
        Compute parameter contributions to loss reduction.

        Call this after loss.backward() but before optimizer.step().

        Args:
            loss: Current loss value
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n in self.delta_params:
                    self.delta_params[n] = p.detach() - self.params_prev[n]
                    self.W[n] -= p.grad * self.delta_params[n]

    def after_task(self) -> None:
        """Update omega values after completing a task."""
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.delta_params:
                delta = p.detach() - self.params_prev[n]

                denominator = delta.pow(2) + self.xi
                self.omega[n] += self.W[n] / denominator

        self.params_current = {
            n: p.detach().clone().to(self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        for n in self.W:
            self.W[n].zero_()

    def penalty(self) -> Tensor:
        """
        Compute SI penalty term.

        Returns:
            SI regularization penalty
        """
        loss = torch.tensor(0.0, device=self.device)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.omega and n in self.params_current:
                diff = p - self.params_current[n]
                loss += (self.omega[n] * diff.pow(2)).sum()

        return self.si_lambda * loss

    def get_importance_ranking(self) -> List[Tuple[str, float]]:
        """Get parameters ranked by importance."""
        rankings = []

        for n, omega in self.omega.items():
            rankings.append((n, omega.sum().item()))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def reset(self) -> None:
        """Reset SI tracking."""
        self._initialize_tracking()


class SIOptimizer:
    """
    Optimizer wrapper that integrates Synaptic Intelligence.

    Provides seamless integration of SI regularization with
    standard PyTorch optimizers.

    Args:
        model: Neural network
        optimizer: Base optimizer (Adam, SGD, etc.)
        si_lambda: SI regularization strength
        xi: Damping parameter
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        si_lambda: float = 1.0,
        xi: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.si = SynapticIntelligence(model, si_lambda, xi, device)
        self.device = device

    def zero_grad(self) -> None:
        """Zero gradients and update tracking."""
        self.optimizer.zero_grad()
        self.si.update_tracking()

    def step(self, loss: Tensor) -> None:
        """
        Perform optimization step with SI regularization.

        Args:
            loss: Current loss
        """
        self.si.compute_weight_contributions(loss)

        total_loss = loss + self.si.penalty()

        total_loss.backward()
        self.optimizer.step()

    def after_task(self) -> None:
        """Call after completing a task."""
        self.si.after_task()

    def compute_penalty(self) -> Tensor:
        """Get current SI penalty value."""
        return self.si.penalty()


class OnlineSI:
    """
    Online Synaptic Intelligence.

    Memory-efficient version of SI that maintains a single
    importance estimate rather than per-task.

    Args:
        model: Neural network
        si_lambda: Regularization strength
        decay: Decay factor for importance
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        si_lambda: float = 1.0,
        decay: float = 0.99,
        device: str = "cpu",
    ):
        self.model = model
        self.si_lambda = si_lambda
        self.decay = decay
        self.device = device

        self.omega: Dict[str, Tensor] = {}
        self.params_ref: Dict[str, Tensor] = {}

        for n, p in model.named_parameters():
            if p.requires_grad:
                self.omega[n] = torch.zeros_like(p, device=device)
                self.params_ref[n] = p.detach().clone().to(device)

    def update(self, loss: Tensor) -> None:
        """Update importance estimates."""
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                delta = p.detach() - self.params_ref[n]

                contribution = -p.grad * delta

                self.omega[n] = self.decay * self.omega[n] + contribution

        self.params_ref = {
            n: p.detach().clone().to(self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def penalty(self) -> Tensor:
        """Compute SI penalty."""
        loss = torch.tensor(0.0, device=self.device)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.omega:
                diff = p - self.params_ref[n]
                loss += (self.omega[n].abs() * diff.pow(2)).sum()

        return self.si_lambda * loss

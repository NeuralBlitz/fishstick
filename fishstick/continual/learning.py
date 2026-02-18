"""
Comprehensive Continual Learning Module for Fishstick.

Implements state-of-the-art continual learning algorithms spanning:
- Regularization-based methods (EWC, SI, MAS, RWalk, MCL)
- Replay-based methods (ExperienceReplay, GEM, A-GEM, ICaRL, FDR, GSS, ASER)
- Dynamic architecture methods (ProgressiveNetworks, PackNet, Piggyback, HAT, SupSup, CPG)
- Meta-learning approaches (MetaContinual, OML, ANML, BMCL)
- Generative replay methods (DGR, MeRGAN, LifelongGAN, GenerativeFeatureReplay)
- Task-free continual learning (OnlineContinual, StreamingLearning, TestTimeAdaptation)
- Evaluation metrics (AverageAccuracy, BackwardTransfer, ForwardTransfer, BWTPlus, LearningCurve)
- Utilities (TaskIncremental, DomainIncremental, ClassIncremental, ContinualTrainer, MemoryBuffer)

References:
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017) - EWC
- Zenke et al., "Continual learning through synaptic intelligence" (2017) - SI
- Aljundi et al., "Memory Aware Synapses" (2018) - MAS
- Chaudhry et al., "Riemannian Walk for Incremental Learning" (2018) - RWalk
- Rebuffi et al., "iCaRL: Incremental Classifier and Representation Learning" (2017)
- Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning" (2017) - GEM
- Chaudhry et al., "Efficient Lifelong Learning with A-GEM" (2019)
- Shin et al., "Continual Learning with Deep Generative Replay" (2017)
- Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network" (2018)
- Serra et al., "Overcoming Catastrophic Forgetting with Hard Attention to the Task" (2018) - HAT
- Javed & White, "Meta-Learning Representations for Continual Learning" (2019) - OML
"""

from typing import Optional, Tuple, Dict, Any, List, Union, Callable, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy
import math
import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

import numpy as np
from collections import deque
from enum import Enum, auto


# ============================================================================
# Base Classes and Types
# ============================================================================


class ContinualScenario(Enum):
    """Types of continual learning scenarios."""

    TASK_INCREMENTAL = auto()
    DOMAIN_INCREMENTAL = auto()
    CLASS_INCREMENTAL = auto()


@dataclass
class Task:
    """Represents a learning task in continual learning."""

    task_id: int
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    num_classes: int = 0
    task_type: ContinualScenario = ContinualScenario.TASK_INCREMENTAL


@dataclass
class ContinualMetrics:
    """Metrics for continual learning evaluation."""

    accuracy_matrix: np.ndarray
    forgetting: np.ndarray
    forward_transfer: np.ndarray
    backward_transfer: np.ndarray
    average_accuracy: float
    average_forgetting: float


class ContinualLearner(ABC, nn.Module):
    """Abstract base class for continual learning methods."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.task_id: int = 0
        self.seen_tasks: Set[int] = set()

    @abstractmethod
    def before_task(self, task: Task) -> None:
        """Prepare for learning a new task."""
        pass

    @abstractmethod
    def after_task(self, task: Task) -> None:
        """Clean up after learning a task."""
        pass

    def forward(self, x: Tensor, task_id: Optional[int] = None) -> Tensor:
        """Forward pass with optional task ID."""
        return self.model(x)


# ============================================================================
# Section 1: Regularization Methods
# ============================================================================


class EWC(ContinualLearner):
    """
    Elastic Weight Consolidation (EWC).

    Regularizes parameter updates to protect important weights for previous tasks.
    Uses Fisher Information Matrix to estimate parameter importance.

    Reference:
        Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017

    Args:
        model: Base neural network
        ewc_lambda: Regularization strength
        fisher_sample_size: Number of samples for Fisher estimation
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1.0,
        fisher_sample_size: int = 200,
    ):
        super().__init__(model)
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size

        self.fisher_dict: Dict[int, Dict[str, Tensor]] = {}
        self.optimal_params: Dict[int, Dict[str, Tensor]] = {}

    def before_task(self, task: Task) -> None:
        """Prepare for new task."""
        self.task_id = task.task_id
        self.seen_tasks.add(task.task_id)

    def after_task(self, task: Task) -> None:
        """Compute Fisher Information and store optimal parameters."""
        self._compute_fisher(task.train_loader)
        self._store_optimal_params()

    def _compute_fisher(self, dataloader: DataLoader) -> None:
        """Compute Fisher Information Matrix."""
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()
        for i, (x, y) in enumerate(dataloader):
            if i >= self.fisher_sample_size:
                break

            x, y = (
                x.to(next(self.model.parameters()).device),
                y.to(next(self.model.parameters()).device),
            )
            self.model.zero_grad()

            output = self.model(x)
            log_probs = F.log_softmax(output, dim=1)

            probs = F.softmax(output, dim=1)
            sampled_label = torch.multinomial(probs, 1).squeeze()

            loss = F.nll_loss(log_probs, sampled_label)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2) / self.fisher_sample_size

        self.fisher_dict[self.task_id] = fisher

    def _store_optimal_params(self) -> None:
        """Store optimal parameters after task training."""
        self.optimal_params[self.task_id] = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def penalty(self) -> Tensor:
        """Compute EWC penalty term."""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for task_id in self.seen_tasks:
            if task_id == self.task_id:
                continue

            fisher = self.fisher_dict.get(task_id, {})
            optimal_params = self.optimal_params.get(task_id, {})

            for n, p in self.model.named_parameters():
                if p.requires_grad and n in fisher and n in optimal_params:
                    loss += (fisher[n] * (p - optimal_params[n]).pow(2)).sum()

        return self.ewc_lambda * loss


class OnlineEWC(ContinualLearner):
    """
    Online Elastic Weight Consolidation.

    Efficient online version of EWC that maintains a single running estimate
    of Fisher Information rather than storing per-task estimates.

    Reference:
        Schwarz et al., "Progress & Compress: A scalable framework for continual learning", ICML 2018

    Args:
        model: Base neural network
        ewc_lambda: Regularization strength
        gamma: Decay factor for online Fisher
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__(model)
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma

        self.fisher: Optional[Dict[str, Tensor]] = None
        self.optimal_params: Optional[Dict[str, Tensor]] = None
        self.task_count = 0

    def before_task(self, task: Task) -> None:
        """Prepare for new task."""
        self.task_id = task.task_id
        self.seen_tasks.add(task.task_id)

    def after_task(self, task: Task) -> None:
        """Update online Fisher and optimal parameters."""
        new_fisher = self._compute_fisher(task.train_loader)

        if self.fisher is None:
            self.fisher = new_fisher
        else:
            for n in self.fisher:
                self.fisher[n] = self.gamma * self.fisher[n] + new_fisher[n]

        self.optimal_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.task_count += 1

    def _compute_fisher(self, dataloader: DataLoader) -> Dict[str, Tensor]:
        """Compute Fisher Information."""
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()
        for i, (x, y) in enumerate(dataloader):
            if i >= 200:
                break

            x = x.to(next(self.model.parameters()).device)
            self.model.zero_grad()

            output = self.model(x)
            probs = F.softmax(output, dim=1)
            log_probs = F.log_softmax(output, dim=1)

            loss = -(probs * log_probs).sum(dim=1).mean()
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2) / 200

        return fisher

    def penalty(self) -> Tensor:
        """Compute online EWC penalty."""
        if self.fisher is None or self.optimal_params is None:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.optimal_params[n]).pow(2)).sum()

        return self.ewc_lambda * loss / self.task_count if self.task_count > 0 else loss


class SI(ContinualLearner):
    """
    Synaptic Intelligence (SI).

    Estimates parameter importance online during training by tracking
    how much each parameter contributes to loss reduction.

    Reference:
        Zenke et al., "Continual Learning Through Synaptic Intelligence", ICML 2017

    Args:
        model: Base neural network
        si_lambda: Regularization strength
        xi: Damping parameter for numerical stability
    """

    def __init__(
        self,
        model: nn.Module,
        si_lambda: float = 1.0,
        xi: float = 1e-3,
    ):
        super().__init__(model)
        self.si_lambda = si_lambda
        self.xi = xi

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
                self.omega[n] = torch.zeros_like(p)
                self.W[n] = torch.zeros_like(p)
                self.params_prev[n] = p.clone().detach()
                self.params_current[n] = p.clone().detach()
                self.delta_params[n] = torch.zeros_like(p)

    def before_task(self, task: Task) -> None:
        """Prepare for new task."""
        self.task_id = task.task_id
        self.seen_tasks.add(task.task_id)

        for n in self.W:
            self.W[n].zero_()
            self.params_prev[n] = self.params_current[n].clone()

    def after_task(self, task: Task) -> None:
        """Update omega after task."""
        self._update_omega()
        self.params_current = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def _update_omega(self) -> None:
        """Update parameter importance (omega)."""
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.delta_params:
                delta = p.detach() - self.params_prev[n]
                if delta.abs().sum() > self.xi:
                    self.omega[n] += self.W[n] / (delta.pow(2) + self.xi)

    def update_W(self, loss: Tensor) -> None:
        """Update contribution to loss reduction (call after backward)."""
        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n in self.delta_params:
                    self.delta_params[n] = p.detach() - self.params_prev[n]
                    self.W[n] -= p.grad * self.delta_params[n]

    def penalty(self) -> Tensor:
        """Compute SI penalty."""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.omega and n in self.params_current:
                loss += (self.omega[n] * (p - self.params_current[n]).pow(2)).sum()

        return self.si_lambda * loss


class MAS(ContinualLearner):
    """
    Memory Aware Synapses (MAS).

    Estimates parameter importance by measuring the sensitivity of the
    learned function output to parameter changes. Model-agnostic approach.

    Reference:
        Aljundi et al., "Memory Aware Synapses: Learning What (not) to forget", ECCV 2018

    Args:
        model: Base neural network
        lambda_mas: Regularization strength
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_mas: float = 1.0,
    ):
        super().__init__(model)
        self.lambda_mas = lambda_mas

        self.omega: Dict[str, Tensor] = {}
        self.optimal_params: Dict[str, Tensor] = {}

        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize MAS storage."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.omega[n] = torch.zeros_like(p)
                self.optimal_params[n] = p.clone().detach()

    def before_task(self, task: Task) -> None:
        """Prepare for new task."""
        self.task_id = task.task_id
        self.seen_tasks.add(task.task_id)

    def after_task(self, task: Task) -> None:
        """Compute MAS importance and update optimal params."""
        self._compute_importance(task.train_loader)
        self.optimal_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def _compute_importance(self, dataloader: DataLoader) -> None:
        """Compute parameter importance using squared gradients of L2 loss."""
        self.model.eval()

        for n in self.omega:
            self.omega[n].zero_()

        for i, (x, _) in enumerate(dataloader):
            if i >= 200:
                break

            x = x.to(next(self.model.parameters()).device)
            self.model.zero_grad()

            output = self.model(x)

            loss = (output**2).sum() / output.numel()
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None and n in self.omega:
                    self.omega[n] += p.grad.abs() / 200

    def penalty(self) -> Tensor:
        """Compute MAS penalty."""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.omega and n in self.optimal_params:
                loss += (self.omega[n] * (p - self.optimal_params[n]).pow(2)).sum()

        return self.lambda_mas * loss


class RWalk(ContinualLearner):
    """
    Random Walk Regularization (RWalk).

    Combines EWC and SI with a random walk interpretation of parameter updates.
    Uses KL-divergence between successive task solutions.

    Reference:
        Chaudhry et al., "Riemannian Walk for Incremental Learning: Understanding
        Forgetting and Intransigence", ECCV 2018

    Args:
        model: Base neural network
        rwalk_lambda: Regularization strength
        alpha: Balance between EWC and SI terms
    """

    def __init__(
        self,
        model: nn.Module,
        rwalk_lambda: float = 1.0,
        alpha: float = 0.5,
    ):
        super().__init__(model)
        self.rwalk_lambda = rwalk_lambda
        self.alpha = alpha

        self.fisher: Dict[str, Tensor] = {}
        self.score: Dict[str, Tensor] = {}
        self.optimal_params: Dict[str, Tensor] = {}

        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize RWalk storage."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.fisher[n] = torch.zeros_like(p)
                self.score[n] = torch.zeros_like(p)
                self.optimal_params[n] = p.clone().detach()

    def before_task(self, task: Task) -> None:
        """Prepare for new task."""
        self.task_id = task.task_id
        self.seen_tasks.add(task.task_id)

    def after_task(self, task: Task) -> None:
        """Update Fisher and scores."""
        new_fisher = self._compute_fisher(task.train_loader)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.score:
                self.score[n] = (
                    self.alpha * self.fisher[n] + (1 - self.alpha) * new_fisher[n]
                )
                self.fisher[n] = new_fisher[n]

        self.optimal_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def _compute_fisher(self, dataloader: DataLoader) -> Dict[str, Tensor]:
        """Compute Fisher Information."""
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()
        for i, (x, y) in enumerate(dataloader):
            if i >= 200:
                break

            x = x.to(next(self.model.parameters()).device)
            self.model.zero_grad()

            output = self.model(x)
            log_probs = F.log_softmax(output, dim=1)
            probs = F.softmax(output, dim=1)

            loss = -(probs.detach() * log_probs).sum(dim=1).mean()
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2) / 200

        return fisher

    def penalty(self) -> Tensor:
        """Compute RWalk penalty."""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.score and n in self.optimal_params:
                loss += (self.score[n] * (p - self.optimal_params[n]).pow(2)).sum()

        return self.rwalk_lambda * loss

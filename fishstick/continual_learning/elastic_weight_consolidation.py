"""
Elastic Weight Consolidation (EWC) Implementations.

Advanced EWC variants including online EWC, diagonal Fisher approximation,
and Kronecker-factored approximate curvature (KFAC) EWC.

Classes:
- EWCRegularizer: Standard EWC with Fisher Information
- OnlineEWC: Online EWC with single Fisher estimate
- DiagonalEWC: EWC with diagonal Fisher approximation
- KFAC_EWC: EWC with K-FAC approximated Fisher
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import copy
from collections import defaultdict


@dataclass
class EWCPersistent:
    """Persistent storage for EWC parameters."""

    optimal_params: Dict[str, Tensor]
    fisher_diagonal: Dict[str, Tensor]


class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC) Regularizer.

    Prevents catastrophic forgetting by penalizing changes to parameters
    that were important for previous tasks.

    Reference:
        Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017

    Args:
        model: Neural network to regularize
        ewc_lambda: Regularization strength (higher = more protection)
        fisher_sample_size: Number of samples for Fisher estimation
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        fisher_sample_size: int = 200,
        device: str = "cpu",
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size
        self.device = device

        self.task_persistent: Dict[int, EWCPersistent] = {}
        self.current_task: int = 0

    def compute_fisher(
        self,
        dataloader: DataLoader,
        task_id: int,
    ) -> Dict[str, Tensor]:
        """
        Compute Fisher Information Matrix diagonal.

        Args:
            dataloader: DataLoader for sampling
            task_id: Task identifier

        Returns:
            Dictionary of Fisher diagonals per parameter
        """
        fisher = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()

        sample_count = 0

        for inputs, targets in dataloader:
            if sample_count >= self.fisher_sample_size:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            log_probs = F.log_softmax(outputs, dim=-1)
            probs = F.softmax(outputs, dim=-1)

            for t in range(len(targets)):
                if sample_count >= self.fisher_sample_size:
                    break

                sampled_idx = torch.multinomial(probs[t], 1).item()
                loss = log_probs[t, sampled_idx]
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.pow(2)

                sample_count += 1

        for n in fisher:
            fisher[n] /= sample_count

        return fisher

    def compute_empirical_fisher(
        self,
        dataloader: DataLoader,
        task_id: int,
    ) -> Dict[str, Tensor]:
        """
        Compute empirical Fisher using gradients of log-likelihood.

        Args:
            dataloader: DataLoader for sampling
            task_id: Task identifier

        Returns:
            Dictionary of empirical Fisher diagonals
        """
        fisher = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()

        sample_count = 0

        for inputs, targets in dataloader:
            if sample_count >= self.fisher_sample_size:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2)

            sample_count += inputs.size(0)

        for n in fisher:
            fisher[n] /= sample_count

        return fisher

    def register_task(
        self, task_id: int, dataloader: Optional[DataLoader] = None
    ) -> None:
        """
        Register a task by computing and storing Fisher and optimal params.

        Args:
            task_id: Task identifier
            dataloader: Optional dataloader for Fisher computation
        """
        if dataloader is not None:
            fisher = self.compute_empirical_fisher(dataloader, task_id)
        else:
            fisher = {
                n: torch.ones_like(p, device=self.device)
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }

        optimal_params = {
            n: p.detach().clone().to(self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.task_persistent[task_id] = EWCPersistent(
            optimal_params=optimal_params,
            fisher_diagonal=fisher,
        )

        self.current_task = task_id

    def penalty(self, task_id: Optional[int] = None) -> Tensor:
        """
        Compute EWC penalty term.

        Args:
            task_id: Optional task ID (uses current if None)

        Returns:
            EWC penalty as scalar tensor
        """
        if len(self.task_persistent) == 0:
            return torch.tensor(0.0, device=self.device)

        if task_id is None:
            task_id = self.current_task

        loss = torch.tensor(0.0, device=self.device)

        for tid, persistent in self.task_persistent.items():
            if tid == task_id:
                continue

            for n, p in self.model.named_parameters():
                if p.requires_grad and n in persistent.optimal_params:
                    diff = p - persistent.optimal_params[n]
                    loss += (persistent.fisher_diagonal[n] * diff.pow(2)).sum()

        return self.ewc_lambda * loss

    def penalty_single_task(self, task_id: int) -> Tensor:
        """Compute penalty for a specific task only."""
        if task_id not in self.task_persistent:
            return torch.tensor(0.0, device=self.device)

        persistent = self.task_persistent[task_id]

        loss = torch.tensor(0.0, device=self.device)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in persistent.optimal_params:
                diff = p - persistent.optimal_params[n]
                loss += (persistent.fisher_diagonal[n] * diff.pow(2)).sum()

        return self.ewc_lambda * loss

    def get_importance_scores(self) -> Dict[str, float]:
        """Get normalized importance scores for each parameter."""
        total_importance = {}

        for tid, persistent in self.task_persistent.items():
            for n, fisher in persistent.fisher_diagonal.items():
                if n not in total_importance:
                    total_importance[n] = 0.0
                total_importance[n] += fisher.sum().item()

        return total_importance


class OnlineEWC:
    """
    Online Elastic Weight Consolidation.

    Efficient online version that maintains a single running estimate
    of Fisher Information rather than storing per-task estimates.

    Reference:
        Schwarz et al., "Progress & Compress: A scalable framework for continual learning", ICML 2018

    Args:
        model: Neural network to regularize
        ewc_lambda: Regularization strength
        gamma: Decay factor for online Fisher
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.device = device

        self.fisher: Optional[Dict[str, Tensor]] = None
        self.optimal_params: Optional[Dict[str, Tensor]] = None
        self.task_count: int = 0

    def compute_fisher(self, dataloader: DataLoader) -> Dict[str, Tensor]:
        """Compute Fisher Information for current data."""
        fisher = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()

        sample_count = 0

        for inputs, _ in dataloader:
            if sample_count >= 200:
                break

            inputs = inputs.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            probs = F.softmax(outputs, dim=-1)
            log_probs = F.log_softmax(outputs, dim=-1)

            loss = -(probs * log_probs).sum(dim=-1).mean()
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2)

            sample_count += inputs.size(0)

        for n in fisher:
            fisher[n] /= sample_count

        return fisher

    def update(self, dataloader: Optional[DataLoader] = None) -> None:
        """
        Update Fisher estimate and optimal parameters.

        Args:
            dataloader: Optional dataloader for Fisher computation
        """
        if dataloader is not None:
            new_fisher = self.compute_fisher(dataloader)
        else:
            new_fisher = {
                n: torch.ones_like(p, device=self.device)
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }

        if self.fisher is None:
            self.fisher = new_fisher
        else:
            for n in self.fisher:
                self.fisher[n] = (
                    self.gamma * self.fisher[n] + (1 - self.gamma) * new_fisher[n]
                )

        self.optimal_params = {
            n: p.detach().clone().to(self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.task_count += 1

    def penalty(self) -> Tensor:
        """Compute online EWC penalty."""
        if self.fisher is None or self.optimal_params is None:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                diff = p - self.optimal_params[n]
                loss += (self.fisher[n] * diff.pow(2)).sum()

        return self.ewc_lambda * loss


class DiagonalEWC:
    """
    EWC with Diagonal Fisher Approximation.

    Uses efficient diagonal approximation of Fisher Information
    for memory efficiency.

    Args:
        model: Neural network to regularize
        ewc_lambda: Regularization strength
        ema_decay: EMA decay for Fisher estimation
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        ema_decay: float = 0.95,
        device: str = "cpu",
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.ema_decay = ema_decay
        self.device = device

        self.fisher: Optional[Dict[str, Tensor]] = None
        self.optimal_params: Optional[Dict[str, Tensor]] = None

    def update_fisher(self, batch_size: int = 32) -> None:
        """Update Fisher using exponential moving average."""
        self.model.eval()

        if self.fisher is None:
            self.fisher = {
                n: torch.zeros_like(p, device=self.device)
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }

        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad_squared = p.grad.pow(2)
                self.fisher[n] = (
                    self.ema_decay * self.fisher[n]
                    + (1 - self.ema_decay) * grad_squared
                )

    def register_params(self) -> None:
        """Register current parameters as optimal."""
        self.optimal_params = {
            n: p.detach().clone().to(self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def penalty(self) -> Tensor:
        """Compute diagonal EWC penalty."""
        if self.fisher is None or self.optimal_params is None:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                diff = p - self.optimal_params[n]
                loss += (self.fisher[n].sqrt() * diff.abs()).sum()

        return self.ewc_lambda * loss


class KFAC_EWC:
    """
    EWC with Kronecker-Factored Approximate Curvature (KFAC).

    Uses K-FAC approximation for more accurate Fisher estimation
    while maintaining computational efficiency.

    Reference:
        Martens & Grosse, "Optimizing Neural Networks with Kronecker-factored Approximate Curvature", ICML 2015

    Args:
        model: Neural network to regularize
        ewc_lambda: Regularization strength
        kfac_stats: Number of batches for KFAC statistics
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        kfac_stats: int = 200,
        device: str = "cpu",
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.kfac_stats = kfac_stats
        self.device = device

        self.fisher_factors: Dict[str, Tuple[Tensor, Tensor]] = {}
        self.optimal_params: Optional[Dict[str, Tensor]] = None

    def compute_kfac_factors(self, dataloader: DataLoader) -> None:
        """Compute K-FAC Fisher factors."""
        self.model.eval()

        a_factors: Dict[str, List[Tensor]] = defaultdict(list)
        g_factors: Dict[str, List[Tensor]] = defaultdict(list)

        count = 0

        for inputs, _ in dataloader:
            if count >= self.kfac_stats:
                break

            inputs = inputs.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            probs = F.softmax(outputs, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)
            loss = F.cross_entropy(outputs, idx)
            loss.backward()

            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    if hasattr(module, "weight") and module.weight.grad is not None:
                        a = module.input_cov if hasattr(module, "input_cov") else None
                        g = module.weight.grad

                        if a is not None:
                            a_factors[name].append(a.detach())
                            g_factors[name].append(g.detach())

            count += inputs.size(0)

        for name in a_factors:
            if len(a_factors[name]) > 0:
                A = torch.stack(a_factors[name]).mean(dim=0)
                G = torch.stack(g_factors[name]).mean(dim=0)

                eigenvalues, eigenvectors = torch.linalg.eigh(A)
                eigenvalues = eigenvalues.clamp(min=1e-6)
                A_inv_sqrt = (
                    eigenvectors @ torch.diag(eigenvalues**-0.5) @ eigenvectors.T
                )

                self.fisher_factors[name] = (A_inv_sqrt, G)

    def register_params(self, dataloader: DataLoader) -> None:
        """Register optimal parameters after computing K-FAC."""
        self.compute_kfac_factors(dataloader)

        self.optimal_params = {
            n: p.detach().clone().to(self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def penalty(self) -> Tensor:
        """Compute K-FAC EWC penalty."""
        if len(self.fisher_factors) == 0 or self.optimal_params is None:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for name, (A_inv_sqrt, G) in self.fisher_factors.items():
            for n, p in self.model.named_parameters():
                if name in n and p.requires_grad and n in self.optimal_params:
                    diff = p - self.optimal_params[n]
                    reshaped_diff = diff.view(diff.size(0), -1)

                    grad_norm = A_inv_sqrt @ reshaped_diff
                    loss += (grad_norm * (G @ grad_norm.T)).sum()

        return self.ewc_lambda * loss

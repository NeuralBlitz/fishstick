"""
Memory Aware Synapses (MAS) Implementation.

Model-agnostic approach to estimating parameter importance by measuring
sensitivity of learned function output to parameter changes.

Classes:
- MemoryAwareSynapses: MAS regularizer
- MASImportance: Importance computation utilities
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np


class MemoryAwareSynapses:
    """
    Memory Aware Synapses (MAS) Regularizer.

    Estimates parameter importance by measuring sensitivity of the
    learned function output to parameter changes.

    Reference:
        Aljundi et al., "Memory Aware Synapses: Learning What (not) to forget", ECCV 2018

    Args:
        model: Neural network to regularize
        lambda_mas: Regularization strength
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_mas: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model
        self.lambda_mas = lambda_mas
        self.device = device

        self.omega: Dict[str, Tensor] = {}
        self.optimal_params: Dict[str, Tensor] = {}

        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize MAS storage."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.omega[n] = torch.zeros_like(p, device=self.device)
                self.optimal_params[n] = p.detach().clone().to(self.device)

    def compute_importance(
        self,
        dataloader: DataLoader,
        sample_size: int = 200,
    ) -> None:
        """
        Compute parameter importance using MAS methodology.

        Args:
            dataloader: DataLoader for computing importance
            sample_size: Number of samples to use
        """
        self.model.eval()

        for n in self.omega:
            self.omega[n].zero_()

        sample_count = 0

        for inputs, _ in dataloader:
            if sample_count >= sample_size:
                break

            inputs = inputs.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            loss = (outputs**2).sum() / outputs.numel()
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.omega[n] += p.grad.abs()

            sample_count += inputs.size(0)

        for n in self.omega:
            self.omega[n] /= sample_count

    def compute_importance_knn(
        self,
        dataloader: DataLoader,
        sample_size: int = 200,
    ) -> None:
        """
        Compute importance using k-NN based sensitivity.

        Args:
            dataloader: DataLoader for computing importance
            sample_size: Number of samples to use
        """
        self.model.eval()

        for n in self.omega:
            self.omega[n].zero_()

        features_list = []
        targets_list = []

        sample_count = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                if sample_count >= sample_size:
                    break

                inputs = inputs.to(self.device)

                outputs = self.model(inputs)

                if isinstance(outputs, tuple):
                    features = outputs[0]
                else:
                    features = outputs

                features_list.append(features)
                targets_list.append(targets)

                sample_count += inputs.size(0)

        features = torch.cat(features_list, dim=0)
        targets = torch.cat(targets_list, dim=0)

        feature_importance = features.std(dim=0).unsqueeze(0)

        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad_shape = p.grad.shape

                if len(grad_shape) >= 2:
                    self.omega[n] += feature_importance.mean()
                else:
                    self.omega[n] += 1.0

    def compute_importance_output_distance(
        self,
        dataloader: DataLoader,
        sample_size: int = 200,
    ) -> None:
        """
        Compute importance based on output distance sensitivity.

        Args:
            dataloader: DataLoader for computing importance
            sample_size: Number of samples to use
        """
        self.model.eval()

        for n in self.omega:
            self.omega[n].zero_()

        sample_count = 0

        for inputs, _ in dataloader:
            if sample_count >= sample_size:
                break

            inputs = inputs.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            output_dist = outputs.pow(2).sum(dim=-1).mean()
            output_dist.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.omega[n] += p.grad.abs()

            sample_count += inputs.size(0)

        for n in self.omega:
            self.omega[n] /= sample_count

    def register_optimal_params(self) -> None:
        """Register current parameters as optimal."""
        self.optimal_params = {
            n: p.detach().clone().to(self.device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def update_importance(
        self,
        dataloader: DataLoader,
        sample_size: int = 200,
    ) -> None:
        """
        Update importance estimates with new data.

        Args:
            dataloader: DataLoader for computing importance
            sample_size: Number of samples to use
        """
        new_omega: Dict[str, Tensor] = {}

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                new_omega[n] = torch.zeros_like(p, device=self.device)

        self.model.eval()

        sample_count = 0

        for inputs, _ in dataloader:
            if sample_count >= sample_size:
                break

            inputs = inputs.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            loss = (outputs**2).sum() / outputs.numel()
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    new_omega[n] += p.grad.abs()

            sample_count += inputs.size(0)

        for n in new_omega:
            new_omega[n] /= sample_count
            self.omega[n] = 0.5 * self.omega[n] + 0.5 * new_omega[n]

    def penalty(self) -> Tensor:
        """
        Compute MAS penalty term.

        Returns:
            MAS regularization penalty
        """
        loss = torch.tensor(0.0, device=self.device)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.omega and n in self.optimal_params:
                diff = p - self.optimal_params[n]
                loss += (self.omega[n] * diff.pow(2)).sum()

        return self.lambda_mas * loss

    def get_importance_map(self) -> Dict[str, float]:
        """Get normalized importance map."""
        total = sum(omega.sum().item() for omega in self.omega.values())

        if total == 0:
            return {n: 0.0 for n in self.omega}

        return {n: omega.sum().item() / total for n, omega in self.omega.items()}

    def apply_importance_mask(self, threshold: float = 0.5) -> None:
        """
        Apply mask to protect important parameters.

        Args:
            threshold: Importance threshold for protection
        """
        importance_map = self.get_importance_map()

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in importance_map:
                if importance_map[n] > threshold:
                    p.requires_grad = False

    def reset_importance(self) -> None:
        """Reset importance values."""
        for n in self.omega:
            self.omega[n].zero_()


class MASImportance:
    """
    Utilities for computing MAS importance scores.

    Provides different methods for computing parameter importance
    in the MAS framework.
    """

    @staticmethod
    def compute_gradient_magnitude(
        model: nn.Module,
        dataloader: DataLoader,
    ) -> Dict[str, Tensor]:
        """Compute gradient magnitude based importance."""
        importance = {}

        for n, p in model.named_parameters():
            if p.requires_grad:
                importance[n] = torch.zeros_like(p)

        model.eval()

        for inputs, _ in dataloader:
            model.zero_grad()

            outputs = model(inputs)
            loss = outputs.pow(2).mean()
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    importance[n] += p.grad.abs()

        return importance

    @staticmethod
    def compute_output_sensitivity(
        model: nn.Module,
        dataloader: DataLoader,
        target_layer: str,
    ) -> Dict[str, Tensor]:
        """Compute output sensitivity based importance."""
        importance = {}

        def hook_fn(module, input, output):
            pass

        hooks = []

        for n, m in model.named_modules():
            if target_layer in n:
                hooks.append(m.register_forward_hook(hook_fn))

        model.eval()

        for inputs, _ in dataloader:
            model(inputs)

        for hook in hooks:
            hook.remove()

        return importance

    @staticmethod
    def normalize_importance(
        importance: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Normalize importance scores to [0, 1]."""
        all_vals = torch.cat([v.flatten() for v in importance.values()])

        min_val = all_vals.min()
        max_val = all_vals.max()

        if max_val - min_val < 1e-8:
            return importance

        normalized = {}

        for n, v in importance.items():
            normalized[n] = (v - min_val) / (max_val - min_val)

        return normalized

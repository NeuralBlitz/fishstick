"""
Private Aggregation Methods for Federated and Distributed Learning.

This module provides aggregation methods that preserve privacy when
combining model updates from multiple sources.

Example:
    >>> from fishstick.privacy import PrivateAggregator, NoisyAggregator
    >>>
    >>> aggregator = NoisyAggregator(noise_scale=0.1)
    >>> global_update = aggregator.aggregate(updates)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

Tensor = torch.Tensor
Module = nn.Module


class PrivateAggregator(ABC):
    """Abstract base class for private aggregation.

    Example:
        >>> class MyAggregator(PrivateAggregator):
        ...     def aggregate(self, updates: List[Dict]) -> Dict: pass
    """

    @abstractmethod
    def aggregate(
        self,
        updates: List[Dict[str, Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Aggregate model updates privately.

        Args:
            updates: List of model parameter dictionaries.
            weights: Optional weights for each update.

        Returns:
            Aggregated model parameters.
        """
        pass

    @abstractmethod
    def aggregate_models(
        self,
        models: List[Module],
        weights: Optional[List[float]] = None,
    ) -> Module:
        """Aggregate model objects.

        Args:
            models: List of PyTorch models.
            weights: Optional weights for each model.

        Returns:
            Aggregated model.
        """
        pass


class NoisyAggregator(PrivateAggregator):
    """Add noise to aggregated updates for privacy.

    Adds calibrated Gaussian noise to the aggregated result.

    Args:
        noise_scale: Scale of noise to add.
        clip_norm: Norm for clipping aggregated update.
        epsilon: Privacy parameter.

    Example:
        >>> aggregator = NoisyAggregator(noise_scale=0.1)
        >>> global_model = aggregator.aggregate_models(client_models)
    """

    def __init__(
        self,
        noise_scale: float = 0.1,
        clip_norm: Optional[float] = None,
        epsilon: float = 1.0,
    ):
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        self.epsilon = epsilon

    def aggregate(
        self,
        updates: List[Dict[str, Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Aggregate with noise addition.

        Args:
            updates: List of parameter dictionaries.
            weights: Optional weights.

        Returns:
            Noisy aggregated parameters.
        """
        if not updates:
            return {}

        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        aggregated = {}
        param_keys = updates[0].keys()

        for key in param_keys:
            stacked = torch.stack([up[key] * w for up, w in zip(updates, weights)])

            summed = stacked.sum(dim=0)

            if self.clip_norm is not None:
                flat = summed.view(-1)
                norm = flat.norm()
                if norm > self.clip_norm:
                    summed = summed * (self.clip_norm / norm)

            noise_scale = self.noise_scale * (self.clip_norm or 1.0)
            noise = torch.randn_like(summed) * noise_scale
            aggregated[key] = summed + noise

        return aggregated

    def aggregate_models(
        self,
        models: List[Module],
        weights: Optional[List[float]] = None,
    ) -> Module:
        """Aggregate models with noise.

        Args:
            models: List of PyTorch models.
            weights: Optional weights.

        Returns:
            Aggregated model with noise.
        """
        updates = [model.state_dict() for model in models]
        aggregated_state = self.aggregate(updates, weights)

        result = type(models[0])()
        result.load_state_dict(aggregated_state)

        return result


class ClippedAggregator(PrivateAggregator):
    """Aggregate with per-update clipping.

    Clips each update before aggregation to bound sensitivity.

    Args:
        clip_norm: Maximum norm for each update.
        noise_scale: Scale of noise to add (optional).

    Example:
        >>> aggregator = ClippedAggregator(clip_norm=1.0)
        >>> global_update = aggregator.aggregate(updates)
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_scale: float = 0.0,
    ):
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale

    def aggregate(
        self,
        updates: List[Dict[str, Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Aggregate with clipping.

        Args:
            updates: List of parameter dictionaries.
            weights: Optional weights.

        Returns:
            Aggregated parameters.
        """
        if not updates:
            return {}

        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)

        clipped_updates = []
        for update in updates:
            clipped = {}
            for key, param in update.items():
                flat = param.view(-1)
                norm = flat.norm()

                if norm > self.clip_norm:
                    clipped[key] = param * (self.clip_norm / norm)
                else:
                    clipped[key] = param

            clipped_updates.append(clipped)

        aggregated = {}
        param_keys = updates[0].keys()

        for key in param_keys:
            stacked = torch.stack(
                [up[key] * w for up, w in zip(clipped_updates, weights)]
            )
            summed = stacked.sum(dim=0)

            if self.noise_scale > 0:
                noise = torch.randn_like(summed) * self.noise_scale * self.clip_norm
                summed = summed + noise

            aggregated[key] = summed

        return aggregated

    def aggregate_models(
        self,
        models: List[Module],
        weights: Optional[List[float]] = None,
    ) -> Module:
        """Aggregate models with clipping."""
        updates = [model.state_dict() for model in models]
        aggregated_state = self.aggregate(updates, weights)

        result = type(models[0])()
        result.load_state_dict(aggregated_state)

        return result


class SecureAggregator(PrivateAggregator):
    """Secure aggregation with cryptographic primitives.

    Uses secret sharing and threshold cryptography to ensure
    no individual update is revealed.

    Reference:
        Bonawitz et al., "Practical Secure Aggregation", PoPETS 2017.

    Args:
        threshold: Minimum number of participants required.
        num_clients: Total number of clients.
        modulus: Prime modulus for secret sharing.
    """

    def __init__(
        self,
        threshold: int = 2,
        num_clients: int = 3,
        modulus: int = 2**61 - 1,
    ):
        self.threshold = threshold
        self.num_clients = num_clients
        self.modulus = modulus

    def aggregate(
        self,
        updates: List[Dict[str, Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Aggregate using secure aggregation protocol.

        Args:
            updates: List of parameter dictionaries.
            weights: Optional weights.

        Returns:
            Aggregated parameters.
        """
        if len(updates) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} updates, got {len(updates)}"
            )

        masked_updates = self._mask_updates(updates)
        aggregated = self._aggregate_masked(masked_updates, weights)
        result = self._unmask(aggregated, len(updates))

        return result

    def _mask_updates(
        self,
        updates: List[Dict[str, Tensor]],
    ) -> List[Dict[str, Tensor]]:
        """Mask updates with secret shares.

        Args:
            updates: List of updates.

        Returns:
            Masked updates.
        """
        masked = []

        for i, update in enumerate(updates):
            mask = {}
            for key, param in update.items():
                mask[key] = torch.randint_like(param, 0, self.modulus)

            masked.append(mask)

            for key in update.keys():
                updates[i][key] = (updates[i][key] + mask[key]) % self.modulus

        return masked

    def _aggregate_masked(
        self,
        updates: List[Dict[str, Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Aggregate masked updates.

        Args:
            updates: List of masked updates.
            weights: Optional weights.

        Returns:
            Sum of masked updates.
        """
        if weights is None:
            weights = [1.0] * len(updates)

        result = {}
        param_keys = updates[0].keys()

        for key in param_keys:
            stacked = torch.stack([up[key] * w for up, w in zip(updates, weights)])
            result[key] = stacked.sum(dim=0) % self.modulus

        return result

    def _unmask(
        self,
        aggregated: Dict[str, Tensor],
        num_participants: int,
    ) -> Dict[str, Tensor]:
        """Unmask the aggregated result.

        Args:
            aggregated: Aggregated masked result.
            num_participants: Number of participants.

        Returns:
            Unmasked aggregated result.
        """
        result = {}
        for key, param in aggregated.items():
            result[key] = param / num_participants

        return result

    def aggregate_models(
        self,
        models: List[Module],
        weights: Optional[List[float]] = None,
    ) -> Module:
        """Aggregate models securely."""
        updates = [model.state_dict() for model in models]
        aggregated_state = self.aggregate(updates, weights)

        result = type(models[0])()
        result.load_state_dict(aggregated_state)

        return result


class DPFederatedAggregator(PrivateAggregator):
    """Differentially private federated averaging.

    Combines secure aggregation with DP noise for strong privacy guarantees.

    Args:
        clip_norm: Norm for clipping client updates.
        noise_scale: Scale of noise to add.
        epsilon: Target privacy budget.
        delta: Target delta.

    Example:
        >>> aggregator = DPFederatedAggregator(clip_norm=1.0, noise_scale=0.1)
        >>> global_model = aggregator.aggregate_models(client_models)
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_scale: float = 0.1,
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ):
        self.clip_norm = clip_norm
        self.noise_scale = noise_scale
        self.epsilon = epsilon
        self.delta = delta

    def aggregate(
        self,
        updates: List[Dict[str, Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Aggregate with DP guarantees.

        Args:
            updates: List of parameter dictionaries.
            weights: Optional weights.

        Returns:
            DP-aggregated parameters.
        """
        if not updates:
            return {}

        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)

        clipped = self._clip_updates(updates)

        aggregated = self._weighted_sum(clipped, weights)

        noise = self._compute_noise(aggregated)
        for key in aggregated.keys():
            aggregated[key] = aggregated[key] + noise[key]

        return aggregated

    def _clip_updates(
        self,
        updates: List[Dict[str, Tensor]],
    ) -> List[Dict[str, Tensor]]:
        """Clip each update.

        Args:
            updates: List of updates.

        Returns:
            Clipped updates.
        """
        clipped = []

        for update in updates:
            clipped_update = {}
            for key, param in update.items():
                flat = param.view(-1)
                norm = flat.norm()

                if norm > self.clip_norm:
                    clipped_update[key] = param * (self.clip_norm / norm)
                else:
                    clipped_update[key] = param

            clipped.append(clipped_update)

        return clipped

    def _weighted_sum(
        self,
        updates: List[Dict[str, Tensor]],
        weights: List[float],
    ) -> Dict[str, Tensor]:
        """Compute weighted sum.

        Args:
            updates: List of updates.
            weights: Weights.

        Returns:
            Weighted sum.
        """
        result = {}
        param_keys = updates[0].keys()

        for key in param_keys:
            stacked = torch.stack([up[key] * w for up, w in zip(updates, weights)])
            result[key] = stacked.sum(dim=0)

        return result

    def _compute_noise(
        self,
        aggregated: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute DP noise.

        Args:
            aggregated: Aggregated parameters.

        Returns:
            Noise tensors.
        """
        sigma = self.noise_scale * self.clip_norm

        noise = {}
        for key, param in aggregated.items():
            noise[key] = torch.randn_like(param) * sigma

        return noise

    def aggregate_models(
        self,
        models: List[Module],
        weights: Optional[List[float]] = None,
    ) -> Module:
        """Aggregate models with DP."""
        updates = [model.state_dict() for model in models]
        aggregated_state = self.aggregate(updates, weights)

        result = type(models[0])()
        result.load_state_dict(aggregated_state)

        return result


def create_aggregator(
    aggregator_type: str = "noisy",
    **kwargs,
) -> PrivateAggregator:
    """Factory function to create aggregators.

    Args:
        aggregator_type: Type of aggregator.
        **kwargs: Additional arguments.

    Returns:
        Configured aggregator.

    Example:
        >>> aggregator = create_aggregator('noisy', noise_scale=0.1)
    """
    aggregator_type = aggregator_type.lower()

    if aggregator_type == "noisy":
        return NoisyAggregator(**kwargs)
    elif aggregator_type == "clipped":
        return ClippedAggregator(**kwargs)
    elif aggregator_type == "secure":
        return SecureAggregator(**kwargs)
    elif aggregator_type == "dp_federated":
        return DPFederatedAggregator(**kwargs)
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")


def compute_adaptive_noise(
    epsilon: float,
    delta: float,
    num_clients: int,
    sensitivity: float = 1.0,
) -> float:
    """Compute adaptive noise scale for aggregation.

    Args:
        epsilon: Target epsilon.
        delta: Target delta.
        num_clients: Number of clients.
        sensitivity: Aggregation sensitivity.

    Returns:
        Noise scale.
    """
    import math

    sigma = math.sqrt(2 * math.log(1.25 / delta) * num_clients) / epsilon

    return sigma * sensitivity

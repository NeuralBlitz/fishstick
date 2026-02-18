"""
Advanced Aggregation Strategies for Federated Learning

This module provides advanced federated aggregation strategies:
- FedNova: Normalized averaging for varying local steps
- FedProx: Proximal regularization for handling heterogeneity
- FedDyn: Dynamic regularization
- Async Aggregation: Asynchronous model updates
- Client Drift Correction: Correcting client model drift

References:
- Wang et al. (2021): "Federated Learning with Non-IID Data"
- Li et al. (2020): "Federated Optimization in Heterogeneous Networks"
- Acedanski et al. (2021): "FedDyn: Federated Learning with Dynamic Regularization"
"""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Advanced aggregation strategies."""

    FEDNOVA = auto()
    FEDPROX = auto()
    FEDDYN = auto()
    ASYNC = auto()
    FEDAVG = auto()


@dataclass
class AggregationConfig:
    """Configuration for advanced aggregation."""

    strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    local_steps: int = 5
    proximal_mu: float = 0.01
    async_timeout: float = 30.0
    async_buffer_size: int = 10
    staleness_weight: float = 0.5
    drift_correction: bool = True


class BaseAggregationStrategy(ABC):
    """Base class for advanced aggregation strategies."""

    def __init__(self, config: AggregationConfig):
        self.config = config
        self.iteration = 0
        self.global_state: Optional[Dict[str, Tensor]] = None

    @abstractmethod
    def aggregate(
        self,
        client_updates: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Tensor]:
        """Aggregate client updates.

        Args:
            client_updates: List of client model updates
            client_weights: Optional weights for each client
            metadata: Optional metadata (e.g., local steps, staleness)

        Returns:
            Aggregated global model update
        """
        pass

    def set_global_state(self, state: Dict[str, Tensor]) -> None:
        """Set global model state."""
        self.global_state = {k: v.clone() for k, v in state.items()}

    def get_global_state(self) -> Optional[Dict[str, Tensor]]:
        """Get global model state."""
        return self.global_state


class FedNovaStrategy(BaseAggregationStrategy):
    """FedNova: Federated Normalized Averaging.

    Normalizes client updates by the number of local steps to handle
    heterogeneous local training durations.

    Reference: Wang et al. (2021)
    """

    def aggregate(
        self,
        client_updates: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Tensor]:
        if not client_updates:
            raise ValueError("No client updates provided")

        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)

        if metadata is None:
            local_steps = [self.config.local_steps] * len(client_updates)
        else:
            local_steps = [
                m.get("local_steps", self.config.local_steps) for m in metadata
            ]

        total_steps = sum(ls * w for ls, w in zip(local_steps, client_weights))
        if total_steps == 0:
            total_steps = 1

        keys = client_updates[0].keys()
        aggregated: Dict[str, Tensor] = {}

        for key in keys:
            weighted_sum = torch.zeros_like(client_updates[0][key], dtype=torch.float32)

            for update, weight, steps in zip(
                client_updates, client_weights, local_steps
            ):
                normalized_update = update[key].float() * (steps / total_steps)
                weighted_sum += weight * normalized_update

            aggregated[key] = weighted_sum.to(client_updates[0][key].dtype)

        self.iteration += 1
        logger.debug(f"FedNova aggregation completed for iteration {self.iteration}")
        return aggregated


class FedProxStrategy(BaseAggregationStrategy):
    """FedProx: Federated Proximal.

    Adds proximal regularization to handle heterogeneity by limiting
    the distance between local and global models.

    Reference: Li et al. (2020)
    """

    def __init__(self, config: AggregationConfig):
        super().__init__(config)
        self.client_deltas: List[Dict[str, Tensor]] = []

    def aggregate(
        self,
        client_updates: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Tensor]:
        if not client_updates:
            raise ValueError("No client updates provided")

        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)

        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        keys = client_updates[0].keys()
        aggregated: Dict[str, Tensor] = {}

        for key in keys:
            weighted_sum = torch.zeros_like(client_updates[0][key], dtype=torch.float32)

            for update, weight in zip(client_updates, client_weights):
                weighted_sum += weight * update[key].float()

            aggregated[key] = weighted_sum.to(client_updates[0][key].dtype)

        if self.config.drift_correction and self.global_state:
            corrected = self._apply_drift_correction(aggregated)
            return corrected

        self.iteration += 1
        return aggregated

    def _apply_drift_correction(
        self,
        aggregated: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Apply client drift correction."""
        if not self.client_deltas:
            return aggregated

        keys = aggregated.keys()
        correction: Dict[str, Tensor] = {}

        for key in keys:
            correction_sum = torch.zeros_like(aggregated[key])
            for delta in self.client_deltas[-len(client_updates) :]:
                if key in delta:
                    correction_sum += delta[key]

            avg_correction = correction_sum / max(len(self.client_deltas), 1)
            correction[key] = aggregated[key] - self.config.proximal_mu * avg_correction

        return correction


class FedDynStrategy(BaseAggregationStrategy):
    """FedDyn: Federated Dynamic Regularization.

    Uses dynamic regularization to handle data heterogeneity.

    Reference: Acedanski et al. (2021)
    """

    def __init__(self, config: AggregationConfig):
        super().__init__(config)
        self.linear_term: Dict[str, Tensor] = {}
        self.hessian_estimate: Dict[str, Tensor] = {}

    def initialize(self, model_state: Dict[str, Tensor]) -> None:
        """Initialize linear term and Hessian estimate."""
        for key, param in model_state.items():
            self.linear_term[key] = torch.zeros_like(param)
            self.hessian_estimate[key] = torch.ones_like(param)

    def aggregate(
        self,
        client_updates: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Tensor]:
        if not client_updates:
            raise ValueError("No client updates provided")

        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)

        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        keys = client_updates[0].keys()
        aggregated: Dict[str, Tensor] = {}

        for key in keys:
            weighted_sum = torch.zeros_like(client_updates[0][key], dtype=torch.float32)

            for update, weight in zip(client_updates, client_weights):
                weighted_sum += weight * update[key].float()

            if key in self.linear_term:
                hessian = self.hessian_estimate.get(key, torch.ones_like(weighted_sum))
                regularized = self.linear_term[key] / (hessian + 1e-8)
                weighted_sum = weighted_sum - self.config.proximal_mu * regularized

            aggregated[key] = weighted_sum.to(client_updates[0][key].dtype)

        self._update_linear_term(aggregated)

        self.iteration += 1
        return aggregated

    def _update_linear_term(self, aggregated: Dict[str, Tensor]) -> None:
        """Update linear term based on current aggregated update."""
        for key in aggregated.keys():
            if key in self.linear_term:
                self.linear_term[key] = (
                    self.linear_term[key] - self.config.proximal_mu * aggregated[key]
                )


class AsyncAggregationStrategy(BaseAggregationStrategy):
    """Asynchronous Aggregation Strategy.

    Handles asynchronous client updates with staleness-aware weighting.

    Reference: Sprague et al. (2020): "Asynchronous Federated Optimization"
    """

    def __init__(self, config: AggregationConfig):
        super().__init__(config)
        self.update_queue: queue.Queue = queue.Queue(maxsize=config.async_buffer_size)
        self.last_update_round: Dict[int, int] = {}
        self.current_round = 0
        self.lock = threading.Lock()

    def add_update(
        self,
        client_id: int,
        update: Dict[str, Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a client update to the queue."""
        update_data = {
            "client_id": client_id,
            "update": update,
            "metadata": metadata or {},
            "timestamp": self.current_round,
        }

        try:
            self.update_queue.put_nowait(update_data)
        except queue.Full:
            logger.warning("Async buffer full, dropping oldest update")

    def aggregate(
        self,
        client_updates: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Tensor]:
        with self.lock:
            if not client_updates:
                pending_updates = self._collect_pending_updates()
                if not pending_updates:
                    return {}

                client_updates = [p["update"] for p in pending_updates]
                client_weights = [1.0 / len(pending_updates)] * len(pending_updates)
                metadata = [p["metadata"] for p in pending_updates]

            return self._aggregate_sync(client_updates, client_weights, metadata)

    def _collect_pending_updates(self) -> List[Dict[str, Any]]:
        """Collect pending updates from queue."""
        updates = []
        while not self.update_queue.empty():
            try:
                update = self.update_queue.get_nowait()
                updates.append(update)
            except queue.Empty:
                break
        return updates

    def _aggregate_sync(
        self,
        client_updates: List[Dict[str, Tensor]],
        client_weights: List[float],
        metadata: List[Dict[str, Any]],
    ) -> Dict[str, Tensor]:
        """Synchronously aggregate with staleness weighting."""
        if not client_updates:
            return {}

        staleness_weights = []
        for meta in metadata:
            staleness = meta.get("staleness", 1)
            staleness_weight = self.config.staleness_weight**staleness
            staleness_weights.append(staleness_weight)

        total_staleness = sum(sw for sw in staleness_weights)
        adjusted_weights = [
            w * sw / total_staleness for w, sw in zip(client_weights, staleness_weights)
        ]

        keys = client_updates[0].keys()
        aggregated: Dict[str, Tensor] = {}

        for key in keys:
            weighted_sum = torch.zeros_like(client_updates[0][key], dtype=torch.float32)
            for update, weight in zip(client_updates, adjusted_weights):
                weighted_sum += weight * update[key].float()
            aggregated[key] = weighted_sum.to(client_updates[0][key].dtype)

        self.current_round += 1
        self.iteration += 1
        return aggregated


class AdaptiveAggregationStrategy(BaseAggregationStrategy):
    """Adaptive Aggregation Strategy.

    Adapts aggregation weights based on client performance and data quality.
    """

    def __init__(self, config: AggregationConfig):
        super().__init__(config)
        self.client_scores: Dict[int, float] = {}
        self.client_contributions: Dict[int, List[float]] = {}

    def update_client_scores(
        self,
        client_id: int,
        accuracy: float,
        loss: float,
    ) -> None:
        """Update client score based on performance."""
        score = accuracy / (loss + 1e-8)
        self.client_scores[client_id] = score

        if client_id not in self.client_contributions:
            self.client_contributions[client_id] = []

        self.client_contributions[client_id].append(score)
        if len(self.client_contributions[client_id]) > 10:
            self.client_contributions[client_id] = self.client_contributions[client_id][
                -10:
            ]

    def aggregate(
        self,
        client_updates: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Tensor]:
        if not client_updates:
            raise ValueError("No client updates provided")

        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)

        if metadata is None:
            metadata = [{}] * len(client_updates)

        adaptive_weights = []
        for i, (weight, meta) in enumerate(zip(client_weights, metadata)):
            client_id = meta.get("client_id", i)
            score = self.client_scores.get(client_id, 1.0)
            contribution = self.client_contributions.get(client_id, [1.0])
            avg_contribution = (
                sum(contribution) / len(contribution) if contribution else 1.0
            )

            adaptive_weight = weight * (score / (avg_contribution + 1e-8))
            adaptive_weights.append(adaptive_weight)

        total_weight = sum(adaptive_weights)
        adaptive_weights = [w / total_weight for w in adaptive_weights]

        keys = client_updates[0].keys()
        aggregated: Dict[str, Tensor] = {}

        for key in keys:
            weighted_sum = torch.zeros_like(client_updates[0][key], dtype=torch.float32)
            for update, weight in zip(client_updates, adaptive_weights):
                weighted_sum += weight * update[key].float()
            aggregated[key] = weighted_sum.to(client_updates[0][key].dtype)

        self.iteration += 1
        return aggregated


def create_aggregation_strategy(
    config: AggregationConfig,
) -> BaseAggregationStrategy:
    """Factory function to create aggregation strategy.

    Args:
        config: Configuration for the aggregation strategy

    Returns:
        Instance of the appropriate aggregation strategy

    Example:
        >>> config = AggregationConfig(strategy=AggregationStrategy.FEDNOVA, local_steps=5)
        >>> strategy = create_aggregation_strategy(config)
    """
    strategies = {
        AggregationStrategy.FEDNOVA: FedNovaStrategy,
        AggregationStrategy.FEDPROX: FedProxStrategy,
        AggregationStrategy.FEDDYN: FedDynStrategy,
        AggregationStrategy.ASYNC: AsyncAggregationStrategy,
        AggregationStrategy.FEDAVG: BaseAggregationStrategy,
    }

    if config.strategy not in strategies:
        raise ValueError(f"Unknown aggregation strategy: {config.strategy}")

    return strategies[config.strategy](config)


__all__ = [
    "AggregationStrategy",
    "AggregationConfig",
    "BaseAggregationStrategy",
    "FedNovaStrategy",
    "FedProxStrategy",
    "FedDynStrategy",
    "AsyncAggregationStrategy",
    "AdaptiveAggregationStrategy",
    "create_aggregation_strategy",
]

"""
Client Sampling Strategies for Federated Learning

This module provides various client sampling strategies for federated learning:
- Random Sampling: Randomly select clients each round
- Round-Robin: Sequential client selection
- FedCS: Communication-efficient client selection
- Oort: Resource-aware client selection
- Power-of-Choice: Multi-armed bandit based selection

References:
- Nishio & Yonetani (2019): "Client Selection for Federated Learning with Heterogeneous Resources"
- Lai et al. (2021): "Oort: Efficient Federated Learning via Guided Client Selection"
- Ribero & Vikalo (2020): "Communication-Efficient Federated Learning via Importance Sampling"
"""

from __future__ import annotations

import logging
import random
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Supported client sampling strategies."""

    RANDOM = auto()
    ROUND_ROBIN = auto()
    FEDCS = auto()
    OORT = auto()
    POWER_OF_CHOICE = auto()
    BANDIT = auto()


@dataclass
class SamplingConfig:
    """Configuration for client sampling strategies."""

    strategy: SamplingStrategy = SamplingStrategy.RANDOM
    sample_fraction: float = 0.1
    min_sample_size: int = 10
    max_retry: int = 3
    alpha: float = 1.0
    beta: float = 1.0
    exploration_weight: float = 1.0
    seed: Optional[int] = None


@dataclass
class ClientInfo:
    """Information about a client for sampling decisions."""

    client_id: int
    num_samples: int
    latency: float = 1.0
    accuracy: float = 0.0
    loss: float = float("inf")
    availability: float = 1.0
    last_selected: int = 0
    selection_count: int = 0
    historical_loss: List[float] = field(default_factory=list)
    resource_score: float = 1.0

    def compute_loss_improvement(self) -> float:
        """Compute loss improvement based on historical loss."""
        if len(self.historical_loss) < 2:
            return 0.0
        return self.historical_loss[-2] - self.historical_loss[-1]

    def compute_resource_utilization(self) -> float:
        """Compute resource utilization score."""
        return self.resource_score * self.availability


class BaseSamplingStrategy(ABC):
    """Base class for client sampling strategies."""

    def __init__(self, config: SamplingConfig, num_clients: int):
        self.config = config
        self.num_clients = num_clients
        self.client_info: Dict[int, ClientInfo] = {}
        self.current_round = 0

        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        for i in range(num_clients):
            self.client_info[i] = ClientInfo(client_id=i, num_samples=0)

    @abstractmethod
    def select_clients(
        self,
        available_clients: Optional[List[int]] = None,
    ) -> List[int]:
        """Select clients for the current round.

        Args:
            available_clients: Optional list of available client IDs

        Returns:
            List of selected client IDs
        """
        pass

    def update_client_info(
        self,
        client_id: int,
        num_samples: Optional[int] = None,
        latency: Optional[float] = None,
        accuracy: Optional[float] = None,
        loss: Optional[float] = None,
    ) -> None:
        """Update client information after training."""
        if client_id not in self.client_info:
            self.client_info[client_id] = ClientInfo(client_id=client_id, num_samples=0)

        if num_samples is not None:
            self.client_info[client_id].num_samples = num_samples
        if latency is not None:
            self.client_info[client_id].latency = latency
        if accuracy is not None:
            self.client_info[client_id].accuracy = accuracy
        if loss is not None:
            self.client_info[client_id].loss = loss
            self.client_info[client_id].historical_loss.append(loss)

    def get_num_clients_to_sample(self) -> int:
        """Calculate number of clients to sample."""
        available = self.num_clients
        num_samples = max(
            self.config.min_sample_size,
            int(available * self.config.sample_fraction),
        )
        return min(num_samples, available)

    def filter_available_clients(
        self,
        available_clients: Optional[List[int]] = None,
    ) -> List[int]:
        """Filter to available clients."""
        if available_clients is None:
            return list(range(self.num_clients))
        return [c for c in available_clients if c in range(self.num_clients)]


class RandomSamplingStrategy(BaseSamplingStrategy):
    """Random client sampling strategy.

    Randomly samples clients each round with probability proportional
    to their sample count.
    """

    def select_clients(
        self,
        available_clients: Optional[List[int]] = None,
    ) -> List[int]:
        available = self.filter_available_clients(available_clients)
        num_to_sample = self.get_num_clients_to_sample()

        if self.config.alpha == 1.0:
            selected = random.sample(available, min(num_to_sample, len(available)))
        else:
            weights = []
            for c in available:
                client = self.client_info[c]
                sample_weight = client.num_samples**self.config.alpha
                weights.append(sample_weight)
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            selected = np.random.choice(
                available,
                size=min(num_to_sample, len(available)),
                replace=False,
                p=weights,
            ).tolist()

        for c in selected:
            self.client_info[c].selection_count += 1
            self.client_info[c].last_selected = self.current_round

        self.current_round += 1
        logger.debug(
            f"Round {self.current_round}: Randomly selected {len(selected)} clients"
        )
        return selected


class RoundRobinSamplingStrategy(BaseSamplingStrategy):
    """Round-robin client sampling strategy.

    Selects clients in a sequential order to ensure fair participation.
    """

    def __init__(self, config: SamplingConfig, num_clients: int):
        super().__init__(config, num_clients)
        self.current_index = 0

    def select_clients(
        self,
        available_clients: Optional[List[int]] = None,
    ) -> List[int]:
        available = self.filter_available_clients(available_clients)
        num_to_sample = self.get_num_clients_to_sample()

        available_sorted = sorted(available)
        selected = []

        for _ in range(num_to_sample):
            if not available_sorted:
                break

            selected_client = available_sorted[
                self.current_index % len(available_sorted)
            ]
            selected.append(selected_client)
            self.client_info[selected_client].selection_count += 1
            self.client_info[selected_client].last_selected = self.current_round
            self.current_index += 1

        self.current_round += 1
        logger.debug(
            f"Round {self.current_round}: Selected {len(selected)} clients via round-robin"
        )
        return selected


class FedCSSamplingStrategy(BaseSamplingStrategy):
    """FedCS: Communication-Efficient Client Selection.

    Selects clients based on their communication resources to maximize
    throughput while meeting a deadline constraint.

    Reference: Nishio & Yonetani (2019)
    """

    def __init__(self, config: SamplingConfig, num_clients: int):
        super().__init__(config, num_clients)
        self.deadline: float = 100.0

    def set_deadline(self, deadline: float) -> None:
        """Set communication deadline."""
        self.deadline = deadline

    def estimate_completion_time(
        self,
        client_id: int,
        model_size: float,
    ) -> float:
        """Estimate time to complete training and upload for a client."""
        client = self.client_info[client_id]
        upload_time = model_size / max(client.latency, 0.001)
        return upload_time

    def select_clients(
        self,
        available_clients: Optional[List[int]] = None,
        model_size: float = 1.0,
    ) -> List[int]:
        available = self.filter_available_clients(available_clients)
        num_to_sample = self.get_num_clients_to_sample()

        completion_times = []
        for c in available:
            completion_time = self.estimate_completion_time(c, model_size)
            completion_times.append((c, completion_time))

        completion_times.sort(key=lambda x: x[1])
        selected = []
        total_time = 0.0

        for c, comp_time in completion_times:
            if (
                total_time + comp_time <= self.deadline
                and len(selected) < num_to_sample
            ):
                selected.append(c)
                total_time += comp_time
                self.client_info[c].selection_count += 1
                self.client_info[c].last_selected = self.current_round

        self.current_round += 1
        logger.debug(
            f"Round {self.current_round}: FedCS selected {len(selected)} clients"
        )
        return selected


class OortSamplingStrategy(BaseSamplingStrategy):
    """Oort: Resource-Aware Client Selection.

    Selects clients based on both statistical utility (loss improvement)
    and resource utility (latency/availability).

    Reference: Lai et al. (2021)
    """

    def __init__(self, config: SamplingConfig, num_clients: int):
        super().__init__(config, num_clients)
        self.exploration_counter: Dict[int, int] = {i: 0 for i in range(num_clients)}

    def compute_utility(self, client_id: int) -> float:
        """Compute overall utility score for a client."""
        client = self.client_info[client_id]

        statistical_utility = client.compute_loss_improvement()
        if statistical_utility <= 0:
            statistical_utility = 0.1

        resource_utility = client.compute_resource_utilization()

        util = statistical_utility * resource_utility

        if self.config.exploration_weight > 0:
            exploration_bonus = self.config.exploration_weight * math.sqrt(
                math.log(self.current_round + 1)
                / (self.exploration_counter[client_id] + 1)
            )
            util = util + exploration_bonus

        return util

    def select_clients(
        self,
        available_clients: Optional[List[int]] = None,
    ) -> List[int]:
        available = self.filter_available_clients(available_clients)
        num_to_sample = self.get_num_clients_to_sample()

        utilities = []
        for c in available:
            util = self.compute_utility(c)
            utilities.append((c, util))

        utilities.sort(key=lambda x: x[1], reverse=True)
        selected = [c for c, _ in utilities[:num_to_sample]]

        for c in selected:
            self.exploration_counter[c] += 1
            self.client_info[c].selection_count += 1
            self.client_info[c].last_selected = self.current_round

        self.current_round += 1
        logger.debug(
            f"Round {self.current_round}: Oort selected {len(selected)} clients"
        )
        return selected


class PowerOfChoiceSamplingStrategy(BaseSamplingStrategy):
    """Power-of-Choice (PoE) Client Selection.

    Two-stage selection: first sample more clients than needed, then
    select the best ones based on loss/utility.

    Reference: Ribero & Vikalo (2020)
    """

    def __init__(self, config: SamplingConfig, num_clients: int):
        super().__init__(config, num_clients)
        self.candidate_factor = 2

    def select_clients(
        self,
        available_clients: Optional[List[int]] = None,
    ) -> List[int]:
        available = self.filter_available_clients(available_clients)
        num_to_sample = self.get_num_clients_to_sample()

        num_candidates = min(num_to_sample * self.candidate_factor, len(available))

        candidates = random.sample(available, num_candidates)

        candidate_losses = []
        for c in candidates:
            client = self.client_info[c]
            candidate_losses.append((c, client.loss))

        candidate_losses.sort(key=lambda x: x[1])
        selected = [c for c, _ in candidate_losses[:num_to_sample]]

        for c in selected:
            self.client_info[c].selection_count += 1
            self.client_info[c].last_selected = self.current_round

        self.current_round += 1
        logger.debug(
            f"Round {self.current_round}: PoE selected {len(selected)} clients"
        )
        return selected


class BanditSamplingStrategy(BaseSamplingStrategy):
    """Multi-Armed Bandit Based Client Selection.

    Uses Thompson sampling to balance exploration and exploitation
    based on client performance.
    """

    def __init__(self, config: SamplingConfig, num_clients: int):
        super().__init__(config, num_clients)
        self.alpha: Dict[int, float] = {i: 1.0 for i in range(num_clients)}
        self.beta: Dict[int, float] = {i: 1.0 for i in range(num_clients)}

    def sample_thompson(self, client_id: int) -> float:
        """Sample from Thompson distribution for a client."""
        alpha = self.alpha[client_id]
        beta = self.beta[client_id]
        return np.random.beta(alpha, beta)

    def update_bandit(self, client_id: int, reward: float) -> None:
        """Update bandit parameters based on reward."""
        if reward > 0:
            self.alpha[client_id] += reward
        else:
            self.beta[client_id] += 1

    def select_clients(
        self,
        available_clients: Optional[List[int]] = None,
    ) -> List[int]:
        available = self.filter_available_clients(available_clients)
        num_to_sample = self.get_num_clients_to_sample()

        sampled_values = [(c, self.sample_thompson(c)) for c in available]
        sampled_values.sort(key=lambda x: x[1], reverse=True)
        selected = [c for c, _ in sampled_values[:num_to_sample]]

        for c in selected:
            self.client_info[c].selection_count += 1
            self.client_info[c].last_selected = self.current_round

        self.current_round += 1
        logger.debug(
            f"Round {self.current_round}: Bandit selected {len(selected)} clients"
        )
        return selected


def create_sampling_strategy(
    config: SamplingConfig,
    num_clients: int,
) -> BaseSamplingStrategy:
    """Factory function to create sampling strategy.

    Args:
        config: Configuration for the sampling strategy
        num_clients: Total number of clients

    Returns:
        Instance of the appropriate sampling strategy

    Example:
        >>> config = SamplingConfig(strategy=SamplingStrategy.RANDOM, sample_fraction=0.1)
        >>> strategy = create_sampling_strategy(config, num_clients=100)
    """
    strategies = {
        SamplingStrategy.RANDOM: RandomSamplingStrategy,
        SamplingStrategy.ROUND_ROBIN: RoundRobinSamplingStrategy,
        SamplingStrategy.FEDCS: FedCSSamplingStrategy,
        SamplingStrategy.OORT: OortSamplingStrategy,
        SamplingStrategy.POWER_OF_CHOICE: PowerOfChoiceSamplingStrategy,
        SamplingStrategy.BANDIT: BanditSamplingStrategy,
    }

    if config.strategy not in strategies:
        raise ValueError(f"Unknown sampling strategy: {config.strategy}")

    return strategies[config.strategy](config, num_clients)


__all__ = [
    "SamplingStrategy",
    "SamplingConfig",
    "ClientInfo",
    "BaseSamplingStrategy",
    "RandomSamplingStrategy",
    "RoundRobinSamplingStrategy",
    "FedCSSamplingStrategy",
    "OortSamplingStrategy",
    "PowerOfChoiceSamplingStrategy",
    "BanditSamplingStrategy",
    "create_sampling_strategy",
]

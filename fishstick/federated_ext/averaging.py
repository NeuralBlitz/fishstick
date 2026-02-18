"""
Federated Averaging Algorithms for fishstick

This module provides various federated averaging algorithms including:
- FedAvg: Standard federated averaging
- FedAvgM: Momentum-based federated averaging
- FedAdam: Adaptive federated optimization
- FedOpt: Optimizer-based aggregation
- Scaffold: Stochastic Controlled Averaging for Federated Learning

References:
- McMahan et al. (2017): "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Hsu et al. (2019): "Measuring the Effects of Non-Identical Data Distribution in Federated Learning"
- Reddi et al. (2021): "Adaptive Federated Optimization"
- Karimireddy et al. (2020): "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class AveragingMethod(Enum):
    """Supported federated averaging methods."""

    FEDAVG = auto()
    FEDAVG_MOMENTUM = auto()
    FEDADAM = auto()
    FEDOPT = auto()
    SCAFFOLD = auto()
    FEDNOVA = auto()
    FEDDYN = auto()


@dataclass
class AveragingConfig:
    """Configuration for federated averaging algorithms."""

    method: AveragingMethod = AveragingMethod.FEDAVG
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    server_momentum: float = 0.0
    proximal_mu: float = 0.0
    use_client_weights: bool = True


class BaseAveraging(ABC):
    """Base class for federated averaging strategies."""

    def __init__(self, config: AveragingConfig):
        self.config = config
        self.server_state: Dict[str, Tensor] = {}
        self.server_momentum: Dict[str, Tensor] = {}
        self.iteration = 0

    @abstractmethod
    def aggregate(
        self,
        client_states: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        """Aggregate client model updates.

        Args:
            client_states: List of client model state dictionaries
            client_weights: Optional weights for each client (e.g., sample counts)

        Returns:
            Aggregated global model state
        """
        pass

    def compute_delta(
        self,
        client_state: Dict[str, Tensor],
        global_state: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute delta between client and global state."""
        return {
            key: client_state[key]
            - global_state.get(key, torch.zeros_like(client_state[key]))
            for key in client_state.keys()
        }

    def apply_delta(
        self,
        global_state: Dict[str, Tensor],
        delta: Dict[str, Tensor],
        lr: float,
    ) -> Dict[str, Tensor]:
        """Apply delta to global state with learning rate."""
        return {key: global_state[key] + lr * delta[key] for key in global_state.keys()}


class FedAvgAveraging(BaseAveraging):
    """Standard Federated Averaging (FedAvg).

    FedAvg aggregates client models by weighted averaging based on
    the number of local training samples.

    Reference: McMahan et al. (2017)
    """

    def aggregate(
        self,
        client_states: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        if not client_states:
            raise ValueError("No client states provided for aggregation")

        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        aggregated: Dict[str, Tensor] = {}
        keys = client_states[0].keys()

        for key in keys:
            weighted_sum = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for client_state, weight in zip(client_states, client_weights):
                weighted_sum += weight * client_state[key].float()
            aggregated[key] = weighted_sum.to(client_states[0][key].dtype)

        self.iteration += 1
        logger.debug(f"FedAvg aggregation completed for iteration {self.iteration}")
        return aggregated


class FedAvgMomentumAveraging(BaseAveraging):
    """FedAvg with Momentum (FedAvgM).

    Applies server-side momentum to accelerate convergence.

    Reference: Hsu et al. (2019)
    """

    def __init__(self, config: AveragingConfig):
        super().__init__(config)
        self.momentum_buffer: Dict[str, Tensor] = {}

    def aggregate(
        self,
        client_states: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        if not client_states:
            raise ValueError("No client states provided for aggregation")

        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        aggregated_delta: Dict[str, Tensor] = {}
        keys = client_states[0].keys()

        for key in keys:
            weighted_sum = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for client_state, weight in zip(client_states, client_weights):
                weighted_sum += weight * client_state[key].float()
            aggregated_delta[key] = weighted_sum.to(client_states[0][key].dtype)

        if self.config.server_momentum > 0:
            for key in keys:
                if key not in self.momentum_buffer:
                    self.momentum_buffer[key] = torch.zeros_like(aggregated_delta[key])

                self.momentum_buffer[key] = (
                    self.config.momentum * self.momentum_buffer[key]
                    + aggregated_delta[key]
                )
                aggregated_delta[key] = self.momentum_buffer[key]

        return aggregated_delta


class FedAdamAveraging(BaseAveraging):
    """Federated Adam (FedAdam).

    Adaptive federated optimization using server-side Adam optimizer.

    Reference: Reddi et al. (2021)
    """

    def __init__(self, config: AveragingConfig):
        super().__init__(config)
        self.m: Dict[str, Tensor] = {}
        self.v: Dict[str, Tensor] = {}
        self.t = 0

    def aggregate(
        self,
        client_states: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        if not client_states:
            raise ValueError("No client states provided for aggregation")

        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        self.t += 1
        keys = client_states[0].keys()

        aggregated_delta: Dict[str, Tensor] = {}
        for key in keys:
            weighted_sum = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for client_state, weight in zip(client_states, client_weights):
                weighted_sum += weight * client_state[key].float()
            aggregated_delta[key] = weighted_sum.to(client_states[0][key].dtype)

        lr_t = self.config.learning_rate * (
            self.config.beta2**self.t / (1 - self.config.beta1**self.t)
        )

        for key in keys:
            if key not in self.m:
                self.m[key] = torch.zeros_like(aggregated_delta[key])
                self.v[key] = torch.zeros_like(aggregated_delta[key])

            self.m[key] = (
                self.config.beta1 * self.m[key]
                + (1 - self.config.beta1) * aggregated_delta[key]
            )
            self.v[key] = self.config.beta2 * self.v[key] + (1 - self.config.beta2) * (
                aggregated_delta[key] ** 2
            )

            m_hat = self.m[key] / (1 - self.config.beta1**self.t)
            v_hat = self.v[key] / (1 - self.config.beta2**self.t)

            aggregated_delta[key] = (
                lr_t * m_hat / (torch.sqrt(v_hat) + self.config.epsilon)
            )

        return aggregated_delta


class FedOptAveraging(BaseAveraging):
    """Federated Optimizer (FedOpt).

    General optimizer-based federated learning that applies any optimizer
    to aggregate client updates.

    Reference: Reddi et al. (2021)
    """

    def __init__(self, config: AveragingConfig):
        super().__init__(config)
        self.m: Dict[str, Tensor] = {}
        self.v: Dict[str, Tensor] = {}
        self.t = 0

    def aggregate(
        self,
        client_states: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        if not client_states:
            raise ValueError("No client states provided for aggregation")

        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        self.t += 1
        keys = client_states[0].keys()

        aggregated_delta: Dict[str, Tensor] = {}
        for key in keys:
            weighted_sum = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for client_state, weight in zip(client_states, client_weights):
                weighted_sum += weight * client_state[key].float()
            aggregated_delta[key] = weighted_sum.to(client_states[0][key].dtype)

        return aggregated_delta


class ScaffoldAveraging(BaseAveraging):
    """SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.

    SCAFFOLD corrects for client drift using control variates.

    Reference: Karimireddy et al. (2020)
    """

    def __init__(self, config: AveragingConfig):
        super().__init__(config)
        self.server_controls: Dict[str, Tensor] = {}
        self.client_controls: List[Dict[str, Tensor]] = []

    def initialize_controls(self, model_state: Dict[str, Tensor]) -> None:
        """Initialize server control variates."""
        for key in model_state.keys():
            self.server_controls[key] = torch.zeros_like(model_state[key])

    def aggregate(
        self,
        client_states: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        if not client_states:
            raise ValueError("No client states provided for aggregation")

        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        keys = client_states[0].keys()

        aggregated_delta: Dict[str, Tensor] = {}
        for key in keys:
            weighted_sum = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for client_state, weight in zip(client_states, client_weights):
                weighted_sum += weight * client_state[key].float()
            aggregated_delta[key] = weighted_sum.to(client_states[0][key].dtype)

        if self.server_controls:
            for key in keys:
                delta_control = torch.zeros_like(aggregated_delta[key])
                for client_weight, client_control in zip(
                    client_weights, self.client_controls
                ):
                    if key in client_control:
                        delta_control += client_weight * client_control[key]

                aggregated_delta[key] = (
                    aggregated_delta[key] - self.server_controls[key] + delta_control
                )

        self.iteration += 1
        return aggregated_delta

    def update_client_controls(
        self,
        client_state: Dict[str, Tensor],
        global_state: Dict[str, Tensor],
        client_id: int,
    ) -> Dict[str, Tensor]:
        """Update client control variates after local training."""
        if len(self.client_controls) <= client_id:
            for key in global_state.keys():
                self.client_controls.append(
                    {
                        key: torch.zeros_like(global_state[key])
                        for key in global_state.keys()
                    }
                )

        client_delta = self.compute_delta(client_state, global_state)
        new_controls: Dict[str, Tensor] = {}

        for key in global_state.keys():
            if key in self.server_controls:
                new_controls[key] = (
                    self.client_controls[client_id].get(
                        key, torch.zeros_like(global_state[key])
                    )
                    - client_delta.get(key, torch.zeros_like(global_state[key]))
                    + self.server_controls[key]
                )
            else:
                new_controls[key] = self.client_controls[client_id].get(
                    key, torch.zeros_like(global_state[key])
                )

        if len(self.client_controls) > client_id:
            self.client_controls[client_id] = new_controls
        else:
            self.client_controls.append(new_controls)

        return new_controls

    def update_server_controls(
        self,
        client_states: List[Dict[str, Tensor]],
        global_state: Dict[str, Tensor],
        client_weights: Optional[List[float]] = None,
    ) -> None:
        """Update server control variates after aggregation."""
        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        keys = global_state.keys()

        for key in keys:
            aggregated_delta = torch.zeros_like(global_state[key], dtype=torch.float32)
            for client_state, weight in zip(client_states, client_weights):
                if key in client_state:
                    delta = client_state[key] - global_state.get(
                        key, torch.zeros_like(client_state[key])
                    )
                    aggregated_delta += weight * delta.float()

            new_server_control = torch.zeros_like(global_state[key])
            for client_weight, client_control in zip(
                client_weights, self.client_controls
            ):
                if key in client_control:
                    new_server_control += client_weight * client_control[key]

            self.server_controls[key] = (
                new_server_control
                - self.config.learning_rate * aggregated_delta.float()
            )


class FedNovaAveraging(BaseAveraging):
    """FedNova: Federated Normalized Averaging.

    Normalizes client updates to account for varying numbers of local steps.

    Reference: Wang et al. (2021)
    """

    def __init__(self, config: AveragingConfig):
        super().__init__(config)
        self.total_steps: Dict[int, int] = {}

    def aggregate(
        self,
        client_states: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
        client_steps: Optional[List[int]] = None,
    ) -> Dict[str, Tensor]:
        if not client_states:
            raise ValueError("No client states provided for aggregation")

        if client_steps is None:
            client_steps = [1] * len(client_states)

        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        total_steps = sum(client_steps)
        keys = client_states[0].keys()

        aggregated: Dict[str, Tensor] = {}
        for key in keys:
            weighted_sum = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for client_state, weight, steps in zip(
                client_states, client_weights, client_steps
            ):
                weighted_sum += weight * steps * client_state[key].float() / total_steps
            aggregated[key] = weighted_sum.to(client_states[0][key].dtype)

        self.iteration += 1
        logger.debug(f"FedNova aggregation completed for iteration {self.iteration}")
        return aggregated


class FedDynAveraging(BaseAveraging):
    """FedDyn: Federated Learning with Dynamic Regularization.

    Uses dynamic regularization to handle heterogeneity.

    Reference: Acedanski et al. (2021)
    """

    def __init__(self, config: AveragingConfig):
        super().__init__(config)
        self.linear_term: Dict[str, Tensor] = {}

    def initialize_linear_term(self, model_state: Dict[str, Tensor]) -> None:
        """Initialize linear term for dynamic regularization."""
        for key in model_state.keys():
            self.linear_term[key] = torch.zeros_like(model_state[key])

    def aggregate(
        self,
        client_states: List[Dict[str, Tensor]],
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, Tensor]:
        if not client_states:
            raise ValueError("No client states provided for aggregation")

        if client_weights is None:
            client_weights = [1.0 / len(client_states)] * len(client_states)
        else:
            total = sum(client_weights)
            client_weights = [w / total for w in client_weights]

        keys = client_states[0].keys()

        aggregated: Dict[str, Tensor] = {}
        for key in keys:
            weighted_sum = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for client_state, weight in zip(client_states, client_weights):
                weighted_sum += weight * client_state[key].float()

            if key in self.linear_term:
                weighted_sum += self.config.proximal_mu * self.linear_term[key]

            aggregated[key] = weighted_sum.to(client_states[0][key].dtype)

        return aggregated


def create_averaging_strategy(
    config: AveragingConfig,
) -> BaseAveraging:
    """Factory function to create averaging strategy.

    Args:
        config: Configuration for the averaging strategy

    Returns:
        Instance of the appropriate averaging strategy

    Example:
        >>> config = AveragingConfig(method=AveragingMethod.FEDAVG, learning_rate=0.01)
        >>> strategy = create_averaging_strategy(config)
    """
    strategies = {
        AveragingMethod.FEDAVG: FedAvgAveraging,
        AveragingMethod.FEDAVG_MOMENTUM: FedAvgMomentumAveraging,
        AveragingMethod.FEDADAM: FedAdamAveraging,
        AveragingMethod.FEDOPT: FedOptAveraging,
        AveragingMethod.SCAFFOLD: ScaffoldAveraging,
        AveragingMethod.FEDNOVA: FedNovaAveraging,
        AveragingMethod.FEDDYN: FedDynAveraging,
    }

    if config.method not in strategies:
        raise ValueError(f"Unknown averaging method: {config.method}")

    return strategies[config.method](config)


__all__ = [
    "AveragingMethod",
    "AveragingConfig",
    "BaseAveraging",
    "FedAvgAveraging",
    "FedAvgMomentumAveraging",
    "FedAdamAveraging",
    "FedOptAveraging",
    "ScaffoldAveraging",
    "FedNovaAveraging",
    "FedDynAveraging",
    "create_averaging_strategy",
]

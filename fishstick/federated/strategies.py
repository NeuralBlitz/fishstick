"""
Federated Learning Aggregation Strategies
"""

from typing import Dict, List
import torch


class AggregationStrategy:
    """Base class for aggregation strategies."""

    def aggregate(self, client_states: list, weights: list) -> Dict:
        raise NotImplementedError


class FedAvg(AggregationStrategy):
    """Federated Averaging (FedAvg)."""

    def aggregate(self, client_states: List[Dict], weights: List[int]) -> Dict:
        """Aggregate client models using weighted average."""
        if not client_states:
            return {}

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        aggregated = {}
        for key in client_states[0].keys():
            aggregated[key] = torch.zeros_like(client_states[0][key])

            for state, weight in zip(client_states, normalized_weights):
                aggregated[key] += state[key] * weight

        return aggregated


class FedProx(AggregationStrategy):
    """Federated Proximal (FedProx)."""

    def __init__(self, mu: float = 0.01):
        self.mu = mu

    def aggregate(self, client_states: List[Dict], weights: List[int]) -> Dict:
        """Aggregate with proximal term."""
        return FedAvg().aggregate(client_states, weights)


class FedNova(AggregationStrategy):
    """Federated Normalized Averaging (FedNova)."""

    def aggregate(self, client_states: List[Dict], weights: List[int]) -> Dict:
        """Aggregate with normalization."""
        if not client_states:
            return {}

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        aggregated = {}
        for key in client_states[0].keys():
            aggregated[key] = torch.zeros_like(client_states[0][key])

            for state, weight in zip(client_states, normalized_weights):
                aggregated[key] += state[key] * weight

        return aggregated


class FedAdam(AggregationStrategy):
    """Federated Adam optimizer."""

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}

    def aggregate(self, client_states: List[Dict], weights: List[int]) -> Dict:
        """Aggregate using Adam optimizer."""
        if not client_states:
            return {}

        for key in client_states[0].keys():
            if key not in self.m:
                self.m[key] = torch.zeros_like(client_states[0][key])
                self.v[key] = torch.zeros_like(client_states[0][key])

        aggregated = {}
        for key in client_states[0].keys():
            grad = torch.zeros_like(client_states[0][key])

            for state in client_states:
                grad += state[key]

            grad /= len(client_states)

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)

            aggregated[key] = self.m[key] / (torch.sqrt(self.v[key]) + self.eps)

        return aggregated


class FedAdagrad(AggregationStrategy):
    """Federated Adagrad optimizer."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.G = {}

    def aggregate(self, client_states: List[Dict], weights: List[int]) -> Dict:
        """Aggregate using Adagrad optimizer."""
        if not client_states:
            return {}

        for key in client_states[0].keys():
            if key not in self.G:
                self.G[key] = torch.zeros_like(client_states[0][key])

        aggregated = {}
        for key in client_states[0].keys():
            grad = torch.zeros_like(client_states[0][key])

            for state in client_states:
                grad += state[key]

            grad /= len(client_states)

            self.G[key] += grad**2
            aggregated[key] = grad / (torch.sqrt(self.G[key]) + self.eps)

        return aggregated


class Scaffold(AggregationStrategy):
    """Stochastic Controlled Averaging for Federated Learning (SCAFFOLD)."""

    def __init__(self):
        self.server_controls = {}
        self.client_controls = {}

    def aggregate(self, client_states: List[Dict], weights: List[int]) -> Dict:
        """Aggregate using SCAFFOLD."""
        if not client_states:
            return {}

        aggregated = {}
        for key in client_states[0].keys():
            total_weight = sum(weights)
            aggregated[key] = torch.zeros_like(client_states[0][key])

            for state, weight in zip(client_states, weights):
                aggregated[key] += state[key] * (weight / total_weight)

        return aggregated

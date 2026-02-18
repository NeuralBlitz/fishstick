import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum


class AggregationStrategy(Enum):
    FEDAVG = "fedavg"
    FEDSCALE = "fedscale"
    FEDPROX = "fedprox"
    FEDNOVA = "fednova"


class BaseFederatedServer(ABC):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_clients: int,
        sample_fraction: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_clients = num_clients
        self.sample_fraction = sample_fraction
        self.current_round = 0
        self.client_states: Dict[int, Dict[str, torch.Tensor]] = {}

    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pass

    def get_global_model(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def set_global_model(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)

    def select_clients(self) -> List[int]:
        num_to_sample = max(1, int(self.num_clients * self.sample_fraction))
        selected = torch.randperm(self.num_clients)[:num_to_sample].tolist()
        return selected


class FedAvgServer(BaseFederatedServer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_clients: int,
        sample_fraction: float = 1.0,
    ):
        super().__init__(model, device, num_clients, sample_fraction)

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return self.get_global_model()

        total_samples = sum(update["num_samples"] for update in client_updates)
        aggregated_params: Dict[str, torch.Tensor] = {}

        param_names = client_updates[0]["parameters"].keys()
        for name in param_names:
            weighted_sum = None
            for update in client_updates:
                weight = update["num_samples"] / total_samples
                params = update["parameters"][name]
                if isinstance(params, torch.Tensor):
                    params = params.to(self.device)

                if weighted_sum is None:
                    weighted_sum = weight * params
                else:
                    weighted_sum += weight * params

            aggregated_params[name] = weighted_sum.cpu() if weighted_sum is not None else None

        self.set_global_model(aggregated_params)
        self.current_round += 1

        return aggregated_params


class FedScaleServer(BaseFederatedServer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_clients: int,
        sample_fraction: float = 1.0,
        alpha: float = 1.0,
        beta: float = 0.5,
    ):
        super().__init__(model, device, num_clients, sample_fraction)
        self.alpha = alpha
        self.beta = beta
        self.client_weights: Dict[int, float] = {}

    def _compute_client_weight(self, num_samples: int, latency: float) -> float:
        latency_penalty = self.beta / (latency + 1e-6)
        return (num_samples ** self.alpha) * latency_penalty

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return self.get_global_model()

        for update in client_updates:
            client_id = update["client_id"]
            num_samples = update["num_samples"]
            latency = update.get("latency", 1.0)
            self.client_weights[client_id] = self._compute_client_weight(num_samples, latency)

        total_weight = sum(self.client_weights.values())
        
        aggregated_params: Dict[str, torch.Tensor] = {}
        param_names = client_updates[0]["parameters"].keys()

        for name in param_names:
            weighted_sum = None
            for update in client_updates:
                client_id = update["client_id"]
                weight = self.client_weights[client_id] / total_weight
                params = update["parameters"][name]
                if isinstance(params, torch.Tensor):
                    params = params.to(self.device)

                if weighted_sum is None:
                    weighted_sum = weight * params
                else:
                    weighted_sum += weight * params

            aggregated_params[name] = weighted_sum.cpu() if weighted_sum is not None else None

        self.set_global_model(aggregated_params)
        self.current_round += 1

        return aggregated_params


class AggregationHandler:
    @staticmethod
    def fednova_aggregate(
        client_updates: List[Dict[str, Any]],
        global_model: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return global_model

        total_a_u = sum(update["a_u"].item() for update in client_updates if "a_u" in update)
        
        if total_a_u == 0:
            total_a_u = sum(update["local_steps"] for update in client_updates)

        aggregated_params: Dict[str, torch.Tensor] = {}
        param_names = client_updates[0]["parameters"].keys()

        for name in param_names:
            normalized_sum = None
            for update in client_updates:
                a_u = update.get("a_u", torch.tensor(update["local_steps"])).item()
                tau_eff = a_u / (total_a_u + 1e-6)
                
                params = update["parameters"][name]
                if isinstance(params, torch.Tensor):
                    params = params

                if normalized_sum is None:
                    normalized_sum = tau_eff * params
                else:
                    normalized_sum += tau_eff * params

            aggregated_params[name] = normalized_sum

        return aggregated_params

    @staticmethod    def fedprox_aggregate(
        client_updates: List[Dict[str, Any]],
        global_model: Dict[str, torch.Tensor],
        mu: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        if not client_updates:
            return global_model

        total_samples = sum(update["num_samples"] for update in client_updates)
        aggregated_params: Dict[str, torch.Tensor] = {}

        param_names = client_updates[0]["parameters"].keys()
        for name in param_names:
            weighted_sum = None
            for update in client_updates:
                weight = update["num_samples"] / total_samples
                params = update["parameters"][name]

                if weighted_sum is None:
                    weighted_sum = weight * params
                else:
                    weighted_sum += weight * params

            aggregated_params[name] = weighted_sum

        return aggregated_params

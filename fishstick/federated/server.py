"""
Federated Learning Server
"""

from typing import Dict, Any, Optional, Callable
import torch
from torch import nn
from fishstick.federated.clients import FederatedClient, ClientManager
from fishstick.federated.strategies import FedAvg, FedProx, FedNova


class AggregationStrategy:
    """Base class for aggregation strategies."""

    def aggregate(self, client_states: list, weights: list) -> Dict:
        raise NotImplementedError


class FederatedServer:
    """Federated learning server."""

    def __init__(
        self,
        model: nn.Module,
        client_manager: ClientManager,
        strategy: Optional[AggregationStrategy] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.client_manager = client_manager
        self.strategy = strategy or FedAvg()
        self.device = device
        self.current_round = 0

    def get_global_model_state(self) -> Dict:
        """Get global model parameters."""
        return self.model.state_dict()

    def set_global_model_state(self, state_dict: Dict) -> None:
        """Set global model parameters."""
        self.model.load_state_dict(state_dict)

    def broadcast_model(self, selected_clients: list) -> None:
        """Broadcast global model to selected clients."""
        global_state = self.get_global_model_state()
        for client in selected_clients:
            client.set_model_state(global_state)

    def train_round(
        self,
        num_clients: int = 10,
        local_epochs: int = 1,
        strategy: str = "random",
    ) -> Dict[str, Any]:
        """Execute one federated training round."""
        self.current_round += 1

        selected_clients = self.client_manager.select_clients(
            num_clients, strategy=strategy
        )

        self.broadcast_model(selected_clients)

        client_results = []
        for client in selected_clients:
            result = client.train(epochs=local_epochs)
            client_results.append(result)

        aggregated_state = self.strategy.aggregate(
            [c.get_model_state() for c in selected_clients],
            [c.train_loader.dataset.__len__() for c in selected_clients],
        )

        self.set_global_model_state(aggregated_state)

        avg_loss = sum(r["loss"] for r in client_results) / len(client_results)

        return {
            "round": self.current_round,
            "num_clients": len(selected_clients),
            "avg_loss": avg_loss,
            "client_results": client_results,
        }

    def evaluate(
        self,
        test_loaders: list,
    ) -> Dict[str, float]:
        """Evaluate global model on client test sets."""
        self.model.eval()

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for test_loader in test_loaders:
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    total_correct += (pred == target).sum().item()
                    total_samples += target.size(0)

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            "accuracy": accuracy,
            "total_samples": total_samples,
        }

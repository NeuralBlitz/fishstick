"""
Federated Learning Clients
"""

from typing import Dict, Any, Optional, Callable
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


class FederatedClient:
    """Federated learning client."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optimizer or torch.optim.Adam(model.parameters())

    def get_model_state(self) -> Dict:
        """Get model parameters."""
        return self.model.state_dict()

    def set_model_state(self, state_dict: Dict) -> None:
        """Set model parameters."""
        self.model.load_state_dict(state_dict)

    def train(
        self,
        epochs: int = 1,
        criterion: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Train on local data."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "client_id": self.client_id,
            "loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "num_samples": len(self.train_loader.dataset),
        }

    def evaluate(
        self,
        test_loader: DataLoader,
        criterion: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Evaluate on local test data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return {
            "client_id": self.client_id,
            "loss": total_loss / len(test_loader),
            "accuracy": correct / total if total > 0 else 0.0,
        }


class ClientManager:
    """Manage federated learning clients."""

    def __init__(self, clients: Optional[list] = None):
        self.clients = clients or []

    def add_client(self, client: FederatedClient) -> None:
        """Add a client."""
        self.clients.append(client)

    def get_client(self, client_id: int) -> Optional[FederatedClient]:
        """Get client by ID."""
        for client in self.clients:
            if client.client_id == client_id:
                return client
        return None

    def select_clients(
        self,
        num_clients: int,
        strategy: str = "random",
    ) -> list:
        """Select clients for training."""
        if strategy == "random":
            import random

            return random.sample(self.clients, min(num_clients, len(self.clients)))
        elif strategy == "all":
            return self.clients
        return self.clients[:num_clients]

    def aggregate_models(
        self,
        selected_clients: list,
        aggregation_strategy: str = "fedavg",
    ) -> Dict:
        """Aggregate models from selected clients."""
        if not selected_clients:
            return {}

        if aggregation_strategy == "fedavg":
            total_samples = sum(
                c.train_loader.dataset.__len__() for c in selected_clients
            )
            aggregated_state = {}

            for key in selected_clients[0].get_model_state().keys():
                aggregated_state[key] = torch.zeros_like(
                    selected_clients[0].get_model_state()[key]
                )

                for client in selected_clients:
                    weight = client.train_loader.dataset.__len__() / total_samples
                    aggregated_state[key] += client.get_model_state()[key] * weight

            return aggregated_state

        return selected_clients[0].get_model_state()

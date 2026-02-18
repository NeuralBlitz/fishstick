import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class BaseFederatedClient(ABC):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        client_id: int,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
    ):
        self.model = model.to(device)
        self.device = device
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_loader = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    @abstractmethod
    def set_parameters(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        pass

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        pass

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def get_num_samples(self) -> int:
        if self.train_loader is not None:
            return len(self.train_loader.dataset)
        return 0


class FedAvgClient(BaseFederatedClient):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        client_id: int,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ):
        super().__init__(
            model, device, client_id, local_epochs, batch_size, learning_rate
        )
        self.momentum = momentum
        self.weight_decay = weight_decay

    def set_parameters(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(global_state_dict)

    def train(self) -> Dict[str, Any]:
        self.model.train()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            for batch_data, batch_labels in self.train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "client_id": self.client_id,
            "num_samples": self.get_num_samples(),
            "avg_loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "parameters": self.get_parameters(),
        }


class FedProxClient(BaseFederatedClient):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        client_id: int,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        mu: float = 0.01,
    ):
        super().__init__(
            model, device, client_id, local_epochs, batch_size, learning_rate
        )
        self.mu = mu
        self.global_parameters: Optional[Dict[str, torch.Tensor]] = None

    def set_parameters(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        self.global_parameters = {
            k: v.cpu().clone() for k, v in global_state_dict.items()
        }
        self.model.load_state_dict(global_state_dict)

    def _prox_term(self) -> torch.Tensor:
        if self.global_parameters is None:
            return torch.tensor(0.0, device=self.device)

        prox_loss = torch.tensor(0.0, device=self.device)
        for (name, param), (global_name, global_param) in zip(
            self.model.named_parameters(), self.global_parameters.items()
        ):
            if param.shape == global_param.shape:
                prox_loss += torch.sum((param - global_param.to(self.device)) ** 2)

        return (self.mu / 2) * prox_loss

    def train(self) -> Dict[str, Any]:
        self.model.train()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            for batch_data, batch_labels in self.train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                ce_loss = self.criterion(outputs, batch_labels)
                prox_loss = self._prox_term()
                loss = ce_loss + prox_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return {
            "client_id": self.client_id,
            "num_samples": self.get_num_samples(),
            "avg_loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "parameters": self.get_parameters(),
        }


class FedNovaClient(BaseFederatedClient):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        client_id: int,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ):
        super().__init__(
            model, device, client_id, local_epochs, batch_size, learning_rate
        )
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.a_u: Optional[torch.Tensor] = None

    def set_parameters(self, global_state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(global_state_dict)

    def _compute_a_u(self) -> torch.Tensor:
        if self.train_loader is None:
            return torch.tensor(0.0)

        total_steps = len(self.train_loader) * self.local_epochs
        a_u = torch.tensor(total_steps, dtype=torch.float32)
        return a_u

    def train(self) -> Dict[str, Any]:
        self.model.train()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            for batch_data, batch_labels in self.train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        self.a_u = self._compute_a_u()

        return {
            "client_id": self.client_id,
            "num_samples": self.get_num_samples(),
            "avg_loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "parameters": self.get_parameters(),
            "a_u": self.a_u,
            "local_steps": num_batches,
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Callable, Any, Tuple
from torch.utils.data import DataLoader
from copy import deepcopy


class DartsArchitect:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        unrolled: bool = False,
    ):
        self.model = model
        self.criterion = criterion
        self.unrolled = unrolled
        self.optimizer = optim.Adam(model.parameters(), lr=3e-4)

    def step(
        self,
        input_train: torch.Tensor,
        target_train: torch.Tensor,
        input_valid: torch.Tensor,
        target_valid: torch.Tensor,
        eta: float = 0.01,
    ) -> float:
        self.optimizer.zero_grad()

        if self.unrolled:
            loss = self._unrolled_step(
                input_train, target_train, input_valid, target_valid
            )
        else:
            loss = self._backward_step(input_valid, target_valid)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _backward_step(
        self,
        input_valid: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.model(input_valid)
        return self.criterion(pred, target_valid)

    def _unrolled_step(
        self,
        input_train: torch.Tensor,
        target_train: torch.Tensor,
        input_valid: torch.Tensor,
        target_valid: torch.Tensor,
        eta: float = 0.01,
    ) -> torch.Tensor:
        optimizer = optim.SGD(self.model.parameters(), eta)
        self.model.train()

        for _ in range(2):
            pred = self.model(input_train)
            loss = self.criterion(pred, target_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        pred = self.model(input_valid)
        loss = self.criterion(pred, target_valid)
        return loss


class ProxylessNASTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: str = "cuda",
    ):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.model.to(device)

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def finetune(
        self,
        train_loader: DataLoader,
        num_epochs: int = 50,
        lr: float = 0.05,
    ) -> nn.Module:
        optimizer = optim.SGD(
            self.model.parameters(), lr, momentum=0.9, weight_decay=3e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            loss = self.train_epoch(train_loader, optimizer, epoch)
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        return self.model


class OnceForAllSupernet(nn.Module):
    def __init__(self, base_model: nn.Module, num_choices: int = 5):
        super().__init__()
        self.base_model = base_model
        self.num_choices = num_choices
        self.depth_choices = [1, 2, 3, 4]
        self.width_choices = [0.25, 0.5, 0.75, 1.0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    def sample_subnet(self) -> Dict[str, Any]:
        return {
            "depth": torch.randint(0, len(self.depth_choices), (1,)).item(),
            "width": torch.randint(0, len(self.width_choices), (1,)).item(),
        }


class OnceForAllTrainer:
    def __init__(
        self,
        supernet: OnceForAllSupernet,
        criterion: nn.Module,
        device: str = "cuda",
    ):
        self.supernet = supernet
        self.criterion = criterion
        self.device = device
        self.supernet.to(device)

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        subnet: Dict[str, Any],
        optimizer: optim.Optimizer,
    ) -> float:
        optimizer.zero_grad()

        self.supernet.train()
        outputs = self.supernet(inputs)

        loss = self.criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 50,
        lr: float = 0.01,
    ) -> nn.Module:
        optimizer = optim.Adam(self.supernet.parameters(), lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            self.supernet.train()
            total_loss = 0.0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                subnet = self.supernet.sample_subnet()
                loss = self.train_step(inputs, targets, subnet, optimizer)
                total_loss += loss

            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
                )

        return self.supernet

    def extract_subnet(self, config: Dict[str, Any]) -> nn.Module:
        return deepcopy(self.supernet.base_model)


class SupernetTrainer:
    def __init__(
        self,
        supernet: nn.Module,
        criterion: nn.Module,
        device: str = "cuda",
    ):
        self.supernet = supernet
        self.criterion = criterion
        self.device = device
        self.supernet.to(device)

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> float:
        optimizer.zero_grad()
        outputs = self.supernet(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        lr: float = 0.01,
    ) -> nn.Module:
        optimizer = optim.Adam(self.supernet.parameters(), lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            self.supernet.train()
            total_loss = 0.0

            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                loss = self.train_step(inputs, targets, optimizer)
                total_loss += loss

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}"
                )

        return self.supernet

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.supernet.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.supernet(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return total_loss / len(val_loader), 100.0 * correct / total


class GradientBasedSearch:
    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.model = model
        self.criterion = criterion

    def compute_gradients(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()

        gradients = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                gradients[name] = param.grad.clone()

        return gradients

    def rank_operations(self, gradients: Dict[str, torch.Tensor]) -> List[str]:
        ranked = sorted(
            gradients.items(),
            key=lambda x: torch.norm(x[1]).item() if x[1] is not None else 0,
            reverse=True,
        )
        return [name for name, _ in ranked]

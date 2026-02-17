"""
Training Loop

High-level training loop with support for various training strategies.
"""

from typing import Any, Callable, Dict, Optional
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm


class Trainer:
    """Main trainer class with flexible configuration."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.current_epoch = 0
        self.global_step = 0
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    def train_epoch(
        self, train_loader: DataLoader, callbacks: Optional[list] = None
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)
            loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            pbar.set_postfix(
                {"loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}"}
            )

        self.current_epoch += 1
        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        callbacks: Optional[list] = None,
    ) -> Dict[str, list]:
        """Full training loop."""
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, callbacks)
            self.history["train_loss"].append(train_loss)

            if val_loader:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        return self.history


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    optimizer: Optional[Optimizer] = None,
    criterion: Optional[Callable] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 10,
    lr: float = 1e-3,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for quick training.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: Optimizer (default: Adam)
        criterion: Loss function (default: CrossEntropyLoss)
        device: Device to train on
        epochs: Number of epochs
        lr: Learning rate
        **kwargs: Additional trainer arguments

    Returns:
        Training history dictionary
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        **kwargs,
    )

    return trainer.fit(train_loader, val_loader, epochs)


class DistributedTrainer:
    """Distributed training support."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable,
        device: str = "cuda",
        world_size: int = 1,
        rank: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.world_size = world_size
        self.rank = rank

        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(self.rank), target.to(self.rank)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)


class AMPTrainer:
    """Automatic Mixed Precision trainer."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)

            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.criterion(output, target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / len(train_loader)


class EnsembleTrainer:
    """Train multiple models as an ensemble."""

    def __init__(
        self,
        models: list[nn.Module],
        optimizer_fn: Callable[[nn.Module], Optimizer],
        criterion: Callable,
        device: str = "cuda",
    ):
        self.models = [m.to(device) for m in models]
        self.optimizers = [optimizer_fn(m) for m in models]
        self.criterion = criterion
        self.device = device

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        for model in self.models:
            model.train()

        total_losses = {f"model_{i}": 0.0 for i in range(len(self.models))}

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)

            for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                total_losses[f"model_{i}"] += loss.item()

        return {k: v / len(train_loader) for k, v in total_losses.items()}

    def predict(self, x: Tensor) -> Tensor:
        """Ensemble prediction by averaging."""
        self.model.eval()
        with torch.no_grad():
            outputs = [model(x) for model in self.models]
            return torch.stack(outputs).mean(dim=0)

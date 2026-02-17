"""
Training Callbacks

Event-driven callback system for training loops.
"""

from typing import Any, Dict, Optional, Callable
from abc import ABC, abstractmethod
import time
import torch
from torch import Tensor
from pathlib import Path


class Callback(ABC):
    """Base callback class."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, trainer: "Trainer", batch: int) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(
        self, trainer: "Trainer", batch: int, loss: Tensor, outputs: Any
    ) -> None:
        """Called at the end of each batch."""
        pass

    def on_validation_begin(self, trainer: "Trainer") -> None:
        """Called before validation."""
        pass

    def on_validation_end(self, trainer: "Trainer", metrics: Dict[str, float]) -> None:
        """Called after validation."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.best_value: Optional[float] = None
        self.counter = 0
        self.should_stop = False
        self.best_state: Optional[Dict] = None

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        current = metrics.get(self.monitor)
        if current is None:
            return

        if self.best_value is None:
            self.best_value = current
            if self.restore_best:
                self.best_state = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
            return

        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta

        if improved:
            self.best_value = current
            self.counter = 0
            if self.restore_best:
                self.best_state = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and self.best_state:
                    trainer.model.load_state_dict(self.best_state)
                    print(
                        f"Early stopping: Restored best model from epoch {epoch - self.patience}"
                    )


class ModelCheckpoint(Callback):
    """Save model checkpoints."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.best_value: Optional[float] = None

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        if self.save_last:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": trainer.model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "metrics": metrics,
                },
                self.filepath.parent / "last.pt",
            )

        if not self.save_best_only:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": trainer.model.state_dict(),
                    "metrics": metrics,
                },
                self.filepath.parent / f"checkpoint_epoch_{epoch}.pt",
            )
            return

        current = metrics.get(self.monitor)
        if current is None:
            return

        if self.best_value is None:
            self.best_value = current
            self._save_checkpoint(trainer, epoch, "best.pt")
            return

        if self.mode == "min":
            improved = current < self.best_value
        else:
            improved = current > self.best_value

        if improved:
            self.best_value = current
            self._save_checkpoint(trainer, epoch, "best.pt")

    def _save_checkpoint(self, trainer: "Trainer", epoch: int, filename: str) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
            },
            self.filepath.parent / filename,
        )


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback."""

    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        self.scheduler = scheduler

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        if hasattr(self.scheduler, "step"):
            if hasattr(self.scheduler, "metric"):
                self.scheduler.step(metrics.get(self.scheduler.monitor))
            else:
                self.scheduler.step()


class GradientClipping(Callback):
    """Gradient clipping callback."""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_batch_end(
        self, trainer: "Trainer", batch: int, loss: Tensor, outputs: Any
    ) -> None:
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), self.max_norm, self.norm_type
        )


class ProgressBar(Callback):
    """Progress bar callback."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        print(f"Epoch {epoch + 1}/{self.total_epochs}")

    def on_batch_end(
        self, trainer: "Trainer", batch: int, loss: Tensor, outputs: Any
    ) -> None:
        if batch % 10 == 0:
            print(f"  Batch {batch}: Loss = {loss.item():.4f}")


class TensorBoardCallback(Callback):
    """TensorBoard logging callback."""

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        except ImportError:
            print("TensorBoard not available. Install tensorboard.")

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self.writer:
            self.writer.close()


class WandBCallback(Callback):
    """Weights & Biases logging callback."""

    def __init__(self, project: str = "fishstick", name: Optional[str] = None):
        self.project = project
        self.name = name
        self.run = None

        try:
            import wandb

            wandb.init(project=project, name=name)
            self.run = wandb
        except ImportError:
            print("WandB not available. Install wandb.")

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        if self.run:
            self.run.log(metrics, step=epoch)

    def on_train_end(self, trainer: "Trainer") -> None:
        if self.run:
            self.run.finish()


class MetricsLogger(Callback):
    """Metrics logger callback."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file) if log_file else None
        self.history: Dict[str, list] = {}

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"Epoch {epoch}: {metrics}\n")

    def get_history(self, metric: str) -> list:
        return self.history.get(metric, [])


class ParameterMonitor(Callback):
    """Monitor model parameters during training."""

    def __init__(self, log_norms: bool = True, log_weights: bool = False):
        self.log_norms = log_norms
        self.log_weights = log_weights

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        if not (self.log_norms or self.log_weights):
            return

        for name, param in trainer.model.named_parameters():
            if self.log_norms and param.grad is not None:
                grad_norm = param.grad.norm().item()
                metrics[f"grad_norm/{name}"] = grad_norm

            if self.log_weights:
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                metrics[f"weight_mean/{name}"] = param_mean
                metrics[f"weight_std/{name}"] = param_std


class EpochTimer(Callback):
    """Time each epoch."""

    def __init__(self):
        self.epoch_times = []
        self.epoch_start_time = 0.0

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        self.epoch_start_time = time.time()

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, metrics: Dict[str, float]
    ) -> None:
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        metrics["epoch_time"] = epoch_time

        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining = avg_time * (trainer.max_epochs - epoch - 1)
        print(
            f"  Epoch time: {epoch_time:.2f}s | Avg: {avg_time:.2f}s | ETA: {remaining:.2f}s"
        )


class Trainer:
    """Simple trainer class for integration with callbacks."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: str = "cuda",
        max_epochs: int = 100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.max_epochs = max_epochs
        self.callbacks: list[Callback] = []

    def add_callback(self, callback: Callback) -> "Trainer":
        self.callbacks.append(callback)
        return self

    def _call_event(self, event: str, **kwargs) -> None:
        for callback in self.callbacks:
            getattr(callback, event)(self, **kwargs)

    def fit(self, train_loader, val_loader=None):
        self._call_event("on_train_begin")

        for epoch in range(self.max_epochs):
            self._call_event("on_epoch_begin", epoch=epoch)

            self.model.train()
            train_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                self._call_event("on_batch_begin", batch=batch_idx)

                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                self._call_event(
                    "on_batch_end", batch=batch_idx, loss=loss, outputs=output
                )

                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            metrics = {"train_loss": train_loss}

            if val_loader:
                self._call_event("on_validation_begin")
                val_loss = self.validate(val_loader)
                metrics["val_loss"] = val_loss
                self._call_event("on_validation_end", metrics=metrics)

            self._call_event("on_epoch_end", epoch=epoch, metrics=metrics)

        self._call_event("on_train_end")

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()

        return val_loss / len(val_loader)

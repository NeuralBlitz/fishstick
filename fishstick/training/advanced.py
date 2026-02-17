"""
Advanced Training System for fishstick

A comprehensive training framework with callbacks, metrics tracking,
checkpointing, and distributed training support.
"""

from typing import Optional, Dict, List, Callable, Any, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
import os
from pathlib import Path
from collections import defaultdict
import warnings
from abc import ABC, abstractmethod


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: "Trainer"):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: "Trainer"):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, trainer: "Trainer", batch: int):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, trainer: "Trainer", batch: int, logs: Dict):
        """Called at the end of each batch."""
        pass

    def on_validation_begin(self, trainer: "Trainer"):
        """Called at the beginning of validation."""
        pass

    def on_validation_end(self, trainer: "Trainer", logs: Dict):
        """Called at the end of validation."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback to stop training when metric stops improving.

    Args:
        monitor: Metric name to monitor
        min_delta: Minimum change to qualify as improvement
        patience: Number of epochs with no improvement to wait
        mode: 'min' or 'max' for the monitored metric
        verbose: Whether to print messages
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
        elif mode == "max":
            self.is_better = lambda score, best: score > best + min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict):
        score = logs.get(self.monitor)
        if score is None:
            return

        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                trainer.stop_training = True
                if self.verbose:
                    print(f"EarlyStopping: Training stopped at epoch {epoch}")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Args:
        filepath: Path to save checkpoints (can use {epoch} and {val_loss:.2f} placeholders)
        monitor: Metric to monitor for best model
        mode: 'min' or 'max'
        save_best_only: Only save when metric improves
        save_weights_only: Only save weights (not full model)
        verbose: Print save messages
    """

    def __init__(
        self,
        filepath: str = "checkpoint.pt",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        self.best_score = None

        if mode == "min":
            self.is_better = lambda score, best: score < best
        else:
            self.is_better = lambda score, best: score > best

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict):
        score = logs.get(self.monitor)

        if score is None:
            return

        filepath = self.filepath.format(epoch=epoch, **logs)

        if self.save_best_only:
            if self.best_score is None or self.is_better(score, self.best_score):
                self.best_score = score
                self._save_model(trainer, filepath)
                if self.verbose:
                    print(f"\nCheckpoint: Saved best model to {filepath}")
        else:
            self._save_model(trainer, filepath)
            if self.verbose:
                print(f"\nCheckpoint: Saved model to {filepath}")

    def _save_model(self, trainer: "Trainer", filepath: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        checkpoint = {
            "epoch": trainer.current_epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "history": trainer.history,
        }

        if self.save_weights_only:
            torch.save(trainer.model.state_dict(), filepath)
        else:
            torch.save(checkpoint, filepath)


class ReduceLROnPlateau(Callback):
    """
    Reduce learning rate when metric stops improving.

    Args:
        monitor: Metric to monitor
        factor: Factor by which to reduce learning rate
        patience: Number of epochs to wait before reducing
        mode: 'min' or 'max'
        min_lr: Minimum learning rate
        verbose: Print reduction messages
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        factor: float = 0.1,
        patience: int = 5,
        mode: str = "min",
        min_lr: float = 0.0,
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.min_lr = min_lr
        self.verbose = verbose

        self.best_score = None
        self.counter = 0

        if mode == "min":
            self.is_better = lambda score, best: score < best
        else:
            self.is_better = lambda score, best: score > best

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict):
        score = logs.get(self.monitor)
        if score is None:
            return

        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                current_lr = trainer.optimizer.param_groups[0]["lr"]
                new_lr = max(current_lr * self.factor, self.min_lr)

                if new_lr < current_lr:
                    for param_group in trainer.optimizer.param_groups:
                        param_group["lr"] = new_lr

                    if self.verbose:
                        print(
                            f"\nReduceLROnPlateau: Reducing learning rate to {new_lr:.2e}"
                        )

                    self.counter = 0


class TensorBoardLogger(Callback):
    """
    Log metrics to TensorBoard.

    Args:
        log_dir: Directory for tensorboard logs
        histogram_freq: Frequency (in epochs) to log histograms (0 to disable)
    """

    def __init__(self, log_dir: str = "runs", histogram_freq: int = 0):
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        except ImportError:
            raise ImportError("tensorboard is required for TensorBoardLogger")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: Dict):
        for name, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, epoch)

        if self.histogram_freq > 0 and epoch % self.histogram_freq == 0:
            for name, param in trainer.model.named_parameters():
                self.writer.add_histogram(name, param, epoch)

    def on_train_end(self, trainer: "Trainer"):
        self.writer.close()


class MetricsTracker:
    """Track and compute training metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(float)
        self.count = 0

    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
            self.epoch_metrics[key] += value
        self.count += 1

    def compute(self) -> Dict[str, float]:
        """Compute average metrics for the epoch."""
        return {key: value / self.count for key, value in self.epoch_metrics.items()}

    def reset(self):
        """Reset metrics for new epoch."""
        self.epoch_metrics = defaultdict(float)
        self.count = 0

    def get_history(self) -> Dict[str, List[float]]:
        """Get full history of metrics."""
        return dict(self.metrics)


class Trainer:
    """
    Advanced trainer with callbacks, metrics tracking, and checkpointing.

    Args:
        model: PyTorch model to train
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to use ('cuda', 'cpu', or None for auto)
        callbacks: List of callback objects
        metrics: Dict of metric functions

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=nn.CrossEntropyLoss(),
        ...     callbacks=[
        ...         EarlyStopping(patience=10),
        ...         ModelCheckpoint('best_model.pt', save_best_only=True)
        ...     ]
        ... )
        >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: Optional[str] = None,
        callbacks: Optional[List[Callback]] = None,
        metrics: Optional[Dict[str, Callable]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.callbacks = callbacks or []
        self.metrics = metrics or {}

        self.model.to(self.device)

        self.history = defaultdict(list)
        self.current_epoch = 0
        self.stop_training = False

        self.metrics_tracker = MetricsTracker()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        self.stop_training = False

        # Call on_train_begin callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(epochs):
            if self.stop_training:
                break

            self.current_epoch = epoch

            # Call on_epoch_begin callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            # Training phase
            train_logs = self._train_epoch(train_loader, verbose)

            # Validation phase
            if val_loader is not None:
                val_logs = self._validate(val_loader, verbose)
                logs = {**train_logs, **val_logs}
            else:
                logs = train_logs

            # Update history
            for key, value in logs.items():
                self.history[key].append(value)

            # Call on_epoch_end callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, logs)

            if verbose:
                log_str = f"Epoch {epoch + 1}/{epochs}"
                for key, value in logs.items():
                    if isinstance(value, float):
                        log_str += f" - {key}: {value:.4f}"
                    else:
                        log_str += f" - {key}: {value}"
                print(log_str)

        # Call on_train_end callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)

        return dict(self.history)

    def _train_epoch(self, train_loader: DataLoader, verbose: bool) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics_tracker.reset()

        for batch_idx, (data, target) in enumerate(train_loader):
            # Call on_batch_begin callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Compute metrics
            batch_metrics = {"loss": loss.item()}

            for name, metric_fn in self.metrics.items():
                batch_metrics[name] = metric_fn(output, target)

            self.metrics_tracker.update(**batch_metrics)

            # Call on_batch_end callbacks
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, batch_metrics)

            if verbose and batch_idx % 10 == 0:
                print(
                    f"  Batch {batch_idx}/{len(train_loader)} - loss: {loss.item():.4f}",
                    end="\r",
                )

        return self.metrics_tracker.compute()

    def _validate(self, val_loader: DataLoader, verbose: bool) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.metrics_tracker.reset()

        # Call on_validation_begin callbacks
        for callback in self.callbacks:
            callback.on_validation_begin(self)

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss_fn(output, target)

                batch_metrics = {"val_loss": loss.item()}

                for name, metric_fn in self.metrics.items():
                    batch_metrics[f"val_{name}"] = metric_fn(output, target)

                self.metrics_tracker.update(**batch_metrics)

        logs = self.metrics_tracker.compute()

        # Call on_validation_end callbacks
        for callback in self.callbacks:
            callback.on_validation_end(self, logs)

        return logs

    def predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions.

        Args:
            data_loader: Data loader

        Returns:
            Tuple of (predictions, targets)
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                all_preds.append(output.cpu())
                all_targets.append(target)

        return torch.cat(all_preds), torch.cat(all_targets)

    def save(self, filepath: str):
        """Save trainer state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "current_epoch": self.current_epoch,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """Load trainer state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = defaultdict(list, checkpoint["history"])
        self.current_epoch = checkpoint["current_epoch"]


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute classification accuracy."""
    pred = output.argmax(dim=1)
    correct = (pred == target).sum().item()
    return correct / len(target)


def top_k_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy."""
    _, pred = output.topk(k, dim=1)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).sum().item() / len(target)


def precision(
    output: torch.Tensor, target: torch.Tensor, average: str = "macro"
) -> float:
    """Compute precision."""
    pred = output.argmax(dim=1)

    if average == "macro":
        classes = torch.unique(target)
        precisions = []
        for c in classes:
            tp = ((pred == c) & (target == c)).sum().item()
            fp = ((pred == c) & (target != c)).sum().item()
            if tp + fp > 0:
                precisions.append(tp / (tp + fp))
        return sum(precisions) / len(precisions) if precisions else 0.0
    else:
        raise NotImplementedError(f"Average '{average}' not implemented")


def recall(output: torch.Tensor, target: torch.Tensor, average: str = "macro") -> float:
    """Compute recall."""
    pred = output.argmax(dim=1)

    if average == "macro":
        classes = torch.unique(target)
        recalls = []
        for c in classes:
            tp = ((pred == c) & (target == c)).sum().item()
            fn = ((pred != c) & (target == c)).sum().item()
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))
        return sum(recalls) / len(recalls) if recalls else 0.0
    else:
        raise NotImplementedError(f"Average '{average}' not implemented")


def f1_score(
    output: torch.Tensor, target: torch.Tensor, average: str = "macro"
) -> float:
    """Compute F1 score."""
    p = precision(output, target, average)
    r = recall(output, target, average)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


__all__ = [
    "Trainer",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "ReduceLROnPlateau",
    "TensorBoardLogger",
    "MetricsTracker",
    "accuracy",
    "top_k_accuracy",
    "precision",
    "recall",
    "f1_score",
]

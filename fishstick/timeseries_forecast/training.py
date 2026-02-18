"""
Training Infrastructure for Time Series Forecasting.

Provides:
- Forecasting metrics (MSE, MAE, MSIS, OWA, etc.)
- ForecastingTrainer with early stopping, LR scheduling
- Checkpointing utilities

Example:
    >>> from fishstick.timeseries_forecast import (
    ...     ForecastingMetrics,
    ...     ForecastingTrainer,
    ...     EarlyStopping,
    ...     LearningRateScheduler,
    ... )
"""

from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from abc import ABC, abstractmethod
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


class ForecastingMetrics:
    """Collection of forecasting metrics.

    Example:
        >>> metrics = ForecastingMetrics()
        >>> mse = metrics.mse(pred, target)
        >>> mae = metrics.mae(pred, target)
        >>> msis = metrics.msis(pred, target, target_train, alpha=0.05)
    """

    @staticmethod
    def mse(pred: Tensor, target: Tensor) -> Tensor:
        """Mean squared error.

        Args:
            pred: Predictions
            target: Ground truth

        Returns:
            MSE
        """
        return F.mse_loss(pred, target)

    @staticmethod
    def mae(pred: Tensor, target: Tensor) -> Tensor:
        """Mean absolute error.

        Args:
            pred: Predictions
            target: Ground truth

        Returns:
            MAE
        """
        return F.l1_loss(pred, target)

    @staticmethod
    def rmse(pred: Tensor, target: Tensor) -> Tensor:
        """Root mean squared error.

        Args:
            pred: Predictions
            target: Ground truth

        Returns:
            RMSE
        """
        return torch.sqrt(F.mse_loss(pred, target))

    @staticmethod
    def mape(pred: Tensor, target: Tensor, epsilon: float = 1e-8) -> Tensor:
        """Mean absolute percentage error.

        Args:
            pred: Predictions
            target: Ground truth
            epsilon: Small value to avoid division by zero

        Returns:
            MAPE
        """
        return torch.mean(torch.abs((target - pred) / (target + epsilon))) * 100

    @staticmethod
    def smape(pred: Tensor, target: Tensor) -> Tensor:
        """Symmetric MAPE.

        Args:
            pred: Predictions
            target: Ground truth

        Returns:
            sMAPE
        """
        denominator = (torch.abs(target) + torch.abs(pred)) / 2
        return torch.mean(torch.abs(target - pred) / (denominator + 1e-8)) * 100

    @staticmethod
    def msis(
        pred: Tensor,
        target: Tensor,
        lower: Tensor,
        upper: Tensor,
        alpha: float = 0.05,
    ) -> Tensor:
        """Mean Scaled Interval Score.

        Args:
            pred: Predictions
            target: Ground truth
            lower: Lower bound
            upper: Upper bound
            alpha: Significance level

        Returns:
            MSIS score
        """
        lower = lower.unsqueeze(1) if lower.dim() == 1 else lower
        upper = upper.unsqueeze(1) if upper.dim() == 1 else upper

        interval_width = upper - lower
        below = (target < lower).float()
        above = (target > upper).float()

        penalty = (
            2
            / alpha
            * torch.cat(
                [
                    (lower - target) * below,
                    (target - upper) * above,
                ],
                dim=-1,
            ).clamp(min=0)
        )

        return (interval_width + penalty).mean()

    @staticmethod
    def owa(
        pred: Tensor,
        target: Tensor,
        naive_pred: Tensor,
    ) -> Tensor:
        """Overall Weighted Average (OWA).

        Args:
            pred: Predictions
            target: Ground truth
            naive_pred: Naive (persistence) predictions

        Returns:
            OWA score
        """
        mse = F.mse_loss(pred, target)
        naive_mse = F.mse_loss(naive_pred, target)

        mae = F.l1_loss(pred, target)
        naive_mae = F.l1_loss(naive_pred, target)

        mse_score = mse / naive_mse
        mae_score = mae / naive_mae

        return (mse_score + mae_score) / 2

    @staticmethod
    def r2_score(pred: Tensor, target: Tensor) -> Tensor:
        """R-squared score.

        Args:
            pred: Predictions
            target: Ground truth

        Returns:
            R2 score
        """
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - target.mean()) ** 2)
        return 1 - ss_res / (ss_tot + 1e-8)

    @staticmethod
    def quantile_loss(
        pred: Tensor,
        target: Tensor,
        quantile: float = 0.5,
    ) -> Tensor:
        """Quantile (pinball) loss.

        Args:
            pred: Predictions
            target: Ground truth
            quantile: Quantile level

        Returns:
            Quantile loss
        """
        error = target - pred
        loss = torch.max((quantile - 1) * error, quantile * error)
        return loss.mean()

    @staticmethod
    def coverage(
        pred_lower: Tensor,
        pred_upper: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Prediction interval coverage probability.

        Args:
            pred_lower: Lower predictions
            pred_upper: Upper predictions
            target: Ground truth

        Returns:
            Coverage
        """
        coverage = ((target >= pred_lower) & (target <= pred_upper)).float()
        return coverage.mean()


class EarlyStopping:
    """Early stopping utility.

    Args:
        patience: Number of epochs to wait
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max'

    Example:
        >>> early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_loss = validate()
        ...     if early_stopping(val_loss):
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value

        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class LearningRateScheduler:
    """Learning rate scheduler for forecasting.

    Args:
        optimizer: PyTorch optimizer
        mode: 'min' or 'max'
        factor: Factor to reduce LR
        patience: Epochs to wait before reducing
        min_lr: Minimum learning rate

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> scheduler = LearningRateScheduler(optimizer, mode='min')
        >>> for epoch in range(100):
        ...     train()
        ...     scheduler.step(val_loss)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.counter = 0
        self.best_score = None

    def step(self, metric: float) -> None:
        """Update learning rate.

        Args:
            metric: Current metric value
        """
        if self.best_score is None:
            self.best_score = metric
            return

        if self.mode == "min":
            improved = metric < self.best_score
        else:
            improved = metric > self.best_score

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0

    def _reduce_lr(self) -> None:
        """Reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group["lr"] = new_lr


class CheckpointManager:
    """Manages model checkpoints.

    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        mode: 'min' or 'max' for metric optimization

    Example:
        >>> manager = CheckpointManager('./checkpoints')
        >>> manager.save(model, epoch, val_loss)
        >>> model = manager.load_best()
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.mode = mode

        self.checkpoints: List[Dict[str, Any]] = []
        self.best_score = None

    def save(
        self,
        model: nn.Module,
        epoch: int,
        metric: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ) -> str:
        """Save checkpoint.

        Args:
            model: Model to save
            epoch: Current epoch
            metric: Current metric value
            optimizer: Optional optimizer state
            **kwargs: Additional data to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metric": metric,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        checkpoint.update(kwargs)

        filename = f"checkpoint_epoch{epoch}_metric{metric:.4f}.pt"
        filepath = self.checkpoint_dir / filename

        torch.save(checkpoint, filepath)

        self.checkpoints.append(
            {
                "path": str(filepath),
                "metric": metric,
                "epoch": epoch,
            }
        )

        self._cleanup_old_checkpoints()

        if self.best_score is None or self._is_better(metric):
            self.best_score = metric
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        return str(filepath)

    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than best."""
        if self.best_score is None:
            return True
        if self.mode == "min":
            return metric < self.best_score
        return metric > self.best_score

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints."""
        if len(self.checkpoints) > self.max_checkpoints:
            sorted_checkpoints = sorted(
                self.checkpoints,
                key=lambda x: x["metric"],
                reverse=self.mode == "max",
            )
            for ckpt in sorted_checkpoints[self.max_checkpoints :]:
                path = Path(ckpt["path"])
                if path.exists() and path.name != "best_model.pt":
                    path.unlink()
            self.checkpoints = sorted_checkpoints[: self.max_checkpoints]

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load best checkpoint.

        Returns:
            Checkpoint dictionary
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return torch.load(best_path)
        return None


class ForecastingTrainer:
    """Trainer for time series forecasting models.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        gradient_clip: Gradient clipping value

    Example:
        >>> model = Informer(input_dim=7, pred_len=24)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> trainer = ForecastingTrainer(model, optimizer)
        >>> trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_clip: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.MSELoss()
        self.device = device
        self.gradient_clip = gradient_clip

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x = batch

            x = x.to(self.device)
            y = y.to(self.device) if isinstance(batch, (list, tuple)) else x

            self.optimizer.zero_grad()

            pred = self.model(x)

            if y.dim() == 3:
                y = y.squeeze(-1) if y.shape[-1] == 1 else y
            if pred.dim() == 3:
                pred = pred.squeeze(-1) if pred.shape[-1] == 1 else pred

            loss = self.criterion(pred, y)

            loss.backward()

            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self, val_loader: DataLoader) -> float:
        """Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x = batch

                x = x.to(self.device)
                y = y.to(self.device) if isinstance(batch, (list, tuple)) else x

                pred = self.model(x)

                if y.dim() == 3:
                    y = y.squeeze(-1) if y.shape[-1] == 1 else y
                if pred.dim() == 3:
                    pred = pred.squeeze(-1) if pred.shape[-1] == 1 else pred

                loss = self.criterion(pred, y)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping: Optional[EarlyStopping] = None,
        lr_scheduler: Optional[LearningRateScheduler] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> Dict[str, List[float]]:
        """Train model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping: Early stopping callback
            lr_scheduler: Learning rate scheduler
            checkpoint_manager: Checkpoint manager

        Returns:
            Training history
        """
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)

                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                if lr_scheduler is not None:
                    lr_scheduler.step(val_loss)

                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        self.model,
                        epoch,
                        val_loss,
                        self.optimizer,
                    )

                if early_stopping is not None and early_stopping(val_loss):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        return self.history

    def predict(self, x: Tensor) -> Tensor:
        """Generate predictions.

        Args:
            x: Input data

        Returns:
            Predictions
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x).cpu()

    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: int = 100,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict with uncertainty estimation.

        Args:
            x: Input data
            n_samples: Number of samples for MC dropout

        Returns:
            (mean, lower, upper)
        """
        self.model.eval()

        if hasattr(self.model, "forward"):
            original_training = self.model.training
            self.model.train()

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(x.to(self.device))
                predictions.append(pred.cpu())

        predictions = torch.stack(predictions)

        if hasattr(self.model, "forward"):
            self.model.train() if original_training else self.model.eval()

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        z = 1.96
        lower = mean - z * std
        upper = mean + z * std

        return mean, lower, upper


def create_trainer(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr: float = 0.001,
    criterion: Optional[nn.Module] = None,
    device: Optional[str] = None,
    **kwargs,
) -> ForecastingTrainer:
    """Factory function to create a trainer.

    Args:
        model: PyTorch model
        optimizer: Optimizer (if None, creates Adam)
        lr: Learning rate
        criterion: Loss function
        device: Device
        **kwargs: Additional trainer arguments

    Returns:
        ForecastingTrainer

    Example:
        >>> model = Informer(input_dim=7, pred_len=24)
        >>> trainer = create_trainer(model, lr=0.0005)
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return ForecastingTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        **kwargs,
    )

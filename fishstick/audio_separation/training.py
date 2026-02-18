"""
Training Utilities for Audio Source Separation

Provides training loop, datasets, dataloaders, and training utilities
for audio source separation models.
"""

from typing import Optional, List, Dict, Any, Callable, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import json
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration for separation training."""
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 100
    gradient_clip: float = 5.0
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5
    device: str = "cuda"
    accumulation_steps: int = 1
    warmup_steps: int = 1000
    scheduler: str = "cosine"


class SeparationDataset(Dataset):
    """Dataset for audio source separation.

    Expects preprocessed audio files or numpy arrays of mixed and
    clean source audio.
    """

    def __init__(
        self,
        mixtures: List[torch.Tensor],
        sources: List[torch.Tensor],
        source_names: Optional[List[str]] = None,
        sample_rate: int = 16000,
        max_length: Optional[int] = None,
        augment: Optional[nn.Module] = None,
    ):
        self.mixtures = mixtures
        self.sources = sources
        self.source_names = source_names or [f"source_{i}" for i in range(len(sources[0]))]
        self.sample_rate = sample_rate
        self.augment = augment

        if max_length:
            self.mixtures, self.sources = self._trim_to_length(
                mixtures, sources, max_length
            )

    def _trim_to_length(
        self,
        mixtures: List[torch.Tensor],
        sources: List[torch.Tensor],
        max_length: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Trim audio to maximum length."""
        trimmed_mix = []
        trimmed_src = []

        for mix, src in zip(mixtures, sources):
            if mix.shape[-1] > max_length:
                start = np.random.randint(0, mix.shape[-1] - max_length)
                trimmed_mix.append(mix[..., start : start + max_length])
                trimmed_src.append(src[..., start : start + max_length])
            else:
                trimmed_mix.append(mix)
                trimmed_src.append(src)

        return trimmed_mix, trimmed_src

    def __len__(self) -> int:
        return len(self.mixtures)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mixture = self.mixtures[idx]
        sources = self.sources[idx]

        if self.augment:
            mixture = self.augment(mixture)
            sources = self.augment(sources)

        return {
            "mixture": mixture,
            "sources": sources,
            "source_names": self.source_names,
        }


class SeparationTrainer:
    """Trainer for audio source separation models.

    Provides a complete training loop with validation, checkpointing,
    and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device or torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.scheduler = scheduler
        self.metrics = metrics

        self.model.to(self.device)

        self.global_step = 0
        self.epoch = 0

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Run full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader:
                val_loss = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if val_loader:
                print(f"Val Loss: {self.history['val_loss'][-1]:.4f}")

            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        return self.history

    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Epoch {self.epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            mixture = batch["mixture"].to(self.device)
            sources = batch["sources"].to(self.device)

            outputs = self.model(mixture)

            if hasattr(outputs, "sources"):
                predictions = outputs.sources
            else:
                predictions = outputs

            loss = self.loss_fn(predictions, sources)

            (loss / self.config.accumulation_steps).backward()

            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.gradient_clip > 0:
                   _grad_norm_(
 torch.nn.utils.clip                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": total_loss / num_batches})

        return total_loss / num_batches

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Validate the model.

        Args:
            loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc="Validation"):
            mixture = batch["mixture"].to(self.device)
            sources = batch["sources"].to(self.device)

            outputs = self.model(mixture)

            if hasattr(outputs, "sources"):
                predictions = outputs.sources
            else:
                predictions = outputs

            loss = self.loss_fn(predictions, sources)

            total_loss += loss.item()
            num_batches += 1

            if self.metrics:
                metrics = self.metrics.compute(predictions, sources)

        return total_loss / num_batches

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.history = checkpoint.get("history", self.history)

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test set.

        Args:
            loader: Test data loader

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_losses = []

        for batch in tqdm(loader, desc="Evaluating"):
            mixture = batch["mixture"].to(self.device)
            sources = batch["sources"].to(self.device)

            outputs = self.model(mixture)

            if hasattr(outputs, "sources"):
                predictions = outputs.sources
            else:
                predictions = outputs

            loss = self.loss_fn(predictions, sources)
            all_losses.append(loss.item())

            all_predictions.append(predictions.cpu())
            all_targets.append(sources.cpu())

        predictions = torch.cat(all_predictions, dim=1)
        targets = torch.cat(all_targets, dim=1)

        results = {
            "loss": np.mean(all_losses),
        }

        if self.metrics:
            metrics = self.metrics.compute(predictions, targets)
            results.update(metrics)

        return results


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create dataloader for separation dataset.

    Args:
        dataset: Separation dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        pin_memory: Whether to pin memory

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for separation batches.

    Args:
        batch: List of samples

    Returns:
        Batched data
    """
    mixtures = torch.stack([item["mixture"] for item in batch])
    sources = torch.stack([item["sources"] for item in batch])

    return {
        "mixture": mixtures,
        "sources": sources,
    }


class EarlyStopping:
    """Early stopping handler.

    Stops training when validation loss stops improving.
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

        self.best_value = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if should stop.

        Args:
            value: Current validation loss

        Returns:
            True if should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop


class LRSchedulerWrapper:
    """Wrapper for learning rate schedulers."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        total_steps: int,
    ):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0

        self.scheduler = self._create_scheduler()

    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.total_steps // 3,
                gamma=0.5,
            )
        elif self.config.scheduler == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95,
            )
        else:
            return None

    def step(self) -> None:
        """Step the scheduler."""
        self.current_step += 1
        if self.scheduler:
            self.scheduler.step()

    def get_last_lr(self) -> List[float]:
        """Get last learning rate."""
        if self.scheduler:
            return self.scheduler.get_last_lr()
        return [self.config.learning_rate]

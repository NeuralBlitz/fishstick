"""
Checkpoint Management Utilities

Comprehensive checkpoint management for model training including saving,
loading, periodic checkpoints, best model tracking, and distributed checkpoint support.
"""

from typing import Optional, Dict, Any, List, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import os
import json
import shutil
import torch
import torch.nn as nn
from torch.optim import Optimizer
from datetime import datetime
import warnings


class CheckpointStrategy(Enum):
    """Checkpoint saving strategies."""

    PERIODIC = "periodic"
    BEST = "best"
    BEST_PERIODIC = "best_periodic"
    LAST = "last"


@dataclass
class Checkpoint:
    """
    Checkpoint data container.

    Holds all information needed to save and load a training checkpoint.
    """

    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    epoch: int = 0
    global_step: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "metrics": self.metrics,
            "config": self.config,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dictionary."""
        return cls(
            model_state=data.get("model_state", {}),
            optimizer_state=data.get("optimizer_state"),
            scheduler_state=data.get("scheduler_state"),
            epoch=data.get("epoch", 0),
            global_step=data.get("global_step", 0),
            metrics=data.get("metrics", {}),
            config=data.get("config", {}),
            timestamp=data.get("timestamp"),
            version=data.get("version", "1.0"),
        )


class CheckpointManager:
    """
    Main checkpoint manager for training.

    Handles saving, loading, and managing checkpoints during training.

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir="checkpoints",
        ...     max_checkpoints=5,
        ... )
        >>> checkpoint = manager.save(model, optimizer, epoch=10, metrics={"val_loss": 0.5})
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_config: bool = True,
        save_best_only: bool = False,
        monitor: str = "val_loss",
        mode: str = "min",
        verbose: bool = True,
        filename_format: str = "checkpoint_{epoch:04d}.pt",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_config = save_config
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        self.filename_format = filename_format
        self.best_value: Optional[float] = None
        self.checkpoints: List[Path] = []
        self.metadata: Dict[str, Any] = {}

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            epoch: Current epoch
            global_step: Current global step
            metrics: Current metrics
            config: Additional configuration

        Returns:
            Path to saved checkpoint
        """
        if metrics is None:
            metrics = {}
        if config is None:
            config = {}

        checkpoint = Checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
            if optimizer and self.save_optimizer
            else None,
            scheduler_state=scheduler.state_dict()
            if scheduler and self.save_scheduler
            else None,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            config=config,
        )

        if self.save_best_only:
            current_value = metrics.get(self.monitor)
            if current_value is not None:
                if self.best_value is None:
                    should_save = True
                elif self.mode == "min":
                    should_save = current_value < self.best_value
                else:
                    should_save = current_value > self.best_value

                if should_save:
                    self.best_value = current_value
                else:
                    if self.verbose:
                        print(f"Skipping checkpoint: {self.monitor} not improved")
                    return self.checkpoints[-1] if self.checkpoints else Path("")

        filename = self.filename_format.format(epoch=epoch, step=global_step, **metrics)
        filepath = self.checkpoint_dir / filename

        torch.save(checkpoint.to_dict(), filepath)

        self.checkpoints.append(filepath)
        self._cleanup_old_checkpoints()

        if self.verbose:
            print(f"Checkpoint saved to {filepath}")

        return filepath

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                if self.verbose:
                    print(f"Removed old checkpoint: {old_checkpoint}")

    def load(
        self,
        filepath: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Checkpoint:
        """
        Load a checkpoint.

        Args:
            filepath: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load checkpoint to

        Returns:
            Loaded Checkpoint object
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint_data = torch.load(filepath, map_location=device)

        checkpoint = Checkpoint.from_dict(checkpoint_data)

        model.load_state_dict(checkpoint.model_state)

        if optimizer and checkpoint.optimizer_state is not None:
            optimizer.load_state_dict(checkpoint.optimizer_state)

        if scheduler and checkpoint.scheduler_state is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state)

        if self.verbose:
            print(f"Checkpoint loaded from {filepath}")

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint."""
        if not self.checkpoints:
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if not checkpoints:
                return None
            return max(checkpoints, key=lambda p: p.stat().st_mtime)
        return self.checkpoints[-1]

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the path to the best checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None

        best_checkpoint = None
        best_value = None

        for ckpt_path in checkpoints:
            try:
                data = torch.load(ckpt_path, map_location="cpu")
                value = data.get("metrics", {}).get(self.monitor)
                if value is not None:
                    if best_value is None:
                        best_value = value
                        best_checkpoint = ckpt_path
                    elif self.mode == "min" and value < best_value:
                        best_value = value
                        best_checkpoint = ckpt_path
                    elif self.mode == "max" and value > best_value:
                        best_value = value
                        best_checkpoint = ckpt_path
            except Exception:
                continue

        return best_checkpoint

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        for ckpt_path in self.checkpoint_dir.glob("checkpoint_*.pt"):
            try:
                data = torch.load(ckpt_path, map_location="cpu")
                checkpoints.append(
                    {
                        "path": str(ckpt_path),
                        "epoch": data.get("epoch", 0),
                        "global_step": data.get("global_step", 0),
                        "metrics": data.get("metrics", {}),
                        "timestamp": data.get("timestamp"),
                    }
                )
            except Exception:
                continue

        return sorted(checkpoints, key=lambda x: x.get("epoch", 0), reverse=True)


class PeriodicCheckpointManager(CheckpointManager):
    """
    Periodic checkpoint manager.

    Saves checkpoints at regular intervals during training.

    Example:
        >>> manager = PeriodicCheckpointManager(
        ...     checkpoint_dir="checkpoints",
        ...     save_interval=5,
        ...     max_checkpoints=10,
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_interval: int = 5,
        max_checkpoints: int = 5,
        **kwargs,
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            **kwargs,
        )
        self.save_interval = save_interval
        self.last_save_epoch = -1

    def should_save(self, epoch: int) -> bool:
        """Check if checkpoint should be saved at this epoch."""
        if epoch - self.last_save_epoch >= self.save_interval:
            self.last_save_epoch = epoch
            return True
        return False


class BestCheckpointManager(CheckpointManager):
    """
    Best checkpoint manager.

    Only saves checkpoints when the monitored metric improves.

    Example:
        >>> manager = BestCheckpointManager(
        ...     checkpoint_dir="checkpoints",
        ...     monitor="val_loss",
        ...     mode="min",
        ...     save_last=True,
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True,
        max_best_checkpoints: int = 3,
        **kwargs,
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            monitor=monitor,
            mode=mode,
            max_checkpoints=max_best_checkpoints,
            **kwargs,
        )

        self.save_last = save_last
        self.last_checkpoint_path: Optional[Path] = None

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint, tracking best and last separately."""
        if metrics is None:
            metrics = {}

        current_value = metrics.get(self.monitor)
        is_best = False

        if current_value is not None:
            if self.best_value is None:
                is_best = True
            elif self.mode == "min":
                is_best = current_value < self.best_value
            else:
                is_best = current_value > self.best_value

            if is_best:
                self.best_value = current_value

        filepath = super().save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            config=config,
        )

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict() if optimizer else None,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                best_path,
            )
            if self.verbose:
                print(f"Best model saved to {best_path}")

        if self.save_last:
            last_path = self.checkpoint_dir / "last_model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict() if optimizer else None,
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch,
                    "global_step": global_step,
                    "metrics": metrics,
                },
                last_path,
            )
            self.last_checkpoint_path = last_path

        return filepath


def save_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    save_optimizer: bool = True,
) -> None:
    """
    Convenience function to save a checkpoint.

    Args:
        model: Model to save
        filepath: Path to save checkpoint
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch
        global_step: Current global step
        metrics: Current metrics
        config: Additional configuration
        save_optimizer: Whether to save optimizer state
    """
    if metrics is None:
        metrics = {}
    if config is None:
        config = {}

    checkpoint = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }

    if optimizer and save_optimizer:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    if scheduler:
        checkpoint["scheduler_state"] = scheduler.state_dict()

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to load a checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to
        strict: Whether to strictly enforce state dict matching

    Returns:
        Checkpoint metadata dictionary
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state"], strict=strict)

    if optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def load_pretrained(
    model: nn.Module,
    pretrained_path: Union[str, Path],
    device: Optional[torch.device] = None,
    strict: bool = False,
    ignore_keys: Optional[List[str]] = None,
) -> nn.Module:
    """
    Load pretrained weights into a model.

    Args:
        model: Model to load weights into
        pretrained_path: Path to pretrained checkpoint
        device: Device to load weights to
        strict: Whether to strictly enforce key matching
        ignore_keys: List of keys to ignore during loading

    Returns:
        Model with loaded weights
    """
    pretrained_path = Path(pretrained_path)

    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

    state_dict = torch.load(pretrained_path, map_location=device)

    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]

    if ignore_keys is not None:
        for key in ignore_keys:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(key)}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if strict and (missing_keys or unexpected_keys):
        warnings.warn(
            f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}"
        )

    return model


class CheckpointCollection:
    """
    Manage a collection of checkpoints with metadata.

    Provides versioning and organization for multiple checkpoint directories.
    """

    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, CheckpointManager] = {}

    def create_experiment(
        self,
        name: str,
        max_checkpoints: int = 5,
    ) -> CheckpointManager:
        """Create a new experiment checkpoint manager."""
        exp_dir = self.base_dir / name
        manager = CheckpointManager(
            checkpoint_dir=exp_dir,
            max_checkpoints=max_checkpoints,
        )
        self.experiments[name] = manager
        return manager

    def get_experiment(self, name: str) -> Optional[CheckpointManager]:
        """Get checkpoint manager for an experiment."""
        return self.experiments.get(name)

    def list_experiments(self) -> List[str]:
        """List all experiments."""
        return list(self.experiments.keys())

    def save_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        """Save experiment metadata."""
        meta_path = self.base_dir / name / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self, name: str) -> Dict[str, Any]:
        """Load experiment metadata."""
        meta_path = self.base_dir / name / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return {}


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""

    path: str
    epoch: int
    global_step: int
    metrics: Dict[str, float]
    size_mb: float
    timestamp: Optional[str] = None


def list_checkpoint_info(checkpoint_dir: Union[str, Path]) -> List[CheckpointInfo]:
    """
    List all checkpoints with detailed information.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of CheckpointInfo objects
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []

    for ckpt_path in checkpoint_dir.glob("checkpoint_*.pt"):
        try:
            data = torch.load(ckpt_path, map_location="cpu")
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)

            checkpoints.append(
                CheckpointInfo(
                    path=str(ckpt_path),
                    epoch=data.get("epoch", 0),
                    global_step=data.get("global_step", 0),
                    metrics=data.get("metrics", {}),
                    size_mb=size_mb,
                    timestamp=data.get("timestamp"),
                )
            )
        except Exception:
            continue

    return sorted(checkpoints, key=lambda x: x.epoch, reverse=True)

"""
Checkpoint Management

Comprehensive checkpoint management for model training.

Classes:
- CheckpointManager: Main checkpoint manager
- Checkpoint: Checkpoint data structure
- CheckpointMetadata: Metadata for checkpoints
- BestModelTracker: Track best model checkpoints
- CheckpointHistory: History of all checkpoints
- CheckpointOptimizer: Optimize checkpoint storage
- LazyCheckpoint: Lazy loading of checkpoints
- DistributedCheckpoint: Distributed checkpoint handling
"""

from typing import Optional, Dict, List, Any, Union, Callable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import shutil
import glob
import re
import threading
import time
import hashlib

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


T = TypeVar("T")


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""

    step: int
    epoch: int
    timestamp: float
    experiment_name: str
    model_class: str
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    file_size: int = 0
    checksum: Optional[str] = None


@dataclass
class Checkpoint:
    """Complete checkpoint containing model state and metadata."""

    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    metadata: Optional[CheckpointMetadata] = None
    extra_state: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Union[str, Path]) -> None:
        """Save checkpoint to disk.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "metadata": {
                "step": self.metadata.step if self.metadata else 0,
                "epoch": self.metadata.epoch if self.metadata else 0,
                "timestamp": self.metadata.timestamp if self.metadata else time.time(),
                "experiment_name": self.metadata.experiment_name
                if self.metadata
                else "",
                "model_class": self.metadata.model_class if self.metadata else "",
                "metrics": self.metadata.metrics if self.metadata else {},
                "tags": self.metadata.tags if self.metadata else {},
            }
            if self.metadata
            else {},
            "extra_state": self.extra_state,
        }

        torch.save(checkpoint_data, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Checkpoint":
        """Load checkpoint from disk.

        Args:
            path: Path to checkpoint

        Returns:
            Checkpoint object
        """
        path = Path(path)
        checkpoint_data = torch.load(path, map_location="cpu")

        metadata = None
        if "metadata" in checkpoint_data and checkpoint_data["metadata"]:
            meta_dict = checkpoint_data["metadata"]
            metadata = CheckpointMetadata(
                step=meta_dict.get("step", 0),
                epoch=meta_dict.get("epoch", 0),
                timestamp=meta_dict.get("timestamp", 0.0),
                experiment_name=meta_dict.get("experiment_name", ""),
                model_class=meta_dict.get("model_class", ""),
                metrics=meta_dict.get("metrics", {}),
                tags=meta_dict.get("tags", {}),
            )

        return cls(
            model_state=checkpoint_data.get("model_state", {}),
            optimizer_state=checkpoint_data.get("optimizer_state"),
            scheduler_state=checkpoint_data.get("scheduler_state"),
            metadata=metadata,
            extra_state=checkpoint_data.get("extra_state", {}),
        )


class BestModelTracker:
    """Track best model based on a metric."""

    def __init__(
        self,
        metric_name: str,
        mode: str = "max",
        save_top_k: int = 1,
    ):
        """Initialize best model tracker.

        Args:
            metric_name: Metric to track (e.g., 'val_accuracy')
            mode: 'max' or 'min'
            save_top_k: Number of top checkpoints to save
        """
        self.metric_name = metric_name
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_checkpoints: List[tuple] = []

    def update(self, metric_value: float, checkpoint_path: Path) -> bool:
        """Update with new metric value.

        Args:
            metric_value: New metric value
            checkpoint_path: Path to checkpoint

        Returns:
            True if this is a new best
        """
        is_best = False

        if self.mode == "max":
            is_best = metric_value > self.best_value
            if is_best:
                self.best_value = metric_value
        else:
            is_best = metric_value < self.best_value
            if is_best:
                self.best_value = metric_value

        self.best_checkpoints.append((metric_value, checkpoint_path))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=self.mode == "max")

        if len(self.best_checkpoints) > self.save_top_k:
            old_checkpoints = self.best_checkpoints[self.save_top_k :]
            self.best_checkpoints = self.best_checkpoints[: self.save_top_k]

            for _, old_path in old_checkpoints:
                if old_path.exists() and old_path != self.best_checkpoints[0][1]:
                    pass

        return is_best

    def get_best_path(self) -> Optional[Path]:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None
        """
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]
        return None

    def get_top_k_paths(self) -> List[Path]:
        """Get paths to top k checkpoints.

        Returns:
            List of paths
        """
        return [path for _, path in self.best_checkpoints[: self.save_top_k]]


class CheckpointHistory:
    """Maintain history of all checkpoints."""

    def __init__(self, history_file: Optional[Path] = None):
        """Initialize checkpoint history.

        Args:
            history_file: Path to history file
        """
        self.history_file = history_file
        self.checkpoints: List[CheckpointMetadata] = []
        self._lock = threading.Lock()

        if history_file and history_file.exists():
            self.load()

    def add(
        self,
        metadata: CheckpointMetadata,
        checkpoint_path: Path,
    ) -> None:
        """Add checkpoint to history.

        Args:
            metadata: Checkpoint metadata
            checkpoint_path: Path to checkpoint
        """
        with self._lock:
            self.checkpoints.append(metadata)
            self.save()

    def get_best(
        self, metric_name: str, mode: str = "max"
    ) -> Optional[CheckpointMetadata]:
        """Get best checkpoint by metric.

        Args:
            metric_name: Metric to find best
            mode: 'max' or 'min'

        Returns:
            Best checkpoint metadata or None
        """
        valid_checkpoints = [
            (
                cp,
                cp.metrics.get(
                    metric_name, float("-inf") if mode == "max" else float("inf")
                ),
            )
            for cp in self.checkpoints
            if metric_name in cp.metrics
        ]

        if not valid_checkpoints:
            return None

        return max(valid_checkpoints, key=lambda x: x[1] if mode == "max" else -x[1])[0]

    def get_recent(self, n: int = 10) -> List[CheckpointMetadata]:
        """Get n most recent checkpoints.

        Args:
            n: Number of checkpoints

        Returns:
            List of recent checkpoints
        """
        sorted_checkpoints = sorted(
            self.checkpoints, key=lambda x: x.timestamp, reverse=True
        )
        return sorted_checkpoints[:n]

    def save(self) -> None:
        """Save history to file."""
        if not self.history_file:
            return

        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "step": cp.step,
                "epoch": cp.epoch,
                "timestamp": cp.timestamp,
                "experiment_name": cp.experiment_name,
                "model_class": cp.model_class,
                "metrics": cp.metrics,
                "tags": cp.tags,
                "file_size": cp.file_size,
                "checksum": cp.checksum,
            }
            for cp in self.checkpoints
        ]

        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load history from file."""
        if not self.history_file or not self.history_file.exists():
            return

        with open(self.history_file, "r") as f:
            data = json.load(f)

        self.checkpoints = [
            CheckpointMetadata(
                step=cp["step"],
                epoch=cp["epoch"],
                timestamp=cp["timestamp"],
                experiment_name=cp["experiment_name"],
                model_class=cp["model_class"],
                metrics=cp.get("metrics", {}),
                tags=cp.get("tags", {}),
                file_size=cp.get("file_size", 0),
                checksum=cp.get("checksum"),
            )
            for cp in data
        ]


class CheckpointOptimizer:
    """Optimize checkpoint storage."""

    @staticmethod
    def calculate_checksum(path: Path) -> str:
        """Calculate MD5 checksum of file.

        Args:
            path: Path to file

        Returns:
            Checksum string
        """
        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def clean_old_checkpoints(
        checkpoint_dir: Path,
        keep_last_n: int = 5,
        keep_best: bool = True,
        metric_name: str = "val_accuracy",
    ) -> List[Path]:
        """Clean old checkpoints, keeping only recent and best.

        Args:
            checkpoint_dir: Directory containing checkpoints
            keep_last_n: Number of recent checkpoints to keep
            keep_best: Whether to keep best checkpoint
            metric_name: Metric to use for best checkpoint

        Returns:
            List of deleted checkpoint paths
        """
        checkpoint_files = sorted(
            checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        deleted = []

        files_to_keep = set(checkpoint_files[:keep_last_n])

        if keep_best and checkpoint_files:
            history_file = checkpoint_dir / "history.json"
            if history_file.exists():
                history = CheckpointHistory(history_file)
                best = history.get_best(metric_name, mode="max")
                if best:
                    pattern = f"checkpoint_step_{best.step}_*.pt"
                    best_files = list(checkpoint_dir.glob(pattern))
                    files_to_keep.update(best_files)

        for checkpoint_file in checkpoint_files:
            if checkpoint_file not in files_to_keep:
                checkpoint_file.unlink()
                deleted.append(checkpoint_file)

        return deleted

    @staticmethod
    def create_checkpoint_archive(
        checkpoint_dir: Path,
        archive_name: str,
        compression: int = 5,
    ) -> Path:
        """Create compressed archive of checkpoints.

        Args:
            checkpoint_dir: Directory containing checkpoints
            archive_name: Name of archive
            compression: Compression level (0-9)

        Returns:
            Path to archive
        """
        archive_path = checkpoint_dir / f"{archive_name}.tar.gz"

        import tarfile

        with tarfile.open(archive_path, "w:gz", compresslevel=compression) as tar:
            tar.add(checkpoint_dir, arcname=Path(checkpoint_dir).name)

        return archive_path


class LazyCheckpoint:
    """Lazy loading of checkpoints."""

    def __init__(self, checkpoint_path: Path, map_location: str = "cpu"):
        """Initialize lazy checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            map_location: Device to map tensors to
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.map_location = map_location
        self._checkpoint = None

    def _load(self) -> Dict[str, Any]:
        """Load checkpoint data."""
        if self._checkpoint is None:
            self._checkpoint = torch.load(
                self.checkpoint_path, map_location=self.map_location
            )
        return self._checkpoint

    @property
    def model_state(self) -> Dict[str, Any]:
        """Get model state dict."""
        return self._load().get("model_state", {})

    @property
    def optimizer_state(self) -> Optional[Dict[str, Any]]:
        """Get optimizer state dict."""
        return self._load().get("optimizer_state")

    @property
    def scheduler_state(self) -> Optional[Dict[str, Any]]:
        """Get scheduler state dict."""
        return self._load().get("scheduler_state")

    @property
    def metadata(self) -> Optional[CheckpointMetadata]:
        """Get checkpoint metadata."""
        data = self._load().get("metadata")
        if data:
            return CheckpointMetadata(
                step=data.get("step", 0),
                epoch=data.get("epoch", 0),
                timestamp=data.get("timestamp", 0.0),
                experiment_name=data.get("experiment_name", ""),
                model_class=data.get("model_class", ""),
                metrics=data.get("metrics", {}),
                tags=data.get("tags", {}),
            )
        return None


class CheckpointManager:
    """Main checkpoint management class."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        experiment_name: str = "experiment",
        max_checkpoints: int = 10,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_frequency: int = 1,
        save_best: bool = True,
        best_metric: str = "val_accuracy",
        best_mode: str = "max",
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            experiment_name: Name of experiment
            max_checkpoints: Maximum checkpoints to keep
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            save_frequency: Save every n epochs/steps
            save_best: Whether to track and save best model
            best_metric: Metric to use for best model
            best_mode: 'max' or 'min' for best metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_frequency = save_frequency
        self.save_best = save_best

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_tracker = (
            BestModelTracker(best_metric, best_mode, save_top_k=3)
            if save_best
            else None
        )

        self.history_file = self.checkpoint_dir / "history.json"
        self.history = CheckpointHistory(self.history_file)

        self._lock = threading.Lock()

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: Model to save
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            step: Current step
            epoch: Current epoch
            metrics: Dictionary of metrics
            tags: Optional tags
            extra_state: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_step_{step}_epoch_{epoch}.pt"
        )

        model_state = model.state_dict()
        optimizer_state = (
            optimizer.state_dict() if optimizer and self.save_optimizer else None
        )
        scheduler_state = (
            scheduler.state_dict() if scheduler and self.save_scheduler else None
        )

        metadata = CheckpointMetadata(
            step=step,
            epoch=epoch,
            timestamp=time.time(),
            experiment_name=self.experiment_name,
            model_class=model.__class__.__name__,
            metrics=metrics or {},
            tags=tags or {},
            file_size=0,
        )

        checkpoint = Checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            metadata=metadata,
            extra_state=extra_state or {},
        )

        checkpoint.save(checkpoint_path)

        metadata.file_size = checkpoint_path.stat().st_size
        metadata.checksum = CheckpointOptimizer.calculate_checksum(checkpoint_path)

        self.history.add(metadata, checkpoint_path)

        if self.best_tracker and metrics:
            best_metric_value = metrics.get(self.best_tracker.metric_name)
            if best_metric_value is not None:
                is_best = self.best_tracker.update(best_metric_value, checkpoint_path)
                if is_best:
                    best_path = self.checkpoint_dir / "best_model.pt"
                    shutil.copy(checkpoint_path, best_path)

        if self.max_checkpoints > 0:
            self._cleanup_old_checkpoints()

        return checkpoint_path

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints exceeding max_checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for checkpoint in checkpoints[self.max_checkpoints :]:
            if checkpoint.name != "best_model.pt":
                checkpoint.unlink()

    def load(
        self,
        checkpoint_path: Optional[Path] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        map_location: str = "cpu",
    ) -> Checkpoint:
        """Load a checkpoint.

        Args:
            checkpoint_path: Specific checkpoint path to load
            step: Load checkpoint at step
            epoch: Load checkpoint at epoch
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            map_location: Device to map tensors to

        Returns:
            Loaded checkpoint
        """
        if checkpoint_path is None:
            if step is not None and epoch is not None:
                checkpoint_path = (
                    self.checkpoint_dir / f"checkpoint_step_{step}_epoch_{epoch}.pt"
                )
            else:
                checkpoints = sorted(
                    self.checkpoint_dir.glob("checkpoint_*.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                checkpoint_path = checkpoints[0] if checkpoints else None

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = Checkpoint.load(checkpoint_path)

        if model is not None:
            model.load_state_dict(checkpoint.model_state)

        if optimizer is not None and checkpoint.optimizer_state is not None:
            optimizer.load_state_dict(checkpoint.optimizer_state)

        if scheduler is not None and checkpoint.scheduler_state is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state)

        return checkpoint

    def load_best(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
    ) -> Checkpoint:
        """Load best checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into

        Returns:
            Loaded checkpoint
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return self.load(
            checkpoint_path=best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def get_latest(self) -> Optional[Path]:
        """Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return checkpoints[0] if checkpoints else None

    def get_best(self) -> Optional[Path]:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return best_path if best_path.exists() else None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints with metadata.

        Returns:
            List of checkpoint info dictionaries
        """
        return [
            {
                "path": str(cp),
                "step": cp.stat().st_size,
                "size_mb": cp.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(cp.stat().st_mtime).isoformat(),
            }
            for cp in sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        ]

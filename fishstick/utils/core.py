"""
Core utilities module for fishstick.

Comprehensive utilities for logging, checkpointing, serialization, device management,
random seeding, configuration, progress tracking, and miscellaneous utilities.
"""

import os
import sys
import json
import yaml
import time
import random
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Iterator
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


# =============================================================================
# Logging
# =============================================================================


class Logger:
    """Base logger class with common logging functionality."""

    def __init__(self, name: str = "fishstick", level: int = logging.INFO):
        self.name = name
        self.level = level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._initialized = False

    def setup(
        self, log_dir: Optional[str] = None, filename: Optional[str] = None
    ) -> None:
        """Setup logger with file and console handlers."""
        if self._initialized:
            return

        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            if filename is None:
                filename = f"{self.name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(os.path.join(log_dir, filename))
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self._initialized = True

    def debug(self, msg: str, *args, **kwargs) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self.logger.critical(msg, *args, **kwargs)


class TensorBoardLogger:
    """TensorBoard logger wrapper."""

    def __init__(self, log_dir: str, name: str = "experiment"):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "TensorBoard not installed. Install with: pip install tensorboard"
            )

        self.log_dir = os.path.join(log_dir, name)
        self.writer = SummaryWriter(self.log_dir)
        self.step = 0

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        """Log scalar value."""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple scalars."""
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(
        self, tag: str, values: torch.Tensor, step: Optional[int] = None
    ) -> None:
        """Log histogram of tensor."""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)

    def log_image(
        self, tag: str, image: torch.Tensor, step: Optional[int] = None
    ) -> None:
        """Log image."""
        if step is None:
            step = self.step
        self.writer.add_image(tag, image, step)

    def log_graph(self, model: nn.Module, input_to_model: torch.Tensor) -> None:
        """Log model graph."""
        self.writer.add_graph(model, input_to_model)

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Log hyperparameters and metrics."""
        self.writer.add_hparams(hparams, metrics)

    def increment_step(self) -> None:
        """Increment global step."""
        self.step += 1

    def close(self) -> None:
        """Close writer."""
        self.writer.close()


class WandbLogger:
    """Weights & Biases logger wrapper."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        **kwargs,
    ):
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")

        self.project = project
        self.run = self.wandb.init(project=project, name=name, config=config, **kwargs)
        self.step = 0

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics/data."""
        if step is None:
            step = self.step
        self.wandb.log(data, step=step)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str = "model",
        name: Optional[str] = None,
    ) -> None:
        """Log artifact."""
        artifact = self.wandb.Artifact(
            name=name or os.path.basename(artifact_path), type=artifact_type
        )
        artifact.add_file(artifact_path)
        self.run.log_artifact(artifact)

    def watch(self, model: nn.Module, log: str = "all", log_freq: int = 100) -> None:
        """Watch model gradients and parameters."""
        self.wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        """Finish run."""
        self.wandb.finish()


class MLflowLogger:
    """MLflow logger wrapper."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        try:
            import mlflow

            self.mlflow = mlflow
        except ImportError:
            raise ImportError("mlflow not installed. Install with: pip install mlflow")

        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)

        experiment = self.mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = self.mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        self.mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)

    def log_param(self, key: str, value: Any) -> None:
        """Log single parameter."""
        self.mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self.mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log single metric."""
        self.mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics."""
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log artifact."""
        self.mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(
        self, local_dir: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log directory of artifacts."""
        self.mlflow.log_artifacts(local_dir, artifact_path)

    def end_run(self) -> None:
        """End current run."""
        self.mlflow.end_run()


class ConsoleLogger:
    """Simple console logger with colored output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def __init__(self, name: str = "fishstick", use_colors: bool = True):
        self.name = name
        self.use_colors = use_colors and sys.stdout.isatty()

    def _log(self, level: str, msg: str) -> None:
        """Internal log method."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            print(f"{color}[{timestamp}] [{self.name}] [{level}] {msg}{reset}")
        else:
            print(f"[{timestamp}] [{self.name}] [{level}] {msg}")

    def debug(self, msg: str) -> None:
        self._log("DEBUG", msg)

    def info(self, msg: str) -> None:
        self._log("INFO", msg)

    def warning(self, msg: str) -> None:
        self._log("WARNING", msg)

    def error(self, msg: str) -> None:
        self._log("ERROR", msg)

    def critical(self, msg: str) -> None:
        self._log("CRITICAL", msg)


# =============================================================================
# Checkpointing
# =============================================================================


class CheckpointManager:
    """Manage model checkpoints with automatic best model tracking."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.monitor = monitor
        self.mode = mode

        self.checkpoints: List[Tuple[float, Path]] = []
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_checkpoint_path: Optional[Path] = None

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Path:
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "timestamp": time.time(),
            **kwargs,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        filename = f"checkpoint_epoch_{epoch:04d}.pt"
        filepath = self.checkpoint_dir / filename

        torch.save(checkpoint, filepath)

        # Track checkpoint
        score = metrics.get(
            self.monitor, float("inf") if self.mode == "min" else float("-inf")
        )
        self.checkpoints.append((score, filepath))

        # Remove old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            _, old_path = self.checkpoints.pop(0)
            if old_path.exists() and old_path != self.best_checkpoint_path:
                old_path.unlink()

        return filepath

    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    def save_best(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Optional[Path]:
        """Save if this is the best model so far."""
        score = metrics.get(
            self.monitor, float("inf") if self.mode == "min" else float("-inf")
        )

        is_best = (self.mode == "min" and score < self.best_score) or (
            self.mode == "max" and score > self.best_score
        )

        if is_best:
            self.best_score = score
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "best_score": score,
                "timestamp": time.time(),
                **kwargs,
            }

            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            self.best_checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, self.best_checkpoint_path)

            return self.best_checkpoint_path

        return None


def save_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict] = None,
    **kwargs,
) -> None:
    """Save checkpoint to path."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "timestamp": time.time(),
        **kwargs,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load checkpoint from path."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def best_checkpoint(
    checkpoint_dir: str, monitor: str = "val_loss", mode: str = "min"
) -> Optional[str]:
    """Find best checkpoint in directory."""
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    best_score = float("inf") if mode == "min" else float("-inf")
    best_path = None

    for ckpt_file in checkpoint_dir.glob("*.pt"):
        checkpoint = torch.load(ckpt_file, map_location="cpu")
        score = checkpoint.get("metrics", {}).get(monitor)

        if score is not None:
            if (mode == "min" and score < best_score) or (
                mode == "max" and score > best_score
            ):
                best_score = score
                best_path = str(ckpt_file)

    return best_path


# =============================================================================
# Serialization
# =============================================================================


def save_model(model: nn.Module, path: str, **kwargs) -> None:
    """Save model to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "timestamp": time.time(),
        **kwargs,
    }

    torch.save(state, path)


def load_model(
    model: nn.Module, path: str, map_location: str = "cpu", strict: bool = True
) -> nn.Module:
    """Load model from file."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"], strict=strict)
    return model


def save_optimizer(optimizer: Optimizer, path: str, **kwargs) -> None:
    """Save optimizer state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_class": optimizer.__class__.__name__,
        "timestamp": time.time(),
        **kwargs,
    }

    torch.save(state, path)


def load_optimizer(
    optimizer: Optimizer, path: str, map_location: str = "cpu"
) -> Optimizer:
    """Load optimizer state."""
    state = torch.load(path, map_location=map_location)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return optimizer


# =============================================================================
# Device Management
# =============================================================================

_device: Optional[torch.device] = None


def get_device() -> torch.device:
    """Get current device (cuda if available, else cpu)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def to_device(data: Any, device: Optional[torch.device] = None) -> Any:
    """Move data to device."""
    if device is None:
        device = get_device()

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, nn.Module):
        return data.to(device)
    else:
        return data


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def set_device(device: Union[str, int, torch.device]) -> None:
    """Set global device."""
    global _device
    if isinstance(device, int):
        _device = torch.device(f"cuda:{device}" if device >= 0 else "cpu")
    else:
        _device = torch.device(device)


# =============================================================================
# Random Seed Management
# =============================================================================


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed
        deterministic: If True, enables deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_seed(seed: int) -> None:
    """Set random seed (alias for seed_everything without deterministic mode)."""
    seed_everything(seed, deterministic=False)


def get_rng_state() -> Dict[str, Any]:
    """Get current random number generator states."""
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Set random number generator states."""
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# =============================================================================
# Configuration
# =============================================================================


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        if path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save configuration to YAML or JSON file."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    with open(path, "w") as f:
        if path.suffix in [".yaml", ".yml"]:
            yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == ".json":
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries."""
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_config(
    config: Dict[str, Any], schema: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate configuration against a schema.

    Schema format:
    {
        'key': {
            'type': type or tuple of types,
            'required': bool,
            'default': default_value,
            'choices': list of valid values (optional),
            'range': (min, max) for numeric values (optional)
        }
    }
    """
    errors = []

    for key, rules in schema.items():
        # Check required
        if rules.get("required", False) and key not in config:
            errors.append(f"Missing required key: {key}")
            continue

        if key not in config:
            continue

        value = config[key]

        # Check type
        expected_type = rules.get("type")
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"Key '{key}': expected {expected_type}, got {type(value)}")

        # Check choices
        if "choices" in rules and value not in rules["choices"]:
            errors.append(f"Key '{key}': value must be one of {rules['choices']}")

        # Check range
        if "range" in rules:
            min_val, max_val = rules["range"]
            if value < min_val or value > max_val:
                errors.append(
                    f"Key '{key}': value must be in range [{min_val}, {max_val}]"
                )

    return len(errors) == 0, errors


# =============================================================================
# Progress Tracking
# =============================================================================


class ProgressBar:
    """Simple progress bar for training loops."""

    def __init__(
        self, total: int, desc: str = "", width: int = 50, display_interval: float = 0.1
    ):
        self.total = total
        self.desc = desc
        self.width = width
        self.display_interval = display_interval
        self.n = 0
        self.start_time = time.time()
        self.last_display = 0

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.n += n
        current_time = time.time()

        if (
            current_time - self.last_display >= self.display_interval
            or self.n >= self.total
        ):
            self._display()
            self.last_display = current_time

    def _display(self) -> None:
        """Display progress bar."""
        if self.total == 0:
            return

        progress = self.n / self.total
        filled = int(self.width * progress)
        bar = "=" * filled + ">" + "." * (self.width - filled - 1)

        elapsed = time.time() - self.start_time
        rate = self.n / elapsed if elapsed > 0 else 0
        eta = (self.total - self.n) / rate if rate > 0 else 0

        percent = int(progress * 100)
        msg = f"\r{self.desc} [{bar}] {percent}% ({self.n}/{self.total}) "
        msg += f"[{elapsed:.1f}s<{eta:.1f}s, {rate:.2f}it/s]"

        print(msg, end="", flush=True)

        if self.n >= self.total:
            print()

    def close(self) -> None:
        """Close progress bar."""
        if self.n < self.total:
            self.n = self.total
            self._display()


def tqdm_callback(total: Optional[int] = None, desc: str = ""):
    """Create a tqdm-compatible callback function."""
    try:
        from tqdm import tqdm

        pbar = tqdm(total=total, desc=desc)

        def callback(n: int = 1, **kwargs):
            pbar.update(n)
            if kwargs:
                pbar.set_postfix(kwargs)

        callback.close = pbar.close
        return callback
    except ImportError:
        pbar = ProgressBar(total=total or 0, desc=desc)

        def callback(n: int = 1, **kwargs):
            pbar.update(n)

        callback.close = pbar.close
        return callback


def print_progress(
    current: int, total: int, desc: str = "", bar_length: int = 50
) -> None:
    """Print simple progress bar."""
    if total == 0:
        return

    progress = current / total
    filled = int(bar_length * progress)
    bar = "█" * filled + "-" * (bar_length - filled)
    percent = int(progress * 100)

    print(f"\r{desc} |{bar}| {percent}% Complete", end="", flush=True)

    if current >= total:
        print()


# =============================================================================
# Miscellaneous Utilities
# =============================================================================


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.elapsed = 0.0
        self._start_time: Optional[float] = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start_time
        if self.verbose:
            name_str = f" [{self.name}]" if self.name else ""
            print(f"Elapsed{name_str}: {self.elapsed:.4f}s")

    def reset(self) -> None:
        """Reset timer."""
        self._start_time = time.time()
        self.elapsed = 0.0


class Counter:
    """Simple counter for tracking occurrences."""

    def __init__(self):
        self.counts: Dict[str, int] = defaultdict(int)

    def increment(self, key: str, value: int = 1) -> int:
        """Increment counter for key."""
        self.counts[key] += value
        return self.counts[key]

    def get(self, key: str) -> int:
        """Get count for key."""
        return self.counts[key]

    def reset(self, key: Optional[str] = None) -> None:
        """Reset counter(s)."""
        if key is None:
            self.counts.clear()
        else:
            self.counts[key] = 0

    def items(self):
        """Get all counter items."""
        return self.counts.items()

    def total(self) -> int:
        """Get total count."""
        return sum(self.counts.values())


class AverageMeter:
    """Compute and store average and current value."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update average with new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def get_average(self) -> float:
        """Get current average."""
        return self.avg

    def get_sum(self) -> float:
        """Get sum of all values."""
        return self.sum

    def __str__(self) -> str:
        name_str = f"{self.name}: " if self.name else ""
        return f"{name_str}{self.avg:.4f}"


class EarlyStopping:
    """Early stopping to stop training when metric stops improving."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = False,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop.

        Args:
            score: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: New best score: {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(
                        f"EarlyStopping: Triggered after {self.patience} epochs without improvement"
                    )

        return self.early_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


# =============================================================================
# Aliases for backward compatibility and convenience
# =============================================================================

# Logging aliases
Logger = Logger
TB = TensorBoardLogger
WandB = WandbLogger
MLflow = MLflowLogger
Console = ConsoleLogger

# Checkpointing aliases
Manager = CheckpointManager
Save = save_checkpoint
Load = load_checkpoint
Best = best_checkpoint

# Serialization aliases
SaveModel = save_model
LoadModel = load_model
SaveOptimizer = save_optimizer
LoadOptimizer = load_optimizer

# Device aliases
GetDevice = get_device
ToDevice = to_device
IsCUDA = is_cuda_available
SetDevice = set_device

# Random aliases
SeedEverything = seed_everything
SetSeed = set_seed
GetRNGState = get_rng_state
SetRNGState = set_rng_state

# Configuration aliases
LoadConfig = load_config
SaveConfig = save_config
MergeConfigs = merge_configs
ValidateConfig = validate_config

# Progress aliases
Bar = ProgressBar
TQDMCallback = tqdm_callback
PrintProgress = print_progress

# Miscellaneous aliases
Timer = Timer
Counter = Counter
Average = AverageMeter
EarlyStop = EarlyStopping

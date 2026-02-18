"""
Metrics Logger

Comprehensive metrics logging utilities for experiment tracking.

Classes:
- MetricsLogger: Core logging class
- MetricTracker: Track metrics over time
- ScalarLogger: Log scalar values
- HistogramLogger: Log histogram distributions
- ImageLogger: Log image data
- TextLogger: Log text/JSON data
- CompositeLogger: Combine multiple loggers
- TensorBoardLogger: TensorBoard integration
- WandBLogger: Weights & Biases integration
- CSVLogger: CSV-based logging
- JSONLogger: JSON-based logging
"""

from typing import Optional, Dict, List, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import csv
import threading
import time

import torch
from torch import Tensor
import numpy as np
from collections import defaultdict


@dataclass
class MetricValue:
    """Single metric value with metadata."""

    value: Union[float, int]
    step: int
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricTracker:
    """Track metrics over time with statistics."""

    def __init__(self, name: str, window_size: int = 100):
        """Initialize metric tracker.

        Args:
            name: Name of the metric
            window_size: Size of moving average window
        """
        self.name = name
        self.window_size = window_size
        self.values: List[MetricValue] = []
        self.history: List[float] = []

    def append(
        self, value: Union[float, int], step: int, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Append a new value.

        Args:
            value: Metric value
            step: Global step
            tags: Optional tags for grouping
        """
        metric_value = MetricValue(
            value=float(value),
            step=step,
            timestamp=time.time(),
            tags=tags or {},
        )
        self.values.append(metric_value)
        self.history.append(float(value))

    def mean(self, last_n: Optional[int] = None) -> float:
        """Get mean of recent values.

        Args:
            last_n: Number of recent values to average

        Returns:
            Mean value
        """
        if not self.history:
            return 0.0
        data = self.history[-last_n:] if last_n else self.history
        return float(np.mean(data))

    def std(self, last_n: Optional[int] = None) -> float:
        """Get standard deviation of recent values.

        Args:
            last_n: Number of recent values

        Returns:
            Standard deviation
        """
        if not self.history:
            return 0.0
        data = self.history[-last_n:] if last_n else self.history
        return float(np.std(data))

    def min(self, last_n: Optional[int] = None) -> float:
        """Get minimum of recent values.

        Args:
            last_n: Number of recent values

        Returns:
            Minimum value
        """
        if not self.history:
            return 0.0
        data = self.history[-last_n:] if last_n else self.history
        return float(np.min(data))

    def max(self, last_n: Optional[int] = None) -> float:
        """Get maximum of recent values.

        Args:
            last_n: Number of recent values

        Returns:
            Maximum value
        """
        if not self.history:
            return 0.0
        data = self.history[-last_n:] if last_n else self.history
        return float(np.max(data))

    def latest(self) -> Optional[float]:
        """Get latest value.

        Returns:
            Latest value or None
        """
        return self.history[-1] if self.history else None

    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get summary statistics.

        Args:
            last_n: Number of recent values

        Returns:
            Dictionary of statistics
        """
        if not self.history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "latest": 0.0}

        data = self.history[-last_n:] if last_n else self.history
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "latest": float(self.history[-1]),
        }


class ScalarLogger:
    """Logger for scalar values."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize scalar logger.

        Args:
            log_dir: Directory for logs
        """
        self.log_dir = log_dir
        self.metrics: Dict[str, MetricTracker] = defaultdict(lambda: MetricTracker(""))
        self._lock = threading.Lock()

    def log_scalar(
        self,
        name: str,
        value: Union[float, int],
        step: int,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log a scalar value.

        Args:
            name: Metric name
            value: Metric value
            step: Global step
            tags: Optional tags
        """
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricTracker(name)
            self.metrics[name].append(value, step, tags)

    def get_metric(self, name: str) -> Optional[MetricTracker]:
        """Get metric tracker by name.

        Args:
            name: Metric name

        Returns:
            MetricTracker or None
        """
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, MetricTracker]:
        """Get all metrics.

        Returns:
            Dictionary of all metrics
        """
        return dict(self.metrics)


class HistogramLogger:
    """Logger for histogram/distribution data."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize histogram logger.

        Args:
            log_dir: Directory for logs
        """
        self.log_dir = log_dir
        self.histograms: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def log_histogram(
        self,
        name: str,
        values: Union[Tensor, np.ndarray, List[float]],
        step: int,
        num_bins: int = 50,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log histogram of values.

        Args:
            name: Histogram name
            values: Values to compute histogram
            step: Global step
            num_bins: Number of bins
            tags: Optional tags
        """
        with self._lock:
            if isinstance(values, Tensor):
                values = values.detach().cpu().numpy()
            elif isinstance(values, list):
                values = np.array(values)

            hist, bin_edges = np.histogram(values, bins=num_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            self.histograms[name].append(
                {
                    "step": step,
                    "hist": hist.tolist(),
                    "bin_edges": bin_edges.tolist(),
                    "bin_centers": bin_centers.tolist(),
                    "tags": tags or {},
                    "timestamp": time.time(),
                }
            )

    def get_histogram(self, name: str) -> List[Dict[str, Any]]:
        """Get histogram data by name.

        Args:
            name: Histogram name

        Returns:
            List of histogram entries
        """
        return self.histograms.get(name, [])


class ImageLogger:
    """Logger for image data."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize image logger.

        Args:
            log_dir: Directory for logs
        """
        self.log_dir = log_dir
        self.images: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def log_image(
        self,
        name: str,
        image: Union[Tensor, np.ndarray],
        step: int,
        caption: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log an image.

        Args:
            name: Image name
            image: Image tensor/array
            step: Global step
            caption: Optional caption
            tags: Optional tags
        """
        with self._lock:
            if isinstance(image, Tensor):
                image = image.detach().cpu().numpy()

            if image.ndim == 4:
                image = image[0]
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))

            image = np.clip(image, 0, 1)

            self.images[name].append(
                {
                    "step": step,
                    "image": image.tolist() if isinstance(image, np.ndarray) else image,
                    "shape": image.shape,
                    "caption": caption,
                    "tags": tags or {},
                    "timestamp": time.time(),
                }
            )

    def get_images(self, name: str) -> List[Dict[str, Any]]:
        """Get images by name.

        Args:
            name: Image name

        Returns:
            List of image entries
        """
        return self.images.get(name, [])


class TextLogger:
    """Logger for text/JSON data."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize text logger.

        Args:
            log_dir: Directory for logs
        """
        self.log_dir = log_dir
        self.texts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()

    def log_text(
        self,
        name: str,
        text: str,
        step: int,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log text.

        Args:
            name: Text log name
            text: Text content
            step: Global step
            tags: Optional tags
        """
        with self._lock:
            self.texts[name].append(
                {
                    "step": step,
                    "text": text,
                    "tags": tags or {},
                    "timestamp": time.time(),
                }
            )

    def log_json(
        self,
        name: str,
        data: Dict[str, Any],
        step: int,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log JSON-serializable data.

        Args:
            name: JSON log name
            data: Data to log
            step: Global step
            tags: Optional tags
        """
        with self._lock:
            self.texts[name].append(
                {
                    "step": step,
                    "data": data,
                    "tags": tags or {},
                    "timestamp": time.time(),
                }
            )

    def get_texts(self, name: str) -> List[Dict[str, Any]]:
        """Get texts by name.

        Args:
            name: Text name

        Returns:
            List of text entries
        """
        return self.texts.get(name, [])


class CSVLogger:
    """CSV-based logger."""

    def __init__(self, log_file: Path, fieldnames: Optional[List[str]] = None):
        """Initialize CSV logger.

        Args:
            log_file: Path to CSV file
            fieldnames: Column names
        """
        self.log_file = Path(log_file)
        self.fieldnames = fieldnames or ["step", "timestamp"]
        self._file = None
        self._writer = None
        self._lock = threading.Lock()

        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                reader = csv.DictReader(f)
                existing = reader.fieldnames or []
                self.fieldnames = list(set(self.fieldnames + list(existing)))

    def _ensure_file(self) -> None:
        """Ensure CSV file is open."""
        if self._file is None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if self.log_file.exists() else "w"
            self._file = open(self.log_file, mode, newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            if mode == "w":
                self._writer.writeheader()
            self._file.flush()

    def log_row(self, data: Dict[str, Any]) -> None:
        """Log a row of data.

        Args:
            data: Row data
        """
        with self._lock:
            self._ensure_file()
            row = {"timestamp": datetime.now().isoformat()}
            row.update(data)
            self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
            self._file.flush()

    def close(self) -> None:
        """Close the CSV file."""
        if self._file:
            self._file.close()
            self._file = None


class JSONLogger:
    """JSON-based logger."""

    def __init__(self, log_file: Path):
        """Initialize JSON logger.

        Args:
            log_file: Path to JSON file
        """
        self.log_file = Path(log_file)
        self.entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                self.entries = json.load(f)

    def log(self, data: Dict[str, Any]) -> None:
        """Log data as JSON.

        Args:
            data: Data to log
        """
        with self._lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                **data,
            }
            self.entries.append(entry)

    def save(self) -> None:
        """Save entries to file."""
        with self._lock:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as f:
                json.dump(self.entries, f, indent=2)

    def load(self) -> List[Dict[str, Any]]:
        """Load entries from file.

        Returns:
            List of entries
        """
        with self._lock:
            if self.log_file.exists():
                with open(self.log_file, "r") as f:
                    return json.load(f)
            return []


class MetricsLogger:
    """Main metrics logging class combining multiple loggers."""

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        experiment_name: str = "experiment",
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        """Initialize metrics logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Name of experiment
            use_tensorboard: Enable TensorBoard logging
            use_wandb: Enable W&B logging
            wandb_project: W&B project name
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.experiment_name = experiment_name
        self.step = 0
        self.start_time = time.time()

        self.scalar_logger = ScalarLogger(self.log_dir / "scalars")
        self.histogram_logger = HistogramLogger(self.log_dir / "histograms")
        self.image_logger = ImageLogger(self.log_dir / "images")
        self.text_logger = TextLogger(self.log_dir / "texts")

        self.csv_loggers: Dict[str, CSVLogger] = {}
        self.json_loggers: Dict[str, JSONLogger] = {}

        self._tensorboard_logger = None
        self._wandb_run = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tensorboard_logger = SummaryWriter(
                    log_dir=str(self.log_dir / "tensorboard" / experiment_name)
                )
            except ImportError:
                pass

        if use_wandb and wandb_project:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    dir=str(self.log_dir),
                )
            except ImportError:
                pass

    def log_scalar(
        self,
        name: str,
        value: Union[float, int],
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log scalar value.

        Args:
            name: Metric name
            value: Metric value
            step: Global step (auto-increments if None)
            tags: Optional tags
        """
        step = step if step is not None else self.step
        self.scalar_logger.log_scalar(name, value, step, tags)

        if self._tensorboard_logger:
            self._tensorboard_logger.add_scalar(name, value, step)

        if self._wandb_run:
            import wandb

            self._wandb_run.log({name: value, "step": step})

    def log_histogram(
        self,
        name: str,
        values: Union[Tensor, np.ndarray, List[float]],
        step: Optional[int] = None,
        num_bins: int = 50,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log histogram.

        Args:
            name: Histogram name
            values: Values to histogram
            step: Global step
            num_bins: Number of bins
            tags: Optional tags
        """
        step = step if step is not None else self.step
        self.histogram_logger.log_histogram(name, values, step, num_bins, tags)

        if self._tensorboard_logger:
            if isinstance(values, Tensor):
                values = values.detach().cpu().numpy()
            self._tensorboard_logger.add_histogram(name, values, step)

        if self._wandb_run:
            import wandb

            self._wandb_run.log({name: wandb.Histogram(values), "step": step})

    def log_image(
        self,
        name: str,
        image: Union[Tensor, np.ndarray],
        step: Optional[int] = None,
        caption: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log image.

        Args:
            name: Image name
            image: Image tensor/array
            step: Global step
            caption: Optional caption
            tags: Optional tags
        """
        step = step if step is not None else self.step
        self.image_logger.log_image(name, image, step, caption, tags)

        if self._tensorboard_logger:
            self._tensorboard_logger.add_image(name, image, step)

        if self._wandb_run:
            import wandb

            self._wandb_run.log(
                {name: wandb.Image(image, caption=caption), "step": step}
            )

    def log_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log text.

        Args:
            name: Text name
            text: Text content
            step: Global step
            tags: Optional tags
        """
        step = step if step is not None else self.step
        self.text_logger.log_text(name, text, step, tags)

    def log_json(
        self,
        name: str,
        data: Dict[str, Any],
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log JSON data.

        Args:
            name: JSON name
            data: Data to log
            step: Global step
            tags: Optional tags
        """
        step = step if step is not None else self.step
        self.text_logger.log_json(name, data, step, tags)

    def log_metrics(
        self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names to values
            step: Global step
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, step)

    def get_metric_summary(
        self, name: str, last_n: Optional[int] = None
    ) -> Dict[str, float]:
        """Get summary statistics for a metric.

        Args:
            name: Metric name
            last_n: Number of recent values

        Returns:
            Summary statistics
        """
        tracker = self.scalar_logger.get_metric(name)
        if tracker:
            return tracker.get_summary(last_n)
        return {}

    def get_csv_logger(
        self, name: str, fieldnames: Optional[List[str]] = None
    ) -> CSVLogger:
        """Get or create CSV logger.

        Args:
            name: CSV logger name
            fieldnames: Column names

        Returns:
            CSVLogger instance
        """
        if name not in self.csv_loggers:
            log_file = self.log_dir / f"{name}.csv"
            self.csv_loggers[name] = CSVLogger(log_file, fieldnames)
        return self.csv_loggers[name]

    def get_json_logger(self, name: str) -> JSONLogger:
        """Get or create JSON logger.

        Args:
            name: JSON logger name

        Returns:
            JSONLogger instance
        """
        if name not in self.json_loggers:
            log_file = self.log_dir / f"{name}.json"
            self.json_loggers[name] = JSONLogger(log_file)
        return self.json_loggers[name]

    def increment_step(self) -> None:
        """Increment global step."""
        self.step += 1

    def reset_step(self) -> None:
        """Reset global step to 0."""
        self.step = 0

    def close(self) -> None:
        """Close all loggers."""
        for logger in self.csv_loggers.values():
            logger.close()

        for logger in self.json_loggers.values():
            logger.save()

        if self._tensorboard_logger:
            self._tensorboard_logger.close()

        if self._wandb_run:
            self._wandb_run.finish()

    def __enter__(self) -> "MetricsLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class CompositeLogger:
    """Composite logger for combining multiple logging backends."""

    def __init__(self, loggers: List[MetricsLogger]):
        """Initialize composite logger.

        Args:
            loggers: List of MetricsLogger instances
        """
        self.loggers = loggers

    def log_scalar(
        self,
        name: str,
        value: Union[float, int],
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log scalar to all loggers."""
        for logger in self.loggers:
            logger.log_scalar(name, value, step, tags)

    def log_histogram(
        self,
        name: str,
        values: Union[Tensor, np.ndarray, List[float]],
        step: Optional[int] = None,
        num_bins: int = 50,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log histogram to all loggers."""
        for logger in self.loggers:
            logger.log_histogram(name, values, step, num_bins, tags)

    def log_image(
        self,
        name: str,
        image: Union[Tensor, np.ndarray],
        step: Optional[int] = None,
        caption: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log image to all loggers."""
        for logger in self.loggers:
            logger.log_image(name, image, step, caption, tags)

    def close(self) -> None:
        """Close all loggers."""
        for logger in self.loggers:
            logger.close()

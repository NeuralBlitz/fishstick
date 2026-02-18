"""
Base Classes for Metrics

Base classes and interfaces for metric computation.

Classes:
- MetricBase: Abstract base class for all metrics
- MetricTracker: Generic tracker for accumulating predictions
- MetricRegistry: Registry for metric functions
"""

from typing import Dict, List, Optional, Any, Callable, Union
from abc import ABC, abstractmethod

import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field


class MetricBase(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(
        self,
        y_true: Union[Tensor, np.ndarray],
        y_pred: Union[Tensor, np.ndarray],
    ) -> float:
        """Compute the metric.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Metric value
        """
        pass

    def __call__(
        self,
        y_true: Union[Tensor, np.ndarray],
        y_pred: Union[Tensor, np.ndarray],
    ) -> float:
        """Call compute method."""
        return self.compute(y_true, y_pred)


class MetricTracker:
    """Generic tracker for accumulating predictions over batches."""

    def __init__(self):
        self.y_true: List[np.ndarray] = []
        self.y_pred: List[np.ndarray] = []

    def update(
        self,
        y_true: Union[Tensor, np.ndarray],
        y_pred: Union[Tensor, np.ndarray],
    ):
        """Update tracker with new batch."""
        if isinstance(y_true, Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, Tensor):
            y_pred = y_pred.cpu().numpy()

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def compute(self, metric_fn: Callable) -> float:
        """Compute metric on all accumulated data."""
        y_true_all = np.concatenate(self.y_true)
        y_pred_all = np.concatenate(self.y_pred)

        return metric_fn(y_true_all, y_pred_all)

    def reset(self):
        """Reset tracker."""
        self.y_true = []
        self.y_pred = []


class MetricRegistry:
    """Registry for metric functions."""

    def __init__(self):
        self._metrics: Dict[str, Callable] = {}

    def register(self, name: str, metric_fn: Callable):
        """Register a metric function.

        Args:
            name: Metric name
            metric_fn: Metric function
        """
        self._metrics[name] = metric_fn

    def get(self, name: str) -> Callable:
        """Get a metric function by name.

        Args:
            name: Metric name

        Returns:
            Metric function
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found in registry")
        return self._metrics[name]

    def list_metrics(self) -> List[str]:
        """List all registered metrics.

        Returns:
            List of metric names
        """
        return list(self._metrics.keys())

    def compute_all(
        self,
        y_true: Union[Tensor, np.ndarray],
        y_pred: Union[Tensor, np.ndarray],
    ) -> Dict[str, float]:
        """Compute all registered metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metric names to values
        """
        results = {}
        for name, metric_fn in self._metrics.items():
            try:
                results[name] = metric_fn(y_true, y_pred)
            except Exception as e:
                results[name] = float("nan")

        return results


@dataclass
class MetricResult:
    """Container for metric computation results."""

    name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricAggregator:
    """Aggregate metrics from multiple sources."""

    def __init__(self):
        self.results: List[MetricResult] = []

    def add_result(self, name: str, value: float, **metadata):
        """Add a metric result."""
        self.results.append(MetricResult(name=name, value=value, metadata=metadata))

    def get_results(self) -> Dict[str, float]:
        """Get all results as dictionary."""
        return {r.name: r.value for r in self.results}

    def get_detailed_results(self) -> List[MetricResult]:
        """Get detailed results."""
        return self.results

    def summarize(self) -> Dict[str, Any]:
        """Summarize results."""
        values = [r.value for r in self.results]
        return {
            "total_metrics": len(self.results),
            "mean": np.mean(values) if values else 0.0,
            "std": np.std(values) if values else 0.0,
            "min": np.min(values) if values else 0.0,
            "max": np.max(values) if values else 0.0,
        }

    def reset(self):
        """Reset aggregator."""
        self.results = []


class StreamingMetricTracker:
    """Track metrics with streaming updates (running mean)."""

    def __init__(self):
        self.values: List[float] = []
        self.counts: List[int] = []

    def update(self, value: float, count: int = 1):
        """Update with new value."""
        self.values.append(value)
        self.counts.append(count)

    def compute(self) -> Dict[str, float]:
        """Compute aggregated statistics."""
        total_count = sum(self.counts)
        if total_count == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        weighted_mean = (
            sum(v * c for v, c in zip(self.values, self.counts)) / total_count
        )

        weighted_var = (
            sum(c * (v - weighted_mean) ** 2 for v, c in zip(self.values, self.counts))
            / total_count
        )
        weighted_std = np.sqrt(weighted_var)

        return {
            "mean": weighted_mean,
            "std": weighted_std,
            "min": min(self.values),
            "max": max(self.values),
        }

    def reset(self):
        """Reset tracker."""
        self.values = []
        self.counts = []


def create_metric_tracker(metric_type: str = "standard") -> MetricTracker:
    """Create a metric tracker.

    Args:
        metric_type: Type of tracker ('standard', 'streaming')

    Returns:
        MetricTracker instance
    """
    if metric_type == "streaming":
        return StreamingMetricTracker()
    return MetricTracker()


__all__ = [
    "MetricBase",
    "MetricTracker",
    "MetricRegistry",
    "MetricResult",
    "MetricAggregator",
    "StreamingMetricTracker",
    "create_metric_tracker",
]

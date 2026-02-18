"""
Streaming Anomaly Detection Module.

This module provides streaming/online anomaly detection methods:
- Online anomaly detection with sliding windows
- Concept drift detection and adaptation
- Incremental/streaming algorithms
- Adaptive thresholding
- Memory-bounded detection

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import deque
import warnings

import numpy as np
from scipy import stats


@dataclass
class StreamingResult:
    """Container for streaming anomaly detection results."""

    scores: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    n_anomalies: int
    anomaly_indices: np.ndarray
    drift_detected: bool = False
    confidence: Optional[np.ndarray] = None


class BaseStreamingDetector(ABC):
    """Base class for streaming anomaly detectors."""

    def __init__(
        self,
        window_size: int = 100,
        contamination: float = 0.1,
        n_init: int = 100,
    ):
        self.window_size = window_size
        self.contamination = contamination
        self.n_init = n_init
        self.is_initialized = False
        self.drift_detected = False
        self.score = deque(maxlen=window_size)

    @abstractmethod
    def partial_fit(self, X: np.ndarray) -> "BaseStreamingDetector":
        """Update model with new data."""
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return (scores > threshold).astype(int)


class SlidingWindowDetector(BaseStreamingDetector):
    """
    Sliding window-based streaming anomaly detector.

    Maintains a sliding window of recent samples and detects
    anomalies based on deviation from window statistics.

    Parameters
    ----------
    window_size : int
        Size of sliding window.
    contamination : float
        Expected proportion of anomalies.
    method : str
        Detection method: 'zscore', 'iqr', 'mad'.
    threshold : float
        Detection threshold in standard deviations.
    """

    def __init__(
        self,
        window_size: int = 100,
        contamination: float = 0.1,
        method: str = "zscore",
        threshold: float = 3.0,
    ):
        super().__init__(
            window_size=window_size,
            contamination=contamination,
        )
        self.method = method
        self.threshold_val = threshold
        self.window: deque = deque(maxlen=window_size)
        self.threshold: Optional[float] = None

    def partial_fit(self, X: np.ndarray) -> "SlidingWindowDetector":
        """Update window with new data."""
        X = np.asarray(X).flatten()
        for x in X:
            self.window.append(x)

        if len(self.window) >= self.n_init:
            self.is_initialized = True

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        X = np.asarray(X).flatten()
        window_array = np.array(self.window)

        if len(window_array) < 2:
            return np.zeros(len(X))

        scores = np.zeros(len(X))

        if self.method == "zscore":
            window_mean = np.mean(window_array)
            window_std = np.std(window_array) + 1e-10
            scores = np.abs((X - window_mean) / window_std)
        elif self.method == "iqr":
            q1 = np.percentile(window_array, 25)
            q3 = np.percentile(window_array, 75)
            iqr = q3 - q1
            median = np.median(window_array)
            scores = np.abs((X - median) / (iqr + 1e-10))
        elif self.method == "mad":
            median = np.median(window_array)
            mad = np.median(np.abs(window_array - median)) + 1e-10
            scores = np.abs((X - median) / (mad * 1.4826))

        return scores


class DriftAdaptiveDetector(BaseStreamingDetector):
    """
    Concept drift-adaptive anomaly detector.

    Detects anomalies while adapting to concept drift in data streams.

    Parameters
    ----------
    window_size : int
        Size of detection window.
    drift_window : int
        Window for drift detection.
    contamination : float
        Expected proportion of anomalies.
    drift_threshold : float
        Threshold for drift detection.
    adaptation_rate : float
        Rate of adaptation to drift.
    """

    def __init__(
        self,
        window_size: int = 100,
        drift_window: int = 50,
        contamination: float = 0.1,
        drift_threshold: float = 0.1,
        adaptation_rate: float = 0.1,
    ):
        super().__init__(
            window_size=window_size,
            contamination=contamination,
        )
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        self.adaptation_rate = adaptation_rate

        self.reference_window: deque = deque(maxlen=drift_window)
        self.current_window: deque = deque(maxlen=window_size)
        self.reference_stats: Optional[Dict] = None

    def partial_fit(self, X: np.ndarray) -> "DriftAdaptiveDetector":
        """Update model and check for drift."""
        X = np.asarray(X).flatten()

        for x in X:
            self.current_window.append(x)

            if len(self.current_window) >= self.n_init:
                self.is_initialized = True

            if self.is_initialized and len(self.reference_window) >= self.drift_window:
                drift_score = self._compute_drift()
                self.drift_detected = drift_score > self.drift_threshold

                if self.drift_detected:
                    self._adapt_to_drift()
                    self.reference_window.clear()

            if len(self.reference_window) < self.drift_window:
                self.reference_window.append(x)

        return self

    def _compute_drift(self) -> float:
        """Compute drift score between reference and current windows."""
        if len(self.reference_window) < 2 or len(self.current_window) < 2:
            return 0.0

        ref_array = np.array(self.reference_window)
        curr_array = np.array(self.current_window)

        mean_diff = abs(np.mean(curr_array) - np.mean(ref_array))
        std_ratio = max(np.std(curr_array), 1e-10) / max(np.std(ref_array), 1e-10)

        drift_score = mean_diff / (np.std(ref_array) + 1e-10) + abs(std_ratio - 1)
        return drift_score

    def _adapt_to_drift(self) -> None:
        """Adapt to detected drift."""
        if self.reference_stats is None:
            self.reference_stats = {
                "mean": np.mean(self.current_window),
                "std": np.std(self.current_window),
            }
        else:
            current_mean = np.mean(self.current_window)
            current_std = np.std(self.current_window)

            self.reference_stats["mean"] = (
                self.adaptation_rate * current_mean
                + (1 - self.adaptation_rate) * self.reference_stats["mean"]
            )
            self.reference_stats["std"] = (
                self.adaptation_rate * current_std
                + (1 - self.adaptation_rate) * self.reference_stats["std"]
            )

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores with drift adaptation."""
        X = np.asarray(X).flatten()

        if self.reference_stats is None:
            window_array = np.array(self.current_window)
            if len(window_array) < 2:
                return np.zeros(len(X))
            mean = np.mean(window_array)
            std = np.std(window_array) + 1e-10
        else:
            mean = self.reference_stats["mean"]
            std = self.reference_stats["std"]

        scores = np.abs((X - mean) / std)
        return scores


class AdaptiveThresholdDetector(BaseStreamingDetector):
    """
    Adaptive threshold streaming detector.

    Uses exponentially weighted statistics for adaptive thresholding.

    Parameters
    ----------
    window_size : int
        Size of initial window.
    contamination : float
        Expected proportion of anomalies.
    alpha : float
        Exponential smoothing factor.
    """

    def __init__(
        self,
        window_size: int = 100,
        contamination: float = 0.1,
        alpha: float = 0.1,
    ):
        super().__init__(
            window_size=window_size,
            contamination=contamination,
        )
        self.alpha = alpha
        self.ewma_mean: Optional[float] = None
        self.ewma_var: Optional[float] = None

    def partial_fit(self, X: np.ndarray) -> "AdaptiveThresholdDetector":
        """Update adaptive statistics."""
        X = np.asarray(X).flatten()

        for x in X:
            if self.ewma_mean is None:
                self.ewma_mean = x
                self.ewma_var = 0
            else:
                delta = x - self.ewma_mean
                self.ewma_mean += self.alpha * delta
                self.ewma_var = (1 - self.alpha) * (
                    self.ewma_var + self.alpha * delta**2
                )

            self.window.append(x)

            if len(self.window) >= self.n_init:
                self.is_initialized = True

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute adaptive anomaly scores."""
        X = np.asarray(X).flatten()

        if self.ewma_mean is None or self.ewma_var is None:
            return np.zeros(len(X))

        std = np.sqrt(self.ewma_var) + 1e-10
        scores = np.abs((X - self.ewma_mean) / std)
        return scores


class CumulativeSumDetector(BaseStreamingDetector):
    """
    Cumulative Sum (CUSUM) streaming detector.

    Detects changes in mean using cumulative sum algorithm.

    Parameters
    ----------
    window_size : int
        Reference window size.
    contamination : float
        Expected proportion of anomalies.
    drift_threshold : float
        Threshold for drift detection.
    """

    def __init__(
        self,
        window_size: int = 100,
        contamination: float = 0.1,
        drift_threshold: float = 5.0,
    ):
        super().__init__(
            window_size=window_size,
            contamination=contamination,
        )
        self.drift_threshold = drift_threshold
        self.reference_mean: Optional[float] = None
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

    def partial_fit(self, X: np.ndarray) -> "CumulativeSumDetector":
        """Update CUSUM statistics."""
        X = np.asarray(X).flatten()

        for x in X:
            if self.reference_mean is None:
                self.reference_mean = x

            self.window.append(x)

            if len(self.window) >= self.n_init:
                self.is_initialized = True
                self.reference_mean = np.mean(self.window)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute CUSUM scores."""
        X = np.asarray(X).flatten()

        if self.reference_mean is None:
            return np.zeros(len(X))

        std = np.std(self.window) + 1e-10
        normalized = (X - self.reference_mean) / std

        scores = np.abs(normalized)
        return scores


class ExponentialWeightedDetector(BaseStreamingDetector):
    """
    Exponentially weighted moving average detector.

    Uses EWMV for detecting anomalies in streaming data.

    Parameters
    ----------
    window_size : int
        Size of reference window.
    contamination : float
        Expected proportion of anomalies.
    span : int
        Span for EWM.
    threshold : float
        Z-score threshold.
    """

    def __init__(
        self,
        window_size: int = 100,
        contamination: float = 0.1,
        span: int = 20,
        threshold: float = 3.0,
    ):
        super().__init__(
            window_size=window_size,
            contamination=contamination,
        )
        self.span = span
        self.threshold_val = threshold
        self.ewma: Optional[float] = None
        self.ewmv: Optional[float] = None

    def partial_fit(self, X: np.ndarray) -> "ExponentialWeightedDetector":
        """Update EWMV statistics."""
        alpha = 2.0 / (self.span + 1)
        X = np.asarray(X).flatten()

        for x in X:
            if self.ewma is None:
                self.ewma = x
                self.ewmv = 0.0
            else:
                diff = x - self.ewma
                self.ewma += alpha * diff
                self.ewmv = (1 - alpha) * (self.ewmv + alpha * diff**2)

            self.window.append(x)
            if len(self.window) >= self.n_init:
                self.is_initialized = True

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        X = np.asarray(X).flatten()

        if self.ewma is None or self.ewmv is None:
            return np.zeros(len(X))

        std = np.sqrt(self.ewmv) + 1e-10
        scores = np.abs((X - self.ewma) / std)
        return scores


class HalfSpaceTreesDetector(BaseStreamingDetector):
    """
    Half-Space Trees for streaming anomaly detection.

    Efficient tree-based method for high-dimensional data streams.

    Parameters
    ----------
    window_size : int
        Window size for initialization.
    contamination : float
        Expected proportion of anomalies.
    num_trees : int
        Number of half-space trees.
    depth : int
        Maximum depth of trees.
    """

    def __init__(
        self,
        window_size: int = 256,
        contamination: float = 0.1,
        num_trees: int = 100,
        depth: int = 15,
    ):
        super().__init__(
            window_size=window_size,
            contamination=contamination,
        )
        self.num_trees = num_trees
        self.depth = depth
        self.trees: List[Dict] = []
        self.window: deque = deque(maxlen=window_size)

    def partial_fit(self, X: np.ndarray) -> "HalfSpaceTreesDetector":
        """Update half-space trees."""
        X = np.asarray(X)

        if X.ndim > 1:
            X = X.flatten()

        for x in X:
            self.window.append(x)

        if len(self.window) >= self.n_init and not self.trees:
            self._init_trees()
            self.is_initialized = True
        elif self.is_initialized and len(self.window) > 0:
            self._update_trees()

        return self

    def _init_trees(self) -> None:
        """Initialize half-space trees."""
        window_array = np.array(list(self.window))

        for _ in range(self.num_trees):
            tree = self._build_tree(window_array, 0)
            self.trees.append(tree)

    def _build_tree(self, data: np.ndarray, depth: int) -> Dict:
        """Build a single half-space tree."""
        if depth >= self.depth or len(data) <= 1:
            return {
                "type": "leaf",
                "mass": len(data),
                "stats": {
                    "mean": np.mean(data, axis=0),
                    "std": np.std(data, axis=0) + 1e-10,
                },
            }

        dim = np.random.randint(0, data.shape[1])
        split_point = np.median(data[:, dim])

        left_mask = data[:, dim] < split_point
        right_mask = ~left_mask

        return {
            "type": "node",
            "dim": dim,
            "split": split_point,
            "left": self._build_tree(data[left_mask], depth + 1)
            if np.any(left_mask)
            else None,
            "right": self._build_tree(data[right_mask], depth + 1)
            if np.any(right_mask)
            else None,
        }

    def _update_trees(self) -> None:
        """Update trees with new data."""
        window_array = np.array(list(self.window))
        for i, tree in enumerate(self.trees):
            if i % 10 == 0:
                self.trees[i] = self._build_tree(window_array, 0)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        X = np.asarray(X)

        if X.ndim > 1:
            X = X.reshape(1, -1)
        elif len(X) == 1:
            X = X.reshape(1, -1)

        if not self.trees:
            return np.zeros(len(X))

        scores = np.zeros(len(X))

        for x in X:
            for tree in self.trees:
                mass = self._get_mass(tree, x)
                scores[len(scores) - 1] += self.depth - mass

        scores /= len(self.trees)
        return scores

    def _get_mass(self, node: Dict, x: np.ndarray) -> float:
        """Get mass (depth) for a sample."""
        if node["type"] == "leaf":
            return self.depth - np.log2(node["mass"] + 1)

        dim = node["dim"]
        if x[dim] < node["split"]:
            if node["left"] is not None:
                return self._get_mass(node["left"], x)
        else:
            if node["right"] is not None:
                return self._get_mass(node["right"], x)

        return 0


class LodaDetector(BaseStreamingDetector):
    """
    Lightweight Online Detector of Anomalies (LODA).

    Uses projection-based density estimation for streaming data.

    Parameters
    ----------
    window_size : int
        Size of sliding window.
    contamination : float
        Expected proportion of anomalies.
    n_bins : int
        Number of histogram bins.
    n_projections : int
        Number of random projections.
    """

    def __init__(
        self,
        window_size: int = 200,
        contamination: float = 0.1,
        n_bins: int = 10,
        n_projections: int = 100,
    ):
        super().__init__(
            window_size=window_size,
            contamination=contamination,
        )
        self.n_bins = n_bins
        self.n_projections = n_projections
        self.projections: List[np.ndarray] = []
        self.histograms: List[Tuple[np.ndarray, np.ndarray]] = []
        self.window: deque = deque(maxlen=window_size)

    def partial_fit(self, X: np.ndarray) -> "LodaDetector":
        """Update LODA model."""
        X = np.asarray(X)

        for x in X:
            self.window.append(x)

        if len(self.window) >= self.n_init and not self.projections:
            self._init_projections()
            self.is_initialized = True
        elif self.is_initialized:
            window_array = np.array(list(self.window))
            self._update_histograms(window_array)

        return self

    def _init_projections(self) -> None:
        """Initialize random projections."""
        window_array = np.array(list(self.window))
        dim = window_array.shape[1]

        for _ in range(self.n_projections):
            projection = np.random.randn(dim)
            projection /= np.linalg.norm(projection)
            self.projections.append(projection)

        self._update_histograms(window_array)

    def _update_histograms(self, data: np.ndarray) -> None:
        """Update histograms for each projection."""
        self.histograms = []

        for proj in self.projections:
            projected = np.dot(data, proj)
            bins = np.percentile(projected, np.linspace(0, 100, self.n_bins + 1))
            hist, _ = np.histogram(projected, bins=bins)
            hist = hist / (len(projected) + 1e-10)
            self.histograms.append((bins, hist))

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        X = np.asarray(X)

        if not self.histograms:
            return np.zeros(len(X))

        scores = np.zeros(len(X))

        for i, x in enumerate(X):
            log_density = 0.0
            for proj, (bins, hist) in zip(self.projections, self.histograms):
                projected = np.dot(x, proj)
                bin_idx = np.searchsorted(bins[1:-1], projected)
                bin_idx = min(max(bin_idx, 0), len(hist) - 1)
                density = hist[bin_idx] + 1e-10
                log_density += np.log(density)

            scores[i] = -log_density / len(self.projections)

        return scores


class ReservoirSamplingDetector(BaseStreamingDetector):
    """
    Reservoir sampling-based anomaly detector.

    Maintains representative sample from stream for adaptive detection.

    Parameters
    ----------
    reservoir_size : int
        Size of reservoir sample.
    contamination : float
        Expected proportion of anomalies.
    """

    def __init__(
        self,
        reservoir_size: int = 1000,
        contamination: float = 0.1,
    ):
        super().__init__(
            window_size=reservoir_size,
            contamination=contamination,
        )
        self.reservoir_size = reservoir_size
        self.reservoir: List[float] = []
        self.n_seen = 0

    def partial_fit(self, X: np.ndarray) -> "ReservoirSamplingDetector":
        """Update reservoir sample."""
        X = np.asarray(X).flatten()

        for x in X:
            if len(self.reservoir) < self.reservoir_size:
                self.reservoir.append(x)
            else:
                j = np.random.randint(0, self.n_seen + 1)
                if j < self.reservoir_size:
                    self.reservoir[j] = x

            self.n_seen += 1

            if len(self.reservoir) >= self.n_init:
                self.is_initialized = True

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        X = np.asarray(X).flatten()

        if len(self.reservoir) < 2:
            return np.zeros(len(X))

        reservoir_array = np.array(self.reservoir)
        mean = np.mean(reservoir_array)
        std = np.std(reservoir_array) + 1e-10

        scores = np.abs((X - mean) / std)
        return scores

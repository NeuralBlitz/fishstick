"""
Isolation Forest Variants Module.

This module provides various isolation forest-based methods for anomaly detection:
- Standard Isolation Forest with optimized splits
- Randomized Binary Tree-based isolation
- Kernelized Isolation Forest for non-linear boundaries
- Deep Isolation Forest with learned representations
- Streaming Isolation Forest for online learning
- Ensemble of Isolation Forests

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_array, check_random_state
from sklearn.ensemble import IsolationForest as SklearnIsolationForest


@dataclass
class IsolationForestResult:
    """Container for isolation forest detection results."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    n_anomalies: int
    anomaly_indices: np.ndarray
    tree_depths: Optional[np.ndarray] = None


class BaseIsolationForest(ABC):
    """Base class for isolation forest detectors."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        max_samples: Union[str, int] = "auto",
        max_features: Union[float, int] = 1.0,
        bootstrap: bool = False,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.threshold: Optional[float] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseIsolationForest":
        """Fit the detector on data."""
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        if self.threshold is None:
            self.threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return (scores > self.threshold).astype(int)

    def fit_predict(self, X: np.ndarray) -> IsolationForestResult:
        """Fit and predict in one call."""
        self.fit(X)
        scores = self.score(X)
        labels = self.predict(X)
        return IsolationForestResult(
            scores=scores,
            labels=labels,
            threshold=self.threshold,
            n_anomalies=int(np.sum(labels)),
            anomaly_indices=np.where(labels == 1)[0],
        )


class RandomizedBinaryTree:
    """Randomized binary tree for efficient isolation."""

    def __init__(
        self,
        max_depth: int = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree: Optional[Dict] = None

    def build_tree(self, X: np.ndarray) -> Dict:
        """Build a randomized binary tree recursively."""
        rng = check_random_state(self.random_state)
        self.tree = self._build_recursive(X, depth=0, rng=rng)
        return self.tree

    def _build_recursive(
        self,
        X: np.ndarray,
        depth: int,
        rng: np.random.RandomState,
    ) -> Dict:
        """Recursively build tree nodes."""
        n_samples = X.shape[0]

        if n_samples <= 1 or (self.max_depth and depth >= self.max_depth):
            return {"type": "leaf", "size": n_samples}

        feature_idx = rng.integers(0, X.shape[1])
        feature_values = X[:, feature_idx]

        if np.all(feature_values == feature_values[0]):
            return {"type": "leaf", "size": n_samples}

        split_value = rng.uniform(feature_values.min(), feature_values.max())

        left_mask = feature_values < split_value
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            split_value = rng.uniform(feature_values.min(), feature_values.max())
            left_mask = feature_values < split_value
            right_mask = ~left_mask

        return {
            "type": "node",
            "feature": feature_idx,
            "split": split_value,
            "depth": depth,
            "left": self._build_recursive(X[left_mask], depth + 1, rng),
            "right": self._build_recursive(X[right_mask], depth + 1, rng),
        }

    def path_length(self, x: np.ndarray) -> float:
        """Compute path length for a single sample."""
        if self.tree is None:
            raise ValueError("Tree not built. Call build_tree first.")

        node = self.tree
        depth = 0

        while node["type"] == "node":
            if x[node["feature"]] < node["split"]:
                node = node["left"]
            else:
                node = node["right"]
            depth += 1
            if depth > 1000:
                break

        return depth + self._c(node["size"])

    def _c(self, n: int) -> float:
        """Compute average path length for unsuccessful search."""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)


class IsolationForestDetector(BaseIsolationForest):
    """
    Optimized Isolation Forest detector.

    Uses randomized binary trees to isolate anomalies. Anomalies are
    easier to isolate (shorter path lengths) than normal points.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees in the forest.
    contamination : float
        Expected proportion of anomalies.
    max_samples : int or str
        Number of samples to draw to train each tree.
    max_features : int or float
        Number of features to consider for splitting.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        max_samples: Union[str, int] = "auto",
        max_features: Union[float, int] = 1.0,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            verbose=verbose,
        )
        self.trees: List[RandomizedBinaryTree] = []
        self.max_samples_: int = 0

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Fit the isolation forest."""
        X = check_array(X, ensure_min_samples=2)
        n_samples = X.shape[0]

        rng = check_random_state(self.random_state)

        if isinstance(self.max_samples, str) and self.max_samples == "auto":
            self.max_samples_ = min(256, n_samples)
        elif isinstance(self.max_samples, int):
            self.max_samples_ = min(self.max_samples, n_samples)
        else:
            self.max_samples_ = n_samples

        self.max_features_ = (
            int(self.max_features * X.shape[1])
            if self.max_features <= 1.0
            else self.max_features
        )

        self.trees = []
        for i in range(self.n_estimators):
            sample_idx = rng.choice(n_samples, self.max_samples_, replace=False)
            X_sample = X[sample_idx]

            if self.max_features_ < X.shape[1]:
                feature_idx = rng.choice(X.shape[1], self.max_features_, replace=False)
                X_sample = X_sample[:, feature_idx]

            tree = RandomizedBinaryTree(
                max_depth=int(np.ceil(np.log2(self.max_samples_))),
                random_state=rng.randint(0, 2**31),
            )
            tree.build_tree(X_sample)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (shorter path = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit first.")

        X = check_array(X)
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        for tree in self.trees:
            for i in range(n_samples):
                path_len = tree.path_length(X[i])
                scores[i] += path_len

        scores /= self.n_estimators

        c_n = 2 * (np.log(self.max_samples_ - 1) + 0.5772156649) - (
            2 * (self.max_samples_ - 1) / self.max_samples_
        )
        anomaly_scores = 2 ** (-scores / c_n)

        return anomaly_scores


class KernelizedIsolationForest(BaseIsolationForest):
    """
    Kernelized Isolation Forest for non-linear boundaries.

    Uses kernel PCA to project data into a higher-dimensional space
    where isolation becomes easier.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.
    contamination : float
        Expected proportion of anomalies.
    kernel : str
        Kernel type: 'rbf', 'linear', 'poly'.
    gamma : float
        Kernel coefficient.
    n_components : int
        Number of kernel PCA components.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        n_components: int = 10,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.kernel = kernel
        self.gamma = gamma
        self.n_components = n_components
        self.kpca: Optional[Any] = None
        self.iforest: Optional[IsolationForestDetector] = None

    def fit(self, X: np.ndarray) -> "KernelizedIsolationForest":
        """Fit the kernelized isolation forest."""
        from sklearn.kernel_approximation import Nystroem
        from sklearn.decomposition import KernelPCA

        rng = check_random_state(self.random_state)

        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]

        try:
            self.kpca = KernelPCA(
                n_components=min(self.n_components, X.shape[1]),
                kernel=self.kernel,
                gamma=self.gamma,
                random_state=rng,
            )
            X_transformed = self.kpca.fit_transform(X)
        except Exception:
            X_transformed = X

        self.iforest = IsolationForestDetector(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=rng,
        )
        self.iforest.fit(X_transformed)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores in kernel space."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit first.")

        try:
            X_transformed = self.kpca.transform(X)
        except Exception:
            X_transformed = X

        return self.iforest.score(X_transformed)


class DeepIsolationForest(BaseIsolationForest):
    """
    Deep Isolation Forest with learned representations.

    Uses a neural network encoder to learn representations before
    applying isolation forest.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.
    contamination : float
        Expected proportion of anomalies.
    encoder_dims : List[int]
        Encoder network architecture.
    latent_dim : int
        Dimension of latent space.
    epochs : int
        Training epochs for encoder.
    lr : float
        Learning rate.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        encoder_dims: List[int] = None,
        latent_dim: int = 16,
        epochs: int = 100,
        lr: float = 1e-3,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.encoder_dims = encoder_dims or [64, 32]
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.encoder: Optional[Any] = None
        self.iforest: Optional[IsolationForestDetector] = None

    def _build_encoder(self, input_dim: int) -> "torch.nn.Module":
        """Build the encoder network."""
        import torch
        import torch.nn as nn

        layers = []
        dims = [input_dim] + self.encoder_dims + [self.latent_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray) -> "DeepIsolationForest":
        """Fit the deep isolation forest."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        rng = check_random_state(self.random_state)
        torch.manual_seed(rng.randint(0, 2**31))

        self.encoder = self._build_encoder(X.shape[1])
        optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        self.encoder.train()
        for epoch in range(self.epochs):
            for batch, (data,) in enumerate(loader):
                optimizer.zero_grad()
                reconstructed = self.encoder(data)
                loss = criterion(reconstructed, data)
                loss.backward()
                optimizer.step()

        self.encoder.eval()
        with torch.no_grad():
            X_latent = self.encoder(torch.FloatTensor(X)).numpy()

        self.iforest = IsolationForestDetector(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=rng,
        )
        self.iforest.fit(X_latent)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores in latent space."""
        import torch

        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit first.")

        self.encoder.eval()
        with torch.no_grad():
            X_latent = self.encoder(torch.FloatTensor(X)).numpy()

        return self.iforest.score(X_latent)


class StreamingIsolationForest(BaseIsolationForest):
    """
    Streaming Isolation Forest for online learning.

    Supports incremental updates for data streams with concept drift detection.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.
    contamination : float
        Expected proportion of anomalies.
    window_size : int
        Sliding window size.
    drift_threshold : float
        Threshold for drift detection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        window_size: int = 1000,
        drift_threshold: float = 0.1,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.buffer: List[np.ndarray] = []
        self.trees: List[RandomizedBinaryTree] = []
        self.score_history: List[float] = []
        self.drift_detected: bool = False

    def fit(self, X: np.ndarray) -> "StreamingIsolationForest":
        """Initialize with initial batch of data."""
        rng = check_random_state(self.random_state)

        sample_size = min(self.window_size, X.shape[0])
        indices = rng.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]

        self.buffer.append(X_sample)

        for _ in range(self.n_estimators):
            tree = RandomizedBinaryTree(
                max_depth=int(np.ceil(np.log2(sample_size))),
                random_state=rng.randint(0, 2**31),
            )
            tree.build_tree(X_sample)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def partial_fit(self, X: np.ndarray) -> "StreamingIsolationForest":
        """Update model with new data."""
        if not self.is_fitted:
            return self.fit(X)

        rng = check_random_state(self.random_state)

        if len(self.buffer) >= self.window_size:
            self.buffer.pop(0)

        self.buffer.append(X)

        combined = np.vstack(self.buffer)
        if combined.shape[0] > self.window_size:
            indices = rng.choice(combined.shape[0], self.window_size, replace=False)
            combined = combined[indices]

        new_trees = []
        for _ in range(max(1, self.n_estimators // 10)):
            tree = RandomizedBinaryTree(
                max_depth=int(np.ceil(np.log2(combined.shape[0]))),
                random_state=rng.randint(0, 2**31),
            )
            tree.build_tree(combined)
            new_trees.append(tree)

        if len(self.trees) > self.n_estimators:
            self.trees = self.trees[: self.n_estimators]

        self.trees.extend(new_trees)

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit first.")

        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        for tree in self.trees:
            for i in range(n_samples):
                path_len = tree.path_length(X[i])
                scores[i] += path_len

        scores /= len(self.trees)

        buffer_size = min(self.window_size, sum(b.shape[0] for b in self.buffer))
        c_n = 2 * (np.log(buffer_size - 1) + 0.5772156649) - (
            2 * (buffer_size - 1) / buffer_size
        )
        anomaly_scores = 2 ** (-scores / c_n)

        return anomaly_scores

    def detect_drift(self) -> bool:
        """Detect concept drift based on score changes."""
        if len(self.score_history) < 2:
            return False

        recent_scores = self.score_history[-min(10, len(self.score_history)) :]
        if len(recent_scores) < 2:
            return False

        score_variance = np.var(recent_scores)
        self.drift_detected = score_variance > self.drift_threshold

        return self.drift_detected


class EnsembleIsolationForest:
    """
    Ensemble of multiple Isolation Forest variants.

    Combines predictions from multiple isolation forest models
    for improved robustness.

    Parameters
    ----------
    detectors : List[BaseIsolationForest]
        List of isolation forest detectors.
    voting : str
        Voting strategy: 'mean', 'max', 'min'.
    """

    def __init__(
        self,
        detectors: List[BaseIsolationForest] = None,
        voting: str = "mean",
    ):
        self.detectors = detectors or []
        self.voting = voting

    def add_detector(self, detector: BaseIsolationForest) -> "EnsembleIsolationForest":
        """Add a detector to the ensemble."""
        self.detectors.append(detector)
        return self

    def fit(self, X: np.ndarray) -> "EnsembleIsolationForest":
        """Fit all detectors."""
        for detector in self.detectors:
            detector.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble anomaly scores."""
        if not self.detectors:
            raise ValueError("No detectors in ensemble.")

        scores = np.column_stack([d.score(X) for d in self.detectors])

        if self.voting == "mean":
            return np.mean(scores, axis=1)
        elif self.voting == "max":
            return np.max(scores, axis=1)
        elif self.voting == "min":
            return np.min(scores, axis=1)
        else:
            raise ValueError(f"Unknown voting type: {self.voting}")

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.score(X)
        if threshold is None:
            threshold = np.percentile(scores, 90)
        return (scores > threshold).astype(int)


class AdaptiveIsolationForest(BaseIsolationForest):
    """
    Adaptive Isolation Forest that adjusts to local data density.

    Uses adaptive thresholding based on local density estimation.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees.
    contamination : float
        Expected proportion of anomalies.
    n_neighbors : int
        Number of neighbors for density estimation.
    alpha : float
        Adaptive scaling factor.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        n_neighbors: int = 10,
        alpha: float = 0.5,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.iforest: Optional[IsolationForestDetector] = None
        self.local_densities: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "AdaptiveIsolationForest":
        """Fit the adaptive isolation forest."""
        from sklearn.neighbors import NearestNeighbors

        rng = check_random_state(self.random_state)

        nbrs = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, X.shape[0]))
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)

        self.local_densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
        self.local_densities /= np.max(self.local_densities)

        self.iforest = IsolationForestDetector(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=rng,
        )
        self.iforest.fit(X)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute adaptive anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit first.")

        if X.shape[0] != self.local_densities.shape[0]:
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, X.shape[0]))
            nbrs.fit(X)
            distances, _ = nbrs.kneighbors(X)
            local_densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
            local_densities /= np.max(local_densities)
        else:
            local_densities = self.local_densities

        if_scores = self.iforest.score(X)
        adaptive_scores = if_scores * (1 + self.alpha * (1 - local_densities))

        return adaptive_scores

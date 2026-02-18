"""
Time Series Anomaly Detection.

Implements various anomaly detection methods:
- Statistical methods (z-score, IQR)
- Isolation Forest
- LSTM Autoencoder
- One-Class SVM
- Ensemble anomaly detection

Example:
    >>> from fishstick.timeseries_forecast import (
    ...     StatisticalAnomalyDetector,
    ...     IsolationForestDetector,
    ...     LSTMAutoencoderDetector,
    ...     TimeSeriesAnomalyDetector,
    ... )
"""

from typing import Optional, List, Tuple, Dict, Any, Callable, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass
from enum import Enum


class AnomalyScoreMode(Enum):
    """Anomaly scoring modes."""

    POINT = "point"
    CONTEXT = "context"
    RECONSTRUCTION = "reconstruction"


@dataclass
class AnomalyDetectionResult:
    """Container for anomaly detection results."""

    scores: Tensor
    labels: Tensor
    is_anomaly: Tensor
    threshold: float


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors.

    Args:
        threshold: Anomaly threshold
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: Tensor) -> "AnomalyDetector":
        """Fit the detector on normal data.

        Args:
            X: Training data

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: Tensor) -> AnomalyDetectionResult:
        """Predict anomalies.

        Args:
            X: Data to evaluate

        Returns:
            Detection results
        """
        pass

    def fit_predict(self, X: Tensor) -> AnomalyDetectionResult:
        """Fit and predict in one step.

        Args:
            X: Training and evaluation data

        Returns:
            Detection results
        """
        return self.fit(X).predict(X)


class StatisticalAnomalyDetector(AnomalyDetector):
    """Statistical anomaly detector using z-score and IQR methods.

    Args:
        threshold: Z-score threshold
        method: 'zscore' or 'iqr'
        window_size: Window size for rolling statistics

    Example:
        >>> detector = StatisticalAnomalyDetector(threshold=3.0, method='zscore')
        >>> detector.fit(train_data)
        >>> result = detector.predict(test_data)
    """

    def __init__(
        self,
        threshold: float = 3.0,
        method: str = "zscore",
        window_size: Optional[int] = None,
    ):
        super().__init__(threshold)
        self.method = method
        self.window_size = window_size

        self.mean = None
        self.std = None
        self.q1 = None
        self.q3 = None

    def fit(self, X: Tensor) -> "StatisticalAnomalyDetector":
        """Fit statistics.

        Args:
            X: Training data [N, D] or [N,]

        Returns:
            self
        """
        if self.method == "zscore":
            self.mean = X.mean(dim=0, keepdim=True)
            self.std = X.std(dim=0, keepdim=True) + 1e-8
        elif self.method == "iqr":
            self.q1 = torch.quantile(X, 0.25, dim=0, keepdim=True)
            self.q3 = torch.quantile(X, 0.75, dim=0, keepdim=True)

        self.is_fitted = True
        return self

    def predict(self, X: Tensor) -> AnomalyDetectionResult:
        """Predict anomalies.

        Args:
            X: Data to evaluate

        Returns:
            Detection results
        """
        if self.method == "zscore":
            if self.window_size is not None:
                scores = self._rolling_zscore(X)
            else:
                scores = torch.abs((X - self.mean) / self.std)
        else:
            iqr = self.q3 - self.q1
            lower = self.q1 - self.threshold * iqr
            upper = self.q3 + self.threshold * iqr

            lower_score = torch.relu(lower - X)
            upper_score = torch.relu(X - upper)
            scores = torch.max(lower_score, upper_score)

        if scores.dim() > 1:
            scores = scores.mean(dim=-1)

        is_anomaly = (scores > self.threshold).int()

        return AnomalyDetectionResult(
            scores=scores,
            labels=is_anomaly,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
        )

    def _rolling_zscore(self, X: Tensor) -> Tensor:
        """Compute rolling z-score.

        Args:
            X: Input data

        Returns:
            Rolling z-scores
        """
        B, L = X.shape
        scores = torch.zeros_like(X)

        for i in range(L):
            start = max(0, i - self.window_size + 1)
            window = X[:, start : i + 1]
            mean = window.mean(dim=1, keepdim=True)
            std = window.std(dim=1, keepdim=True) + 1e-8
            scores[:, i] = torch.abs((X[:, i : i + 1] - mean) / std).squeeze(-1)

        return scores


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest-based anomaly detector.

    Args:
        n_estimators: Number of trees
        max_samples: Number of samples per tree
        contamination: Expected proportion of anomalies
        random_state: Random seed
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Optional[int] = None,
        contamination: float = 0.1,
        random_state: Optional[int] = None,
    ):
        super().__init__(threshold=0.5)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.trees: List[Dict[str, Any]] = []

    def fit(self, X: Tensor) -> "IsolationForestDetector":
        """Fit isolation forest.

        Args:
            X: Training data

        Returns:
            self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        N, D = X.shape

        if self.max_samples is None:
            self.max_samples = min(N, 256)

        for _ in range(self.n_estimators):
            indices = torch.randperm(N)[: self.max_samples]
            sample = X[indices]

            tree = self._build_tree(
                sample, depth=0, max_depth=int(np.log2(self.max_samples))
            )
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def _build_tree(
        self,
        X: Tensor,
        depth: int,
        max_depth: int,
    ) -> Dict[str, Any]:
        """Build isolation tree.

        Args:
            X: Data at this node
            depth: Current depth
            max_depth: Maximum tree depth

        Returns:
            Tree node
        """
        N = X.shape[0]

        if N <= 1 or depth >= max_depth:
            return {"type": "leaf", "size": N}

        split_dim = torch.randint(0, X.shape[1], (1,)).item()
        split_val = X[:, split_dim].median().item()

        left_mask = X[:, split_dim] < split_val
        right_mask = ~left_mask

        return {
            "type": "node",
            "split_dim": split_dim,
            "split_val": split_val,
            "left": self._build_tree(X[left_mask], depth + 1, max_depth),
            "right": self._build_tree(X[right_mask], depth + 1, max_depth),
        }

    def _path_length(self, x: Tensor, tree: Dict[str, Any], depth: int = 0) -> float:
        """Compute path length for a point.

        Args:
            x: Data point
            tree: Tree node
            depth: Current depth

        Returns:
            Path length
        """
        if tree["type"] == "leaf":
            return depth + self._avg_path_length(tree["size"])

        if x[tree["split_dim"]] < tree["split_val"]:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)

    def _avg_path_length(self, n: int) -> float:
        """Average path length for n samples.

        Args:
            n: Number of samples

        Returns:
            Average path length
        """
        if n <= 1:
            return 0.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)

    def predict(self, X: Tensor) -> AnomalyDetectionResult:
        """Predict anomalies.

        Args:
            X: Data to evaluate

        Returns:
            Detection results
        """
        scores = self.score_samples(X)

        threshold_idx = int(len(scores) * (1 - self.contamination))
        threshold = torch.kthvalue(scores, len(scores) - threshold_idx)[0].item()

        is_anomaly = (scores > threshold).int()

        return AnomalyDetectionResult(
            scores=scores,
            labels=is_anomaly,
            is_anomaly=is_anomaly,
            threshold=threshold,
        )

    def score_samples(self, X: Tensor) -> Tensor:
        """Compute anomaly scores.

        Args:
            X: Data to score

        Returns:
            Anomaly scores
        """
        N = X.shape[0]
        path_lengths = torch.zeros(N, self.n_estimators)

        for i, tree in enumerate(self.trees):
            for j in range(N):
                path_lengths[j, i] = self._path_length(X[j], tree)

        avg_path_lengths = path_lengths.mean(dim=1)

        c = self._avg_path_length(self.max_samples)

        scores = torch.pow(2, -avg_path_lengths / c)

        return scores


class LSTMAutoencoderDetector(nn.Module, AnomalyDetector):
    """LSTM Autoencoder for anomaly detection.

    Learns normal patterns and flags high reconstruction error as anomalies.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        latent_dim: Latent dimension
        n_layers: Number of LSTM layers
        threshold: Anomaly threshold
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_layers: int = 2,
        threshold: float = 0.5,
    ):
        nn.Module.__init__(self)
        AnomalyDetector.__init__(self, threshold)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.encoder_proj = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

        self.output_proj = nn.Linear(hidden_dim, input_dim)

        self.threshold_value = None

    def fit(
        self,
        X: Tensor,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
    ) -> "LSTMAutoencoderDetector":
        """Train autoencoder on normal data.

        Args:
            X: Training data [N, L, D]
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size

        Returns:
            self
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        self.train()

        for epoch in range(epochs):
            total_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()

                recon = self.forward(batch)
                loss = criterion(recon, batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}"
                )

        self.eval()
        self.is_fitted = True
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input [B, L, D]

        Returns:
            Reconstructed output [B, L, D]
        """
        _, (h, _) = self.encoder(x)

        h = h[-1]
        z = self.encoder_proj(h)
        z = z.unsqueeze(1).expand(-1, x.shape[1], -1)

        dec_out, _ = self.decoder(z)
        recon = self.output_proj(dec_out)

        return recon

    def predict(self, X: Tensor) -> AnomalyDetectionResult:
        """Predict anomalies.

        Args:
            X: Data to evaluate [N, L, D]

        Returns:
            Detection results
        """
        self.eval()
        with torch.no_grad():
            recon = self.forward(X)
            mse = F.mse_loss(recon, X, reduction="none")
            scores = mse.mean(dim=(1, 2))

        if self.threshold_value is None:
            self.threshold_value = self._compute_threshold(scores)

        is_anomaly = (scores > self.threshold_value).int()

        return AnomalyDetectionResult(
            scores=scores,
            labels=is_anomaly,
            is_anomaly=is_anomaly,
            threshold=self.threshold_value,
        )

    def _compute_threshold(self, scores: Tensor) -> float:
        """Compute threshold from scores.

        Args:
            scores: Anomaly scores

        Returns:
            Threshold value
        """
        return scores.quantile(1 - self.contamination).item()


class OneClassSVMDetector(AnomalyDetector):
    """One-Class SVM for anomaly detection.

    Args:
        kernel: Kernel type ('rbf', 'linear', 'poly')
        gamma: Kernel coefficient
        nu: Nu parameter (upper bound on fraction of outliers)
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: float = "auto",
        nu: float = 0.1,
    ):
        super().__init__(threshold=0.5)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu

        self.support_vectors: Optional[Tensor] = None
        self.dual_coef: Optional[Tensor] = None
        self.rho: float = 0.0

    def fit(self, X: Tensor) -> "OneClassSVMDetector":
        """Fit One-Class SVM.

        Args:
            X: Training data

        Returns:
            self
        """
        raise NotImplementedError("One-Class SVM requires sklearn")

    def predict(self, X: Tensor) -> AnomalyDetectionResult:
        """Predict anomalies.

        Args:
            X: Data to evaluate

        Returns:
            Detection results
        """
        if self.support_vectors is None:
            raise ValueError("Model not fitted")

        with torch.no_grad():
            scores = self._decision_function(X)

        is_anomaly = (scores < 0).int()

        return AnomalyDetectionResult(
            scores=-scores,
            labels=is_anomaly,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
        )

    def _decision_function(self, X: Tensor) -> Tensor:
        """Compute decision function.

        Args:
            X: Input data

        Returns:
            Decision function values
        """
        if self.kernel == "rbf":
            if self.gamma == "auto":
                gamma = 1.0 / X.shape[1]
            else:
                gamma = self.gamma

            dist = torch.cdist(X, self.support_vectors)
            kernel_matrix = torch.exp(-gamma * dist**2)

            return (kernel_matrix @ self.dual_coef).squeeze() + self.rho

        return torch.zeros(X.shape[0])


class EnsembleAnomalyDetector(AnomalyDetector):
    """Ensemble anomaly detector combining multiple methods.

    Args:
        detectors: List of anomaly detectors
        aggregation: 'mean', 'max', or 'majority'

    Example:
        >>> detectors = [
        ...     StatisticalAnomalyDetector(threshold=3.0),
        ...     IsolationForestDetector(n_estimators=50),
        ... ]
        >>> ensemble = EnsembleAnomalyDetector(detectors, aggregation='mean')
    """

    def __init__(
        self,
        detectors: List[AnomalyDetector],
        aggregation: str = "mean",
        threshold: float = 0.5,
    ):
        super().__init__(threshold)
        self.detectors = (
            nn.ModuleList(detectors)
            if hasattr(detectors[0], "parameters")
            else detectors
        )
        self.aggregation = aggregation

    def fit(self, X: Tensor) -> "EnsembleAnomalyDetector":
        """Fit all detectors.

        Args:
            X: Training data

        Returns:
            self
        """
        for detector in self.detectors:
            if hasattr(detector, "fit"):
                detector.fit(X)

        self.is_fitted = True
        return self

    def predict(self, X: Tensor) -> AnomalyDetectionResult:
        """Predict anomalies using ensemble.

        Args:
            X: Data to evaluate

        Returns:
            Detection results
        """
        all_scores = []
        all_labels = []

        for detector in self.detectors:
            result = detector.predict(X)
            all_scores.append(result.scores)
            all_labels.append(result.is_anomaly)

        scores_tensor = torch.stack(all_scores)
        labels_tensor = torch.stack(all_labels)

        if self.aggregation == "mean":
            scores = scores_tensor.mean(dim=0)
            labels = labels_tensor.float().mean(dim=0) > 0.5
        elif self.aggregation == "max":
            scores = scores_tensor.max(dim=0)[0]
            labels = labels_tensor.max(dim=0)[0]
        elif self.aggregation == "majority":
            scores = scores_tensor.mean(dim=0)
            labels = labels_tensor.float().mean(dim=0) > 0.5
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return AnomalyDetectionResult(
            scores=scores,
            labels=labels.int(),
            is_anomaly=labels.int(),
            threshold=self.threshold,
        )


class TimeSeriesAnomalyDetector:
    """High-level interface for time series anomaly detection.

    Args:
        method: Detection method ('statistical', 'isolation_forest', 'autoencoder', 'ensemble')
        threshold: Anomaly threshold
        **kwargs: Additional method-specific arguments
    """

    METHODS = {
        "statistical": StatisticalAnomalyDetector,
        "isolation_forest": IsolationForestDetector,
        "autoencoder": LSTMAutoencoderDetector,
    }

    def __init__(
        self,
        method: str = "statistical",
        threshold: float = 0.5,
        **kwargs,
    ):
        self.method = method
        self.threshold = threshold
        self.kwargs = kwargs

        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}")

        self.detector = self.METHODS[method](
            threshold=threshold,
            **kwargs,
        )

    def fit(
        self,
        X: Tensor,
        epochs: Optional[int] = None,
    ) -> "TimeSeriesAnomalyDetector":
        """Fit detector.

        Args:
            X: Training data
            epochs: Training epochs (for autoencoder)

        Returns:
            self
        """
        if epochs is not None and self.method == "autoencoder":
            self.detector.fit(X, epochs=epochs)
        else:
            self.detector.fit(X)

        return self

    def predict(self, X: Tensor) -> AnomalyDetectionResult:
        """Predict anomalies.

        Args:
            X: Data to evaluate

        Returns:
            Detection results
        """
        return self.detector.predict(X)

    def detect(
        self,
        X: Tensor,
        return_indices: bool = True,
    ) -> Union[Tensor, List[int]]:
        """Detect anomaly indices.

        Args:
            X: Data to evaluate
            return_indices: Whether to return indices

        Returns:
            Anomaly indices or boolean mask
        """
        result = self.predict(X)

        if return_indices:
            return result.is_anomaly.nonzero(as_tuple=True)[0].tolist()
        return result.is_anomaly.bool()


def create_anomaly_detector(
    method: str = "statistical",
    **kwargs,
) -> AnomalyDetector:
    """Factory function to create anomaly detectors.

    Args:
        method: Detection method
        **kwargs: Additional arguments

    Returns:
        Initialized detector

    Example:
        >>> detector = create_anomaly_detector('autoencoder', input_dim=7)
    """
    detectors = {
        "statistical": StatisticalAnomalyDetector,
        "isolation_forest": IsolationForestDetector,
        "autoencoder": LSTMAutoencoderDetector,
    }

    if method not in detectors:
        raise ValueError(f"Unknown method: {method}")

    return detectors[method](**kwargs)

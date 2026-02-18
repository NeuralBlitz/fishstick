"""
Comprehensive Anomaly Detection Module for Fishstick.

This module provides a wide range of anomaly detection algorithms including:
- Reconstruction-based methods (Autoencoder, VAE, DeepSVDD, AnoGAN, MemAE)
- Distance-based methods (KNN, LOF, Isolation Forest, OCSVM)
- Density-based methods (GMM, KDE, Normalizing Flows)
- Self-supervised methods (Contrastive, Rotation, Jigsaw, Relative Position)
- Attention-based methods (Transformers, Attention mechanisms)

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# =============================================================================
# Base Classes
# =============================================================================


class BaseAnomalyDetector(ABC):
    """Base class for sklearn-based anomaly detectors."""

    def __init__(self, contamination: float = 0.1, random_state: Optional[int] = None):
        super().__init__()
        self.contamination = contamination
        self.random_state = random_state
        self.is_fitted = False
        self.threshold: Optional[float] = None

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "BaseAnomalyDetector":
        """Fit the detector on training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1 for anomaly, 0 for normal)."""
        pass

    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous)."""
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        pass

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and predict in one call."""
        self.fit(X, y)
        return self.predict(X)


class BaseTorchAnomalyDetector(BaseAnomalyDetector, nn.Module):
    """Base class for PyTorch-based anomaly detectors."""

    def __init__(
        self,
        input_dim: int,
        contamination: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 100,
        device: str = "auto",
        random_state: Optional[int] = None,
    ):
        BaseAnomalyDetector.__init__(self, contamination, random_state)
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = self._get_device(device)

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

    def _get_device(self, device: str) -> torch.device:
        """Get torch device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _to_tensor(self, X: np.ndarray) -> Tensor:
        """Convert numpy array to tensor."""
        return torch.FloatTensor(X).to(self.device)

    def _create_dataloader(self, X: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create DataLoader from numpy array."""
        dataset = TensorDataset(torch.FloatTensor(X))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        return self.decision_function(X)


# =============================================================================
# Reconstruction-Based Methods
# =============================================================================


class AutoencoderAnomalyDetector(BaseTorchAnomalyDetector):
    """
    Standard Autoencoder for anomaly detection.

    Uses reconstruction error as anomaly score.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        latent_dim: int = 16,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(input_dim, **kwargs)

        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(self.device)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU())

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to input space."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning reconstruction."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "AutoencoderAnomalyDetector":
        """Fit the autoencoder on normal data."""
        dataloader = self._create_dataloader(X)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                x_hat = self.forward(batch_x)
                loss = F.mse_loss(x_hat, batch_x)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(dataloader):.4f}"
                )

        # Compute threshold
        self.eval()
        scores = self.decision_function(X)
        self.threshold = float(np.percentile(scores, (1 - self.contamination) * 100))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error as anomaly score."""
        self.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            reconstructed = self.forward(X_tensor)
            scores = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        return scores.cpu().numpy()


# =============================================================================
# Distance-Based Methods
# =============================================================================


class KNNAnomalyDetector(BaseAnomalyDetector):
    """K-Nearest Neighbors Anomaly Detection."""

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "minkowski",
        method: str = "largest",
        contamination: float = 0.1,
        random_state: Optional[int] = None,
    ):
        super().__init__(contamination=contamination, random_state=random_state)

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.method = method
        self.neigh_: Optional[NearestNeighbors] = None
        self.X_train_: Optional[np.ndarray] = None

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "KNNAnomalyDetector":
        """Fit KNN model on training data."""
        self.X_train_ = X.copy()
        self.neigh_ = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            metric=self.metric,
            algorithm="auto",
        )
        self.neigh_.fit(X)

        scores = self.decision_function(X)
        self.threshold = float(np.percentile(scores, (1 - self.contamination) * 100))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.decision_function(X)
        return np.where(scores > self.threshold, 1, 0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute KNN distance as anomaly score."""
        if self.neigh_ is None:
            raise RuntimeError("Detector must be fitted before prediction")
        distances, indices = self.neigh_.kneighbors(X)

        if X is self.X_train_:
            distances = distances[:, 1:]
        else:
            distances = distances[:, : self.n_neighbors]

        if self.method == "largest":
            return distances[:, -1]
        elif self.method == "mean":
            return np.mean(distances, axis=1)
        elif self.method == "median":
            return np.median(distances, axis=1)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        return self.decision_function(X)


class LOFDetector(BaseAnomalyDetector):
    """Local Outlier Factor Anomaly Detection."""

    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = "minkowski",
        contamination: float = 0.1,
        novelty: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__(contamination=contamination, random_state=random_state)

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.novelty = novelty
        self.lof_: Optional[LocalOutlierFactor] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "LOFDetector":
        """Fit LOF on training data."""
        self.lof_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            contamination=self.contamination,
            novelty=self.novelty,
        )
        self.lof_.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        if self.lof_ is None:
            raise RuntimeError("Detector must be fitted before prediction")
        predictions = self.lof_.predict(X)
        return np.where(predictions == -1, 1, 0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute LOF score."""
        if self.lof_ is None:
            raise RuntimeError("Detector must be fitted before prediction")
        return -self.lof_.score_samples(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        return self.decision_function(X)


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest Anomaly Detection."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float, str] = "auto",
        max_features: float = 1.0,
        contamination: float = 0.1,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
    ):
        super().__init__(contamination=contamination, random_state=random_state)

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.iforest_: Optional[IsolationForest] = None

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "IsolationForestDetector":
        """Fit Isolation Forest on training data."""
        self.iforest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.iforest_.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        if self.iforest_ is None:
            raise RuntimeError("Detector must be fitted before prediction")
        predictions = self.iforest_.predict(X)
        return np.where(predictions == -1, 1, 0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly score."""
        if self.iforest_ is None:
            raise RuntimeError("Detector must be fitted before prediction")
        return -self.iforest_.score_samples(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        return self.decision_function(X)


class OCSVMDetector(BaseAnomalyDetector):
    """One-Class SVM Anomaly Detection."""

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        nu: float = 0.1,
        contamination: float = 0.1,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(contamination=contamination, random_state=random_state)

        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.svm_kwargs = kwargs
        self.svm_: Optional[OneClassSVM] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OCSVMDetector":
        """Fit One-Class SVM on training data."""
        self.svm_ = OneClassSVM(
            kernel=self.kernel,
            gamma=self.gamma,
            nu=self.nu,
            **self.svm_kwargs,
        )
        self.svm_.fit(X)

        scores = self.decision_function(X)
        self.threshold = float(np.percentile(scores, (1 - self.contamination) * 100))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        if self.svm_ is None:
            raise RuntimeError("Detector must be fitted before prediction")
        scores = self.decision_function(X)
        return np.where(scores > self.threshold, 1, 0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function."""
        if self.svm_ is None:
            raise RuntimeError("Detector must be fitted before prediction")
        return -self.svm_.decision_function(X).ravel()

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        return self.decision_function(X)


# =============================================================================
# Density-Based Methods
# =============================================================================


class GMMDetector(BaseAnomalyDetector):
    """Gaussian Mixture Model Anomaly Detection."""

    def __init__(
        self,
        n_components: int = 5,
        covariance_type: str = "full",
        max_iter: int = 100,
        contamination: float = 0.1,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(contamination=contamination, random_state=random_state)

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.gmm_kwargs = kwargs
        self.gmm_: Optional[GaussianMixture] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "GMMDetector":
        """Fit GMM on training data."""
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.gmm_kwargs,
        )
        self.gmm_.fit(X)

        scores = self.decision_function(X)
        self.threshold = float(np.percentile(scores, (1 - self.contamination) * 100))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.decision_function(X)
        return np.where(scores > self.threshold, 1, 0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute negative log-likelihood as anomaly score."""
        if self.gmm_ is None:
            raise RuntimeError("Detector must be fitted before prediction")
        return -self.gmm_.score_samples(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        return self.decision_function(X)


class KernelDensityEstimator(BaseAnomalyDetector):
    """Kernel Density Estimation Anomaly Detection."""

    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        contamination: float = 0.1,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(contamination=contamination, random_state=random_state)

        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde_kwargs = kwargs
        self.kde_: Optional[KernelDensity] = None

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "KernelDensityEstimator":
        """Fit KDE on training data."""
        self.kde_ = KernelDensity(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            **self.kde_kwargs,
        )
        self.kde_.fit(X)

        scores = self.decision_function(X)
        self.threshold = float(np.percentile(scores, (1 - self.contamination) * 100))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        scores = self.decision_function(X)
        return np.where(scores > self.threshold, 1, 0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute negative log-density as anomaly score."""
        if self.kde_ is None:
            raise RuntimeError("Detector must be fitted before prediction")
        return -self.kde_.score_samples(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        return self.decision_function(X)


class NormalizingFlowDetector(BaseTorchAnomalyDetector):
    """
    Normalizing Flow-based Anomaly Detection.

    Learns an invertible transformation to a simple distribution,
    using log-probability as anomaly score.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_flows: int = 4,
        **kwargs,
    ):
        """
        Initialize Normalizing Flow detector.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for coupling layers
            n_flows: Number of flow layers
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, **kwargs)

        self.hidden_dim = hidden_dim
        self.n_flows = n_flows

        # Simple affine coupling flows
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(AffineCouplingLayer(input_dim, hidden_dim, i % 2 == 0))

        self.to(self.device)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through flow.

        Returns:
            z: Latent representation
            log_det: Log determinant of Jacobian
        """
        log_det_total = torch.zeros(x.size(0), device=self.device)
        z = x

        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det

        return z, log_det_total

    def inverse(self, z: Tensor) -> Tensor:
        """Inverse transformation."""
        x = z
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability."""
        z, log_det = self.forward(x)

        # Base distribution (standard Gaussian)
        log_pz = -0.5 * torch.sum(z**2, dim=1) - 0.5 * self.input_dim * np.log(
            2 * np.pi
        )

        # Change of variables formula
        log_px = log_pz + log_det

        return log_px

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "NormalizingFlowDetector":
        """Fit normalizing flow on training data."""
        self.train()
        dataloader = self._create_dataloader(X)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                log_prob = self.log_prob(batch_x)
                loss = -torch.mean(log_prob)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(dataloader):.6f}"
                )

        # Compute threshold
        self.eval()
        scores = self.decision_function(X)
        self.threshold = float(np.percentile(scores, (1 - self.contamination) * 100))
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels."""
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")

        scores = self.decision_function(X)
        return np.where(scores > self.threshold, 1, 0)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute negative log-likelihood as anomaly score."""
        self.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            log_prob = self.log_prob(X_tensor)
        return (-log_prob).cpu().numpy()


class RealNVPAnomaly(NormalizingFlowDetector):
    """
    RealNVP-based Anomaly Detection.

    Specific implementation of normalizing flows using RealNVP architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_coupling_layers: int = 6,
        **kwargs,
    ):
        """
        Initialize RealNVP anomaly detector.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            n_coupling_layers: Number of coupling layers
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, hidden_dim, n_coupling_layers, **kwargs)


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flows."""

    def __init__(self, dim: int, hidden_dim: int, mask_type: bool = True):
        super().__init__()
        self.dim = dim
        self.mask_type = mask_type

        # Mask for splitting
        self.register_buffer("mask", self._create_mask(dim, mask_type))

        # Scale and translation networks
        d_in = dim // 2 + dim % 2
        self.scale_net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - d_in),
            nn.Tanh(),
        )
        self.translation_net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - d_in),
        )

    def _create_mask(self, dim: int, mask_type: bool) -> Tensor:
        """Create alternating mask."""
        mask = torch.zeros(dim)
        mask[::2] = 1 if mask_type else 0
        mask[1::2] = 0 if mask_type else 1
        return mask

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)

        s = self.scale_net(x_masked[self.mask.bool()].view(x.size(0), -1))
        t = self.translation_net(x_masked[self.mask.bool()].view(x.size(0), -1))

        y_unmasked = (
            x_unmasked[~self.mask.bool()].view(x.size(0), -1) * torch.exp(s) + t
        )
        y = x_masked + (1 - self.mask) * y_unmasked.view(x.size(0), -1)

        log_det = torch.sum(s, dim=1)

        return y, log_det

    def inverse(self, y: Tensor) -> Tensor:
        """Inverse pass."""
        y_masked = y * self.mask
        y_unmasked = y * (1 - self.mask)

        s = self.scale_net(y_masked[self.mask.bool()].view(y.size(0), -1))
        t = self.translation_net(y_masked[self.mask.bool()].view(y.size(0), -1))

        x_unmasked = (
            y_unmasked[~self.mask.bool()].view(y.size(0), -1) - t
        ) * torch.exp(-s)
        x = y_masked + (1 - self.mask) * x_unmasked.view(y.size(0), -1)

        return x


# =============================================================================
# Anomaly Scoring Functions
# =============================================================================


def reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    metric: str = "mse",
    reduction: str = "mean",
) -> np.ndarray:
    """
    Compute reconstruction error.

    Args:
        original: Original data
        reconstructed: Reconstructed data
        metric: Error metric ('mse', 'mae', 'rmse')
        reduction: Reduction method ('mean', 'sum', 'max')

    Returns:
        Reconstruction errors per sample
    """
    if metric == "mse":
        error = (original - reconstructed) ** 2
    elif metric == "mae":
        error = np.abs(original - reconstructed)
    elif metric == "rmse":
        error = np.sqrt((original - reconstructed) ** 2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if reduction == "mean":
        return np.mean(error, axis=tuple(range(1, error.ndim)))
    elif reduction == "sum":
        return np.sum(error, axis=tuple(range(1, error.ndim)))
    elif reduction == "max":
        return np.max(error, axis=tuple(range(1, error.ndim)))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def mahalanobis_distance(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Mahalanobis distance.

    Args:
        X: Data points
        mean: Mean vector (computed if None)
        cov: Covariance matrix (computed if None)

    Returns:
        Mahalanobis distances
    """
    if mean is None:
        mean = np.mean(X, axis=0)

    if cov is None:
        cov = np.cov(X, rowvar=False)

    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))

    diff = X - mean
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    return distances


def isolation_score(
    X: np.ndarray,
    n_estimators: int = 100,
    max_samples: Union[int, str] = "auto",
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Compute isolation score based on tree path length.

    Args:
        X: Data points
        n_estimators: Number of trees
        max_samples: Number of samples per tree
        random_state: Random seed

    Returns:
        Isolation scores (higher = more anomalous)
    """
    clf = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
    )
    clf.fit(X)

    return -clf.score_samples(X)


def energy_score(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute energy score for out-of-distribution detection.

    Args:
        logits: Model logits
        temperature: Temperature parameter

    Returns:
        Energy scores (higher = more anomalous)
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def entropy_score(
    probs: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute entropy-based anomaly score.

    Args:
        probs: Probability distributions

    Returns:
        Entropy scores (higher = more uncertain)
    """
    if isinstance(probs, torch.Tensor):
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    else:
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

    return entropy


# =============================================================================
# Evaluation Metrics
# =============================================================================


def precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    """
    Compute precision at top-k anomalies.

    Args:
        y_true: True labels (1 for anomaly, 0 for normal)
        scores: Anomaly scores (higher = more anomalous)
        k: Number of top samples to consider

    Returns:
        Precision at k
    """
    top_k_indices = np.argsort(scores)[-k:]
    n_anomalies = np.sum(y_true[top_k_indices])
    return n_anomalies / k


def recall_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    """
    Compute recall at top-k anomalies.

    Args:
        y_true: True labels (1 for anomaly, 0 for normal)
        scores: Anomaly scores (higher = more anomalous)
        k: Number of top samples to consider

    Returns:
        Recall at k
    """
    total_anomalies = np.sum(y_true)

    if total_anomalies == 0:
        return 0.0

    top_k_indices = np.argsort(scores)[-k:]
    n_detected = np.sum(y_true[top_k_indices])

    return n_detected / total_anomalies


def f1_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> float:
    """
    Compute F1 score at a given threshold.

    Args:
        y_true: True labels (1 for anomaly, 0 for normal)
        scores: Anomaly scores (higher = more anomalous)
        threshold: Decision threshold (if None, uses median)

    Returns:
        F1 score
    """
    if threshold is None:
        threshold = np.median(scores)

    y_pred = (scores >= threshold).astype(int)

    return f1_score(y_true, y_pred)


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_anomaly_scores(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Anomaly Score Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot distribution of anomaly scores.

    Args:
        scores: Anomaly scores
        labels: True labels (optional)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        ax.hist(normal_scores, bins=50, alpha=0.6, label="Normal", color="blue")
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label="Anomaly", color="red")
        ax.legend()
    else:
        ax.hist(scores, bins=50, alpha=0.6, color="blue")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels (1 for anomaly, 0 for normal)
        scores: Anomaly scores
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_precision_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels (1 for anomaly, 0 for normal)
        scores: Anomaly scores
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, linewidth=2, label=f"PR Curve (AP = {ap:.3f})")

    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(
        baseline, linestyle="--", color="gray", label=f"Baseline ({baseline:.3f})"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def highlight_anomalies(
    data: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    contamination: float = 0.1,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """
    Highlight anomalies in time series or feature data.

    Args:
        data: Input data (n_samples, n_features) or (n_samples,) for 1D
        scores: Anomaly scores
        threshold: Decision threshold (if None, computed from contamination)
        contamination: Expected proportion of anomalies
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if threshold is None:
        threshold = np.percentile(scores, (1 - contamination) * 100)

    anomaly_indices = scores > threshold

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    if data.ndim == 1:
        axes[0].plot(data, color="blue", alpha=0.7, label="Data")
        axes[0].scatter(
            np.where(anomaly_indices)[0],
            data[anomaly_indices],
            color="red",
            s=50,
            zorder=5,
            label="Anomalies",
        )
    else:
        axes[0].plot(data[:, 0], color="blue", alpha=0.7, label="Data (Feature 1)")
        axes[0].scatter(
            np.where(anomaly_indices)[0],
            data[anomaly_indices, 0],
            color="red",
            s=50,
            zorder=5,
            label="Anomalies",
        )

    axes[0].set_ylabel("Value")
    axes[0].set_title("Data with Anomaly Highlighting")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(scores, color="green", alpha=0.7, label="Anomaly Score")
    axes[1].axhline(
        threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.3f})"
    )
    axes[1].fill_between(
        range(len(scores)),
        scores,
        threshold,
        where=anomaly_indices,
        color="red",
        alpha=0.3,
    )
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Anomaly Score")
    axes[1].set_title("Anomaly Scores")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Base classes
    "BaseAnomalyDetector",
    "BaseTorchAnomalyDetector",
    # Reconstruction-based
    "AutoencoderAnomalyDetector",
    "VAEAnomalyDetector",
    "DeepSVDD",
    "AnoGAN",
    "MemAE",
    # Distance-based
    "KNNAnomalyDetector",
    "LOFDetector",
    "IsolationForestDetector",
    "OCSVMDetector",
    # Density-based
    "GMMDetector",
    "KernelDensityEstimator",
    "NormalizingFlowDetector",
    "RealNVPAnomaly",
    "AffineCouplingLayer",
    # Scoring functions
    "reconstruction_error",
    "mahalanobis_distance",
    "isolation_score",
    "energy_score",
    "entropy_score",
    # Evaluation metrics
    "precision_at_k",
    "recall_at_k",
    "f1_at_threshold",
    # Visualization
    "plot_anomaly_scores",
    "plot_roc_curve",
    "plot_precision_recall",
    "highlight_anomalies",
]


class DeepSVDD(BaseTorchAnomalyDetector):
    """
    Deep One-Class Classification (Deep SVDD).

    Learns a neural network transformation that maps normal data
    close to a hypersphere center.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 32,
        activation: str = "relu",
        device: str = "cpu",
    ):
        super().__init__(input_dim, device)

        # Build network
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    self._get_activation(activation),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.network = nn.Sequential(*layers)

        # Hypersphere center
        self.center: Optional[Tensor] = None

        self.to(device)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning latent representation and anomaly score."""
        z = self.network(x)

        if self.center is None:
            return z, torch.zeros(x.size(0), device=x.device)

        # Distance from center as anomaly score
        score = torch.sum((z - self.center) ** 2, dim=-1)
        return z, score

    def fit(
        self,
        x: Union[np.ndarray, Tensor],
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        verbose: bool = True,
    ) -> "DeepSVDD":
        """Fit Deep SVDD on normal data."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Compute center as mean of initial forward pass
        self.eval()
        with torch.no_grad():
            x_init = x[: min(1000, len(x))].to(self.device)
            z_init = self.network(x_init)
            self.center = z_init.mean(dim=0)

        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                z = self.network(batch_x)
                loss = torch.sum((z - self.center) ** 2)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}"
                )

        self.is_fitted = True
        return self


class AnoGAN(nn.Module):
    """
    Anomaly Detection with Generative Adversarial Networks.

    Uses a trained GAN to generate normal data and measures
    reconstruction error in both pixel and feature space.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 100,
        generator_dims: List[int] = [128, 256],
        discriminator_dims: List[int] = [256, 128],
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        # Generator
        generator_layers = []
        prev_dim = latent_dim
        for dim in generator_dims:
            generator_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = dim
        generator_layers.append(nn.Linear(prev_dim, input_dim))
        self.generator = nn.Sequential(*generator_layers)

        # Discriminator
        discriminator_layers = []
        prev_dim = input_dim
        for dim in discriminator_dims:
            discriminator_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            prev_dim = dim
        discriminator_layers.append(nn.Linear(prev_dim, 1))
        self.discriminator = nn.Sequential(*discriminator_layers)

        self.to(device)
        self.is_fitted = False

    def generate(self, z: Tensor) -> Tensor:
        """Generate samples from latent code."""
        return self.generator(z)

    def discriminate(self, x: Tensor) -> Tensor:
        """Discriminate real vs fake."""
        return self.discriminator(x)

    def fit(
        self,
        x: Union[np.ndarray, Tensor],
        epochs: int = 100,
        batch_size: int = 256,
        g_lr: float = 2e-4,
        d_lr: float = 2e-4,
        verbose: bool = True,
    ) -> "AnoGAN":
        """Train the GAN on normal data."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999)
        )
        d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999)
        )

        criterion = nn.BCEWithLogitsLoss()

        self.train()
        for epoch in range(epochs):
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                batch_size = batch_x.size(0)

                # Train discriminator
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                d_optimizer.zero_grad()

                real_loss = criterion(self.discriminate(batch_x), real_labels)

                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_data = self.generate(z)
                fake_loss = criterion(
                    self.discriminate(fake_data.detach()), fake_labels
                )

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                d_optimizer.step()

                # Train generator
                g_optimizer.zero_grad()
                g_loss = criterion(self.discriminate(fake_data), real_labels)
                g_loss.backward()
                g_optimizer.step()

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
                )

        self.is_fitted = True
        return self

    def detect(
        self,
        x: Union[np.ndarray, Tensor],
        n_iterations: int = 100,
        learning_rate: float = 0.01,
        lambda_feat: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies by finding optimal latent code.

        Args:
            x: Input data
            n_iterations: Number of optimization steps
            learning_rate: Learning rate for latent code optimization
            lambda_feat: Weight for feature matching loss

        Returns:
            anomaly_scores: Anomaly scores
            reconstructed: Reconstructed samples
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)

        self.eval()

        # Initialize latent codes
        z = torch.randn(
            x.size(0), self.latent_dim, device=self.device, requires_grad=True
        )
        optimizer = torch.optim.Adam([z], lr=learning_rate)

        for _ in range(n_iterations):
            optimizer.zero_grad()

            x_hat = self.generate(z)

            # Residual loss
            residual_loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1).sum()

            # Feature matching loss
            feat_real = self.discriminator[:-1](x)
            feat_fake = self.discriminator[:-1](x_hat)
            feat_loss = (
                F.mse_loss(feat_fake, feat_real, reduction="none").mean(dim=-1).sum()
            )

            loss = residual_loss + lambda_feat * feat_loss
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x_hat = self.generate(z)
            scores = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

        return scores.cpu().numpy(), x_hat.cpu().numpy()


class MemAE(nn.Module):
    """
    Memory-Augmented Autoencoder for Anomaly Detection.

    Uses an external memory module to record prototypical normal patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        latent_dim: int = 32,
        mem_dim: int = 50,
        shrink_threshold: float = 1.0 / 50,
        device: str = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mem_dim = mem_dim
        self.shrink_threshold = shrink_threshold
        self.device = device

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Memory
        self.memory = nn.Parameter(torch.randn(mem_dim, latent_dim))
        nn.init.xavier_uniform_(self.memory)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(device)
        self.is_fitted = False

    def query_memory(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Query memory with attention mechanism.

        Returns:
            z_hat: Reconstructed latent code
            w: Attention weights
        """
        # Compute attention scores
        w = torch.matmul(z, self.memory.t())  # [batch, mem_dim]
        w = F.softmax(w, dim=-1)

        # Sparse addressing (shrinkage)
        if self.shrink_threshold > 0:
            w = F.relu(w - self.shrink_threshold)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

        # Read from memory
        z_hat = torch.matmul(w, self.memory)  # [batch, latent_dim]

        return z_hat, w

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Returns:
            x_hat: Reconstruction
            score: Anomaly score
            w: Attention weights
        """
        z = self.encoder(x)
        z_hat, w = self.query_memory(z)
        x_hat = self.decoder(z_hat)

        score = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

        return x_hat, score, w

    def fit(
        self,
        x: Union[np.ndarray, Tensor],
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        entropy_weight: float = 0.0002,
        verbose: bool = True,
    ) -> "MemAE":
        """Fit MemAE on normal data."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()
                x_hat, _, w = self.forward(batch_x)

                # Reconstruction loss
                recon_loss = F.mse_loss(x_hat, batch_x)

                # Entropy loss to encourage sparse memory access
                entropy = -torch.sum(w * torch.log(w + 1e-8), dim=-1).mean()

                loss = recon_loss + entropy_weight * entropy
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}"
                )

        self.is_fitted = True
        return self

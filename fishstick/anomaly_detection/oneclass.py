"""
One-Class Classification Module.

This module provides one-class classification methods for anomaly detection:
- One-Class SVM with various kernels
- Support Vector Data Description (SVDD)
- Kernel PCA-based one-class classification
- Gaussian Process one-class classifier
- Deep One-Class classification

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


@dataclass
class OneClassResult:
    """Container for one-class classification results."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float
    n_anomalies: int
    decision_values: Optional[np.ndarray] = None


class BaseOneClassClassifier(ABC):
    """Base class for one-class classifiers."""

    def __init__(self, contamination: float = 0.1, nu: float = 0.1):
        self.contamination = contamination
        self.nu = nu
        self.threshold: Optional[float] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseOneClassClassifier":
        """Fit the classifier on normal data."""
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

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values."""
        return self.score(X)

    def fit_predict(self, X: np.ndarray) -> OneClassResult:
        """Fit and predict in one call."""
        self.fit(X)
        scores = self.score(X)
        labels = self.predict(X)
        return OneClassResult(
            scores=scores,
            labels=labels,
            threshold=self.threshold,
            n_anomalies=int(np.sum(labels)),
            decision_values=scores,
        )


class KernelOneClassSVM(BaseOneClassClassifier):
    """
    Kernel One-Class SVM for anomaly detection.

    Uses kernel trick to learn a boundary around normal data.
    Points outside the boundary are flagged as anomalies.

    Parameters
    ----------
    kernel : str
        Kernel type: 'rbf', 'linear', 'poly', 'sigmoid'.
    gamma : float or str
        Kernel coefficient. 'scale' = 1 / (n_features * X.var()).
    nu : float
        Upper bound on fraction of outliers.
    contamination : float
        Expected proportion of outliers.
    degree : int
        Degree for polynomial kernel.
    coef0 : float
        Independent term for polynomial/sigmoid kernels.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: Union[float, str] = "scale",
        nu: float = 0.1,
        contamination: float = 0.1,
        degree: int = 3,
        coef0: float = 0.0,
    ):
        super().__init__(contamination, nu)
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.degree = degree
        self.coef0 = coef0
        self.scaler = StandardScaler()
        self.model: Optional[OneClassSVM] = None

    def fit(self, X: np.ndarray) -> "KernelOneClassSVM":
        """Fit the One-Class SVM on normal data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model = OneClassSVM(
            kernel=self.kernel,
            gamma=self.gamma,
            nu=self.nu,
            degree=self.degree,
            coef0=self.coef0,
        )
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (negative of decision function)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        X_scaled = self.scaler.transform(X)
        decision = self.model.decision_function(X_scaled)
        return -decision  # Negate so higher = more anomalous


class SVDDClassifier(BaseOneClassClassifier):
    """
    Support Vector Data Description (SVDD).

    Finds the smallest hypersphere that encloses normal data.
    Points outside the sphere are flagged as anomalies.

    Parameters
    ----------
    C : float
        Regularization parameter (trade-off between sphere size and errors).
    gamma : float
        Kernel coefficient.
    kernel : str
        Kernel type: 'rbf', 'linear', 'poly'.
    contamination : float
        Expected proportion of outliers.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: float = 0.1,
        kernel: str = "rbf",
        contamination: float = 0.1,
    ):
        super().__init__(contamination, nu=0.1)
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.scaler = StandardScaler()
        self.center: Optional[np.ndarray] = None
        self.radius: float = 0.0
        self.alpha: Optional[np.ndarray] = None
        self.support_vectors: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SVDDClassifier":
        """Fit SVDD by finding the smallest enclosing sphere."""
        X_scaled = self.scaler.fit_transform(X)
        n_samples = X_scaled.shape[0]

        K = self._compute_kernel(X_scaled, X_scaled)
        ones = np.ones(n_samples)
        P = K
        q = -np.diag(K)

        A_eq = ones.T
        b_eq = 1.0

        bounds = [(0, self.C) for _ in range(n_samples)]

        from scipy.optimize import minimize

        def objective(alpha):
            return 0.5 * alpha @ (P * np.outer(ones, ones)) @ alpha + q @ alpha

        constraints = {"type": "eq", "fun": lambda a: ones @ a - 1}
        result = minimize(
            objective,
            np.zeros(n_samples),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        self.alpha = result.x
        self.support_vectors = X_scaled[self.alpha > 1e-7]

        self.center = (
            np.sum(
                self.alpha[:, np.newaxis] * X_scaled * K[:, :, np.newaxis], axis=(0, 1)
            )
            / self.alpha.sum()
        )

        distances = np.linalg.norm(self.support_vectors - self.center, axis=1)
        self.radius = np.max(distances) if len(distances) > 0 else 1.0

        self.is_fitted = True
        return self

    def _compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix."""
        if self.kernel == "rbf":
            pairwise_sq_dists = (
                np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * X @ Y.T
            )
            return np.exp(-self.gamma * pairwise_sq_dists)
        elif self.kernel == "linear":
            return X @ Y.T
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute distance from center as anomaly score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        X_scaled = self.scaler.transform(X)
        distances = np.linalg.norm(X_scaled - self.center, axis=1)
        return np.maximum(distances - self.radius, 0)


class KernelPCAOneClass(BaseOneClassClassifier):
    """
    Kernel PCA based one-class classification.

    Projects data to kernel PCA space and detects anomalies
    based on reconstruction error in that space.

    Parameters
    ----------
    n_components : int
        Number of kernel PCA components.
    kernel : str
        Kernel type: 'rbf', 'linear', 'poly'.
    gamma : float
        Kernel coefficient.
    contamination : float
        Expected proportion of outliers.
    """

    def __init__(
        self,
        n_components: int = 10,
        kernel: str = "rbf",
        gamma: Union[float, str] = "scale",
        contamination: float = 0.1,
    ):
        super().__init__(contamination, nu=0.1)
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.kpca: Optional[KernelPCA] = None

    def fit(self, X: np.ndarray) -> "KernelPCAOneClass":
        """Fit kernel PCA on normal data."""
        X_scaled = self.scaler.fit_transform(X)
        self.kpca = KernelPCA(
            n_components=min(self.n_components, X.shape[1]),
            kernel=self.kernel,
            gamma=self.gamma,
        )
        self.latent_train = self.kpca.fit_transform(X_scaled)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction error in kernel PCA space."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        X_scaled = self.scaler.transform(X)
        latent_new = self.kpca.transform(X_scaled)

        reconstructed = self.kpca.inverse_transform(latent_new)
        errors = np.linalg.norm(X_scaled - reconstructed, axis=1)
        return errors


class NystroemOneClassSVM(BaseOneClassClassifier):
    """
    Nystroem approximation for scalable One-Class SVM.

    Uses Nystroem method to approximate kernel SVM for large datasets.

    Parameters
    ----------
    n_components : int
        Number of Nystroem components (random Fourier features).
    gamma : float
        Kernel coefficient.
    kernel : str
        Kernel type: 'rbf', 'linear'.
    nu : float
        Upper bound on fraction of outliers.
    contamination : float
        Expected proportion of outliers.
    """

    def __init__(
        self,
        n_components: int = 100,
        gamma: float = 0.1,
        kernel: str = "rbf",
        nu: float = 0.1,
        contamination: float = 0.1,
    ):
        super().__init__(contamination, nu)
        self.n_components = n_components
        self.gamma = gamma
        self.kernel = kernel
        self.scaler = StandardScaler()
        self.nystroem: Optional[Nystroem] = None
        self.clf: Optional[OneClassSVM] = None

    def fit(self, X: np.ndarray) -> "NystroemOneClassSVM":
        """Fit Nystroem approximation then One-Class SVM."""
        X_scaled = self.scaler.fit_transform(X)

        self.nystroem = Nystroem(
            kernel=self.kernel,
            gamma=self.gamma,
            n_components=min(self.n_components, X.shape[0]),
            random_state=42,
        )
        X_transformed = self.nystroem.fit_transform(X_scaled)

        self.clf = OneClassSVM(kernel="linear", nu=self.nu)
        self.clf.fit(X_transformed)
        self.is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        X_scaled = self.scaler.transform(X)
        X_transformed = self.nystroem.transform(X_scaled)
        decision = self.clf.decision_function(X_transformed)
        return -decision


class DeepSVDDOneClass(nn.Module):
    """
    Deep SVDD for one-class classification.

    Uses a neural network to learn a hypersphere that encloses
    normal data in the latent space.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list
        Hidden layer dimensions.
    latent_dim : int
        Latent space dimension.
    activation : str
        Activation function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        latent_dim: int = 16,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.ReLU

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    act_fn(),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

        self.center = nn.Parameter(torch.zeros(latent_dim))
        self.radius = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder."""
        return self.encoder(x)

    def get_latent(self, x: Tensor) -> Tensor:
        """Get latent representation."""
        with torch.no_grad():
            return self.forward(x)


class DeepSVDDClassifier(BaseOneClassClassifier):
    """
    Deep Support Vector Data Description.

    Uses deep neural network to learn compact representation
    of normal data in latent space.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dims : list
        Hidden layer dimensions.
    latent_dim : int
        Latent space dimension.
    contamination : float
        Expected proportion of outliers.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    device : str
        Device to use.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        latent_dim: int = 16,
        contamination: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        device: str = "auto",
    ):
        super().__init__(contamination, nu=0.1)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = self._get_device(device)

        self.scaler = StandardScaler()
        self.model = DeepSVDDOneClass(input_dim, hidden_dims, latent_dim).to(
            self.device
        )

    def _get_device(self, device: str) -> torch.device:
        """Get torch device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def fit(self, X: np.ndarray) -> "DeepSVDDClassifier":
        """Train Deep SVDD on normal data."""
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            indices = torch.randperm(len(X_tensor))
            epoch_loss = 0.0

            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[indices[i : i + self.batch_size]]
                latent = self.model(batch)

                dist = torch.sum((latent - self.model.center) ** 2, dim=1)
                loss = torch.mean(torch.clamp(dist - self.model.radius**2, min=0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        self._set_radius(X_tensor)
        self.is_fitted = True
        return self

    def _set_radius(self, X: Tensor) -> None:
        """Set radius based on training data distances."""
        self.model.eval()
        with torch.no_grad():
            latent = self.model(X)
            distances = torch.sum((latent - self.model.center) ** 2, dim=1)
            self.model.radius.data = torch.quantile(distances, 1 - self.contamination)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute distance from center in latent space."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            latent = self.model(X_tensor)
            distances = torch.sum((latent - self.model.center) ** 2, dim=1)
            return distances.cpu().numpy()


class OneClassForest(BaseOneClassClassifier):
    """
    One-Class Random Forest for anomaly detection.

    Uses ensemble of random trees where anomalies are points
    that fall quickly into leaf nodes (isolated quickly).

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_samples : int
        Max samples per tree.
    contamination : float
        Expected proportion of outliers.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Optional[int] = None,
        contamination: float = 0.1,
        random_state: Optional[int] = None,
    ):
        super().__init__(contamination, nu=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.trees: List[Dict[str, Any]] = []

    def fit(self, X: np.ndarray) -> "OneClassForest":
        """Fit ensemble of random trees."""
        X_scaled = self.scaler.fit_transform(X)
        n_samples = X_scaled.shape[0]

        if self.max_samples is None:
            self.max_samples = min(256, n_samples)

        np.random.seed(self.random_state)

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            X_boot = X_scaled[indices]
            tree = self._build_tree(X_boot, depth=0, max_depth=10)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def _build_tree(
        self,
        X: np.ndarray,
        depth: int,
        max_depth: int,
    ) -> Dict[str, Any]:
        """Build a random binary tree."""
        if len(X) <= 1 or depth >= max_depth:
            return {"leaf": True, "size": len(X)}

        feat_idx = np.random.randint(X.shape[1])
        split_val = np.random.uniform(X[:, feat_idx].min(), X[:, feat_idx].max())

        left_mask = X[:, feat_idx] <= split_val
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": feat_idx,
            "split": split_val,
            "left": self._build_tree(X[left_mask], depth + 1, max_depth),
            "right": self._build_tree(X[right_mask], depth + 1, max_depth),
        }

    def _get_path_length(self, x: np.ndarray, tree: Dict, depth: int = 0) -> float:
        """Get path length for a point in a tree."""
        if tree.get("leaf", False):
            return depth + self._c_factor(tree["size"])

        if x[tree["feature"]] <= tree["split"]:
            return self._get_path_length(x, tree["left"], depth + 1)
        else:
            return self._get_path_length(x, tree["right"], depth + 1)

    def _c_factor(self, n: int) -> float:
        """Compute harmonic number approximation."""
        if n <= 1:
            return 1.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute average path length (anomaly score)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        X_scaled = self.scaler.transform(X)
        path_lengths = np.zeros(len(X_scaled))

        for tree in self.trees:
            for i, x in enumerate(X_scaled):
                path_lengths[i] += self._get_path_length(x, tree)

        path_lengths /= self.n_estimators
        return path_lengths


__all__ = [
    "BaseOneClassClassifier",
    "OneClassResult",
    "KernelOneClassSVM",
    "SVDDClassifier",
    "KernelPCAOneClass",
    "NystroemOneClassSVM",
    "DeepSVDDOneClass",
    "DeepSVDDClassifier",
    "OneClassForest",
]

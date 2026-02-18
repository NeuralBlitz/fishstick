"""
Feature Engineering Utilities Module for fishstick

Provides comprehensive feature engineering tools including polynomial features,
interaction features, binning, target encoding, and automated feature selection.

Features:
- Polynomial feature generation
- Feature interactions
- Discretization/binning
- Target encoding
- Automated feature selection
- PCA-based features
"""

from __future__ import annotations

from typing import (
    Optional,
    Callable,
    List,
    Union,
    Dict,
    Any,
    Tuple,
    Sequence,
    TypeVar,
    Generic,
)
from dataclasses import dataclass, field
from enum import Enum
import warnings
import numpy as np
import torch
from torch import Tensor
from scipy import stats
from sklearn.decomposition import PCA as SKPCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler


T = TypeVar("T")
ArrayLike = Union[np.ndarray, Tensor]


class PolynomialFeatures:
    """
    Generate polynomial and interaction features.

    Creates polynomial features up to specified degree and includes
    interaction terms between features.
    """

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = True,
        order: str = "C",
    ):
        """
        Args:
            degree: Maximum polynomial degree
            interaction_only: If True, only interaction terms
            include_bias: Include bias term
            order: Output array order
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order
        self._n_features: int = 0

    def fit(self, X: ArrayLike) -> "PolynomialFeatures":
        """Fit the transformer."""
        X_arr = self._to_numpy(X)
        self._n_features = X_arr.shape[1]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform to polynomial features."""
        X_arr = self._to_numpy(X)

        if self._n_features == 0:
            self.fit(X_arr)

        n_samples = X_arr.shape[0]
        n_features = self._n_features

        if self.degree == 1:
            return X_arr

        features = (
            [X_arr] if not self.include_bias else [np.ones((n_samples, 1)), X_arr]
        )

        for d in range(2, self.degree + 1):
            if self.interaction_only:
                combs = self._interaction_combinations(n_features, d)
            else:
                combs = self._polynomial_combinations(n_features, d)

            if combs:
                new_features = X_arr[:, combs].prod(axis=2)
                features.append(new_features)

        return np.hstack(features)

    def _to_numpy(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, Tensor):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def _polynomial_combinations(self, n_features: int, degree: int) -> List[Tuple]:
        indices = np.arange(n_features)
        result = []

        for combo in self._combinations_with_replacement(indices, degree):
            result.append(combo)

        return result

    def _interaction_combinations(self, n_features: int, degree: int) -> List[Tuple]:
        indices = np.arange(n_features)
        result = []

        for combo in self._combinations_without_replacement(indices, degree):
            if len(set(combo)) == degree:
                result.append(combo)

        return result

    def _combinations_with_replacement(
        self, indices: np.ndarray, r: int
    ) -> List[Tuple]:
        from itertools import combinations_with_replacement

        return list(combinations_with_replacement(indices, r))

    def _combinations_without_replacement(
        self, indices: np.ndarray, r: int
    ) -> List[Tuple]:
        from itertools import combinations

        return list(combinations(indices, r))

    def __call__(self, X: ArrayLike) -> np.ndarray:
        return self.transform(X)


class InteractionFeatures:
    """
    Generate feature interaction terms.

    Creates pairwise and higher-order interaction features between
    specified feature groups.
    """

    def __init__(
        self,
        interaction_degree: int = 2,
        max_features: Optional[int] = None,
        sparse: bool = False,
    ):
        """
        Args:
            interaction_degree: Maximum interaction order
            max_features: Maximum number of features to use
            sparse: Return sparse matrix
        """
        self.interaction_degree = interaction_degree
        self.max_features = max_features
        self.sparse = sparse

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "InteractionFeatures":
        """Fit the transformer."""
        X_arr = self._to_numpy(X)
        self._n_features = X_arr.shape[1]

        if self.max_features and self._n_features > self.max_features:
            self._selected_indices = self._select_features(X_arr, y)
        else:
            self._selected_indices = np.arange(self._n_features)

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Generate interaction features."""
        X_arr = self._to_numpy(X)[:, self._selected_indices]

        interactions = [X_arr]

        for degree in range(2, self.interaction_degree + 1):
            combos = self._generate_combinations(degree)
            for combo in combos:
                interaction = X_arr[:, list(combo)].prod(axis=1, keepdims=True)
                interactions.append(interaction)

        return np.hstack(interactions)

    def _to_numpy(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, Tensor):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def _select_features(self, X: np.ndarray, y: Optional[np.ndarray]) -> np.ndarray:
        if y is None:
            return np.random.choice(X.shape[1], self.max_features, replace=False)

        scores = mutual_info_classif(X, y, random_state=42)
        return np.argsort(scores)[-self.max_features :]

    def _generate_combinations(self, degree: int) -> List[Tuple]:
        from itertools import combinations

        n = len(self._selected_indices)
        return list(combinations(range(n), degree))

    def __call__(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        if not hasattr(self, "_selected_indices"):
            self.fit(X, y)
        return self.transform(X)


class BinningTransformer:
    """
    Discretize continuous features into bins.

    Supports quantile, uniform, and k-means binning strategies.
    """

    class Strategy(Enum):
        UNIFORM = "uniform"
        QUANTILE = "quantile"
        KMEANS = "kmeans"

    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = "quantile",
        encode: str = "ordinal",
        smooth: bool = False,
    ):
        """
        Args:
            n_bins: Number of bins
            strategy: Binning strategy
            encode: Encoding method (onehot, ordinal)
            smooth: Apply label smoothing
        """
        self.n_bins = n_bins
        self.strategy = self.Strategy(strategy)
        self.encode = encode
        self.smooth = smooth
        self._bin_edges: Dict[int, np.ndarray] = {}
        self._fitted = False

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "BinningTransformer":
        """Fit the binner."""
        X_arr = self._to_numpy(X)
        self._n_features = X_arr.shape[1]

        for i in range(self._n_features):
            self._bin_edges[i] = self._compute_bins(X_arr[:, i])

        self._fitted = True
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform to binned features."""
        X_arr = self._to_numpy(X)
        result = np.zeros_like(X_arr, dtype=int)

        for i in range(self._n_features):
            result[:, i] = np.digitize(X_arr[:, i], self._bin_edges[i]) - 1
            result[:, i] = np.clip(result[:, i], 0, self.n_bins - 1)

        if self.encode == "onehot":
            return self._onehot_encode(result)

        return result

    def _compute_bins(self, x: np.ndarray) -> np.ndarray:
        if self.strategy == self.Strategy.UNIFORM:
            return np.linspace(x.min(), x.max(), self.n_bins + 1)
        elif self.strategy == self.Strategy.QUANTILE:
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            return np.percentile(x, quantiles)
        else:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=self.n_bins, random_state=42).fit(
                x.reshape(-1, 1)
            )
            return np.sort(kmeans.cluster_centers_.flatten())

    def _onehot_encode(self, x: np.ndarray) -> np.ndarray:
        n_samples, n_features = x.shape
        result = np.zeros((n_samples, n_features * self.n_bins))

        for i in range(n_features):
            for j in range(self.n_bins):
                mask = x[:, i] == j
                result[mask, i * self.n_bins + j] = 1

        return result

    def _to_numpy(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, Tensor):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def __call__(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        if not self._fitted:
            self.fit(X, y)
        return self.transform(X)


class TargetEncoder:
    """
    Target encoding for categorical features.

    Encodes categorical features using target statistics with optional
    smoothing to prevent overfitting.
    """

    def __init__(
        self,
        smoothing: float = 1.0,
        min_samples: int = 1,
        noise: float = 0.0,
    ):
        """
        Args:
            smoothing: Smoothing factor
            min_samples: Minimum samples for encoding
            noise: Gaussian noise to add
        """
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.noise = noise
        self._encodings: Dict[int, Dict[Any, float]] = {}
        self._global_mean: float = 0.0
        self._fitted = False

    def fit(self, X: ArrayLike, y: ArrayLike) -> "TargetEncoder":
        """Fit the encoder."""
        X_arr = self._to_numpy(X)
        y_arr = self._to_numpy(y).ravel()

        self._global_mean = y_arr.mean()
        self._n_features = X_arr.shape[1]

        for i in range(self._n_features):
            self._encodings[i] = self._compute_encoding(X_arr[:, i], y_arr)

        self._fitted = True
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform to target-encoded features."""
        X_arr = self._to_numpy(X)
        result = np.zeros_like(X_arr, dtype=float)

        for i in range(self._n_features):
            for j, val in enumerate(X_arr[:, i]):
                result[j, i] = self._encodings[i].get(val, self._global_mean)

        if self.noise > 0:
            result += np.random.normal(0, self.noise, result.shape)

        return result

    def _compute_encoding(self, x: np.ndarray, y: np.ndarray) -> Dict[Any, float]:
        unique_vals = np.unique(x)
        encoding = {}

        for val in unique_vals:
            mask = x == val
            n = mask.sum()

            if n < self.min_samples:
                encoding[val] = self._global_mean
            else:
                mean = y[mask].mean()
                smooth = 1 / (1 + np.exp(-(n - self.smoothing) / self.smoothing))
                encoding[val] = smooth * mean + (1 - smooth) * self._global_mean

        return encoding

    def _to_numpy(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, Tensor):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def __call__(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        if not self._fitted:
            if y is None:
                raise ValueError("y required for fitting")
            self.fit(X, y)
        return self.transform(X)


class FeatureSelector:
    """
    Automated feature selection using various methods.

    Supports variance threshold, correlation, and mutual information.
    """

    class Method(Enum):
        VARIANCE = "variance"
        CORRELATION = "correlation"
        MUTUAL_INFO = "mutual_info"
        RECURSIVE = "recursive"

    def __init__(
        self,
        method: str = "variance",
        threshold: float = 0.0,
        k: Optional[int] = None,
    ):
        """
        Args:
            method: Selection method
            threshold: Threshold for filtering
            k: Number of features to select
        """
        self.method = self.Method(method)
        self.threshold = threshold
        self.k = k
        self._selected_indices: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "FeatureSelector":
        """Fit the selector."""
        X_arr = self._to_numpy(X)
        self._n_features = X_arr.shape[1]

        if self.method == self.Method.VARIANCE:
            variances = X_arr.var(axis=0)
            self._selected_indices = np.where(variances > self.threshold)[0]
        elif self.method == self.Method.CORRELATION:
            self._selected_indices = self._correlation_selection(X_arr)
        elif self.method == self.Method.MUTUAL_INFO:
            if y is None:
                raise ValueError("y required for mutual info selection")
            self._selected_indices = self._mutual_info_selection(X_arr, y)
        else:
            self._selected_indices = np.arange(self._n_features)

        if self.k and len(self._selected_indices) > self.k:
            self._selected_indices = self._selected_indices[: self.k]

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform to selected features."""
        X_arr = self._to_numpy(X)
        return X_arr[:, self._selected_indices]

    def _correlation_selection(self, X: np.ndarray) -> np.ndarray:
        corr_matrix = np.corrcoef(X.T)
        n = corr_matrix.shape[0]
        to_remove = set()

        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > self.threshold:
                    to_remove.add(i if X[:, i].var() < X[:, j].var() else j)

        return np.array([i for i in range(n) if i not in to_remove])

    def _mutual_info_selection(self, X: np.ndarray, y: ArrayLike) -> np.ndarray:
        y_arr = self._to_numpy(y).ravel()
        scores = mutual_info_classif(X, y_arr, random_state=42)
        indices = np.argsort(scores)[::-1]
        return indices[: self.k] if self.k else indices

    def _to_numpy(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, Tensor):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def __call__(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        if not self._fitted:
            self.fit(X, y)
        return self.transform(X)


class PCAFeatures:
    """
    PCA-based feature extraction.

    Extracts principal components as new features.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
        standardize: bool = True,
    ):
        """
        Args:
            n_components: Number of components
            whiten: Whether to whiten
            standardize: Whether to standardize first
        """
        self.n_components = n_components
        self.whiten = whiten
        self.standardize = standardize
        self._pca: Optional[SKPCA] = None
        self._scaler: Optional[StandardScaler] = None

    def fit(self, X: ArrayLike) -> "PCAFeatures":
        """Fit the transformer."""
        X_arr = self._to_numpy(X)

        if self.standardize:
            self._scaler = StandardScaler()
            X_arr = self._scaler.fit_transform(X_arr)

        self._pca = SKPCA(
            n_components=self.n_components,
            whiten=self.whiten,
        )
        self._pca.fit(X_arr)

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform to PCA features."""
        X_arr = self._to_numpy(X)

        if self.standardize and self._scaler:
            X_arr = self._scaler.transform(X_arr)

        return self._pca.transform(X_arr)

    def _to_numpy(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, Tensor):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def __call__(self, X: ArrayLike) -> np.ndarray:
        if self._pca is None:
            self.fit(X)
        return self.transform(X)


class FourierFeatures:
    """
    Random Fourier features for kernel approximation.

    Approximates RBF kernel features using random Fourier basis.
    """

    def __init__(
        self,
        n_components: int = 100,
        gamma: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            n_components: Number of Fourier features
            gamma: RBF kernel parameter
            random_state: Random seed
        """
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self._W: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike) -> "FourierFeatures":
        """Fit the transformer."""
        X_arr = self._to_numpy(X)
        n_features = X_arr.shape[1]

        rng = np.random.RandomState(self.random_state)
        self._W = rng.normal(
            0, np.sqrt(2 * self.gamma), (n_features, self.n_components)
        )
        self._b = rng.uniform(0, 2 * np.pi, self.n_components)

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform to Fourier features."""
        X_arr = self._to_numpy(X)
        projection = X_arr @ self._W + self._b
        return np.sqrt(2 / self.n_components) * np.cos(projection)

    def _to_numpy(self, X: ArrayLike) -> np.ndarray:
        if isinstance(X, Tensor):
            return X.detach().cpu().numpy()
        return np.asarray(X)

    def __call__(self, X: ArrayLike) -> np.ndarray:
        if self._W is None:
            self.fit(X)
        return self.transform(X)


@dataclass
class FeatureStatistics:
    """Statistics for feature analysis."""

    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    median: np.ndarray
    skewness: Optional[np.ndarray] = None
    kurtosis: Optional[np.ndarray] = None
    n_unique: Optional[np.ndarray] = None
    n_missing: int = 0


def compute_feature_statistics(X: ArrayLike) -> FeatureStatistics:
    """
    Compute comprehensive statistics for features.

    Args:
        X: Input features

    Returns:
        FeatureStatistics object
    """
    X_arr = X if isinstance(X, np.ndarray) else X.detach().cpu().numpy()

    return FeatureStatistics(
        mean=X_arr.mean(axis=0),
        std=X_arr.std(axis=0),
        min=X_arr.min(axis=0),
        max=X_arr.max(axis=0),
        median=np.median(X_arr, axis=0),
        skewness=stats.skew(X_arr, axis=0),
        kurtosis=stats.kurtosis(X_arr, axis=0),
        n_unique=np.array([len(np.unique(X_arr[:, i])) for i in range(X_arr.shape[1])]),
    )

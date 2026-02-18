"""
Feature Engineering Module for fishstick

Comprehensive feature engineering utilities including:
- Numerical Features: Scaling, normalization, binning, transformations
- Categorical Features: Encoding strategies
- Text Features: Vectorization and embeddings
- Time Features: Date extraction, cyclical encoding, lag/rolling features
- Feature Selection: Various selection methods
- Feature Generation: Automated feature generation
- Utilities: Pipelines and transformers

Mathematical Foundations:
- Information theory for feature selection
- Symplectic geometry for feature interactions
- Category theory for feature composition
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    Tuple,
    Set,
    Iterator,
    Sequence,
)
from dataclasses import dataclass, field
from enum import Enum, auto
import warnings
from collections import defaultdict
import hashlib
import json

import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Try to import optional dependencies
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import (
        StandardScaler as SklearnStandardScaler,
        MinMaxScaler as SklearnMinMaxScaler,
        RobustScaler as SklearnRobustScaler,
        Normalizer as SklearnNormalizer,
        OneHotEncoder as SklearnOneHotEncoder,
        LabelEncoder as SklearnLabelEncoder,
        OrdinalEncoder as SklearnOrdinalEncoder,
        PolynomialFeatures as SklearnPolynomialFeatures,
    )
    from sklearn.feature_selection import (
        VarianceThreshold as SklearnVarianceThreshold,
        SelectKBest as SklearnSelectKBest,
        mutual_info_classif,
        mutual_info_regression,
        f_classif,
        f_regression,
        chi2,
    )
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Lasso, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    # Define dummy base classes if sklearn not available
    class BaseEstimator:
        pass

    class TransformerMixin:
        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)


try:
    from scipy import stats
    from scipy.sparse import csr_matrix, issparse

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from gensim.models import Word2Vec

    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Import fishstick core types
try:
    from ..core.types import MetricTensor
    from ..geometric.fisher import FisherInformationMetric

    FISHSTICK_CORE_AVAILABLE = True
except ImportError:
    FISHSTICK_CORE_AVAILABLE = False


# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================


class ScalerType(Enum):
    """Types of scalers for numerical features."""

    STANDARD = auto()
    MINMAX = auto()
    ROBUST = auto()
    MAXABS = auto()
    NORMALIZER = auto()
    QUANTILE = auto()
    POWER = auto()


class EncoderType(Enum):
    """Types of encoders for categorical features."""

    ONEHOT = auto()
    LABEL = auto()
    ORDINAL = auto()
    TARGET = auto()
    HASHING = auto()
    BINARY = auto()
    FREQUENCY = auto()


class SelectionMethod(Enum):
    """Feature selection methods."""

    VARIANCE = auto()
    KBEST = auto()
    RFE = auto()
    LASSO = auto()
    MUTUAL_INFO = auto()
    PCA = auto()
    CORRELATION = auto()


class TextVectorizerType(Enum):
    """Text vectorization methods."""

    TFIDF = auto()
    COUNT = auto()
    HASHING = auto()


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    numerical_strategy: ScalerType = ScalerType.STANDARD
    categorical_strategy: EncoderType = EncoderType.ONEHOT
    text_strategy: TextVectorizerType = TextVectorizerType.TFIDF
    selection_method: SelectionMethod = SelectionMethod.VARIANCE
    max_features: int = 100
    handle_missing: str = "mean"
    random_state: int = 42

    # Numerical settings
    binning_bins: int = 10
    polynomial_degree: int = 2

    # Categorical settings
    target_encoding_smoothing: float = 1.0
    hashing_n_features: int = 100

    # Text settings
    max_df: float = 0.95
    min_df: float = 0.01
    ngram_range: Tuple[int, int] = (1, 2)

    # Time settings
    max_lag: int = 10
    rolling_window: int = 5

    # Selection settings
    variance_threshold: float = 0.01
    k_best_k: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numerical_strategy": self.numerical_strategy.name,
            "categorical_strategy": self.categorical_strategy.name,
            "text_strategy": self.text_strategy.name,
            "selection_method": self.selection_method.name,
            "max_features": self.max_features,
            "handle_missing": self.handle_missing,
            "random_state": self.random_state,
        }


# ============================================================================
# BASE CLASSES
# ============================================================================


class BaseFeatureTransformer(ABC):
    """Base class for all feature transformers."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._is_fitted = False
        self._feature_names_out: Optional[List[str]] = None

    @abstractmethod
    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "BaseFeatureTransformer":
        """Fit the transformer to the data."""
        pass

    @abstractmethod
    def transform(
        self, X: Union[NDArray, pd.DataFrame]
    ) -> Union[NDArray, pd.DataFrame]:
        """Transform the data."""
        pass

    def fit_transform(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> Union[NDArray, pd.DataFrame]:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        """Get output feature names."""
        if self._feature_names_out is not None:
            return self._feature_names_out
        if input_features is not None:
            return input_features
        return []

    def _validate_input(self, X: Union[NDArray, pd.DataFrame]) -> pd.DataFrame:
        """Validate and convert input to DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

    def _check_is_fitted(self):
        """Check if transformer is fitted."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet. Call fit first.")


# ============================================================================
# NUMERICAL FEATURES
# ============================================================================


class StandardScaler(BaseFeatureTransformer):
    """
    Z-score normalization: (x - mean) / std

    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(
        self, with_mean: bool = True, with_std: bool = True, name: Optional[str] = None
    ):
        super().__init__(name)
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[NDArray] = None
        self.std_: Optional[NDArray] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "StandardScaler":
        X = self._validate_input(X)
        self.mean_ = X.mean(axis=0).values if self.with_mean else np.zeros(X.shape[1])
        self.std_ = X.std(axis=0).values if self.with_std else np.ones(X.shape[1])
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        X_scaled = (X.values - self.mean_) / self.std_
        return X_scaled


class MinMaxScaler(BaseFeatureTransformer):
    """
    Min-max scaling: (x - min) / (max - min)

    Scales features to a given range (default [0, 1]).
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.feature_range = feature_range
        self.min_: Optional[NDArray] = None
        self.scale_: Optional[NDArray] = None
        self.data_min_: Optional[NDArray] = None
        self.data_max_: Optional[NDArray] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "MinMaxScaler":
        X = self._validate_input(X)
        self.data_min_ = X.min(axis=0).values
        self.data_max_ = X.max(axis=0).values
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (
            self.data_max_ - self.data_min_ + 1e-8
        )
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        X_scaled = X.values * self.scale_ + self.min_
        return np.clip(X_scaled, self.feature_range[0], self.feature_range[1])


class RobustScaler(BaseFeatureTransformer):
    """
    Robust scaling using median and IQR.

    Uses statistics that are robust to outliers:
    center = median, scale = IQR (75th percentile - 25th percentile)
    """

    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_: Optional[NDArray] = None
        self.scale_: Optional[NDArray] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "RobustScaler":
        X = self._validate_input(X)
        if self.with_centering:
            self.center_ = X.median(axis=0).values
        else:
            self.center_ = np.zeros(X.shape[1])

        if self.with_scaling:
            q_low = X.quantile(self.quantile_range[0] / 100.0, axis=0).values
            q_high = X.quantile(self.quantile_range[1] / 100.0, axis=0).values
            self.scale_ = q_high - q_low
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
        else:
            self.scale_ = np.ones(X.shape[1])

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        X_scaled = (X.values - self.center_) / self.scale_
        return X_scaled


class Normalizer(BaseFeatureTransformer):
    """
    Normalize samples to unit norm.

    Each sample (row) is normalized independently.
    """

    def __init__(self, norm: str = "l2", name: Optional[str] = None):
        super().__init__(name)
        self.norm = norm

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "Normalizer":
        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        X_arr = X.values

        if self.norm == "l1":
            norms = np.sum(np.abs(X_arr), axis=1, keepdims=True)
        elif self.norm == "l2":
            norms = np.sqrt(np.sum(X_arr**2, axis=1, keepdims=True))
        elif self.norm == "max":
            norms = np.max(np.abs(X_arr), axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        norms = np.where(norms == 0, 1, norms)
        return X_arr / norms


class BinningTransformer(BaseFeatureTransformer):
    """
    Discretize continuous features into bins.

    Supports uniform, quantile, and kmeans binning strategies.
    """

    def __init__(
        self,
        n_bins: int = 10,
        strategy: str = "quantile",
        encode: str = "onehot",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.bin_edges_: Optional[List[NDArray]] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "BinningTransformer":
        X = self._validate_input(X)
        self.bin_edges_ = []

        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx].dropna().values

            if self.strategy == "uniform":
                edges = np.linspace(col_data.min(), col_data.max(), self.n_bins + 1)
            elif self.strategy == "quantile":
                edges = np.percentile(col_data, np.linspace(0, 100, self.n_bins + 1))
            elif self.strategy == "kmeans":
                # Simple k-means binning
                from scipy.cluster.vq import kmeans

                centroids, _ = kmeans(col_data.reshape(-1, 1), self.n_bins)
                edges = np.sort(centroids.flatten())
                edges = np.concatenate([[col_data.min()], edges, [col_data.max()]])
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            self.bin_edges_.append(edges)

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        X_arr = X.values

        binned = np.zeros_like(X_arr, dtype=int)
        for col_idx in range(X_arr.shape[1]):
            binned[:, col_idx] = np.digitize(
                X_arr[:, col_idx], self.bin_edges_[col_idx][1:-1]
            )

        if self.encode == "onehot":
            # One-hot encode bins
            n_samples = X_arr.shape[0]
            n_features = X_arr.shape[1]
            result = np.zeros((n_samples, n_features * self.n_bins))

            for col_idx in range(n_features):
                for bin_idx in range(self.n_bins):
                    mask = binned[:, col_idx] == bin_idx
                    result[mask, col_idx * self.n_bins + bin_idx] = 1

            return result
        elif self.encode == "ordinal":
            return binned
        else:
            return binned


class PolynomialFeatures(BaseFeatureTransformer):
    """
    Generate polynomial and interaction features.

    Creates features: x1, x2, x1^2, x1*x2, x2^2, ...
    """

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.n_input_features_: Optional[int] = None
        self.combinations_: Optional[List[Tuple[int, ...]]] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "PolynomialFeatures":
        X = self._validate_input(X)
        self.n_input_features_ = X.shape[1]

        from itertools import combinations_with_replacement, combinations

        self.combinations_ = []
        start_degree = 0 if self.include_bias else 1

        for d in range(start_degree, self.degree + 1):
            if d == 0:
                self.combinations_.append(())
            else:
                if self.interaction_only:
                    self.combinations_.extend(
                        combinations(range(self.n_input_features_), d)
                    )
                else:
                    self.combinations_.extend(
                        combinations_with_replacement(range(self.n_input_features_), d)
                    )

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        X_arr = X.values
        n_samples = X_arr.shape[0]

        result = np.ones((n_samples, len(self.combinations_)))

        for idx, comb in enumerate(self.combinations_):
            if len(comb) == 0:
                result[:, idx] = 1.0
            else:
                result[:, idx] = np.prod(X_arr[:, list(comb)], axis=1)

        return result


class LogTransformer(BaseFeatureTransformer):
    """
    Log transformation for skewed distributions.

    Handles zero and negative values with shift parameter.
    """

    def __init__(
        self,
        base: float = np.e,
        handle_zeros: bool = True,
        offset: float = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.base = base
        self.handle_zeros = handle_zeros
        self.offset = offset
        self.min_values_: Optional[NDArray] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "LogTransformer":
        X = self._validate_input(X)
        if self.handle_zeros:
            self.min_values_ = X.min(axis=0).values
        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        X_arr = X.values

        if self.handle_zeros:
            # Shift data to be positive
            X_shifted = X_arr - self.min_values_ + self.offset
        else:
            X_shifted = X_arr

        if self.base == np.e:
            return np.log(X_shifted)
        else:
            return np.log(X_shifted) / np.log(self.base)

    def inverse_transform(self, X: NDArray) -> NDArray:
        """Inverse log transform."""
        if self.base == np.e:
            result = np.exp(X)
        else:
            result = self.base**X

        if self.handle_zeros:
            result = result + self.min_values_ - self.offset

        return result


# ============================================================================
# CATEGORICAL FEATURES
# ============================================================================


class OneHotEncoder(BaseFeatureTransformer):
    """
    One-hot encode categorical features.

    Creates binary columns for each category.
    """

    def __init__(
        self,
        drop: Optional[str] = None,
        sparse: bool = False,
        handle_unknown: str = "ignore",
        min_frequency: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.drop = drop
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.min_frequency = min_frequency
        self.categories_: Optional[List[NDArray]] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "OneHotEncoder":
        X = self._validate_input(X)
        self.categories_ = []

        for col_idx in range(X.shape[1]):
            categories = pd.Series(X.iloc[:, col_idx]).value_counts()

            if self.min_frequency is not None:
                categories = categories[categories >= self.min_frequency]

            cats = categories.index.values

            if self.drop == "first" and len(cats) > 0:
                cats = cats[1:]

            self.categories_.append(cats)

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        result_list = []
        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx]
            categories = self.categories_[col_idx]

            # Create one-hot matrix
            encoded = np.zeros((len(col_data), len(categories)))
            for i, cat in enumerate(categories):
                encoded[:, i] = (col_data == cat).astype(float).values

            result_list.append(encoded)

        return (
            np.hstack(result_list)
            if result_list
            else np.array([]).reshape(X.shape[0], 0)
        )


class LabelEncoder(BaseFeatureTransformer):
    """
    Encode labels with value between 0 and n_classes-1.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.classes_: Optional[NDArray] = None
        self.class_to_index_: Optional[Dict[Any, int]] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame, pd.Series],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "LabelEncoder":
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]  # Take first column if DataFrame

        self.classes_ = np.unique(X)
        self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame, pd.Series]) -> NDArray:
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        return np.array([self.class_to_index_.get(x, -1) for x in X])

    def inverse_transform(self, y: NDArray) -> NDArray:
        """Convert labels back to original values."""
        return self.classes_[y]


class OrdinalEncoder(BaseFeatureTransformer):
    """
    Encode categorical features as ordinal integers.
    """

    def __init__(
        self,
        handle_unknown: str = "use_encoded_value",
        unknown_value: int = -1,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.categories_: Optional[List[NDArray]] = None
        self.mappings_: Optional[List[Dict[Any, int]]] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "OrdinalEncoder":
        X = self._validate_input(X)
        self.categories_ = []
        self.mappings_ = []

        for col_idx in range(X.shape[1]):
            unique_vals = pd.Series(X.iloc[:, col_idx]).unique()
            self.categories_.append(unique_vals)
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            self.mappings_.append(mapping)

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        result = np.zeros((X.shape[0], X.shape[1]), dtype=int)
        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx]
            mapping = self.mappings_[col_idx]

            for i, val in enumerate(col_data):
                result[i, col_idx] = mapping.get(val, self.unknown_value)

        return result


class TargetEncoder(BaseFeatureTransformer):
    """
    Target encoding: replace category with mean target value.

    Uses smoothing to handle rare categories.
    """

    def __init__(
        self,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        noise: float = 0.0,
        handle_unknown: str = "value",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise = noise
        self.handle_unknown = handle_unknown
        self.encodings_: Optional[List[Dict[Any, float]]] = None
        self.global_mean_: Optional[float] = None

    def fit(
        self, X: Union[NDArray, pd.DataFrame], y: Union[NDArray, pd.Series]
    ) -> "TargetEncoder":
        X = self._validate_input(X)

        if isinstance(y, pd.Series):
            y = y.values

        self.global_mean_ = np.mean(y)
        self.encodings_ = []

        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx]
            encoding = {}

            for category in col_data.unique():
                mask = col_data == category
                n_samples = mask.sum()
                category_mean = y[mask.values].mean()

                # Smoothing: blend category mean with global mean
                smoothing_factor = 1 / (
                    1 + np.exp(-(n_samples - self.min_samples_leaf) / self.smoothing)
                )
                encoded_value = (
                    smoothing_factor * category_mean
                    + (1 - smoothing_factor) * self.global_mean_
                )
                encoding[category] = encoded_value

            self.encodings_.append(encoding)

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        result = np.zeros((X.shape[0], X.shape[1]))
        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx]
            encoding = self.encodings_[col_idx]

            for i, val in enumerate(col_data):
                result[i, col_idx] = encoding.get(val, self.global_mean_)

            if self.noise > 0:
                result[:, col_idx] += np.random.normal(
                    0, self.noise, size=len(col_data)
                )

        return result


class HashingEncoder(BaseFeatureTransformer):
    """
    Hashing trick for high-cardinality categorical features.

    Uses hash function to map categories to fixed-size vectors.
    """

    def __init__(
        self,
        n_features: int = 100,
        hash_method: str = "md5",
        alternate_sign: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.n_features = n_features
        self.hash_method = hash_method
        self.alternate_sign = alternate_sign

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "HashingEncoder":
        self._is_fitted = True
        return self

    def _hash(self, value: Any) -> int:
        """Hash a value to an integer."""
        if self.hash_method == "md5":
            hash_obj = hashlib.md5(str(value).encode())
        elif self.hash_method == "sha256":
            hash_obj = hashlib.sha256(str(value).encode())
        else:
            hash_obj = hashlib.md5(str(value).encode())

        return int(hash_obj.hexdigest(), 16)

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        n_samples = X.shape[0]
        n_cols = X.shape[1]
        result = np.zeros((n_samples, n_cols * self.n_features))

        for col_idx in range(n_cols):
            col_data = X.iloc[:, col_idx]

            for i, val in enumerate(col_data):
                hash_val = self._hash(val)
                feature_idx = hash_val % self.n_features

                if self.alternate_sign:
                    sign = 1 if (hash_val // self.n_features) % 2 == 0 else -1
                else:
                    sign = 1

                result[i, col_idx * self.n_features + feature_idx] = sign

        return result


class EmbeddingEncoder(BaseFeatureTransformer):
    """
    Learnable embedding encoder for categorical features.

    Creates dense vector representations of categories.
    Requires torch.
    """

    def __init__(
        self,
        embedding_dim: int = 10,
        max_categories: int = 1000,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.embedding_dim = embedding_dim
        self.max_categories = max_categories

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for EmbeddingEncoder")

        self.embeddings_: Optional[nn.ModuleDict] = None
        self.category_to_idx_: Optional[List[Dict[Any, int]]] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "EmbeddingEncoder":
        X = self._validate_input(X)

        self.category_to_idx_ = []
        embedding_modules = {}

        for col_idx in range(X.shape[1]):
            unique_cats = pd.Series(X.iloc[:, col_idx]).unique()[: self.max_categories]
            cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
            self.category_to_idx_.append(cat_to_idx)

            embedding_modules[f"emb_{col_idx}"] = nn.Embedding(
                len(unique_cats), self.embedding_dim
            )

        self.embeddings_ = nn.ModuleDict(embedding_modules)
        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        n_samples = X.shape[0]
        n_cols = X.shape[1]
        result = np.zeros((n_samples, n_cols * self.embedding_dim))

        with torch.no_grad():
            for col_idx in range(n_cols):
                col_data = X.iloc[:, col_idx]
                cat_to_idx = self.category_to_idx_[col_idx]

                indices = torch.LongTensor([cat_to_idx.get(val, 0) for val in col_data])

                embeddings = self.embeddings_[f"emb_{col_idx}"](indices).numpy()
                result[
                    :, col_idx * self.embedding_dim : (col_idx + 1) * self.embedding_dim
                ] = embeddings

        return result


# ============================================================================
# TEXT FEATURES
# ============================================================================


class TFIDF(BaseFeatureTransformer):
    """
    TF-IDF vectorizer for text features.

    Term Frequency - Inverse Document Frequency.
    """

    def __init__(
        self,
        max_features: int = 10000,
        min_df: float = 0.01,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Optional[Union[str, List[str]]] = None,
        lowercase: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.vocabulary_: Optional[Dict[str, int]] = None
        self.idf_: Optional[NDArray] = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        if self.lowercase:
            text = text.lower()

        # Simple word tokenization
        tokens = text.split()

        # Remove stop words
        if self.stop_words == "english":
            stop_words = {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "shall",
                "can",
                "need",
                "dare",
                "ought",
                "used",
                "to",
                "of",
                "in",
                "for",
                "on",
                "with",
                "at",
                "by",
                "from",
                "as",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
                "under",
                "and",
                "but",
                "or",
                "yet",
                "so",
                "if",
                "because",
                "although",
                "though",
                "while",
                "where",
                "when",
                "that",
                "which",
                "who",
                "whom",
                "whose",
                "what",
                "this",
                "these",
                "those",
                "i",
                "me",
                "my",
                "myself",
                "we",
                "our",
                "ours",
                "ourselves",
                "you",
                "your",
                "yours",
                "yourself",
                "yourselves",
                "he",
                "him",
                "his",
                "himself",
                "she",
                "her",
                "hers",
                "herself",
                "it",
                "its",
                "itself",
                "they",
                "them",
                "their",
                "theirs",
                "themselves",
                "what",
                "which",
                "who",
                "whom",
                "this",
                "that",
                "these",
                "those",
                "am",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "having",
                "do",
                "does",
                "did",
                "doing",
                "a",
                "an",
                "the",
                "and",
                "but",
                "if",
                "or",
                "because",
                "as",
                "until",
                "while",
                "of",
                "at",
                "by",
                "for",
                "with",
                "about",
                "against",
                "between",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "to",
                "from",
                "up",
                "down",
                "in",
                "out",
                "on",
                "off",
                "over",
                "under",
                "again",
                "further",
                "then",
                "once",
            }
            tokens = [t for t in tokens if t not in stop_words]
        elif isinstance(self.stop_words, list):
            tokens = [t for t in tokens if t not in self.stop_words]

        return tokens

    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        """Extract n-grams from tokens."""
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])
                ngrams.append(ngram)
        return ngrams

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame, pd.Series, List[str]],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "TFIDF":
        # Convert to list of strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        # Build vocabulary
        term_doc_count = defaultdict(int)
        n_docs = len(X)

        for doc in X:
            tokens = self._tokenize(str(doc))
            ngrams = self._extract_ngrams(tokens)
            unique_ngrams = set(ngrams)

            for ngram in unique_ngrams:
                term_doc_count[ngram] += 1

        # Filter by min_df and max_df
        min_count = self.min_df * n_docs if self.min_df < 1 else self.min_df
        max_count = self.max_df * n_docs if self.max_df < 1 else self.max_df

        filtered_terms = {
            term: count
            for term, count in term_doc_count.items()
            if min_count <= count <= max_count
        }

        # Select top terms
        sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
        top_terms = sorted_terms[: self.max_features]

        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(top_terms)}

        # Compute IDF
        self.idf_ = (
            np.log(n_docs / (np.array([count for _, count in top_terms]) + 1)) + 1
        )

        self._is_fitted = True
        return self

    def transform(
        self, X: Union[NDArray, pd.DataFrame, pd.Series, List[str]]
    ) -> NDArray:
        self._check_is_fitted()

        # Convert to list of strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        n_docs = len(X)
        n_features = len(self.vocabulary_)
        result = np.zeros((n_docs, n_features))

        for i, doc in enumerate(X):
            tokens = self._tokenize(str(doc))
            ngrams = self._extract_ngrams(tokens)

            # Count term frequencies
            term_counts = defaultdict(int)
            for ngram in ngrams:
                term_counts[ngram] += 1

            # Compute TF-IDF
            total_terms = len(ngrams)
            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    tf = count / total_terms if total_terms > 0 else 0
                    result[i, idx] = tf * self.idf_[idx]

        # L2 normalize
        norms = np.sqrt(np.sum(result**2, axis=1, keepdims=True))
        norms = np.where(norms == 0, 1, norms)
        result = result / norms

        return result


class CountVectorizer(BaseFeatureTransformer):
    """
    Count vectorizer for text features.

    Creates bag-of-words representation.
    """

    def __init__(
        self,
        max_features: int = 10000,
        min_df: float = 0.01,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Optional[Union[str, List[str]]] = None,
        lowercase: bool = True,
        binary: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.binary = binary
        self.vocabulary_: Optional[Dict[str, int]] = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        if self.lowercase:
            text = text.lower()

        tokens = text.split()

        if self.stop_words == "english":
            stop_words = {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "shall",
                "can",
                "need",
                "dare",
                "ought",
                "used",
                "to",
                "of",
                "in",
                "for",
                "on",
                "with",
                "at",
                "by",
                "from",
                "as",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
                "under",
                "and",
                "but",
                "or",
                "yet",
                "so",
                "if",
                "because",
                "although",
                "though",
                "while",
                "where",
                "when",
                "that",
                "which",
                "who",
                "whom",
                "whose",
                "what",
                "this",
                "these",
                "those",
                "i",
                "me",
                "my",
                "myself",
                "we",
                "our",
                "ours",
                "ourselves",
                "you",
                "your",
                "yours",
                "yourself",
                "yourselves",
                "he",
                "him",
                "his",
                "himself",
                "she",
                "her",
                "hers",
                "herself",
                "it",
                "its",
                "itself",
                "they",
                "them",
                "their",
                "theirs",
                "themselves",
                "what",
                "which",
                "who",
                "whom",
                "this",
                "that",
                "these",
                "those",
                "am",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "having",
                "do",
                "does",
                "did",
                "doing",
                "a",
                "an",
                "the",
                "and",
                "but",
                "if",
                "or",
                "because",
                "as",
                "until",
                "while",
                "of",
                "at",
                "by",
                "for",
                "with",
                "about",
                "against",
                "between",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "to",
                "from",
                "up",
                "down",
                "in",
                "out",
                "on",
                "off",
                "over",
                "under",
                "again",
                "further",
                "then",
                "once",
            }
            tokens = [t for t in tokens if t not in stop_words]
        elif isinstance(self.stop_words, list):
            tokens = [t for t in tokens if t not in self.stop_words]

        return tokens

    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        """Extract n-grams from tokens."""
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])
                ngrams.append(ngram)
        return ngrams

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame, pd.Series, List[str]],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "CountVectorizer":
        # Convert to list of strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        # Build vocabulary
        term_doc_count = defaultdict(int)
        n_docs = len(X)

        for doc in X:
            tokens = self._tokenize(str(doc))
            ngrams = self._extract_ngrams(tokens)
            unique_ngrams = set(ngrams)

            for ngram in unique_ngrams:
                term_doc_count[ngram] += 1

        # Filter by min_df and max_df
        min_count = self.min_df * n_docs if self.min_df < 1 else self.min_df
        max_count = self.max_df * n_docs if self.max_df < 1 else self.max_df

        filtered_terms = {
            term: count
            for term, count in term_doc_count.items()
            if min_count <= count <= max_count
        }

        # Select top terms
        sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
        top_terms = sorted_terms[: self.max_features]

        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(top_terms)}

        self._is_fitted = True
        return self

    def transform(
        self, X: Union[NDArray, pd.DataFrame, pd.Series, List[str]]
    ) -> NDArray:
        self._check_is_fitted()

        # Convert to list of strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        n_docs = len(X)
        n_features = len(self.vocabulary_)
        result = np.zeros((n_docs, n_features))

        for i, doc in enumerate(X):
            tokens = self._tokenize(str(doc))
            ngrams = self._extract_ngrams(tokens)

            for ngram in ngrams:
                if ngram in self.vocabulary_:
                    idx = self.vocabulary_[ngram]
                    if self.binary:
                        result[i, idx] = 1
                    else:
                        result[i, idx] += 1

        return result


class Word2VecFeatures(BaseFeatureTransformer):
    """
    Word2Vec embeddings for text features.

    Uses gensim Word2Vec to learn embeddings.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        epochs: int = 5,
        aggregation: str = "mean",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.aggregation = aggregation

        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for Word2VecFeatures")

        self.model_: Optional[Any] = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame, pd.Series, List[str]],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "Word2VecFeatures":
        # Convert to list of strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        # Tokenize sentences
        sentences = [self._tokenize(str(doc)) for doc in X]

        # Train Word2Vec
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
        )

        self._is_fitted = True
        return self

    def transform(
        self, X: Union[NDArray, pd.DataFrame, pd.Series, List[str]]
    ) -> NDArray:
        self._check_is_fitted()

        # Convert to list of strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        n_docs = len(X)
        result = np.zeros((n_docs, self.vector_size))

        for i, doc in enumerate(X):
            tokens = self._tokenize(str(doc))
            vectors = []

            for token in tokens:
                if token in self.model_.wv:
                    vectors.append(self.model_.wv[token])

            if vectors:
                vectors = np.array(vectors)
                if self.aggregation == "mean":
                    result[i] = np.mean(vectors, axis=0)
                elif self.aggregation == "sum":
                    result[i] = np.sum(vectors, axis=0)
                elif self.aggregation == "max":
                    result[i] = np.max(vectors, axis=0)
                elif self.aggregation == "min":
                    result[i] = np.min(vectors, axis=0)
            else:
                # If no words found, use zero vector
                result[i] = np.zeros(self.vector_size)

        return result


class BERTFeatures(BaseFeatureTransformer):
    """
    BERT embeddings for text features.

    Uses pre-trained BERT model to extract embeddings.
    Requires transformers library.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        pooling: str = "cls",
        batch_size: int = 32,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.batch_size = batch_size

        self.tokenizer_: Optional[Any] = None
        self.model_: Optional[Any] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame, pd.Series, List[str]],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "BERTFeatures":
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("transformers library is required for BERTFeatures")

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModel.from_pretrained(self.model_name)

        if TORCH_AVAILABLE:
            self.model_.eval()

        self._is_fitted = True
        return self

    def transform(
        self, X: Union[NDArray, pd.DataFrame, pd.Series, List[str]]
    ) -> NDArray:
        self._check_is_fitted()

        # Convert to list of strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        embeddings = []

        with torch.no_grad() if TORCH_AVAILABLE else contextlib.nullcontext():
            for i in range(0, len(X), self.batch_size):
                batch = X[i : i + self.batch_size]

                encoded = self.tokenizer_(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                outputs = self.model_(**encoded)

                if self.pooling == "cls":
                    # Use [CLS] token embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                elif self.pooling == "mean":
                    # Mean pooling
                    attention_mask = encoded["attention_mask"]
                    mask_expanded = (
                        attention_mask.unsqueeze(-1)
                        .expand(outputs.last_hidden_state.size())
                        .float()
                    )
                    sum_embeddings = torch.sum(
                        outputs.last_hidden_state * mask_expanded, 1
                    )
                    batch_embeddings = (
                        sum_embeddings / torch.clamp(mask_expanded.sum(1), min=1e-9)
                    ).numpy()
                else:
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


# ============================================================================
# TIME FEATURES
# ============================================================================


class DateTimeExtractor(BaseFeatureTransformer):
    """
    Extract components from datetime features.

    Extracts: year, month, day, hour, minute, second, dayofweek, quarter, etc.
    """

    def __init__(
        self, components: Optional[List[str]] = None, name: Optional[str] = None
    ):
        super().__init__(name)
        self.components = components or [
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "dayofweek",
            "quarter",
            "dayofyear",
            "weekofyear",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
        ]

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "DateTimeExtractor":
        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        result_list = []

        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx]

            # Convert to datetime
            if not pd.api.types.is_datetime64_any_dtype(col_data):
                col_data = pd.to_datetime(col_data, errors="coerce")

            # Extract components
            for component in self.components:
                if hasattr(col_data.dt, component):
                    result_list.append(getattr(col_data.dt, component).values)
                elif component == "weekofyear":
                    result_list.append(col_data.dt.isocalendar().week.values)
                elif component.startswith("is_"):
                    # Boolean features
                    if component == "is_month_start":
                        result_list.append(
                            col_data.dt.is_month_start.astype(int).values
                        )
                    elif component == "is_month_end":
                        result_list.append(col_data.dt.is_month_end.astype(int).values)
                    elif component == "is_quarter_start":
                        result_list.append(
                            col_data.dt.is_quarter_start.astype(int).values
                        )
                    elif component == "is_quarter_end":
                        result_list.append(
                            col_data.dt.is_quarter_end.astype(int).values
                        )
                    elif component == "is_year_start":
                        result_list.append(col_data.dt.is_year_start.astype(int).values)
                    elif component == "is_year_end":
                        result_list.append(col_data.dt.is_year_end.astype(int).values)

        if result_list:
            return np.column_stack(result_list)
        else:
            return np.array([]).reshape(X.shape[0], 0)


class CyclicalEncoder(BaseFeatureTransformer):
    """
    Encode cyclical features using sine and cosine.

    Transforms: x -> (sin(2*pi*x/period), cos(2*pi*x/period))
    """

    def __init__(
        self, periods: Optional[Dict[int, float]] = None, name: Optional[str] = None
    ):
        super().__init__(name)
        self.periods = periods or {}
        self._default_periods = {
            "hour": 24,
            "day": 31,
            "month": 12,
            "dayofweek": 7,
            "quarter": 4,
        }

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "CyclicalEncoder":
        # Infer periods if not provided
        X = self._validate_input(X)

        for col_idx in range(X.shape[1]):
            if col_idx not in self.periods:
                col_data = X.iloc[:, col_idx]
                max_val = col_data.max()
                min_val = col_data.min()
                self.periods[col_idx] = max_val - min_val + 1

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        result_list = []

        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx].values
            period = self.periods.get(col_idx, col_data.max() - col_data.min() + 1)

            sin_component = np.sin(2 * np.pi * col_data / period)
            cos_component = np.cos(2 * np.pi * col_data / period)

            result_list.append(sin_component)
            result_list.append(cos_component)

        return (
            np.column_stack(result_list)
            if result_list
            else np.array([]).reshape(X.shape[0], 0)
        )


class LagFeatures(BaseFeatureTransformer):
    """
    Create lag features for time series.

    Creates features: x(t-1), x(t-2), ..., x(t-n)
    """

    def __init__(
        self,
        lags: Union[int, List[int]] = 5,
        drop_na: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.lags = [lags] if isinstance(lags, int) else lags
        self.drop_na = drop_na

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "LagFeatures":
        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        result_list = []
        max_lag = max(self.lags)

        for col_idx in range(X.shape[1]):
            col_data = X.iloc[:, col_idx].values

            for lag in self.lags:
                lagged = np.full(len(col_data), np.nan)
                lagged[lag:] = col_data[:-lag]
                result_list.append(lagged)

        result = np.column_stack(result_list)

        if self.drop_na:
            result = result[max_lag:]

        return result


class RollingFeatures(BaseFeatureTransformer):
    """
    Create rolling statistics features for time series.

    Computes: mean, std, min, max, sum over rolling window.
    """

    def __init__(
        self,
        windows: Union[int, List[int]] = 5,
        statistics: List[str] = None,
        min_periods: Optional[int] = None,
        center: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.windows = [windows] if isinstance(windows, int) else windows
        self.statistics = statistics or ["mean", "std", "min", "max"]
        self.min_periods = min_periods
        self.center = center

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "RollingFeatures":
        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        result_list = []

        for col_idx in range(X.shape[1]):
            col_data = pd.Series(X.iloc[:, col_idx].values)

            for window in self.windows:
                for stat in self.statistics:
                    rolling = col_data.rolling(
                        window=window,
                        min_periods=self.min_periods or window,
                        center=self.center,
                    )

                    if stat == "mean":
                        result_list.append(rolling.mean().values)
                    elif stat == "std":
                        result_list.append(rolling.std().values)
                    elif stat == "min":
                        result_list.append(rolling.min().values)
                    elif stat == "max":
                        result_list.append(rolling.max().values)
                    elif stat == "sum":
                        result_list.append(rolling.sum().values)
                    elif stat == "median":
                        result_list.append(rolling.median().values)
                    elif stat == "var":
                        result_list.append(rolling.var().values)
                    elif stat == "skew":
                        result_list.append(rolling.skew().values)
                    elif stat == "kurt":
                        result_list.append(rolling.kurt().values)

        return (
            np.column_stack(result_list)
            if result_list
            else np.array([]).reshape(X.shape[0], 0)
        )


# ============================================================================
# FEATURE SELECTION
# ============================================================================


class VarianceThreshold(BaseFeatureTransformer):
    """
    Remove features with low variance.

    Variance threshold selection.
    """

    def __init__(self, threshold: float = 0.01, name: Optional[str] = None):
        super().__init__(name)
        self.threshold = threshold
        self.variances_: Optional[NDArray] = None
        self.support_: Optional[NDArray] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "VarianceThreshold":
        X = self._validate_input(X)
        self.variances_ = X.var(axis=0).values
        self.support_ = self.variances_ > self.threshold
        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        return X.values[:, self.support_]


class SelectKBest(BaseFeatureTransformer):
    """
    Select K best features using statistical tests.

    Supports: f_classif, f_regression, chi2, mutual_info
    """

    def __init__(
        self,
        score_func: Union[str, Callable] = "f_classif",
        k: int = 10,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.score_func = score_func
        self.k = k
        self.scores_: Optional[NDArray] = None
        self.pvalues_: Optional[NDArray] = None
        self.support_: Optional[NDArray] = None

    def _get_score_func(self, task_type: str = "classification"):
        """Get scoring function."""
        if callable(self.score_func):
            return self.score_func

        funcs = {
            "f_classif": f_classif if SKLEARN_AVAILABLE else None,
            "f_regression": f_regression if SKLEARN_AVAILABLE else None,
            "chi2": chi2 if SKLEARN_AVAILABLE else None,
            "mutual_info_classif": mutual_info_classif if SKLEARN_AVAILABLE else None,
            "mutual_info_regression": mutual_info_regression
            if SKLEARN_AVAILABLE
            else None,
        }

        return funcs.get(self.score_func, funcs.get("f_classif"))

    def fit(
        self, X: Union[NDArray, pd.DataFrame], y: Union[NDArray, pd.Series]
    ) -> "SelectKBest":
        X = self._validate_input(X)

        if isinstance(y, pd.Series):
            y = y.values

        score_func = self._get_score_func()

        if score_func is not None and SKLEARN_AVAILABLE:
            self.scores_, self.pvalues_ = score_func(X.values, y)
        else:
            # Simple variance-based fallback
            self.scores_ = X.var(axis=0).values
            self.pvalues_ = np.zeros_like(self.scores_)

        # Select top k features
        k = min(self.k, len(self.scores_))
        top_indices = np.argsort(self.scores_)[-k:]
        self.support_ = np.zeros(len(self.scores_), dtype=bool)
        self.support_[top_indices] = True

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        return X.values[:, self.support_]


class RFE(BaseFeatureTransformer):
    """
    Recursive Feature Elimination.

    Iteratively removes weakest features.
    """

    def __init__(
        self,
        estimator=None,
        n_features_to_select: int = 10,
        step: int = 1,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.support_: Optional[NDArray] = None
        self.ranking_: Optional[NDArray] = None

    def fit(
        self, X: Union[NDArray, pd.DataFrame], y: Union[NDArray, pd.Series]
    ) -> "RFE":
        X = self._validate_input(X)
        X_arr = X.values

        if isinstance(y, pd.Series):
            y = y.values

        n_features = X_arr.shape[1]
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)

        # Use random forest as default estimator
        if self.estimator is None and SKLEARN_AVAILABLE:
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(n_estimators=10, random_state=42)

        while np.sum(support) > self.n_features_to_select:
            # Fit estimator
            X_selected = X_arr[:, support]
            self.estimator.fit(X_selected, y)

            # Get feature importances
            if hasattr(self.estimator, "feature_importances_"):
                importances = self.estimator.feature_importances_
            elif hasattr(self.estimator, "coef_"):
                importances = np.abs(self.estimator.coef_).flatten()
            else:
                importances = np.ones(X_selected.shape[1])

            # Mark eliminated features
            n_to_remove = max(
                self.step, int(np.sum(support) - self.n_features_to_select)
            )
            weakest_indices = np.argsort(importances)[:n_to_remove]

            # Map back to original indices
            support_indices = np.where(support)[0]
            for idx in weakest_indices:
                support[support_indices[idx]] = False
                ranking[support_indices[idx]] = n_features - np.sum(support) + 1

        self.support_ = support
        self.ranking_ = ranking

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        return X.values[:, self.support_]


class LassoFeature(BaseFeatureTransformer):
    """
    Feature selection using L1 regularization (Lasso).

    Lasso tends to produce sparse solutions, effectively selecting features.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_iter: int = 1000,
        task: str = "regression",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.alpha = alpha
        self.max_iter = max_iter
        self.task = task
        self.coef_: Optional[NDArray] = None
        self.support_: Optional[NDArray] = None

    def fit(
        self, X: Union[NDArray, pd.DataFrame], y: Union[NDArray, pd.Series]
    ) -> "LassoFeature":
        X = self._validate_input(X)

        if isinstance(y, pd.Series):
            y = y.values

        if SKLEARN_AVAILABLE:
            if self.task == "regression":
                from sklearn.linear_model import Lasso

                model = Lasso(alpha=self.alpha, max_iter=self.max_iter, random_state=42)
            else:
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression(
                    penalty="l1",
                    C=1 / self.alpha,
                    solver="liblinear",
                    max_iter=self.max_iter,
                    random_state=42,
                )

            model.fit(X.values, y)
            self.coef_ = np.abs(model.coef_)
            if self.coef_.ndim > 1:
                self.coef_ = self.coef_.mean(axis=0)
            self.support_ = self.coef_ > 1e-10
        else:
            # Simple correlation-based fallback
            correlations = np.array(
                [np.corrcoef(X.values[:, i], y)[0, 1] for i in range(X.shape[1])]
            )
            self.coef_ = np.abs(correlations)
            self.support_ = ~np.isnan(self.coef_)

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        return X.values[:, self.support_]


class MutualInformation(BaseFeatureTransformer):
    """
    Feature selection using mutual information.

    Measures dependency between features and target.
    """

    def __init__(
        self,
        k: int = 10,
        task: str = "regression",
        random_state: int = 42,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.k = k
        self.task = task
        self.random_state = random_state
        self.scores_: Optional[NDArray] = None
        self.support_: Optional[NDArray] = None

    def _compute_mi(self, X: NDArray, y: NDArray) -> NDArray:
        """Compute mutual information for each feature."""
        if SKLEARN_AVAILABLE:
            if self.task == "regression":
                return mutual_info_regression(X, y, random_state=self.random_state)
            else:
                return mutual_info_classif(X, y, random_state=self.random_state)
        else:
            # Simple entropy-based approximation
            scores = []
            for i in range(X.shape[1]):
                # Compute mutual information using histogram
                hist_2d, _, _ = np.histogram2d(X[:, i], y, bins=20)
                # Compute mutual information
                p_xy = hist_2d / float(np.sum(hist_2d))
                p_x = np.sum(p_xy, axis=1)
                p_y = np.sum(p_xy, axis=0)

                mi = 0.0
                for i_x in range(len(p_x)):
                    for i_y in range(len(p_y)):
                        if p_xy[i_x, i_y] > 0:
                            mi += p_xy[i_x, i_y] * np.log(
                                p_xy[i_x, i_y] / (p_x[i_x] * p_y[i_y] + 1e-10)
                            )
                scores.append(mi)
            return np.array(scores)

    def fit(
        self, X: Union[NDArray, pd.DataFrame], y: Union[NDArray, pd.Series]
    ) -> "MutualInformation":
        X = self._validate_input(X)

        if isinstance(y, pd.Series):
            y = y.values

        self.scores_ = self._compute_mi(X.values, y)

        # Select top k features
        k = min(self.k, len(self.scores_))
        top_indices = np.argsort(self.scores_)[-k:]
        self.support_ = np.zeros(len(self.scores_), dtype=bool)
        self.support_[top_indices] = True

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        return X.values[:, self.support_]


# ============================================================================
# FEATURE GENERATION
# ============================================================================


class AutoFeatureGenerator(BaseFeatureTransformer):
    """
    Automated feature generation.

    Generates features based on statistical tests and domain knowledge.
    """

    def __init__(
        self,
        max_features: int = 100,
        include_interactions: bool = True,
        include_polynomials: bool = True,
        include_transforms: bool = True,
        random_state: int = 42,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.max_features = max_features
        self.include_interactions = include_interactions
        self.include_polynomials = include_polynomials
        self.include_transforms = include_transforms
        self.random_state = random_state
        self.generated_features_: List[str] = []

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "AutoFeatureGenerator":
        X = self._validate_input(X)

        # Store column names if available
        if isinstance(X, pd.DataFrame):
            self.generated_features_ = list(X.columns)

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        X_arr = X.values

        features = [X_arr]
        n_features = X_arr.shape[1]

        # Polynomial features
        if self.include_polynomials:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(X_arr)
            features.append(poly_features[:, n_features:])  # Exclude original features

        # Transform features
        if self.include_transforms:
            transforms = []
            for i in range(n_features):
                col = X_arr[:, i]
                # Log transform (shift to positive)
                if np.min(col) > -1:
                    transforms.append(np.log1p(col - np.min(col) + 1).reshape(-1, 1))
                # Square root
                if np.min(col) >= 0:
                    transforms.append(np.sqrt(col).reshape(-1, 1))
                # Square
                transforms.append((col**2).reshape(-1, 1))

            if transforms:
                features.append(np.hstack(transforms))

        # Interaction features
        if self.include_interactions:
            interactions = []
            for i in range(n_features):
                for j in range(i + 1, min(i + 5, n_features)):  # Limit interactions
                    interactions.append((X_arr[:, i] * X_arr[:, j]).reshape(-1, 1))

            if interactions:
                features.append(np.hstack(interactions))

        result = np.hstack(features)

        # Limit to max features
        if result.shape[1] > self.max_features:
            # Select features with highest variance
            variances = np.var(result, axis=0)
            top_indices = np.argsort(variances)[-self.max_features :]
            result = result[:, top_indices]

        return result


class FeatureInteraction(BaseFeatureTransformer):
    """
    Create feature interactions (products, ratios, etc.).

    Generates pairwise feature interactions.
    """

    def __init__(
        self,
        interaction_type: str = "product",
        max_interactions: int = 100,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.interaction_type = interaction_type
        self.max_interactions = max_interactions
        self.selected_pairs_: Optional[List[Tuple[int, int]]] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "FeatureInteraction":
        X = self._validate_input(X)
        n_features = X.shape[1]

        # Generate all pairs
        all_pairs = [
            (i, j) for i in range(n_features) for j in range(i + 1, n_features)
        ]

        # Select top pairs based on correlation if y is provided
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values

            scores = []
            for i, j in all_pairs:
                interaction = self._compute_interaction(X.values[:, i], X.values[:, j])
                score = np.abs(np.corrcoef(interaction, y)[0, 1])
                if not np.isnan(score):
                    scores.append((score, (i, j)))

            scores.sort(reverse=True)
            self.selected_pairs_ = [pair for _, pair in scores[: self.max_interactions]]
        else:
            # Random selection
            np.random.seed(42)
            indices = np.random.choice(
                len(all_pairs),
                min(self.max_interactions, len(all_pairs)),
                replace=False,
            )
            self.selected_pairs_ = [all_pairs[i] for i in indices]

        self._is_fitted = True
        return self

    def _compute_interaction(self, x1: NDArray, x2: NDArray) -> NDArray:
        """Compute interaction between two features."""
        if self.interaction_type == "product":
            return x1 * x2
        elif self.interaction_type == "ratio":
            return np.where(x2 != 0, x1 / (x2 + 1e-10), 0)
        elif self.interaction_type == "sum":
            return x1 + x2
        elif self.interaction_type == "diff":
            return x1 - x2
        elif self.interaction_type == "max":
            return np.maximum(x1, x2)
        elif self.interaction_type == "min":
            return np.minimum(x1, x2)
        else:
            return x1 * x2

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        X_arr = X.values

        interactions = []
        for i, j in self.selected_pairs_:
            interaction = self._compute_interaction(X_arr[:, i], X_arr[:, j])
            interactions.append(interaction.reshape(-1, 1))

        if interactions:
            return np.hstack(interactions)
        else:
            return np.array([]).reshape(X_arr.shape[0], 0)


class FeatureCrossing(BaseFeatureTransformer):
    """
    Feature crossing for categorical features.

    Creates crossed features: cat1 x cat2 -> new categorical feature.
    """

    def __init__(
        self,
        crossing_pairs: Optional[List[Tuple[int, int]]] = None,
        hash_space: int = 1000,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.crossing_pairs = crossing_pairs
        self.hash_space = hash_space
        self.combinations_: Optional[Dict[Tuple[int, int], Dict]] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "FeatureCrossing":
        X = self._validate_input(X)

        # Auto-detect crossing pairs if not provided
        if self.crossing_pairs is None:
            n_features = X.shape[1]
            self.crossing_pairs = [
                (i, j) for i in range(n_features) for j in range(i + 1, n_features)
            ]

        self.combinations_ = {}
        for pair in self.crossing_pairs:
            i, j = pair
            col_i = X.iloc[:, i]
            col_j = X.iloc[:, j]

            # Find all combinations
            combinations = {}
            for val_i, val_j in zip(col_i, col_j):
                key = (str(val_i), str(val_j))
                if key not in combinations:
                    combinations[key] = len(combinations) % self.hash_space

            self.combinations_[pair] = combinations

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        crossed_features = []
        for pair, combinations in self.combinations_.items():
            i, j = pair
            col_i = X.iloc[:, i]
            col_j = X.iloc[:, j]

            # Create one-hot encoded crossed feature
            crossed = np.zeros((len(col_i), self.hash_space))
            for idx, (val_i, val_j) in enumerate(zip(col_i, col_j)):
                key = (str(val_i), str(val_j))
                if key in combinations:
                    crossed[idx, combinations[key]] = 1

            crossed_features.append(crossed)

        if crossed_features:
            return np.hstack(crossed_features)
        else:
            return np.array([]).reshape(X.shape[0], 0)


# ============================================================================
# UTILITIES
# ============================================================================


class FeaturePipeline(BaseFeatureTransformer):
    """
    Pipeline of feature transformers.

    Applies transformers sequentially.
    """

    def __init__(
        self,
        steps: List[Tuple[str, BaseFeatureTransformer]],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.steps = steps

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "FeaturePipeline":
        X_current = X.copy() if isinstance(X, pd.DataFrame) else X

        for name, transformer in self.steps:
            transformer.fit(X_current, y)
            X_current = transformer.transform(X_current)

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X_current = X.copy() if isinstance(X, pd.DataFrame) else X

        for name, transformer in self.steps:
            X_current = transformer.transform(X_current)

        return X_current if isinstance(X_current, np.ndarray) else X_current.values

    def fit_transform(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> NDArray:
        return self.fit(X, y).transform(X)


class FeatureUnion(BaseFeatureTransformer):
    """
    Union of feature transformers.

    Applies transformers in parallel and concatenates results.
    """

    def __init__(
        self,
        transformer_list: List[Tuple[str, BaseFeatureTransformer]],
        n_jobs: int = 1,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "FeatureUnion":
        for name, transformer in self.transformer_list:
            transformer.fit(X, y)

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()

        results = []
        for name, transformer in self.transformer_list:
            result = transformer.transform(X)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            results.append(result)

        return np.hstack(results)


class ColumnTransformer(BaseFeatureTransformer):
    """
    Apply different transformers to different columns.

    Allows different preprocessing for different feature types.
    """

    def __init__(
        self,
        transformers: List[
            Tuple[str, BaseFeatureTransformer, Union[List[int], List[str]]]
        ],
        remainder: str = "drop",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.transformers = transformers
        self.remainder = remainder
        self._fitted_transformers: Dict[str, BaseFeatureTransformer] = {}

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "ColumnTransformer":
        X = self._validate_input(X)

        for name, transformer, columns in self.transformers:
            if isinstance(columns[0], str):
                X_subset = X[columns]
            else:
                X_subset = X.iloc[:, columns]

            transformer.fit(X_subset, y)
            self._fitted_transformers[name] = transformer

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        results = []
        used_columns = set()

        for name, transformer, columns in self.transformers:
            if isinstance(columns[0], str):
                X_subset = X[columns]
                used_columns.update(columns)
            else:
                X_subset = X.iloc[:, columns]
                used_columns.update(columns)

            result = transformer.transform(X_subset)
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            results.append(result)

        # Handle remainder columns
        if self.remainder == "passthrough":
            remaining_cols = [c for c in range(X.shape[1]) if c not in used_columns]
            if remaining_cols:
                results.append(X.iloc[:, remaining_cols].values)
        elif self.remainder == "drop":
            pass
        elif isinstance(self.remainder, BaseFeatureTransformer):
            remaining_cols = [c for c in range(X.shape[1]) if c not in used_columns]
            if remaining_cols:
                result = self.remainder.transform(X.iloc[:, remaining_cols])
                if result.ndim == 1:
                    result = result.reshape(-1, 1)
                results.append(result)

        return np.hstack(results)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def infer_feature_types(
    df: pd.DataFrame, categorical_threshold: int = 10
) -> Dict[str, List[str]]:
    """
    Infer feature types from a DataFrame.

    Args:
        df: Input DataFrame
        categorical_threshold: Max unique values for categorical

    Returns:
        Dictionary mapping feature types to column names
    """
    numerical = []
    categorical = []
    datetime = []
    text = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= categorical_threshold:
                categorical.append(col)
            else:
                numerical.append(col)
        else:
            # Check if text (long strings) or categorical
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > 50:
                text.append(col)
            else:
                categorical.append(col)

    return {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime,
        "text": text,
    }


def create_default_pipeline(
    feature_types: Optional[Dict[str, List[str]]] = None,
    config: Optional[FeatureConfig] = None,
) -> ColumnTransformer:
    """
    Create a default feature engineering pipeline.

    Args:
        feature_types: Dictionary of feature types to column names
        config: Feature configuration

    Returns:
        Configured ColumnTransformer
    """
    config = config or FeatureConfig()

    transformers = []

    if feature_types:
        if feature_types.get("numerical"):
            num_transformer = FeaturePipeline(
                [
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=config.polynomial_degree)),
                ]
            )
            transformers.append(("num", num_transformer, feature_types["numerical"]))

        if feature_types.get("categorical"):
            cat_transformer = OneHotEncoder()
            transformers.append(("cat", cat_transformer, feature_types["categorical"]))

        if feature_types.get("datetime"):
            dt_transformer = DateTimeExtractor()
            transformers.append(("datetime", dt_transformer, feature_types["datetime"]))

        if feature_types.get("text"):
            text_transformer = TFIDF(max_features=config.max_features)
            transformers.append(("text", text_transformer, feature_types["text"]))

    return ColumnTransformer(transformers, remainder="passthrough")


def select_features_by_variance(
    X: Union[NDArray, pd.DataFrame], threshold: float = 0.01
) -> NDArray:
    """
    Select features based on variance threshold.

    Args:
        X: Input features
        threshold: Minimum variance threshold

    Returns:
        Transformed features
    """
    selector = VarianceThreshold(threshold=threshold)
    return selector.fit_transform(X)


def select_features_by_correlation(
    X: Union[NDArray, pd.DataFrame], y: Union[NDArray, pd.Series], k: int = 10
) -> NDArray:
    """
    Select top k features by correlation with target.

    Args:
        X: Input features
        y: Target variable
        k: Number of features to select

    Returns:
        Selected features
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    correlations = np.abs(np.corrcoef(X.T, y)[:-1, -1])
    top_indices = np.argsort(correlations)[-k:]

    return X[:, top_indices]


def generate_polynomial_features(
    X: Union[NDArray, pd.DataFrame], degree: int = 2, interaction_only: bool = False
) -> NDArray:
    """
    Generate polynomial features.

    Args:
        X: Input features
        degree: Polynomial degree
        interaction_only: Include only interaction terms

    Returns:
        Polynomial features
    """
    transformer = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
    return transformer.fit_transform(X)


def create_interaction_features(
    X: Union[NDArray, pd.DataFrame],
    pairs: Optional[List[Tuple[int, int]]] = None,
    interaction_type: str = "product",
) -> NDArray:
    """
    Create interaction features.

    Args:
        X: Input features
        pairs: Pairs of feature indices to interact
        interaction_type: Type of interaction

    Returns:
        Interaction features
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if pairs is None:
        n_features = X.shape[1]
        pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]

    interactions = []
    for i, j in pairs:
        if interaction_type == "product":
            interactions.append(X[:, i] * X[:, j])
        elif interaction_type == "ratio":
            interactions.append(X[:, i] / (X[:, j] + 1e-10))
        elif interaction_type == "sum":
            interactions.append(X[:, i] + X[:, j])
        elif interaction_type == "diff":
            interactions.append(X[:, i] - X[:, j])

    return np.column_stack(interactions)


def encode_cyclical_features(
    X: Union[NDArray, pd.Series], period: float, feature_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Encode cyclical features using sine and cosine.

    Args:
        X: Input feature
        period: Period of the cyclical feature
        feature_name: Name of the feature

    Returns:
        DataFrame with sin and cos components
    """
    if isinstance(X, pd.Series):
        X = X.values

    sin_component = np.sin(2 * np.pi * X / period)
    cos_component = np.cos(2 * np.pi * X / period)

    name = feature_name or "feature"
    return pd.DataFrame({f"{name}_sin": sin_component, f"{name}_cos": cos_component})


def extract_datetime_features(
    datetime_series: pd.Series, components: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract datetime components.

    Args:
        datetime_series: Datetime series
        components: List of components to extract

    Returns:
        DataFrame with datetime features
    """
    if components is None:
        components = ["year", "month", "day", "hour", "dayofweek", "quarter"]

    features = {}

    for component in components:
        if hasattr(datetime_series.dt, component):
            features[component] = getattr(datetime_series.dt, component)
        elif component == "weekofyear":
            features[component] = datetime_series.dt.isocalendar().week

    return pd.DataFrame(features)


# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================


class SymplecticFeatureTransformer(BaseFeatureTransformer):
    """
    Feature transformation based on symplectic geometry.

    Inspired by Hamiltonian dynamics and symplectic forms.
    """

    def __init__(
        self,
        n_pairs: int = 5,
        preserve_structure: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.n_pairs = n_pairs
        self.preserve_structure = preserve_structure
        self.pairs_: Optional[List[Tuple[int, int]]] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "SymplecticFeatureTransformer":
        X = self._validate_input(X)
        n_features = X.shape[1]

        # Select feature pairs based on correlation
        correlations = np.corrcoef(X.values.T)
        pairs = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                pairs.append((abs(correlations[i, j]), i, j))

        pairs.sort(reverse=True)
        self.pairs_ = [(i, j) for _, i, j in pairs[: self.n_pairs]]

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        X_arr = X.values

        symplectic_features = []

        for i, j in self.pairs_:
            x_i = X_arr[:, i]
            x_j = X_arr[:, j]

            # Create symplectic pair (q, p) -> (q, p)
            if self.preserve_structure:
                # Symplectic invariant: q^2 + p^2 (like energy)
                energy = x_i**2 + x_j**2
                symplectic_features.append(energy.reshape(-1, 1))

                # Symplectic area element
                area = x_i * x_j
                symplectic_features.append(area.reshape(-1, 1))
            else:
                # Simple concatenation
                symplectic_features.append(x_i.reshape(-1, 1))
                symplectic_features.append(x_j.reshape(-1, 1))

        return (
            np.hstack(symplectic_features)
            if symplectic_features
            else np.array([]).reshape(X_arr.shape[0], 0)
        )


class FisherInformationFeatureSelection(BaseFeatureTransformer):
    """
    Feature selection based on Fisher Information.

    Uses Fisher Information Matrix to select most informative features.
    """

    def __init__(
        self,
        n_features: int = 10,
        regularization: float = 0.01,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.n_features = n_features
        self.regularization = regularization
        self.fisher_scores_: Optional[NDArray] = None
        self.support_: Optional[NDArray] = None

    def fit(
        self, X: Union[NDArray, pd.DataFrame], y: Union[NDArray, pd.Series]
    ) -> "FisherInformationFeatureSelection":
        X = self._validate_input(X)
        X_arr = X.values

        if isinstance(y, pd.Series):
            y = y.values

        # Compute Fisher Information scores
        # Approximation: diagonal of Fisher Information Matrix
        n_features = X_arr.shape[1]
        fisher_scores = np.zeros(n_features)

        for i in range(n_features):
            # Compute variance as proxy for Fisher information
            variance = np.var(X_arr[:, i])
            # Regularize
            fisher_scores[i] = variance / (1 + self.regularization * variance)

        self.fisher_scores_ = fisher_scores

        # Select top features
        top_indices = np.argsort(fisher_scores)[-self.n_features :]
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[top_indices] = True

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)
        return X.values[:, self.support_]


class CategoricalFeatureComposer(BaseFeatureTransformer):
    """
    Compose categorical features using category theory concepts.

    Creates functor-like mappings between categorical spaces.
    """

    def __init__(
        self,
        composition_depth: int = 2,
        min_support: float = 0.01,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.composition_depth = composition_depth
        self.min_support = min_support
        self.compositions_: Optional[Dict] = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Optional[Union[NDArray, pd.Series]] = None,
    ) -> "CategoricalFeatureComposer":
        X = self._validate_input(X)

        # Find frequent itemsets (simple version)
        self.compositions_ = {}
        n_samples = len(X)

        for depth in range(2, self.composition_depth + 1):
            from itertools import combinations

            for col_indices in combinations(range(X.shape[1]), depth):
                # Count co-occurrences
                value_counts = X.groupby([X.columns[i] for i in col_indices]).size()

                # Filter by minimum support
                frequent = value_counts[value_counts / n_samples >= self.min_support]

                if len(frequent) > 0:
                    key = tuple(col_indices)
                    self.compositions_[key] = frequent.index.tolist()

        self._is_fitted = True
        return self

    def transform(self, X: Union[NDArray, pd.DataFrame]) -> NDArray:
        self._check_is_fitted()
        X = self._validate_input(X)

        composed_features = []

        for col_indices, frequent_combinations in self.compositions_.items():
            # Create binary features for each frequent combination
            for combo in frequent_combinations:
                mask = np.ones(len(X), dtype=bool)
                for i, col_idx in enumerate(col_indices):
                    mask &= X.iloc[:, col_idx] == combo[i]

                composed_features.append(mask.astype(int).reshape(-1, 1))

        if composed_features:
            return np.hstack(composed_features)
        else:
            return np.array([]).reshape(len(X), 0)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Configuration
    "FeatureConfig",
    "ScalerType",
    "EncoderType",
    "SelectionMethod",
    "TextVectorizerType",
    # Base
    "BaseFeatureTransformer",
    # Numerical
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "Normalizer",
    "BinningTransformer",
    "PolynomialFeatures",
    "LogTransformer",
    # Categorical
    "OneHotEncoder",
    "LabelEncoder",
    "OrdinalEncoder",
    "TargetEncoder",
    "HashingEncoder",
    "EmbeddingEncoder",
    # Text
    "TFIDF",
    "CountVectorizer",
    "Word2VecFeatures",
    "BERTFeatures",
    # Time
    "DateTimeExtractor",
    "CyclicalEncoder",
    "LagFeatures",
    "RollingFeatures",
    # Selection
    "VarianceThreshold",
    "SelectKBest",
    "RFE",
    "LassoFeature",
    "MutualInformation",
    # Generation
    "AutoFeatureGenerator",
    "FeatureInteraction",
    "FeatureCrossing",
    # Utilities
    "FeaturePipeline",
    "FeatureUnion",
    "ColumnTransformer",
    # Advanced
    "SymplecticFeatureTransformer",
    "FisherInformationFeatureSelection",
    "CategoricalFeatureComposer",
    # Utility functions
    "infer_feature_types",
    "create_default_pipeline",
    "select_features_by_variance",
    "select_features_by_correlation",
    "generate_polynomial_features",
    "create_interaction_features",
    "encode_cyclical_features",
    "extract_datetime_features",
]

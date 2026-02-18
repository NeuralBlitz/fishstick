"""
Fisher Score Feature Selector for fishstick

Uses Fisher score for feature ranking and selection.
"""

from typing import Optional, Union
import numpy as np
import torch
from scipy.spatial.distance import cdist

from .base import BaseSelector, SupervisedSelector, SelectionResult


class FisherScoreSelector(SupervisedSelector):
    """
    Fisher score feature selector.

    The Fisher score measures the ratio of between-class variance
    to within-class variance. Higher scores indicate more
    discriminative features.

    For multi-class problems, uses the generalized Fisher score.

    Args:
        n_features_to_select: Number of features to select

    Example:
        >>> selector = FisherScoreSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
    ):
        super().__init__(n_features_to_select=n_features_to_select)

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "FisherScoreSelector":
        """
        Compute Fisher score for each feature.

        Args:
            X: Input features (n_samples, n_features)
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        classes = np.unique(y_np)
        n_classes = len(classes)

        if n_classes == 2:
            self.scores_ = self._binary_fisher_score(X_np, y_np)
        else:
            self.scores_ = self._multi_class_fisher_score(X_np, y_np)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]
        self.selected_features_ = indices

        return self

    def _binary_fisher_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute Fisher score for binary classification."""
        scores = np.zeros(X.shape[1])

        class_0 = X[y == 0]
        class_1 = X[y == 1]

        mean_0 = np.mean(class_0, axis=0)
        mean_1 = np.mean(class_1, axis=0)

        var_0 = np.var(class_0, axis=0)
        var_1 = np.var(class_1, axis=0)

        between_class = (mean_0 - mean_1) ** 2
        within_class = var_0 + var_1

        scores = between_class / (within_class + 1e-10)

        return scores

    def _multi_class_fisher_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute generalized Fisher score for multi-class."""
        n_samples, n_features = X.shape
        classes = np.unique(y)

        overall_mean = np.mean(X, axis=0)

        between_class = np.zeros(n_features)
        within_class = np.zeros(n_features)

        for c in classes:
            class_X = X[y == c]
            n_c = len(class_X)

            class_mean = np.mean(class_X, axis=0)
            between_class += n_c * (class_mean - overall_mean) ** 2

            class_var = np.var(class_X, axis=0)
            within_class += n_c * class_var

        between_class /= n_samples
        within_class /= n_samples

        scores = between_class / (within_class + 1e-10)

        return scores


class GeneralizedFisherScore(SupervisedSelector):
    """
    Generalized Fisher Score (GFS) for multi-class problems.

    Implements the trace ratio formulation for optimal feature selection.

    Args:
        n_features_to_select: Number of features to select
        iterations: Number of iterations for trace ratio optimization

    Example:
        >>> selector = GeneralizedFisherScore(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        iterations: int = 100,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.iterations = iterations

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "GeneralizedFisherScore":
        """
        Compute generalized Fisher score.

        Args:
            X: Input features
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        self.scores_ = self._compute_gfs(X_np, y_np)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]
        self.selected_features_ = indices

        return self

    def _compute_gfs(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute generalized Fisher score."""
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        S_b = np.zeros((n_features, n_features))
        S_w = np.zeros((n_features, n_features))

        overall_mean = np.mean(X, axis=0)

        for c in classes:
            class_X = X[y == c]
            n_c = len(class_X)

            class_mean = np.mean(class_X, axis=0)

            S_b += n_c * np.outer(class_mean - overall_mean, class_mean - overall_mean)

            centered = class_X - class_mean
            S_w += centered.T @ centered

        S_b /= n_samples
        S_w /= n_samples

        try:
            eigenvalues, eigenvectors = np.linalg.eig(
                np.linalg.inv(S_w + 1e-10 * np.eye(n_features)) @ S_b
            )
            scores = np.real(eigenvalues)
        except np.linalg.LinAlgError:
            scores = np.ones(n_features)

        return scores


class TraceRatioSelector(SupervisedSelector):
    """
    Trace Ratio feature selector.

    Uses the trace ratio criterion for feature selection,
    which is optimal for discriminative feature selection.

    Args:
        n_features_to_select: Number of features to select
        gamma: Regularization parameter

    Example:
        >>> selector = TraceRatioSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        gamma: float = 0.0,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.gamma = gamma

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "TraceRatioSelector":
        """Compute trace ratio scores."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        self.scores_ = self._trace_ratio_score(X_np, y_np)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]
        self.selected_features_ = indices

        return self

    def _trace_ratio_score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute trace ratio for each feature."""
        n_features = X.shape[1]
        scores = np.zeros(n_features)

        classes = np.unique(y)

        for i in range(n_features):
            feature_X = X[:, i].reshape(-1, 1)

            overall_mean = np.mean(feature_X)

            between_class = 0
            within_class = 0

            for c in classes:
                class_X = feature_X[y == c]
                n_c = len(class_X)

                class_mean = np.mean(class_X)
                between_class += n_c * (class_mean - overall_mean) ** 2

                within_class += np.sum((class_X - class_mean) ** 2)

            scores[i] = between_class / (within_class + 1e-10)

        return scores


def fisher_score_selector(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    n_features: Optional[int] = None,
) -> SelectionResult:
    """
    Functional interface for Fisher score feature selection.

    Args:
        X: Input features
        y: Target labels
        n_features: Number of features to select

    Returns:
        SelectionResult
    """
    selector = FisherScoreSelector(n_features_to_select=n_features)
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="fisher_score",
    )

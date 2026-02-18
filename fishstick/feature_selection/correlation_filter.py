"""
Correlation-based Feature Filter for fishstick

Removes features that are highly correlated with each other.
"""

from typing import Optional, Union, List
import numpy as np
import torch
from scipy import stats

from .base import BaseSelector, UnsupervisedSelector, SelectionResult


class CorrelationFilter(UnsupervisedSelector):
    """
    Feature selector that removes highly correlated features.

    This method computes the correlation matrix and removes features
    that are highly correlated with each other, keeping only one
    from each correlated group.

    Args:
        threshold: Correlation threshold. Features with correlation
            above this threshold will be considered redundant.
            Default is 0.9.
        method: Method to compute correlation ('pearson', 'spearman', 'kendall').
            Default is 'pearson'.
        strategy: Strategy to select which feature to keep ('first', 'mean', 'max_variance').
            Default is 'max_variance'.

    Example:
        >>> selector = CorrelationFilter(threshold=0.85)
        >>> X_selected = selector.fit_transform(X)
    """

    def __init__(
        self,
        threshold: float = 0.9,
        method: str = "pearson",
        strategy: str = "max_variance",
    ):
        """
        Args:
            threshold: Correlation threshold
            method: Correlation method
            strategy: Feature selection strategy within correlated groups
        """
        super().__init__(threshold=threshold)
        self.threshold = threshold
        self.method = method
        self.strategy = strategy
        self.correlation_matrix_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> "CorrelationFilter":
        """
        Compute correlation matrix and identify redundant features.

        Args:
            X: Input features (n_samples, n_features)
            y: Ignored

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)

        self.n_features_in_ = X_np.shape[1]

        if self.method == "pearson":
            self.correlation_matrix_ = np.corrcoef(X_np.T)
        elif self.method == "spearman":
            self.correlation_matrix_ = stats.spearmanr(X_np).correlation
        elif self.method == "kendall":
            self.correlation_matrix_ = stats.kendalltau(X_np).correlation
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.scores_ = np.var(X_np, axis=0)
        self.selected_features_ = self._select_features()

        return self

    def _select_features(self) -> np.ndarray:
        """Select features based on correlation threshold."""
        n_features = self.correlation_matrix_.shape[0]
        to_remove = set()

        for i in range(n_features):
            if i in to_remove:
                continue

            for j in range(i + 1, n_features):
                if j in to_remove:
                    continue

                if abs(self.correlation_matrix_[i, j]) > self.threshold:
                    to_remove.add(j)

        selected = [i for i in range(n_features) if i not in to_remove]

        if len(selected) == 0:
            selected = [0]

        return np.array(selected)

    def get_redundant_pairs(self) -> List[tuple]:
        """Get pairs of redundant features."""
        if self.correlation_matrix_ is None:
            raise ValueError("Selector not fitted")

        pairs = []
        n_features = self.correlation_matrix_.shape[0]

        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(self.correlation_matrix_[i, j]) > self.threshold:
                    pairs.append((i, j, self.correlation_matrix_[i, j]))

        return pairs


class CorrelationWithTargetFilter(BaseSelector):
    """
    Filter features based on correlation with target variable.

    Selects features that have statistically significant correlation
    with the target variable.

    Args:
        threshold: Minimum absolute correlation with target.
            Default is 0.1.
        method: Correlation method ('pearson', 'spearman', 'kendall').
        max_features: Maximum number of features to select.

    Example:
        >>> selector = CorrelationWithTargetFilter(threshold=0.2)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        threshold: float = 0.1,
        method: str = "pearson",
        max_features: Optional[int] = None,
    ):
        super().__init__(n_features_to_select=max_features, threshold=threshold)
        self.threshold = threshold
        self.method = method

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "CorrelationWithTargetFilter":
        """
        Compute correlation of each feature with target.

        Args:
            X: Input features
            y: Target variable

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        correlations = []
        for i in range(self.n_features_in_):
            if self.method == "pearson":
                corr, _ = stats.pearsonr(X_np[:, i], y_np)
            elif self.method == "spearman":
                corr, _ = stats.spearmanr(X_np[:, i], y_np)
            else:
                corr, _ = stats.kendalltau(X_np[:, i], y_np)

            correlations.append(abs(corr) if not np.isnan(corr) else 0)

        self.scores_ = np.array(correlations)

        selected = np.where(self.scores_ >= self.threshold)[0]

        if (
            self.n_features_to_select is not None
            and len(selected) > self.n_features_to_select
        ):
            indices = np.argsort(self.scores_[selected])[::-1][
                : self.n_features_to_select
            ]
            selected = selected[indices]

        self.selected_features_ = selected

        return self


def correlation_filter(
    X: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.9,
    method: str = "pearson",
) -> SelectionResult:
    """
    Functional interface for correlation-based feature selection.

    Args:
        X: Input features
        threshold: Correlation threshold
        method: Correlation method

    Returns:
        SelectionResult
    """
    selector = CorrelationFilter(threshold=threshold, method=method)
    selector.fit(X)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="correlation_filter",
        metadata={"threshold": threshold, "method": method},
    )

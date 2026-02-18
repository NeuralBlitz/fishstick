"""
ANOVA F-Test Feature Selector for fishstick

Univariate filter method using F-statistic for feature selection.
"""

from typing import Optional, Union, Tuple
import numpy as np
import torch
from scipy import stats

from . import BaseSelector, SupervisedSelector, SelectionResult


class ANOVAFTestSelector(SupervisedSelector):
    """
    ANOVA F-test feature selector.

    Computes the F-statistic between each feature and target variable.
    For classification, tests if means differ across classes.
    For regression, tests linear relationship with target.

    Args:
        n_features_to_select: Number of features to select
        alpha: Significance level for hypothesis testing

    Example:
        >>> selector = ANOVAFTestSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        alpha: float = 0.05,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.alpha = alpha

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "ANOVAFTestSelector":
        """
        Compute ANOVA F-scores for each feature.

        Args:
            X: Input features (n_samples, n_features)
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        n_classes = len(np.unique(y_np))

        if n_classes > 1:
            self.scores_, self.p_values_ = self._anova_f_score(X_np, y_np)
        else:
            self.scores_ = np.var(X_np, axis=0)
            self.p_values_ = np.ones(self.n_features_in_)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]
        self.selected_features_ = indices

        return self

    def _anova_f_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ANOVA F-scores for all features."""
        scores = np.zeros(X.shape[1])
        p_values = np.zeros(X.shape[1])

        classes = np.unique(y)

        for i in range(X.shape[1]):
            groups = [X[y == c, i] for c in classes]
            f_stat, p_val = stats.f_oneway(*groups)
            scores[i] = f_stat if not np.isnan(f_stat) else 0.0
            p_values[i] = p_val if not np.isnan(p_val) else 1.0

        return scores, p_values

    def get_significant_features(self) -> np.ndarray:
        """Get features with p-value below alpha."""
        if not hasattr(self, "p_values_"):
            raise ValueError("Selector has not been fitted yet.")
        return np.where(self.p_values_ < self.alpha)[0]


class FRegressionSelector(SupervisedSelector):
    """
    F-regression feature selector.

    Tests linear relationship between each feature and target.
    Uses Pearson correlation coefficient converted to F-statistic.

    Args:
        n_features_to_select: Number of features to select

    Example:
        >>> selector = FRegressionSelector(n_features_to_select=10)
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
    ) -> "FRegressionSelector":
        """
        Compute F-regression scores.

        Args:
            X: Input features
            y: Target values

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        y_centered = y_np - np.mean(y_np)

        correlations = np.array(
            [np.corrcoef(X_np[:, i], y_np)[0, 1] for i in range(X_np.shape[1])]
        )

        correlations = np.nan_to_num(correlations, nan=0.0)

        n_samples = X_np.shape[0]
        self.scores_ = (
            (correlations**2) * (n_samples - 2) / (1 - correlations**2 + 1e-10)
        )

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]
        self.selected_features_ = indices

        return self


def anova_f_test(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    n_features: Optional[int] = None,
) -> SelectionResult:
    """
    Functional interface for ANOVA F-test feature selection.

    Args:
        X: Input features
        y: Target labels
        n_features: Number of features to select

    Returns:
        SelectionResult
    """
    selector = ANOVAFTestSelector(n_features_to_select=n_features)
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="anova_f_test",
    )

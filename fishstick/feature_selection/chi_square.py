"""
Chi-Square Feature Selector for fishstick

Uses chi-squared test for feature selection.
"""

from typing import Optional, Union
import numpy as np
import torch
from sklearn.feature_selection import chi2, SelectKBest

from .base import BaseSelector, SupervisedSelector, SelectionResult


class ChiSquareSelector(SupervisedSelector):
    """
    Chi-squared feature selector for classification.

    Uses chi-squared test to select features with highest chi-squared
    statistic. Works only with non-negative features.

    Args:
        n_features_to_select: Number of features to select
        alpha: Significance level for hypothesis testing

    Example:
        >>> selector = ChiSquareSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        alpha: float = 0.05,
    ):
        """
        Args:
            n_features_to_select: Number of features to select
            alpha: Significance level
        """
        super().__init__(n_features_to_select=n_features_to_select)
        self.alpha = alpha
        self.p_values_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "ChiSquareSelector":
        """
        Compute chi-squared scores for each feature.

        Args:
            X: Input features (n_samples, n_features). Must be non-negative.
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        if np.any(X_np < 0):
            raise ValueError("Chi-square selector requires non-negative features")

        self.n_features_in_ = X_np.shape[1]

        chi2_scores, p_values = chi2(X_np, y_np)

        self.scores_ = chi2_scores
        self.p_values_ = p_values

        n_select = self._parse_n_features(self.n_features_in_)

        if self.alpha is not None:
            from scipy.stats import chi2 as chi2_dist

            significant = p_values < self.alpha
            significant_indices = np.where(significant)[0]

            if len(significant_indices) > 0:
                sorted_idx = np.argsort(chi2_scores[significant_indices])[::-1]
                self.selected_features_ = significant_indices[sorted_idx[:n_select]]
            else:
                indices = np.argsort(chi2_scores)[::-1][:n_select]
                self.selected_features_ = indices
        else:
            indices = np.argsort(chi2_scores)[::-1][:n_select]
            self.selected_features_ = indices

        return self


class SelectKBestWrapper:
    """
    Wrapper for sklearn's SelectKBest with various scoring functions.

    Args:
        score_func: Scoring function ('chi2', 'f_classif', 'f_regression', 'mutual_info')
        k: Number of features to select

    Example:
        >>> selector = SelectKBestWrapper(score_func='chi2', k=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        score_func: str = "chi2",
        k: int = 10,
    ):
        """
        Args:
            score_func: Scoring function name
            k: Number of features
        """
        self.score_func = score_func
        self.k = k
        self.selector_ = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "SelectKBestWrapper":
        """Fit the selector."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        if self.score_func == "chi2":
            from sklearn.feature_selection import chi2

            func = chi2
        elif self.score_func == "f_classif":
            from sklearn.feature_selection import f_classif

            func = f_classif
        elif self.score_func == "f_regression":
            from sklearn.feature_selection import f_regression

            func = f_regression
        else:
            raise ValueError(f"Unknown score function: {self.score_func}")

        self.selector_ = SelectKBest(score_func=func, k=min(self.k, X_np.shape[1]))
        self.selector_.fit(X_np, y_np)

        return self

    def transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Transform to selected features."""
        X_np, is_torch = self._to_numpy(X)
        X_sel = self.selector_.transform(X_np)

        return self._to_torch(X_sel, is_torch)

    def fit_transform(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def get_support(self) -> np.ndarray:
        """Get boolean mask of selected features."""
        return self.selector_.get_support()


def chi_square_selector(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    k: int = 10,
) -> SelectionResult:
    """
    Functional interface for chi-square feature selection.

    Args:
        X: Input features (non-negative)
        y: Target labels
        k: Number of features to select

    Returns:
        SelectionResult
    """
    selector = ChiSquareSelector(n_features_to_select=k)
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="chi_square",
        metadata={"p_values": selector.p_values_},
    )

"""
Variance Threshold Feature Selector for fishstick

Removes features with variance below a threshold.
"""

from typing import Optional, Union
import numpy as np
import torch

from .base import BaseSelector, UnsupervisedSelector, SelectionResult


class VarianceThresholdSelector(UnsupervisedSelector):
    """
    Feature selector that removes all low-variance features.

    This is an unsupervised method that only considers the variance
    of each feature. Features with variance below the threshold
    are removed.

    From: scikit-learn VarianceThreshold

    Args:
        threshold: Features with variance below this threshold are removed.
            Default is 0.0 (remove constant features).

    Example:
        >>> selector = VarianceThresholdSelector(threshold=0.1)
        >>> X_selected = selector.fit_transform(X)
        >>> print(selector.scores_)  # Variance of each feature
    """

    def __init__(
        self,
        threshold: float = 0.0,
    ):
        """
        Args:
            threshold: Variance threshold
        """
        super().__init__(threshold=threshold)
        self.threshold = threshold

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> "VarianceThresholdSelector":
        """
        Compute variance for each feature and select features.

        Args:
            X: Input features (n_samples, n_features)
            y: Ignored (present for API consistency)

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)

        self.n_features_in_ = X_np.shape[1]
        self.scores_ = np.var(X_np, axis=0)

        if self.threshold is not None:
            self.selected_features_ = np.where(self.scores_ > self.threshold)[0]
        else:
            self.selected_features_ = np.arange(self.n_features_in_)

        return self


class VarianceThresholdCVSelector(UnsupervisedSelector):
    """
    Variance threshold selector with cross-validation for threshold selection.

    Automatically finds optimal variance threshold using cross-validation
    with a downstream classifier.

    Args:
        n_thresholds: Number of thresholds to try
        cv: Number of cross-validation folds
        estimator: Classifier to use for evaluation

    Example:
        >>> selector = VarianceThresholdCVSelector(n_thresholds=20, cv=5)
        >>> selector.fit(X, y)
        >>> X_selected = selector.transform(X)
    """

    def __init__(
        self,
        n_thresholds: int = 20,
        cv: int = 5,
        estimator: Optional[any] = None,
    ):
        super().__init__()
        self.n_thresholds = n_thresholds
        self.cv = cv
        self.estimator = estimator
        self.best_threshold_: float = 0.0

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "VarianceThresholdCVSelector":
        """
        Find optimal variance threshold using cross-validation.

        Args:
            X: Input features (n_samples, n_features)
            y: Target labels

        Returns:
            self
        """
        from sklearn.model_selection import cross_val_score

        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]
        variances = np.var(X_np, axis=0)
        self.scores_ = variances

        thresholds = np.linspace(variances.min(), variances.max(), self.n_thresholds)

        if self.estimator is None:
            from sklearn.linear_model import LogisticRegression

            self.estimator = LogisticRegression(max_iter=1000)

        best_score = -np.inf
        best_threshold = 0.0

        for threshold in thresholds:
            selected = variances > threshold
            if selected.sum() == 0:
                continue

            X_sel = X_np[:, selected]
            scores = cross_val_score(self.estimator, X_sel, y_np, cv=self.cv)
            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_threshold = threshold

        self.best_threshold_ = best_threshold
        self.selected_features_ = np.where(variances > best_threshold)[0]

        return self


def variance_threshold_selector(
    X: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.0,
) -> SelectionResult:
    """
    Functional interface for variance threshold selection.

    Args:
        X: Input features
        threshold: Variance threshold

    Returns:
        SelectionResult with selected features
    """
    selector = VarianceThresholdSelector(threshold=threshold)
    selector.fit(X)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1] if isinstance(X, np.ndarray) else X.shape[1],
        method="variance_threshold",
        metadata={"threshold": threshold},
    )

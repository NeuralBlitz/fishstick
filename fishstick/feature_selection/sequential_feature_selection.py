"""
Sequential Feature Selection for fishstick

Wrapper method for forward/backward sequential selection.
"""

from typing import Optional, Union, Callable
import numpy as np
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score

from .base import BaseSelector, SupervisedSelector, SelectionResult


class SequentialFeatureSelector(SupervisedSelector):
    """
    Sequential Feature Selection (SFS) selector.

    Implements both Forward Selection and Backward Elimination:
    - Forward: Start with no features, add one at a time
    - Backward: Start with all features, remove one at a time

    Args:
        n_features_to_select: Number of features to select
        direction: 'forward' or 'backward'
        scoring: Scoring metric for evaluation
        cv: Number of CV folds
        estimator: Base estimator
        n_jobs: Parallel jobs

    Example:
        >>> selector = SequentialFeatureSelector(n_features_to_select=10, direction='forward')
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        direction: str = "forward",
        scoring: str = "accuracy",
        cv: int = 5,
        estimator: Optional[BaseEstimator] = None,
        n_jobs: int = -1,
    ):
        """
        Args:
            n_features_to_select: Number of features to select
            direction: Selection direction
            scoring: Scoring metric
            cv: CV folds
            estimator: Base estimator
            n_jobs: Parallel jobs
        """
        super().__init__(n_features_to_select=n_features_to_select, cv=cv)
        self.direction = direction
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.support_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "SequentialFeatureSelector":
        """
        Fit sequential feature selector.

        Args:
            X: Input features (n_samples, n_features)
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        if self.estimator is None:
            from sklearn.linear_model import LogisticRegression

            self.estimator = LogisticRegression(max_iter=1000)

        n_features_to_select = self._parse_n_features(self.n_features_in_)

        if self.direction == "forward":
            self._forward_selection(X_np, y_np, n_features_to_select)
        else:
            self._backward_elimination(X_np, y_np, n_features_to_select)

        return self

    def _forward_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features_to_select: int,
    ):
        """Implement forward sequential selection."""
        n_features = X.shape[1]
        selected = []
        remaining = list(range(n_features))

        best_scores = []

        for _ in range(n_features_to_select):
            best_score = -np.inf
            best_feature = None

            for feature in remaining:
                candidate = selected + [feature]
                score = self._evaluate_features(X, y, candidate)

                if score > best_score:
                    best_score = score
                    best_feature = feature

            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
                best_scores.append(best_score)

        self.selected_features_ = np.array(selected)
        self.scores_ = np.array(best_scores)
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[selected] = True

    def _backward_elimination(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features_to_select: int,
    ):
        """Implement backward sequential elimination."""
        n_features = X.shape[1]
        selected = list(range(n_features))

        best_scores = []

        initial_score = self._evaluate_features(X, y, selected)
        best_scores.append(initial_score)

        while len(selected) > n_features_to_select:
            best_score = -np.inf
            worst_feature = None

            for feature in selected:
                candidate = [f for f in selected if f != feature]
                score = self._evaluate_features(X, y, candidate)

                if score > best_score:
                    best_score = score
                    worst_feature = feature

            if worst_feature is not None:
                selected.remove(worst_feature)
                best_scores.append(best_score)

        self.selected_features_ = np.array(selected)
        self.scores_ = np.array(best_scores)
        self.support_ = np.zeros(n_features, dtype=bool)
        self.support_[selected] = True

    def _evaluate_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: list,
    ) -> float:
        """Evaluate feature set using cross-validation."""
        if len(feature_indices) == 0:
            return -np.inf

        X_sel = X[:, feature_indices]

        try:
            scores = cross_val_score(
                clone(self.estimator),
                X_sel,
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            return scores.mean()
        except Exception:
            return -np.inf


class SequentialFloatingSelector(SupervisedSelector):
    """
    Sequential Floating Forward Selection (SFFS).

    Adds features one at a time and optionally removes features
    that become redundant after addition.

    Args:
        n_features_to_select: Number of features to select
        scoring: Scoring metric
        cv: CV folds
        threshold: Threshold for feature removal

    Example:
        >>> selector = SequentialFloatingSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        scoring: str = "accuracy",
        cv: int = 5,
        threshold: float = 0.001,
        estimator: Optional[BaseEstimator] = None,
    ):
        super().__init__(n_features_to_select=n_features_to_select, cv=cv)
        self.scoring = scoring
        self.threshold = threshold
        self.estimator = estimator

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "SequentialFloatingSelector":
        """Fit floating selector."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        if self.estimator is None:
            from sklearn.linear_model import LogisticRegression

            self.estimator = LogisticRegression(max_iter=1000)

        n_features_to_select = self._parse_n_features(self.n_features_in_)

        self._sffs(X_np, y_np, n_features_to_select)

        return self

    def _sffs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features_to_select: int,
    ):
        """Implement SFFS algorithm."""
        selected = []
        remaining = list(range(self.n_features_in_))

        while len(selected) < n_features_to_select:
            best_feature = None
            best_score = -np.inf

            for feature in remaining:
                candidate = selected + [feature]
                score = self._evaluate_features(X, y, candidate)

                if score > best_score:
                    best_score = score
                    best_feature = feature

            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)

            if len(selected) > 2:
                improved = True
                while improved and len(selected) > 2:
                    improved = False

                    for feature in selected:
                        candidate = [f for f in selected if f != feature]
                        score_with = self._evaluate_features(X, y, selected)
                        score_without = self._evaluate_features(X, y, candidate)

                        if score_without > score_with + self.threshold:
                            selected.remove(feature)
                            remaining.append(feature)
                            improved = True

        self.selected_features_ = np.array(selected)
        self.scores_ = np.ones(len(selected))

    def _evaluate_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: list,
    ) -> float:
        """Evaluate feature set."""
        if len(feature_indices) == 0:
            return -np.inf

        X_sel = X[:, feature_indices]

        try:
            scores = cross_val_score(
                clone(self.estimator),
                X_sel,
                y,
                cv=self.cv,
                scoring=self.scoring,
            )
            return scores.mean()
        except Exception:
            return -np.inf


def sequential_feature_selector(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    n_features: Optional[int] = None,
    direction: str = "forward",
) -> SelectionResult:
    """
    Functional interface for sequential feature selection.

    Args:
        X: Input features
        y: Target labels
        n_features: Number of features to select
        direction: Forward or backward

    Returns:
        SelectionResult
    """
    selector = SequentialFeatureSelector(
        n_features_to_select=n_features,
        direction=direction,
    )
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method=f"sequential_{direction}",
    )

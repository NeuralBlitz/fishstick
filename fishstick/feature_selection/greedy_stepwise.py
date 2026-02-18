"""
Greedy Stepwise Feature Selector for fishstick

Wrapper method using forward or backward greedy selection.
"""

from typing import Optional, Union, Callable
import numpy as np
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score

from . import SupervisedSelector, SelectionResult


class GreedyStepwiseSelector(SupervisedSelector):
    """
    Greedy Stepwise feature selector.

    Implements both forward and backward greedy selection.
    Forward: Start with no features, add best at each step.
    Backward: Start with all features, remove worst at each step.

    Args:
        estimator: Base estimator for evaluation
        n_features_to_select: Number of features to select
        direction: 'forward' or 'backward'
        cv: Cross-validation folds
        scoring: Scoring metric
        floating: Whether to use floating search (forward only)
        tolerance: Minimum improvement threshold
        random_state: Random seed
        n_jobs: Parallel jobs

    Example:
        >>> selector = GreedyStepwiseSelector(
        ...     estimator=RandomForestClassifier(),
        ...     n_features_to_select=10,
        ...     direction='forward'
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_features_to_select: Optional[Union[int, float]] = None,
        direction: str = "forward",
        cv: int = 5,
        scoring: str = "accuracy",
        floating: bool = False,
        tolerance: float = 1e-4,
        random_state: Optional[int] = 42,
        n_jobs: int = 1,
    ):
        super().__init__(n_features_to_select=n_features_to_select, cv=cv)
        self.estimator = estimator
        self.direction = direction
        self.scoring = scoring
        self.floating = floating
        self.tolerance = tolerance
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scores_history_: list = []

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "GreedyStepwiseSelector":
        """
        Run greedy stepwise selection.

        Args:
            X: Input features (n_samples, n_features)
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]
        n_target = self._parse_n_features(self.n_features_in_)

        if self.estimator is None:
            from sklearn.linear_model import LogisticRegression

            self.estimator = LogisticRegression(random_state=42, max_iter=1000)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.scores_history_ = []

        if self.direction == "forward":
            self._forward_selection(X_np, y_np, n_target)
        else:
            self._backward_selection(X_np, y_np, n_target)

        return self

    def _forward_selection(self, X: np.ndarray, y: np.ndarray, n_target: int):
        """Forward greedy selection."""
        n_features = X.shape[1]
        selected = []
        remaining = list(range(n_features))

        best_score = -np.inf

        while len(selected) < n_target and remaining:
            best_feature = None
            best_candidate_score = -np.inf

            for feature in remaining:
                candidate = selected + [feature]

                try:
                    score = cross_val_score(
                        clone(self.estimator),
                        X[:, candidate],
                        y,
                        cv=self.cv,
                        scoring=self.scoring,
                        n_jobs=self.n_jobs,
                    ).mean()
                except Exception:
                    score = -np.inf

                if score > best_candidate_score:
                    best_candidate_score = score
                    best_feature = feature

                if self.floating and len(selected) > 0:
                    if score > best_score + self.tolerance:
                        candidate_without = selected[:-1] + [feature]

                        try:
                            score_without = cross_val_score(
                                clone(self.estimator),
                                X[:, candidate_without],
                                y,
                                cv=self.cv,
                                scoring=self.scoring,
                                n_jobs=self.n_jobs,
                            ).mean()
                        except Exception:
                            score_without = -np.inf

                        if score_without > best_candidate_score:
                            best_candidate_score = score_without
                            best_feature = feature
                            selected = candidate_without

            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
                best_score = best_candidate_score
                self.scores_history_.append(best_score)
            else:
                break

        self.selected_features_ = np.array(selected)
        self.scores_ = np.zeros(n_features)
        self.scores_[selected] = self.scores_history_[-1] if self.scores_history_ else 0

    def _backward_selection(self, X: np.ndarray, y: np.ndarray, n_target: int):
        """Backward greedy selection."""
        n_features = X.shape[1]
        selected = list(range(n_features))

        best_score = cross_val_score(
            clone(self.estimator),
            X,
            y,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        ).mean()

        self.scores_history_.append(best_score)

        while len(selected) > n_target:
            worst_feature = None
            best_candidate_score = -np.inf

            for feature in selected:
                candidate = [f for f in selected if f != feature]

                try:
                    score = cross_val_score(
                        clone(self.estimator),
                        X[:, candidate],
                        y,
                        cv=self.cv,
                        scoring=self.scoring,
                        n_jobs=self.n_jobs,
                    ).mean()
                except Exception:
                    score = -np.inf

                if score > best_candidate_score:
                    best_candidate_score = score
                    worst_feature = feature

            if (
                worst_feature is not None
                and best_candidate_score >= best_score - self.tolerance
            ):
                selected.remove(worst_feature)
                best_score = best_candidate_score
                self.scores_history_.append(best_score)
            else:
                break

        self.selected_features_ = np.array(selected)
        self.scores_ = np.zeros(n_features)
        self.scores_[selected] = self.scores_history_[-1]


class SequentialForwardSelector(GreedyStepwiseSelector):
    """
    Sequential Forward Selection (SFS).

    Simplified forward selection without floating search.

    Args:
        estimator: Base estimator
        n_features_to_select: Number of features to select
        cv: Cross-validation folds
        scoring: Scoring metric

    Example:
        >>> selector = SequentialForwardSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_features_to_select: Optional[Union[int, float]] = None,
        cv: int = 5,
        scoring: str = "accuracy",
    ):
        super().__init__(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            direction="forward",
            cv=cv,
            scoring=scoring,
            floating=False,
        )


class SequentialBackwardSelector(GreedyStepwiseSelector):
    """
    Sequential Backward Selection (SBS).

    Simplified backward selection without floating search.

    Args:
        estimator: Base estimator
        n_features_to_select: Number of features to select
        cv: Cross-validation folds
        scoring: Scoring metric
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_features_to_select: Optional[Union[int, float]] = None,
        cv: int = 5,
        scoring: str = "accuracy",
    ):
        super().__init__(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            direction="backward",
            cv=cv,
            scoring=scoring,
            floating=False,
        )


def greedy_stepwise(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    estimator: Optional[BaseEstimator] = None,
    n_features: Optional[int] = None,
    direction: str = "forward",
) -> SelectionResult:
    """
    Functional interface for greedy stepwise selection.

    Args:
        X: Input features
        y: Target labels
        estimator: Base estimator
        n_features: Number of features to select
        direction: 'forward' or 'backward'

    Returns:
        SelectionResult
    """
    selector = GreedyStepwiseSelector(
        estimator=estimator,
        n_features_to_select=n_features,
        direction=direction,
    )
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method=f"greedy_{direction}",
        metadata={"scores_history": selector.scores_history_},
    )

"""
Recursive Feature Elimination (RFE) for fishstick

Wrapper method for feature selection using recursive elimination.
"""

from typing import Optional, Union, Callable
import numpy as np
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score

from .base import BaseSelector, SupervisedSelector, SelectionResult


class RecursiveFeatureElimination(SupervisedSelector):
    """
    Recursive Feature Elimination (RFE) selector.

    RFE works by recursively considering smaller and smaller sets of features.
    First, the estimator is trained on the initial set of features, then
    features are eliminated based on their importance scores.

    From: scikit-learn RFE

    Args:
        estimator: Base estimator with fit and feature_importances_ or coef_
        n_features_to_select: Number of features to select
        step: Number of features to eliminate at each iteration
        cv: Number of cross-validation folds
        scoring: Scoring metric
        n_jobs: Number of parallel jobs

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> estimator = RandomForestClassifier(n_estimators=100)
        >>> selector = RecursiveFeatureElimination(estimator, n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_features_to_select: Optional[Union[int, float]] = None,
        step: int = 1,
        cv: int = 5,
        scoring: str = "accuracy",
        n_jobs: int = -1,
    ):
        """
        Args:
            estimator: Base estimator
            n_features_to_select: Number of features to select
            step: Features to eliminate per iteration
            cv: CV folds
            scoring: Scoring metric
            n_jobs: Parallel jobs
        """
        super().__init__(n_features_to_select=n_features_to_select, cv=cv)
        self.estimator = estimator
        self.step = step
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.ranking_: Optional[np.ndarray] = None
        self.support_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "RecursiveFeatureElimination":
        """
        Fit RFE selector.

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
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        n_features_to_select = self._parse_n_features(self.n_features_in_)

        if n_features_to_select >= self.n_features_in_:
            self.selected_features_ = np.arange(self.n_features_in_)
            self.ranking_ = np.ones(self.n_features_in_, dtype=int)
            return self

        self._rfe_fit(X_np, y_np, n_features_to_select)

        return self

    def _rfe_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features_to_select: int,
    ):
        """Implement RFE algorithm."""
        n_features = X.shape[1]

        if self.step > 1:
            elimination_step = self.step
        else:
            elimination_step = 1

        ranking = np.ones(n_features, dtype=int)
        support = np.ones(n_features, dtype=bool)

        current_features = np.arange(n_features)

        while current_features.shape[0] > n_features_to_select:
            n_to_remove = min(
                elimination_step, current_features.shape[0] - n_features_to_select
            )

            estimator = clone(self.estimator)
            estimator.fit(X[:, current_features], y)

            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                importances = np.abs(estimator.coef_)
                if importances.ndim > 1:
                    importances = np.max(importances, axis=1)
            else:
                raise ValueError("Estimator must have feature_importances_ or coef_")

            ranks = np.argsort(importances)[:-n_to_remove]

            ranking[current_features[ranks]] += 1

            current_features = current_features[ranks]

        support = np.zeros(n_features, dtype=bool)
        support[current_features] = True

        self.selected_features_ = np.where(support)[0]
        self.ranking_ = ranking
        self.support_ = support


class RFECV(SupervisedSelector):
    """
    Recursive Feature Elimination with Cross-Validation.

    Automatically selects the optimal number of features using CV.

    Args:
        estimator: Base estimator
        min_features_to_select: Minimum features to consider
        cv: Number of CV folds
        scoring: Scoring metric
        n_jobs: Parallel jobs

    Example:
        >>> selector = RFECV(estimator=RandomForestClassifier())
        >>> selector.fit(X, y)
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        min_features_to_select: int = 1,
        cv: int = 5,
        scoring: str = "accuracy",
        n_jobs: int = -1,
    ):
        super().__init__(n_features_to_select=None, cv=cv)
        self.estimator = estimator
        self.min_features_to_select = min_features_to_select
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv_results_: Optional[Dict] = None
        self.best_n_features_: int = 0

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "RFECV":
        """Fit RFECV selector."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        cv_scores = []
        n_features_range = range(self.min_features_to_select, self.n_features_in_ + 1)

        for n_features in n_features_range:
            scores = []
            from sklearn.model_selection import StratifiedKFold

            for train_idx, test_idx in StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=42
            ).split(X_np, y_np):
                X_train, X_test = X_np[train_idx], X_np[test_idx]
                y_train, y_test = y_np[train_idx], y_np[test_idx]

                selector = RecursiveFeatureElimination(
                    estimator=clone(self.estimator),
                    n_features_to_select=n_features,
                )
                selector.fit(X_train, y_train)

                estimator = clone(self.estimator)
                estimator.fit(selector.transform(X_train), y_train)

                score = cross_val_score(
                    estimator, selector.transform(X_test), y_test, scoring=self.scoring
                )
                scores.append(score.mean())

            cv_scores.append(np.mean(scores))

        self.cv_results_ = {
            "n_features": list(n_features_range),
            "scores": cv_scores,
        }

        best_idx = np.argmax(cv_scores)
        self.best_n_features_ = n_features_range[best_idx]

        final_selector = RecursiveFeatureElimination(
            estimator=clone(self.estimator),
            n_features_to_select=self.best_n_features_,
        )
        final_selector.fit(X_np, y_np)

        self.selected_features_ = final_selector.selected_features_
        self.scores_ = final_selector.scores_

        return self


def recursive_feature_elimination(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    estimator: Optional[BaseEstimator] = None,
    n_features: Optional[int] = None,
) -> SelectionResult:
    """
    Functional interface for RFE.

    Args:
        X: Input features
        y: Target labels
        estimator: Base estimator
        n_features: Number of features to select

    Returns:
        SelectionResult
    """
    selector = RecursiveFeatureElimination(
        estimator=estimator,
        n_features_to_select=n_features,
    )
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.ranking_,
        n_features=X.shape[1],
        method="rfe",
    )

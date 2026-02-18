"""
Elastic Net and Tree-based Feature Selectors for fishstick

Embedded methods using L1/L2 regularization and tree ensembles.
"""

from typing import Optional, Union
import numpy as np
import torch
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from . import SupervisedSelector, SelectionResult


class ElasticNetSelector(SupervisedSelector):
    """
    Elastic Net feature selector.

    Uses combined L1 and L2 regularization.
    Balances feature selection (Lasso-like) with ridge-like shrinkage.

    Args:
        alpha: Regularization strength
        l1_ratio: Mix of L1/L2 (0 = ridge, 1 = Lasso)
        n_features_to_select: Number of features to select
        max_iter: Maximum iterations
        cv: Cross-validation folds

    Example:
        >>> selector = ElasticNetSelector(alpha=0.1, l1_ratio=0.5, n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        n_features_to_select: Optional[Union[int, float]] = None,
        max_iter: int = 10000,
        cv: int = 5,
    ):
        super().__init__(n_features_to_select=n_features_to_select, cv=cv)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.model_: Optional[ElasticNet] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "ElasticNetSelector":
        """Fit Elastic Net selector."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        self.model_ = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=42,
        )

        self.model_.fit(X_np, y_np)

        self.scores_ = np.abs(self.model_.coef_)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]

        self.selected_features_ = indices

        return self


class ElasticNetCVSelector(SupervisedSelector):
    """
    Elastic Net with automatic parameter tuning.

    Uses cross-validation for alpha and l1_ratio selection.

    Args:
        n_features_to_select: Number of features to select
        l1_ratios: List of l1_ratio values to try
        cv: Cross-validation folds
        n_alphas: Number of alphas to try

    Example:
        >>> selector = ElasticNetCVSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        l1_ratios: Optional[list] = None,
        cv: int = 5,
        n_alphas: int = 100,
    ):
        super().__init__(n_features_to_select=n_features_to_select, cv=cv)
        self.l1_ratios = l1_ratios or [0.1, 0.3, 0.5, 0.7, 0.9]
        self.cv = cv
        self.n_alphas = n_alphas
        self.model_: Optional[ElasticNetCV] = None
        self.best_params_: Optional[dict] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "ElasticNetCVSelector":
        """Fit ElasticNetCV selector."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        self.model_ = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            cv=self.cv,
            n_alphas=self.n_alphas,
            random_state=42,
        )

        self.model_.fit(X_np, y_np)

        self.best_params_ = {
            "alpha": self.model_.alpha_,
            "l1_ratio": self.model_.l1_ratio_,
        }

        self.scores_ = np.abs(self.model_.coef_)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]

        self.selected_features_ = indices

        return self


class RandomForestImportanceSelector(SupervisedSelector):
    """
    Random Forest feature importance selector.

    Uses feature importance from Random Forest for selection.

    Args:
        n_features_to_select: Number of features to select
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        random_state: Random seed
        n_jobs: Parallel jobs

    Example:
        >>> selector = RandomForestImportanceSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = 42,
        n_jobs: int = 1,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: Optional[RandomForestClassifier] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "RandomForestImportanceSelector":
        """Fit Random Forest importance selector."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        n_classes = len(np.unique(y_np))

        if n_classes > 2:
            self.model_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

        self.model_.fit(X_np, y_np)

        self.scores_ = self.model_.feature_importances_

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]

        self.selected_features_ = indices

        return self


class GBDTImportanceSelector(SupervisedSelector):
    """
    Gradient Boosting feature importance selector.

    Uses feature importance from Gradient Boosting for selection.

    Args:
        n_features_to_select: Number of features to select
        n_estimators: Number of boosting stages
        learning_rate: Learning rate
        max_depth: Maximum tree depth
        random_state: Random seed

    Example:
        >>> selector = GBDTImportanceSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model_: Optional[GradientBoostingClassifier] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "GBDTImportanceSelector":
        """Fit GBDT importance selector."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        n_classes = len(np.unique(y_np))

        if n_classes > 2:
            self.model_ = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
        else:
            self.model_ = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )

        self.model_.fit(X_np, y_np)

        self.scores_ = self.model_.feature_importances_

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]

        self.selected_features_ = indices

        return self


class PermutationImportanceSelector(SupervisedSelector):
    """
    Permutation Importance feature selector.

    Model-agnostic importance using feature permutation.
    Measures performance drop when feature values are shuffled.

    Args:
        estimator: Base estimator
        n_features_to_select: Number of features to select
        n_repeats: Number of times to permute each feature
        random_state: Random seed
        n_jobs: Parallel jobs

    Example:
        >>> selector = PermutationImportanceSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: Optional = None,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_repeats: int = 10,
        random_state: Optional[int] = 42,
        n_jobs: int = 1,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.estimator = estimator
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "PermutationImportanceSelector":
        """Compute permutation importance."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(n_estimators=50, random_state=42)

        rng = np.random.RandomState(self.random_state)

        self.estimator.fit(X_np, y_np)

        from sklearn.model_selection import cross_val_score

        baseline_score = cross_val_score(
            self.estimator, X_np, y_np, cv=3, scoring="accuracy"
        ).mean()

        self.scores_ = np.zeros(self.n_features_in_)

        for i in range(self.n_features_in_):
            scores = []

            for _ in range(self.n_repeats):
                X_permuted = X_np.copy()
                X_permuted[:, i] = rng.permutation(X_permuted[:, i])

                try:
                    score = cross_val_score(
                        self.estimator, X_permuted, y_np, cv=3, scoring="accuracy"
                    ).mean()
                    scores.append(baseline_score - score)
                except Exception:
                    scores.append(0)

            self.scores_[i] = np.mean(scores)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]

        self.selected_features_ = indices

        return self


def elastic_net_selector(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    n_features: Optional[int] = None,
) -> SelectionResult:
    """
    Functional interface for Elastic Net feature selection.
    """
    selector = ElasticNetSelector(
        alpha=alpha, l1_ratio=l1_ratio, n_features_to_select=n_features
    )
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="elastic_net",
    )

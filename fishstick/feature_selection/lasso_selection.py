"""
Lasso-based Feature Selector for fishstick

Embedded method using L1 regularization.
"""

from typing import Optional, Union, List
import numpy as np
import torch
from sklearn.linear_model import Lasso, LassoCV

from . import SupervisedSelector, SelectionResult


class LassoSelector(SupervisedSelector):
    """
    Lasso feature selector.

    Uses L1 regularization (Lasso) to perform feature selection.
    Features with non-zero coefficients are selected.

    Args:
        alpha: Regularization strength (higher = more sparsity)
        n_features_to_select: Number of features to select (overrides threshold)
        max_iter: Maximum iterations for Lasso
        tol: Tolerance for optimization
        selection: Coefficient selection method ('cyclic' or 'random')
        cv: Cross-validation folds for alpha tuning

    Example:
        >>> selector = LassoSelector(alpha=0.1, n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_features_to_select: Optional[Union[int, float]] = None,
        max_iter: int = 10000,
        tol: float = 1e-4,
        selection: str = "cyclic",
        cv: int = 5,
    ):
        super().__init__(n_features_to_select=n_features_to_select, cv=cv)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection
        self.cv = cv
        self.lasso_model_: Optional[Lasso] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "LassoSelector":
        """
        Fit Lasso selector.

        Args:
            X: Input features (n_samples, n_features)
            y: Target values

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        self.lasso_model_ = Lasso(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            selection=self.selection,
            random_state=42,
        )

        self.lasso_model_.fit(X_np, y_np)

        self.scores_ = np.abs(self.lasso_model_.coef_)

        if self.n_features_to_select is not None:
            n_select = self._parse_n_features(self.n_features_in_)
            indices = np.argsort(self.scores_)[::-1][:n_select]
        else:
            indices = np.where(self.scores_ > 0)[0]

        self.selected_features_ = indices

        return self


class LassoCVSelector(SupervisedSelector):
    """
    LassoCV feature selector with automatic alpha tuning.

    Uses cross-validation to find optimal regularization strength.

    Args:
        n_features_to_select: Number of features to select
        max_iter: Maximum iterations
        tol: Tolerance
        cv: Cross-validation folds
        n_alphas: Number of alphas to try

    Example:
        >>> selector = LassoCVSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        max_iter: int = 10000,
        tol: float = 1e-4,
        cv: int = 5,
        n_alphas: int = 100,
    ):
        super().__init__(n_features_to_select=n_features_to_select, cv=cv)
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.n_alphas = n_alphas
        self.lasso_model_: Optional[LassoCV] = None
        self.best_alpha_: Optional[float] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "LassoCVSelector":
        """Fit LassoCV selector."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        self.lasso_model_ = LassoCV(
            cv=self.cv,
            max_iter=self.max_iter,
            tol=self.tol,
            n_alphas=self.n_alphas,
            random_state=42,
        )

        self.lasso_model_.fit(X_np, y_np)

        self.best_alpha_ = self.lasso_model_.alpha_
        self.scores_ = np.abs(self.lasso_model_.coef_)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]

        self.selected_features_ = indices

        return self


class LassoStabilitySelector(SupervisedSelector):
    """
    Lasso Stability Selection.

    Uses bootstrap sampling to select stable features with Lasso.
    Features selected consistently across bootstrap samples are chosen.

    Args:
        alpha: Regularization strength
        n_features_to_select: Number of features to select
        n_bootstrap: Number of bootstrap samples
        threshold: Selection frequency threshold (0-1)
        random_state: Random seed

    Example:
        >>> selector = LassoStabilitySelector(n_features_to_select=10, n_bootstrap=50)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_bootstrap: int = 50,
        threshold: float = 0.6,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.random_state = random_state
        self.selection_frequency_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "LassoStabilitySelector":
        """Fit stability selection."""
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        n_features = X_np.shape[1]
        selection_counts = np.zeros(n_features)

        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_bootstrap):
            indices = rng.choice(len(X_np), size=len(X_np), replace=True)

            X_boot = X_np[indices]
            y_boot = y_np[indices]

            lasso = Lasso(alpha=self.alpha, max_iter=5000, random_state=42)

            try:
                lasso.fit(X_boot, y_boot)
                selected = np.where(np.abs(lasso.coef_) > 0)[0]
                selection_counts[selected] += 1
            except Exception:
                continue

        self.selection_frequency_ = selection_counts / self.n_bootstrap

        stable_features = np.where(self.selection_frequency_ >= self.threshold)[0]

        if len(stable_features) == 0:
            stable_features = np.argsort(self.selection_frequency_)[::-1][:10]

        self.scores_ = self.selection_frequency_

        if self.n_features_to_select is not None:
            n_select = self._parse_n_features(self.n_features_in_)
            indices = np.argsort(self.scores_)[::-1][:n_select]
        else:
            indices = stable_features

        self.selected_features_ = indices

        return self


def lasso_selector(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    alpha: float = 1.0,
    n_features: Optional[int] = None,
) -> SelectionResult:
    """
    Functional interface for Lasso feature selection.

    Args:
        X: Input features
        y: Target values
        alpha: Regularization strength
        n_features: Number of features to select

    Returns:
        SelectionResult
    """
    selector = LassoSelector(alpha=alpha, n_features_to_select=n_features)
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="lasso",
    )

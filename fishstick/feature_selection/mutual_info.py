"""
Mutual Information Feature Selector for fishstick

Uses mutual information to select features.
"""

from typing import Optional, Union
import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .base import BaseSelector, SupervisedSelector, SelectionResult


class MutualInfoSelector(SupervisedSelector):
    """
    Feature selector based on mutual information.

    Computes mutual information between each feature and the target
    to measure the dependency. Features with higher mutual information
    are more informative.

    Args:
        n_features_to_select: Number of features to select
        method: 'classification' or 'regression'
        n_neighbors: Number of neighbors for MI estimation
        random_state: Random seed

    Example:
        >>> selector = MutualInfoSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        method: str = "classification",
        n_neighbors: int = 3,
        random_state: Optional[int] = 42,
    ):
        """
        Args:
            n_features_to_select: Number of features to select
            method: Classification or regression
            n_neighbors: Number of neighbors for estimation
            random_state: Random seed
        """
        super().__init__(n_features_to_select=n_features_to_select)
        self.method = method
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "MutualInfoSelector":
        """
        Compute mutual information for each feature.

        Args:
            X: Input features (n_samples, n_features)
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        if self.method == "classification":
            mi_scores = mutual_info_classif(
                X_np,
                y_np,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )
        else:
            mi_scores = mutual_info_regression(
                X_np,
                y_np,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )

        self.scores_ = mi_scores

        n_select = self._parse_n_features(self.n_features_in_)
        if n_select < self.n_features_in_:
            indices = np.argsort(mi_scores)[::-1][:n_select]
            self.selected_features_ = indices
        else:
            self.selected_features_ = np.arange(self.n_features_in_)

        return self


class MutualInfoRegressionSelector(MutualInfoSelector):
    """
    Mutual information selector specifically for regression tasks.
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_neighbors: int = 3,
        random_state: Optional[int] = 42,
    ):
        super().__init__(
            n_features_to_select=n_features_to_select,
            method="regression",
            n_neighbors=n_neighbors,
            random_state=random_state,
        )


class MutualInfoClassificationSelector(MutualInfoSelector):
    """
    Mutual information selector specifically for classification tasks.
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_neighbors: int = 3,
        random_state: Optional[int] = 42,
    ):
        super().__init__(
            n_features_to_select=n_features_to_select,
            method="classification",
            n_neighbors=n_neighbors,
            random_state=random_state,
        )


class ConditionalMutualInfoSelector(SupervisedSelector):
    """
    Conditional mutual information feature selector.

    Computes conditional mutual information I(X;Y|Z) to capture
    features that are informative about Y given Z.

    Args:
        n_features_to_select: Number of features to select
        n_neighbors: Number of neighbors for estimation
        random_state: Random seed

    Example:
        >>> selector = ConditionalMutualInfoSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_neighbors: int = 3,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "ConditionalMutualInfoSelector":
        """
        Compute conditional mutual information.

        Args:
            X: Input features
            y: Target

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        cmi_scores = np.zeros(self.n_features_in_)

        for i in range(self.n_features_in_):
            mi_xy = self._compute_mi(X_np[:, i], y_np)

            cmi = 0.0
            for j in range(self.n_features_in_):
                if i != j:
                    mi_xz = self._compute_mi(X_np[:, i], X_np[:, j])
                    if mi_xz > 0:
                        mi_xyz = self._compute_mi(X_np[:, i], y_np, X_np[:, j])
                        cmi += max(0, mi_xy - mi_xyz)

            cmi_scores[i] = cmi

        self.scores_ = cmi_scores

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(cmi_scores)[::-1][:n_select]
        self.selected_features_ = indices

        return self

    def _compute_mi(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray] = None,
    ) -> float:
        """Compute mutual information."""
        from sklearn.feature_selection import mutual_info_classif

        if z is None:
            mi = mutual_info_classif(
                x.reshape(-1, 1),
                y,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )
            return mi[0]
        return 0.0


def mutual_info_selector(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    n_features: Optional[int] = None,
    method: str = "classification",
) -> SelectionResult:
    """
    Functional interface for mutual information feature selection.

    Args:
        X: Input features
        y: Target variable
        n_features: Number of features to select
        method: Classification or regression

    Returns:
        SelectionResult
    """
    selector = MutualInfoSelector(
        n_features_to_select=n_features,
        method=method,
    )
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="mutual_info",
    )

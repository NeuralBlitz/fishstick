"""
ReliefF Feature Selector for fishstick

Instance-based filter algorithm for feature selection.
"""

from typing import Optional, Union, Dict, Any
import numpy as np
import torch
from scipy.spatial.distance import cdist

from . import BaseSelector, SupervisedSelector, SelectionResult


class ReliefFSelector(SupervisedSelector):
    """
    ReliefF feature selector.

    Instance-based algorithm that estimates feature quality based on
    how well they distinguish between nearest neighbors of different classes.

    Works well for classification with nominal and continuous features.
    Handles multi-class problems natively.

    Args:
        n_features_to_select: Number of features to select
        n_neighbors: Number of neighbors to use (default 5)
        sample_size: Number of samples to use (None = all)
        random_state: Random seed

    Example:
        >>> selector = ReliefFSelector(n_features_to_select=10, n_neighbors=5)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_neighbors: int = 5,
        sample_size: Optional[int] = None,
        random_state: Optional[int] = 42,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.n_neighbors = n_neighbors
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "ReliefFSelector":
        """
        Compute ReliefF scores for each feature.

        Args:
            X: Input features (n_samples, n_features)
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = X_np.shape[0]

        if self.sample_size is not None:
            sample_indices = np.random.choice(
                n_samples, min(self.sample_size, n_samples), replace=False
            )
        else:
            sample_indices = np.arange(n_samples)

        X_sample = X_np[sample_indices]
        y_sample = y_np[sample_indices]

        classes = np.unique(y_sample)
        class_probs = np.array([np.mean(y_sample == c) for c in classes])

        self.scores_ = self._compute_relief_scores(
            X_sample, y_sample, classes, class_probs
        )

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[::-1][:n_select]
        self.selected_features_ = indices

        return self

    def _compute_relief_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classes: np.ndarray,
        class_probs: np.ndarray,
    ) -> np.ndarray:
        """Compute ReliefF scores."""
        n_samples, n_features = X.shape
        scores = np.zeros(n_features)

        distances = cdist(X, X, metric="euclidean")
        np.fill_diagonal(distances, np.inf)

        for i in range(n_samples):
            sample = X[i]
            label = y[i]

            nearHits = []
            nearMisses = {c: [] for c in classes if c != label}

            sorted_indices = np.argsort(distances[i])

            for idx in sorted_indices:
                if len(nearHits) < self.n_neighbors and y[idx] == label:
                    nearHits.append(X[idx])
                elif y[idx] != label:
                    if len(nearMisses[y[idx]]) < self.n_neighbors:
                        nearMisses[y[idx]].append(X[idx])
                    if all(len(v) >= self.n_neighbors for v in nearMisses.values()):
                        break

            if len(nearHits) < self.n_neighbors:
                continue

            diff_hit = np.mean([np.abs(sample - hit) for hit in nearHits], axis=0)

            for c in classes:
                if c != label and len(nearMisses[c]) == self.n_neighbors:
                    diff_miss = np.mean(
                        [np.abs(sample - miss) for miss in nearMisses[c]], axis=0
                    )
                    scores += (diff_miss - diff_hit) * class_probs[c]

        scores /= n_samples

        return scores


class ReliefSelector(ReliefFSelector):
    """
    Original Relief algorithm for binary classification.

    Simplified version of ReliefF for binary classification problems.

    Args:
        n_features_to_select: Number of features to select
        n_neighbors: Number of neighbors to use

    Example:
        >>> selector = ReliefSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_neighbors: int = 5,
    ):
        super().__init__(
            n_features_to_select=n_features_to_select,
            n_neighbors=n_neighbors,
        )


def relieff_selector(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    n_features: Optional[int] = None,
    n_neighbors: int = 5,
) -> SelectionResult:
    """
    Functional interface for ReliefF feature selection.

    Args:
        X: Input features
        y: Target labels
        n_features: Number of features to select
        n_neighbors: Number of neighbors

    Returns:
        SelectionResult
    """
    selector = ReliefFSelector(
        n_features_to_select=n_features,
        n_neighbors=n_neighbors,
    )
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="relieff",
    )

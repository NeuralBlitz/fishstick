"""
Laplacian Score Feature Selector for fishstick

Manifold-based feature selection using graph Laplacian.
"""

from typing import Optional, Union
import numpy as np
import torch
from scipy.spatial.distance import cdist

from . import UnsupervisedSelector, SelectionResult


class LaplacianScoreSelector(UnsupervisedSelector):
    """
    Laplacian Score feature selector.

    Selects features that best preserve the local manifold structure.
    Features with lower Laplacian scores preserve neighborhood relationships better.

    Based on the assumption that nearby samples should have similar feature values.

    Args:
        n_features_to_select: Number of features to select
        n_neighbors: Number of neighbors for graph construction
        t: Heat kernel parameter (None = auto)

    Example:
        >>> selector = LaplacianScoreSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_neighbors: int = 5,
        t: Optional[float] = None,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.n_neighbors = n_neighbors
        self.t = t

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> "LaplacianScoreSelector":
        """
        Compute Laplacian scores for each feature.

        Args:
            X: Input features (n_samples, n_features)
            y: Ignored (unsupervised method)

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)

        self.n_features_in_ = X_np.shape[1]

        self.scores_ = self._compute_laplacian_scores(X_np)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[:n_select]
        self.selected_features_ = indices

        return self

    def _compute_laplacian_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute Laplacian score for each feature."""
        n_samples, n_features = X.shape

        S = self._construct_similarity_matrix(X)

        D = np.diag(S.sum(axis=1))
        L = D - S

        X_normalized = X - np.mean(X, axis=0)

        scores = np.zeros(n_features)

        for i in range(n_features):
            f = X_normalized[:, i]

            numerator = f @ L @ f
            denominator = f @ D @ f

            if denominator > 1e-10:
                scores[i] = numerator / denominator
            else:
                scores[i] = 0.0

        return scores

    def _construct_similarity_matrix(self, X: np.ndarray) -> np.ndarray:
        """Construct similarity matrix using heat kernel."""
        n_samples = X.shape[0]

        distances = cdist(X, X, metric="euclidean")
        np.fill_diagonal(distances, np.inf)

        k = min(self.n_neighbors, n_samples - 1)

        sorted_distances = np.sort(distances, axis=1)
        sigma = sorted_distances[:, k - 1 : k].mean()

        if sigma < 1e-10:
            sigma = 1.0

        if self.t is not None:
            t = self.t
        else:
            t = sigma**2

        S = np.exp(-(distances**2) / (2 * t))
        np.fill_diagonal(S, 0)

        S[S < 0] = 0

        return S


class MultiClusterLaplacianScoreSelector(UnsupervisedSelector):
    """
    Multi-cluster Laplacian Score selector.

    Extends Laplacian Score to handle multiple clusters in the data.

    Args:
        n_features_to_select: Number of features to select
        n_clusters: Number of clusters
        n_neighbors: Number of neighbors

    Example:
        >>> selector = MultiClusterLaplacianScoreSelector(n_features_to_select=10)
        >>> X_selected = selector.fit_transform(X)
    """

    def __init__(
        self,
        n_features_to_select: Optional[Union[int, float]] = None,
        n_clusters: int = 3,
        n_neighbors: int = 5,
    ):
        super().__init__(n_features_to_select=n_features_to_select)
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> "MultiClusterLaplacianScoreSelector":
        """Compute multi-cluster Laplacian scores."""
        X_np, is_torch = self._to_numpy(X)

        self.n_features_in_ = X_np.shape[1]

        if y is None:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            y = kmeans.fit_predict(X_np)

        y_np = self._to_numpy(y)[0] if not isinstance(y, np.ndarray) else y

        self.scores_ = self._compute_multi_cluster_scores(X_np, y_np)

        n_select = self._parse_n_features(self.n_features_in_)
        indices = np.argsort(self.scores_)[:n_select]
        self.selected_features_ = indices

        return self

    def _compute_multi_cluster_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Laplacian scores for each cluster."""
        n_samples, n_features = X.shape
        classes = np.unique(y)

        scores = np.zeros(n_features)

        for c in classes:
            mask = y == c
            X_c = X[mask]

            n_c = X_c.shape[0]

            distances = cdist(X_c, X_c, metric="euclidean")
            np.fill_diagonal(distances, np.inf)

            k = min(self.n_neighbors, n_c - 1)
            sorted_distances = np.sort(distances, axis=1)
            sigma = sorted_distances[:, k - 1 : k].mean()

            if sigma < 1e-10:
                sigma = 1.0

            S = np.exp(-(distances**2) / (2 * sigma**2))
            np.fill_diagonal(S, 0)

            D = np.diag(S.sum(axis=1))
            L = D - S

            X_normalized = X_c - np.mean(X_c, axis=0)

            for i in range(n_features):
                f = X_normalized[:, i]
                numerator = f @ L @ f
                denominator = f @ D @ f

                if denominator > 1e-10:
                    scores[i] += (n_c / n_samples) * (numerator / denominator)

        return scores


def laplacian_score(
    X: Union[np.ndarray, torch.Tensor],
    n_features: Optional[int] = None,
    n_neighbors: int = 5,
) -> SelectionResult:
    """
    Functional interface for Laplacian Score feature selection.

    Args:
        X: Input features
        n_features: Number of features to select
        n_neighbors: Number of neighbors

    Returns:
        SelectionResult
    """
    selector = LaplacianScoreSelector(
        n_features_to_select=n_features,
        n_neighbors=n_neighbors,
    )
    selector.fit(X)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="laplacian_score",
    )

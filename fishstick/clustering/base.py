"""
Base classes and utilities for clustering algorithms.

Provides common infrastructure for:
- Cluster result containers
- Distance metrics
- Cluster validity indices
- Base clusterer classes
"""

from typing import Optional, Tuple, List, Union, Callable
from dataclasses import dataclass, field
import torch
from torch import Tensor
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


@dataclass
class ClusterResult:
    """Result of a clustering algorithm."""

    labels: Tensor
    centroids: Optional[Tensor] = None
    n_clusters: int = 0
    inertia: Optional[float] = None
    n_iter: int = 0
    method: str = ""
    cluster_centers_: Optional[Tensor] = None
    labels_: Optional[Tensor] = None


class DistanceMetric:
    """Computes various distance metrics for clustering."""

    def __init__(self, metric: str = "euclidean", p: float = 2.0):
        self.metric = metric
        self.p = p

    def pairwise(
        self, X: Tensor, Y: Optional[Tensor] = None
    ) -> Union[Tensor, np.ndarray]:
        """Compute pairwise distances."""
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        Y_np = (
            Y.cpu().numpy()
            if Y is not None and Y.is_cuda
            else (Y.numpy() if Y is not None else None)
        )

        if self.metric == "precomputed":
            return X_np if Y is None else None

        D = pairwise_distances(X_np, Y_np, metric=self.metric, p=self.p)
        result = torch.from_numpy(D)
        if X.is_cuda:
            result = result.cuda()
        return result

    def cdist(self, X: Tensor, Y: Tensor) -> Tensor:
        """Compute distance between sets of vectors."""
        return self.pairwise(X, Y)


class GraphBuilder:
    """Constructs neighborhood graphs for density-based clustering."""

    def __init__(
        self,
        n_neighbors: int = 10,
        metric: str = "euclidean",
        method: str = "knn",
        epsilon: Optional[float] = None,
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.method = method
        self.epsilon = epsilon

    def fit(self, X: Tensor) -> "GraphBuilder":
        """Fit the graph builder to data."""
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        self.nn_ = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            metric=self.metric,
        )
        self.nn_.fit(X_np)
        return self

    def kneighbors(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Find k-neighbor distances and indices."""
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        distances, indices = self.nn_.kneighbors(X_np)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        return (
            torch.from_numpy(distances),
            torch.from_numpy(indices),
        )

    def radius_neighbors(self, X: Tensor, radius: float) -> Tuple[List, List]:
        """Find neighbors within radius."""
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        distances, indices = self.nn_.radius_neighbors(X_np, radius)
        return distances, indices


class ClusterValidityIndex:
    """Computes cluster validity indices for evaluating clustering."""

    @staticmethod
    def silhouette_score(X: Tensor, labels: Tensor) -> float:
        """Compute silhouette coefficient."""
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
        from sklearn.metrics import silhouette_score

        return silhouette_score(X_np, labels_np)

    @staticmethod
    def davies_bouldin_score(X: Tensor, labels: Tensor) -> float:
        """Compute Davies-Bouldin index."""
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
        from sklearn.metrics import davies_bouldin_score

        return davies_bouldin_score(X_np, labels_np)

    @staticmethod
    def calinski_harabasz_score(X: Tensor, labels: Tensor) -> float:
        """Compute Calinski-Harabasz score."""
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
        from sklearn.metrics import calinski_harabasz_score

        return calinski_harabasz_score(X_np, labels_np)

    @staticmethod
    def dunn_index(X: Tensor, labels: Tensor) -> float:
        """Compute Dunn index (higher is better)."""
        unique_labels = torch.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return 0.0

        inter_cluster_distances = []
        intra_cluster_diameters = []

        for i in range(n_clusters):
            cluster_i = X[labels == i]
            if len(cluster_i) == 0:
                continue

            for j in range(i + 1, n_clusters):
                cluster_j = X[labels == j]
                if len(cluster_j) == 0:
                    continue

                dist = torch.cdist(cluster_i, cluster_j).min()
                inter_cluster_distances.append(dist)

            diameter = torch.cdist(cluster_i, cluster_i).max()
            intra_cluster_diameters.append(diameter)

        if not inter_cluster_diameters or not intra_cluster_diameters:
            return 0.0

        return min(inter_cluster_distances) / max(intra_cluster_diameters)


class ClustererBase:
    """Base class for all clustering algorithms."""

    def __init__(self, n_clusters: int = 8, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_: Optional[Tensor] = None
        self.n_iter_: int = 0

    def fit(self, X: Tensor) -> "ClustererBase":
        """Fit the clustering algorithm to data."""
        raise NotImplementedError("Subclasses must implement fit()")

    def fit_predict(self, X: Tensor) -> Tensor:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_

    def predict(self, X: Tensor) -> Tensor:
        """Predict cluster labels for new data."""
        raise NotImplementedError("Subclasses must implement predict()")


def compute_distance_matrix(
    X: Tensor, metric: str = "euclidean", p: float = 2.0
) -> Tensor:
    """Compute pairwise distance matrix."""
    metric_obj = DistanceMetric(metric=metric, p=p)
    return metric_obj.pairwise(X)


def initialize_centroids(
    X: Tensor,
    n_clusters: int,
    method: str = "kmeans++",
    random_state: Optional[int] = None,
) -> Tensor:
    """Initialize cluster centroids using various methods."""
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    n_samples = X.shape[0]

    if method == "random":
        indices = torch.randperm(n_samples)[:n_clusters]
        return X[indices].clone()

    elif method == "kmeans++":
        centroids = []
        idx = torch.randint(0, n_samples, (1,)).item()
        centroids.append(X[idx].clone())

        for _ in range(n_clusters - 1):
            D = torch.cdist(X, torch.stack(centroids))
            min_distances = D.min(dim=1)[0]
            probs = min_distances / min_distances.sum()
            idx = torch.multinomial(probs, 1).item()
            centroids.append(X[idx].clone())

        return torch.stack(centroids)

    elif method == "first":
        return X[:n_clusters].clone()

    return X[torch.randperm(n_samples)[:n_clusters]].clone()


def compute_inertia(X: Tensor, centroids: Tensor, labels: Tensor) -> float:
    """Compute within-cluster sum of squares (inertia)."""
    total = 0.0
    for k in range(centroids.shape[0]):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            distances = torch.cdist(cluster_points, centroids[k : k + 1])
            total += (distances**2).sum().item()
    return total


def relabel_clusters(labels: Tensor) -> Tensor:
    """Relabel cluster IDs to be consecutive starting from 0."""
    unique = torch.unique(labels)
    mapping = {old.item(): new for new, old in enumerate(unique)}
    return torch.tensor(
        [mapping[l.item()] for l in labels],
        dtype=labels.dtype,
        device=labels.device,
    )

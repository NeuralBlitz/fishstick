"""
Hierarchical/Agglomerative clustering algorithms.

Implements:
- Agglomerative clustering with various linkage methods
- Divisive clustering
- BIRCH-like hierarchical clustering
- Dendrogram utilities
- Distance-based tree cutting
"""

from typing import Optional, List, Tuple, Callable, Union
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram
from scipy.spatial.distance import pdist

from .base import (
    ClustererBase,
    ClusterResult,
    DistanceMetric,
    relabel_clusters,
)


@dataclass
class HierarchicalResult(ClusterResult):
    """Result of hierarchical clustering."""

    n_leaves: int = 0
    n_components: int = 0
    distance_threshold: Optional[float] = None


class AgglomerativeClustering(ClustererBase):
    """
    Agglomerative hierarchical clustering.

    Bottom-up approach that merges clusters iteratively.

    Args:
        n_clusters: Number of clusters (or None for distance_threshold)
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        linkage: Linkage method ('single', 'complete', 'average', 'ward')
        distance_threshold: Distance threshold for cutting dendrogram
        memory: Memory for caching linkage computation
    """

    def __init__(
        self,
        n_clusters: Optional[int] = 8,
        metric: str = "euclidean",
        linkage: str = "ward",
        distance_threshold: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)
        self.metric = metric
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.children_: Optional[Tensor] = None
        self.distances_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "AgglomerativeClustering":
        """Fit agglomerative clustering to data."""
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()

        if self.linkage == "ward":
            from sklearn.cluster import AgglomerativeClustering as SklearnAgg

            sk = SklearnAgg(
                n_clusters=self.n_clusters,
                metric=self.metric,
                linkage=self.linkage,
                distance_threshold=self.distance_threshold,
            )
            sk.fit(X_np)
            self.labels_ = torch.from_numpy(sk.labels_).to(X.device)
            if hasattr(sk, "n_clusters_"):
                self.n_clusters = sk.n_clusters_
        else:
            from sklearn.cluster import AgglomerativeClustering as SklearnAgg

            sk = SklearnAgg(
                n_clusters=self.n_clusters,
                metric=self.metric,
                linkage=self.linkage,
                distance_threshold=self.distance_threshold,
            )
            sk.fit(X_np)
            self.labels_ = torch.from_numpy(sk.labels_).to(X.device)
            self.n_clusters = len(torch.unique(self.labels_))

        return self

    def fit_predict(self, X: Tensor) -> Tensor:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class DivisiveClustering(ClustererBase):
    """
    Divisive hierarchical clustering.

    Top-down approach that splits clusters recursively.

    Args:
        n_clusters: Number of clusters
        random_state: Random seed
    """

    def __init__(
        self,
        n_clusters: int = 8,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)

    def fit(self, X: Tensor) -> "DivisiveClustering":
        """Fit divisive clustering to data."""
        n_samples = X.shape[0]
        device = X.device

        active_clusters = [torch.arange(n_samples, device=device)]

        while len(active_clusters) < self.n_clusters:
            best_split = None
            best_inertia = -float("inf")

            for idx, cluster_indices in enumerate(active_clusters):
                if len(cluster_indices) < 2:
                    continue

                cluster_data = X[cluster_indices]
                centroid = cluster_data.mean(dim=0)
                distances = torch.norm(cluster_data - centroid, dim=1)
                inertia = distances.sum().item()

                if inertia > best_inertia:
                    best_inertia = inertia
                    best_split = (idx, cluster_indices)

            if best_split is None:
                break

            idx, cluster_indices = best_split
            cluster_data = X[cluster_indices]
            centroid = cluster_data.mean(dim=0)

            left_mask = torch.norm(cluster_data - centroid, dim=1) < torch.median(
                torch.norm(cluster_data - centroid, dim=1)
            )
            right_mask = ~left_mask

            left_indices = cluster_indices[left_mask]
            right_indices = cluster_indices[right_mask]

            active_clusters.pop(idx)
            active_clusters.append(left_indices)
            active_clusters.append(right_indices)

        labels = torch.zeros(n_samples, dtype=torch.long, device=device)
        for k, cluster_indices in enumerate(active_clusters):
            labels[cluster_indices] = k

        self.labels_ = relabel_clusters(labels)
        self.n_clusters = len(torch.unique(self.labels_))

        return self


class BIRCH(ClustererBase):
    """
    BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies).

    Memory-efficient hierarchical clustering for large datasets.

    Args:
        n_clusters: Number of clusters
        threshold: Threshold for radius of subcluster
        branching_factor: Maximum number of subclusters per node
        random_state: Random seed
    """

    def __init__(
        self,
        n_clusters: int = 8,
        threshold: float = 0.5,
        branching_factor: int = 50,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)
        self.threshold = threshold
        self.branching_factor = branching_factor

    def fit(self, X: Tensor) -> "BIRCH":
        """Fit BIRCH to data."""
        from sklearn.cluster import Birch

        birch = Birch(
            n_clusters=self.n_clusters,
            threshold=self.threshold,
            branching_factor=self.branching_factor,
        )
        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        birch.fit(X_np)
        self.labels_ = torch.from_numpy(birch.labels_).to(X.device)
        self.n_clusters = len(torch.unique(self.labels_))
        return self


class HDBSCAN:
    """
    HDBSCAN (Hierarchical DBSCAN).

    Robust density-based clustering that builds a hierarchy of clusterings.

    Args:
        min_cluster_size: Minimum cluster size
        min_samples: Number of samples in neighborhood for core points
        metric: Distance metric
        cluster_selection_epsilon: Distance threshold for cluster selection
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        cluster_selection_epsilon: float = 0.0,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.labels_: Optional[Tensor] = None
        self.n_clusters_: int = 0

    def fit(self, X: Tensor) -> "HDBSCAN":
        """Fit HDBSCAN to data."""
        import hdbscan

        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )
        clusterer.fit(X_np)
        self.labels_ = torch.from_numpy(clusterer.labels_).to(X.device)
        self.n_clusters_ = len(torch.unique(self.labels_[self.labels_ >= 0]))
        return self

    def fit_predict(self, X: Tensor) -> Tensor:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class SpectralClustering(ClustererBase):
    """
    Spectral clustering using graph Laplacian.

    Works well for non-convex clusters.

    Args:
        n_clusters: Number of clusters
        affinity: Affinity type ('nearest_neighbors', 'rbf', 'precomputed')
        gamma: Kernel coefficient for rbf
        n_neighbors: Number of neighbors for nearest_neighbors
    """

    def __init__(
        self,
        n_clusters: int = 8,
        affinity: str = "rbf",
        gamma: float = 1.0,
        n_neighbors: int = 10,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors

    def fit(self, X: Tensor) -> "SpectralClustering":
        """Fit spectral clustering to data."""
        from sklearn.cluster import SpectralClustering as SklearnSpectral

        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        spectral = SklearnSpectral(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=self.gamma,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        spectral.fit(X_np)
        self.labels_ = torch.from_numpy(spectral.labels_).to(X.device)
        self.n_clusters = len(torch.unique(self.labels_))
        return self


def compute_linkage_matrix(X: Tensor, method: str = "ward") -> Tensor:
    """Compute hierarchical clustering linkage matrix."""
    X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
    Z = scipy_linkage(X_np, method=method)
    return torch.from_numpy(Z)


def cut_tree(
    Z: Tensor, n_clusters: Optional[int] = None, height: Optional[float] = None
) -> Tensor:
    """Cut hierarchical clustering tree at specified level."""
    from scipy.cluster.hierarchy import cut_tree

    X_np = Z.cpu().numpy() if Z.is_cuda else Z.numpy()
    cuts = cut_tree(X_np, n_clusters=n_clusters, height=height)
    return torch.from_numpy(cuts.flatten())


def fcluster(Z: Tensor, t: Union[int, float], criterion: str = "maxclust") -> Tensor:
    """Form flat clusters from hierarchical clustering."""
    from scipy.cluster.hierarchy import fcluster

    X_np = Z.cpu().numpy() if Z.is_cuda else Z.numpy()
    labels = fcluster(X_np, t=t, criterion=criterion)
    return torch.from_numpy(labels)


def create_agglomerative_clustering(
    n_clusters: Optional[int] = 8,
    linkage: str = "ward",
    **kwargs,
) -> AgglomerativeClustering:
    """Factory function for agglomerative clustering."""
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **kwargs)


def create_divisive_clustering(
    n_clusters: int = 8,
    **kwargs,
) -> DivisiveClustering:
    """Factory function for divisive clustering."""
    return DivisiveClustering(n_clusters=n_clusters, **kwargs)


def create_birch(
    n_clusters: int = 8,
    **kwargs,
) -> BIRCH:
    """Factory function for BIRCH clustering."""
    return BIRCH(n_clusters=n_clusters, **kwargs)

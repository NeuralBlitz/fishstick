"""
K-Means and variants for clustering.

Implements:
- Standard K-Means
- Mini-Batch K-Means
- K-Means++
- Bisecting K-Means
- Elkan's algorithm
- K-Means with multiple initializations
"""

from typing import Optional, Tuple, Callable
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass

from .base import (
    ClustererBase,
    ClusterResult,
    initialize_centroids,
    compute_inertia,
    relabel_clusters,
    DistanceMetric,
)


@dataclass
class KMeansResult(ClusterResult):
    """Extended result for K-Means clustering."""

    best_init: int = 0
    init_centroids: Optional[Tensor] = None
    silhouette_score: Optional[float] = None


class KMeans(ClustererBase):
    """
    K-Means clustering algorithm.

    Partitions data into k clusters by minimizing within-cluster variance.
    Uses Lloyd's algorithm with optional Elkan's optimization.

    Args:
        n_clusters: Number of clusters
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        init: Initialization method ('kmeans++', 'random', 'first')
        n_init: Number of initializations to try
        random_state: Random seed for reproducibility
        verbose: Print progress information
        algorithm: 'lloyd' or 'elkan'
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: str = "kmeans++",
        n_init: int = 10,
        random_state: Optional[int] = None,
        verbose: bool = False,
        algorithm: str = "lloyd",
    ):
        super().__init__(n_clusters, random_state)
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.algorithm = algorithm
        self.centroids_: Optional[Tensor] = None
        self.inertia_: Optional[float] = None

    def fit(self, X: Tensor) -> "KMeans":
        """Fit K-Means to data."""
        best_result = None
        best_inertia = float("inf")

        for init_idx in range(self.n_init):
            if self.init == "random":
                centroids = initialize_centroids(
                    X,
                    self.n_clusters,
                    method="random",
                    random_state=self.random_state + init_idx,
                )
            elif self.init == "kmeans++":
                centroids = initialize_centroids(
                    X,
                    self.n_clusters,
                    method="kmeans++",
                    random_state=self.random_state + init_idx,
                )
            else:
                centroids = initialize_centroids(X, self.n_clusters, method="first")

            if self.algorithm == "elkan":
                centroids, labels, inertia, n_iter = self._elkan_iterate(X, centroids)
            else:
                centroids, labels, inertia, n_iter = self._lloyd_iterate(X, centroids)

            if inertia < best_inertia:
                best_inertia = inertia
                best_result = KMeansResult(
                    labels=labels,
                    centroids=centroids,
                    n_clusters=self.n_clusters,
                    inertia=inertia,
                    n_iter=n_iter,
                    method="kmeans",
                    best_init=init_idx,
                    init_centroids=centroids if init_idx == 0 else None,
                )

        self.labels_ = best_result.labels
        self.centroids_ = best_result.centroids
        self.inertia_ = best_result.inertia
        self.n_iter_ = best_result.n_iter
        self.n_clusters = len(torch.unique(self.labels_))

        return self

    def _lloyd_iterate(
        self, X: Tensor, centroids: Tensor
    ) -> Tuple[Tensor, Tensor, float, int]:
        """Standard Lloyd's algorithm iteration."""
        for i in range(self.max_iter):
            distances = torch.cdist(X, centroids)
            labels = distances.argmin(dim=1)

            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(dim=0)

            if torch.allclose(centroids, new_centroids, atol=self.tol):
                return centroids, labels, compute_inertia(X, centroids, labels), i + 1

            centroids = new_centroids

        return centroids, labels, compute_inertia(X, centroids, labels), self.max_iter

    def _elkan_iterate(
        self, X: Tensor, centroids: Tensor
    ) -> Tuple[Tensor, Tensor, float, int]:
        """Elkan's algorithm with triangle inequality optimization."""
        n_samples = X.shape[0]
        distances = torch.cdist(X, centroids)
        labels = distances.argmin(dim=1)
        centroid_distances = torch.cdist(centroids, centroids)

        for i in range(self.max_iter):
            new_centroids = torch.zeros_like(centroids)
            new_labels = labels.clone()

            for j in range(n_samples):
                k = labels[j]
                c1 = centroids[k]
                d1 = distances[j, k]

                for m in range(self.n_clusters):
                    if m == k:
                        continue

                    if 2 * d1 <= centroid_distances[k, m]:
                        continue

                    d2 = torch.norm(X[j] - centroids[m]).item()
                    distances[j, m] = d2

                    if d2 < d1:
                        new_labels[j] = m
                        d1 = d2
                        k = m

            for k in range(self.n_clusters):
                mask = new_labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(dim=0)

            labels = new_labels

            if torch.allclose(centroids, new_centroids, atol=self.tol):
                return centroids, labels, compute_inertia(X, centroids, labels), i + 1

            centroids = new_centroids
            distances = torch.cdist(X, centroids)
            centroid_distances = torch.cdist(centroids, centroids)

        return centroids, labels, compute_inertia(X, centroids, labels), self.max_iter

    def predict(self, X: Tensor) -> Tensor:
        """Predict cluster labels for new data."""
        distances = torch.cdist(X, self.centroids_)
        return distances.argmin(dim=1)

    def transform(self, X: Tensor) -> Tensor:
        """Transform data to distance space."""
        return torch.cdist(X, self.centroids_)


class MiniBatchKMeans(ClustererBase):
    """
    Mini-Batch K-Means clustering.

    Uses random mini-batches to speed up convergence for large datasets.

    Args:
        n_clusters: Number of clusters
        max_iter: Maximum number of iterations
        batch_size: Size of mini-batches
        init: Initialization method
        random_state: Random seed
        reassignment_ratio: Ratio for cluster reassignment
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        batch_size: int = 1024,
        init: str = "kmeans++",
        random_state: Optional[int] = None,
        reassignment_ratio: float = 0.01,
        verbose: bool = False,
    ):
        super().__init__(n_clusters, random_state)
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.init = init
        self.reassignment_ratio = reassignment_ratio
        self.verbose = verbose
        self.centroids_: Optional[Tensor] = None
        self.inertia_: Optional[float] = None
        self.counts_: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "MiniBatchKMeans":
        """Fit Mini-Batch K-Means to data."""
        n_samples = X.shape[0]
        device = X.device

        centroids = initialize_centroids(
            X, self.n_clusters, method=self.init, random_state=self.random_state
        )
        counts = torch.ones(self.n_clusters, device=device)

        for iteration in range(self.max_iter):
            indices = torch.randint(0, n_samples, (self.batch_size,), device=device)
            batch = X[indices]

            distances = torch.cdist(batch, centroids)
            batch_labels = distances.argmin(dim=1)

            for k in range(self.n_clusters):
                mask = batch_labels == k
                if mask.sum() > 0:
                    batch_points = batch[mask]
                    old_centroid = centroids[k]
                    new_centroid = batch_points.mean(dim=0)

                    counts[k] += mask.sum().item()
                    eta = 1.0 / counts[k]
                    centroids[k] = (1 - eta) * old_centroid + eta * new_centroid

            if iteration % 10 == 0:
                all_distances = torch.cdist(X, centroids)
                labels = all_distances.argmin(dim=1)
                inertia = compute_inertia(X, centroids, labels)

                if self.verbose:
                    print(f"Iteration {iteration}: Inertia = {inertia:.4f}")

        all_distances = torch.cdist(X, centroids)
        labels = all_distances.argmin(dim=1)

        self.labels_ = labels
        self.centroids_ = centroids
        self.counts_ = counts
        self.inertia_ = compute_inertia(X, centroids, labels)
        self.n_clusters = len(torch.unique(labels))

        return self

    def predict(self, X: Tensor) -> Tensor:
        """Predict cluster labels."""
        distances = torch.cdist(X, self.centroids_)
        return distances.argmin(dim=1)

    def transform(self, X: Tensor) -> Tensor:
        """Transform data to distance space."""
        return torch.cdist(X, self.centroids_)


class BisectingKMeans(ClustererBase):
    """
    Bisecting K-Means clustering.

    Recursively splits clusters using K-Means with k=2.

    Args:
        n_clusters: Number of clusters
        max_iter: Maximum iterations per K-Means step
        random_state: Random seed
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)
        self.max_iter = max_iter

    def fit(self, X: Tensor) -> "BisectingKMeans":
        """Fit Bisecting K-Means to data."""
        device = X.device
        n_samples = X.shape[0]

        clusters = [torch.arange(n_samples, device=device)]

        while len(clusters) < self.n_clusters:
            best_split = None
            best_inertia = -float("inf")

            for idx, cluster_indices in enumerate(clusters):
                if len(cluster_indices) < 2:
                    continue

                cluster_data = X[cluster_indices]
                kmeans = KMeans(
                    n_clusters=2,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    n_init=1,
                )
                kmeans.fit(cluster_data)
                inertia = kmeans.inertia_

                if inertia > best_inertia:
                    best_inertia = inertia
                    best_split = (idx, cluster_indices, kmeans.labels_)

            if best_split is None:
                break

            idx, cluster_indices, split_labels = best_split
            left_mask = split_labels == 0
            right_mask = split_labels == 1

            left_indices = cluster_indices[left_mask]
            right_indices = cluster_indices[right_mask]

            clusters.pop(idx)
            clusters.append(left_indices)
            clusters.append(right_indices)

        labels = torch.zeros(n_samples, dtype=torch.long, device=device)
        for k, cluster_indices in enumerate(clusters):
            labels[cluster_indices] = k

        self.labels_ = relabel_clusters(labels)

        centroids = torch.stack(
            [X[self.labels_ == k].mean(dim=0) for k in range(len(clusters))]
        )
        self.centroids_ = centroids
        self.inertia_ = compute_inertia(X, centroids, self.labels_)

        return self

    def predict(self, X: Tensor) -> Tensor:
        """Predict cluster labels."""
        distances = torch.cdist(X, self.centroids_)
        return distances.argmin(dim=1)


class KMeansPlusPlus(ClustererBase):
    """
    K-Means++ initialization and standalone clustering.

    Uses K-Means++ for smart initialization.

    Args:
        n_clusters: Number of clusters
        max_iter: Maximum iterations
        random_state: Random seed
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        random_state: Optional[int] = None,
    ):
        super().__init__(n_clusters, random_state)
        self.max_iter = max_iter

    def fit(self, X: Tensor) -> "KMeansPlusPlus":
        """Fit with K-Means++ initialization."""
        centroids = initialize_centroids(
            X, self.n_clusters, method="kmeans++", random_state=self.random_state
        )

        for i in range(self.max_iter):
            distances = torch.cdist(X, centroids)
            labels = distances.argmin(dim=1)

            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(dim=0)

            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break

            centroids = new_centroids

        distances = torch.cdist(X, centroids)
        self.labels_ = distances.argmin(dim=1)
        self.centroids_ = centroids
        self.inertia_ = compute_inertia(X, centroids, self.labels_)

        return self

    def predict(self, X: Tensor) -> Tensor:
        """Predict cluster labels."""
        distances = torch.cdist(X, self.centroids_)
        return distances.argmin(dim=1)


def create_kmeans(
    n_clusters: int = 8,
    algorithm: str = "lloyd",
    **kwargs,
) -> KMeans:
    """Factory function to create K-Means instance."""
    return KMeans(n_clusters=n_clusters, algorithm=algorithm, **kwargs)


def create_minibatch_kmeans(
    n_clusters: int = 8,
    batch_size: int = 1024,
    **kwargs,
) -> MiniBatchKMeans:
    """Factory function to create Mini-Batch K-Means instance."""
    return MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, **kwargs)


def create_bisecting_kmeans(
    n_clusters: int = 8,
    **kwargs,
) -> BisectingKMeans:
    """Factory function to create Bisecting K-Means instance."""
    return BisectingKMeans(n_clusters=n_clusters, **kwargs)

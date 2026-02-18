"""
DBSCAN and density-based clustering algorithms.

Implements:
- DBSCAN (Density-Based Spatial Clustering)
- OPTICS (Ordering Points To Identify Clustering Structure)
- Mean Shift clustering
- Density peaking clustering
"""

from typing import Optional, List, Tuple, Callable
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass

from .base import (
    ClustererBase,
    ClusterResult,
    GraphBuilder,
    relabel_clusters,
)


@dataclass
class DBSCANResult(ClusterResult):
    """Result of DBSCAN clustering."""

    core_sample_indices: Optional[Tensor] = None
    components_: Optional[Tensor] = None
    n_noise: int = 0


class DBSCAN(ClustererBase):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Finds core samples of high density and expands clusters from them.
    Does not require specifying number of clusters upfront.

    Args:
        eps: Maximum distance between samples for neighborhood
        min_samples: Number of samples in neighborhood for core points
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        algorithm: Algorithm for nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
        n_jobs: Number of parallel jobs
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
        n_jobs: Optional[int] = None,
    ):
        super().__init__(n_clusters=0, random_state=None)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self.core_sample_indices_: Optional[Tensor] = None
        self.components_: Optional[Tensor] = None
        self.n_noise_: int = 0

    def fit(self, X: Tensor) -> "DBSCAN":
        """Fit DBSCAN to data."""
        from sklearn.cluster import DBSCAN as SklearnDBSCAN

        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()

        db = SklearnDBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
        )
        db.fit(X_np)

        self.labels_ = torch.from_numpy(db.labels_).to(X.device)
        self.core_sample_indices_ = (
            torch.from_numpy(db.core_sample_indices_)
            if hasattr(db, "core_sample_indices_")
            and db.core_sample_indices_ is not None
            else None
        )
        self.components_ = (
            torch.from_numpy(db.components_)
            if hasattr(db, "components_") and db.components_ is not None
            else None
        )
        self.n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        self.n_noise_ = (db.labels_ == -1).sum()

        return self

    def fit_predict(self, X: Tensor) -> Tensor:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class OPTICS(ClustererBase):
    """
    OPTICS (Ordering Points To Identify Clustering Structure).

    Density-based clustering that produces a hierarchical structure.
    More robust than DBSCAN for varying density clusters.

    Args:
        min_samples: Minimum number of samples in neighborhood
        max_eps: Maximum distance for clustering
        metric: Distance metric
        cluster_method: Method to extract clusters ('xi', 'dbscan')
        xi: Xi parameter for xi clustering
        predecessor_correction: Use predecessor correction
    """

    def __init__(
        self,
        min_samples: int = 5,
        max_eps: float = float("inf"),
        metric: str = "euclidean",
        cluster_method: str = "xi",
        xi: float = 0.05,
        predecessor_correction: bool = True,
        min_cluster_size: Optional[float] = None,
    ):
        super().__init__(n_clusters=0, random_state=None)
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.cluster_method = cluster_method
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size = min_cluster_size

    def fit(self, X: Tensor) -> "OPTICS":
        """Fit OPTICS to data."""
        from sklearn.cluster import OPTICS as SklearnOPTICS

        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()

        optics = SklearnOPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric=self.metric,
            cluster_method=self.cluster_method,
            xi=self.xi,
            predecessor_correction=self.predecessor_correction,
            min_cluster_size=self.min_cluster_size,
        )
        optics.fit(X_np)

        self.labels_ = torch.from_numpy(optics.labels_).to(X.device)
        self.n_clusters_ = len(set(optics.labels_)) - (1 if -1 in optics.labels_ else 0)
        self.reachability_ = (
            torch.from_numpy(optics.reachability_)
            if hasattr(optics, "reachability_") and optics.reachability_ is not None
            else None
        )
        self.ordering_ = (
            torch.from_numpy(optics.ordering_)
            if hasattr(optics, "ordering_") and optics.ordering_ is not None
            else None
        )

        return self

    def fit_predict(self, X: Tensor) -> Tensor:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class MeanShift(ClustererBase):
    """
    Mean Shift clustering.

    Non-parametric clustering that finds cluster centers as modes of the density.
    Automatically determines number of clusters.

    Args:
        bandwidth: Bandwidth of the Gaussian kernel
        seeds: Seeds for clustering
        bin_seeding: Use bin seeding for initialization
        n_jobs: Number of parallel jobs
    """

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        seeds: Optional[Tensor] = None,
        bin_seeding: bool = False,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(n_clusters=0, random_state=None)
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.n_jobs = n_jobs

    def fit(self, X: Tensor) -> "MeanShift":
        """Fit Mean Shift to data."""
        from sklearn.cluster import MeanShift as SklearnMeanShift

        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()

        ms = SklearnMeanShift(
            bandwidth=self.bandwidth,
            bin_seeding=self.bin_seeding,
            n_jobs=self.n_jobs,
        )
        ms.fit(X_np)

        self.labels_ = torch.from_numpy(ms.labels_).to(X.device)
        self.cluster_centers_ = (
            torch.from_numpy(ms.cluster_centers_)
            if ms.cluster_centers_ is not None
            else None
        )
        self.n_clusters_ = ms.n_clusters_

        return self

    def fit_predict(self, X: Tensor) -> Tensor:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class MeanShiftTorch:
    """
    Mean Shift clustering implemented in PyTorch.

    Pure PyTorch implementation for GPU acceleration.

    Args:
        bandwidth: Bandwidth for the kernel
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
    """

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        max_iter: int = 300,
        tolerance: float = 1e-4,
    ):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.labels_: Optional[Tensor] = None
        self.cluster_centers_: Optional[Tensor] = None
        self.n_clusters_: int = 0

    def fit(self, X: Tensor) -> "MeanShiftTorch":
        """Fit Mean Shift to data using PyTorch."""
        if self.bandwidth is None:
            self.bandwidth = self._estimate_bandwidth(X)

        n_samples = X.shape[0]
        centroids = X.clone()
        device = X.device

        for iteration in range(self.max_iter):
            new_centroids = torch.zeros_like(centroids)

            for i in range(n_samples):
                distances = torch.norm(centroids - centroids[i], dim=1)
                weights = torch.exp(-(distances**2) / (2 * self.bandwidth**2))
                new_centroids[i] = (weights @ centroids) / weights.sum()

            shift = torch.norm(new_centroids - centroids, dim=1).max()
            centroids = new_centroids

            if shift < self.tolerance:
                break

        unique_centroids = self._unique_clusters(centroids)
        self.cluster_centers_ = unique_centroids

        labels = torch.zeros(X.shape[0], dtype=torch.long, device=device)
        for i, centroid in enumerate(unique_centroids):
            distances = torch.norm(X - centroid, dim=1)
            mask = distances == distances.min()
            labels[mask] = i

        self.labels_ = labels
        self.n_clusters_ = len(unique_centroids)

        return self

    def _estimate_bandwidth(self, X: Tensor, n_samples: int = 1000) -> float:
        """Estimate bandwidth using sklearn."""
        from sklearn.cluster import estimate_bandwidth

        X_np = X.cpu().numpy() if X.is_cuda else X.numpy()
        if X_np.shape[0] > n_samples:
            indices = np.random.choice(X_np.shape[0], n_samples, replace=False)
            X_np = X_np[indices]
        return estimate_bandwidth(X_np)

    def _unique_clusters(self, centroids: Tensor, tolerance: float = 1e-3) -> Tensor:
        """Remove duplicate cluster centers."""
        unique = []
        for c in centroids:
            is_unique = True
            for u in unique:
                if torch.norm(c - u) < tolerance:
                    is_unique = False
                    break
            if is_unique:
                unique.append(c)
        return torch.stack(unique) if unique else centroids[:1]

    def fit_predict(self, X: Tensor) -> Tensor:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class DensityPeakClustering(ClustererBase):
    """
    Density Peak Clustering.

    Identifies cluster centers as density peaks.
    Works well for datasets with varying densities.

    Args:
        rho: Density threshold (percentile)
        delta: Distance threshold (percentile)
        n_clusters: Number of clusters (auto if None)
    """

    def __init__(
        self,
        rho: float = 2.0,
        delta: float = 0.5,
        n_clusters: Optional[int] = None,
    ):
        super().__init__(n_clusters=n_clusters or 0, random_state=None)
        self.rho = rho
        self.delta = delta

    def fit(self, X: Tensor) -> "DensityPeakClustering":
        """Fit density peak clustering to data."""
        n_samples = X.shape[0]
        device = X.device

        distances = torch.cdist(X, X)
        distances[torch.arange(n_samples), torch.arange(n_samples)] = float("inf")

        rho = self._compute_density(distances)
        delta = self._compute_delta(distances, rho)

        if self.n_clusters > 0:
            cluster_centers = self._select_centers(rho, delta, self.n_clusters)
        else:
            cluster_centers = self._auto_select_centers(rho, delta)

        labels = torch.zeros(n_samples, dtype=torch.long, device=device)
        for i, center in enumerate(cluster_centers):
            distances_to_center = torch.norm(X - X[center], dim=1)
            mask = distances_to_center == distances_to_center.min()
            labels[mask] = i

        self.labels_ = labels
        self.cluster_centers_ = cluster_centers
        self.n_clusters_ = len(cluster_centers)
        self.rho_ = rho
        self.delta_ = delta

        return self

    def _compute_density(self, distances: Tensor) -> Tensor:
        """Compute local density."""
        dc = torch.kthvalue(distances.flatten(), int(distances.numel() * 0.02))[0]
        rho = torch.exp(-((distances / dc) ** 2)).sum(dim=1)
        return rho

    def _compute_delta(self, distances: Tensor, rho: Tensor) -> Tensor:
        """Compute distance to higher density point."""
        n = distances.shape[0]
        delta = torch.zeros(n, device=distances.device)

        sorted_indices = torch.argsort(rho, descending=True)
        for i, idx in enumerate(sorted_indices[1:], 1):
            min_dist = float("inf")
            for j in sorted_indices[:i]:
                d = distances[idx, j]
                if d < min_dist:
                    min_dist = d
            delta[idx] = min_dist

        delta[sorted_indices[0]] = delta.max()
        return delta

    def _select_centers(self, rho: Tensor, delta: Tensor, n_centers: int) -> Tensor:
        """Select cluster centers manually."""
        scores = rho * delta
        return torch.argsort(scores, descending=True)[:n_centers]

    def _auto_select_centers(self, rho: Tensor, delta: Tensor) -> Tensor:
        """Automatically select cluster centers."""
        rho_threshold = torch.kthvalue(rho, int(len(rho) * 0.1))[0]
        delta_threshold = torch.kthvalue(delta, int(len(delta) * 0.1))[0]

        centers = []
        for i in range(len(rho)):
            if rho[i] > rho_threshold and delta[i] > delta_threshold:
                centers.append(i)

        return torch.tensor(centers, dtype=torch.long, device=rho.device)

    def fit_predict(self, X: Tensor) -> Tensor:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


def create_dbscan(
    eps: float = 0.5,
    min_samples: int = 5,
    **kwargs,
) -> DBSCAN:
    """Factory function to create DBSCAN instance."""
    return DBSCAN(eps=eps, min_samples=min_samples, **kwargs)


def create_optics(
    min_samples: int = 5,
    cluster_method: str = "xi",
    **kwargs,
) -> OPTICS:
    """Factory function to create OPTICS instance."""
    return OPTICS(min_samples=min_samples, cluster_method=cluster_method, **kwargs)


def create_meanshift(
    bandwidth: Optional[float] = None,
    **kwargs,
) -> MeanShift:
    """Factory function to create MeanShift instance."""
    return MeanShift(bandwidth=bandwidth, **kwargs)

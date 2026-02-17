"""
Mapper Algorithm Implementation.

Provides the Mapper algorithm for topological summarization of high-dimensional data.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict
import torch
from torch import Tensor
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE


@dataclass
class MapperCover:
    """
    Cover configuration for Mapper algorithm.

    Defines how to cover the image space with overlapping intervals.
    """

    n_intervals: int
    overlap_percent: float = 0.5

    def get_intervals(self, values: np.ndarray) -> List[Tuple[float, float]]:
        """Generate overlapping intervals covering the value range."""
        min_val = np.min(values)
        max_val = np.max(values)

        if max_val - min_val < 1e-10:
            return [(min_val, max_val)]

        interval_length = (max_val - min_val) / self.n_intervals
        overlap = interval_length * self.overlap_percent

        intervals = []
        for i in range(self.n_intervals):
            start = min_val + i * interval_length - overlap
            end = min_val + (i + 1) * interval_length + overlap
            intervals.append((start, end))

        return intervals


@dataclass
class MapperCluster:
    """Represents a cluster in the Mapper graph."""

    points: List[int]
    simplices: List[Tuple[int, ...]] = field(default_factory=list)
    center: Optional[np.ndarray] = None


class Mapper:
    """
    Mapper Algorithm Implementation.

    The Mapper algorithm provides a topological summarization of high-dimensional data:
    1. Map data through a filter function (e.g., density, projection)
    2. Cover the image with overlapping intervals
    3. Cluster within each preimage
    4. Build simplicial complex from clusters

    This produces a simplified representation that preserves
    topological features of the data.
    """

    def __init__(
        self,
        n_cubes: int = 10,
        overlap: float = 0.5,
        clusterer: str = "dbscan",
        n_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
    ):
        """
        Initialize Mapper algorithm.

        Args:
            n_cubes: Number of intervals in cover
            overlap: Overlap percentage between intervals
            clusterer: Clustering method ('dbscan', 'agglomerative')
            n_clusters: Number of clusters for agglomerative clustering
            distance_threshold: Distance threshold for DBSCAN
        """
        self.n_cubes = n_cubes
        self.overlap = overlap
        self.clusterer = clusterer
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold

    def fit_transform(
        self,
        points: Tensor,
        filter_fn: Optional[Tensor] = None,
    ) -> Tuple[List[MapperCluster], List[Tuple[int, int]]]:
        """
        Apply Mapper algorithm to point cloud.

        Args:
            points: Input point cloud [n_points, dimension]
            filter_fn: Optional filter function values [n_points]

        Returns:
            Tuple of (clusters, edges) representing the Mapper graph
        """
        points_np = points.numpy() if isinstance(points, Tensor) else points

        if filter_fn is None:
            filter_fn = self._default_filter(points_np)
        else:
            filter_fn = (
                filter_fn.numpy() if isinstance(filter_fn, Tensor) else filter_fn
            )

        cover = MapperCover(self.n_cubes, self.overlap)
        intervals = cover.get_intervals(filter_fn)

        clusters = []
        cluster_id = 0

        for start, end in intervals:
            mask = (filter_fn >= start) & (filter_fn <= end)

            if np.sum(mask) == 0:
                continue

            subset_indices = np.where(mask)[0]
            subset_points = points_np[subset_indices]

            cluster_labels = self._cluster_points(subset_points)

            for label in np.unique(cluster_labels):
                if label == -1:
                    continue

                point_indices = subset_indices[cluster_labels == label].tolist()

                cluster = MapperCluster(
                    points=point_indices,
                    center=np.mean(points_np[point_indices], axis=0),
                )
                clusters.append(cluster)
                cluster_id += 1

        edges = self._build_edges(clusters, points_np)

        return clusters, edges

    def _default_filter(self, points: np.ndarray) -> np.ndarray:
        """Default filter: density estimation."""
        from scipy.stats import gaussian_kde

        if points.shape[0] < 10:
            return np.zeros(points.shape[0])

        try:
            kde = gaussian_kde(points.T)
            return kde(points.T)
        except:
            return np.sum(points**2, axis=1)

    def _cluster_points(self, points: np.ndarray) -> np.ndarray:
        """Cluster points using configured method."""
        if len(points) == 0:
            return np.array([])

        if self.clusterer == "dbscan":
            if self.distance_threshold is None:
                self.distance_threshold = 0.5

            clusterer = DBSCAN(eps=self.distance_threshold, min_samples=1)
            return clusterer.fit_predict(points)

        elif self.clusterer == "agglomerative":
            if self.n_clusters is None:
                self.n_clusters = min(3, len(points))

            if self.n_clusters >= len(points):
                return np.arange(len(points))

            clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage="single",
            )
            return clusterer.fit_predict(points)

        else:
            return np.zeros(len(points), dtype=int)

    def _build_edges(
        self,
        clusters: List[MapperCluster],
        points: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Build edges between overlapping clusters."""
        edges = []

        for i, cluster_i in enumerate(clusters):
            for j, cluster_j in enumerate(clusters):
                if i >= j:
                    continue

                if self._clusters_overlap(cluster_i, cluster_j):
                    edges.append((i, j))

        return edges

    def _clusters_overlap(
        self,
        cluster1: MapperCluster,
        cluster2: MapperCluster,
    ) -> bool:
        """Check if two clusters share points."""
        set1 = set(cluster1.points)
        set2 = set(cluster2.points)

        overlap = len(set1.intersection(set2))

        return overlap > 0


class SimplicialComplexBuilder:
    """
    Builds simplicial complex from Mapper output.

    Constructs the nerve complex from overlapping clusters
    using the nerve theorem.
    """

    def __init__(self, min_overlap: int = 1):
        self.min_overlap = min_overlap

    def build_nerve(
        self,
        clusters: List[MapperCluster],
    ) -> Tuple[List[Tuple[int, ...]], List[int]]:
        """
        Build nerve simplicial complex.

        The nerve contains a k-simplex for each k+1 clusters
        with non-empty intersection.

        Args:
            clusters: List of clusters from Mapper

        Returns:
            Tuple of (simplices, dimensions)
        """
        simplices = []
        dimensions = []

        for k in range(1, len(clusters) + 1):
            from itertools import combinations

            for combo in combinations(range(len(clusters)), k):
                intersection = self._get_intersection(combo, clusters)

                if len(intersection) >= self.min_overlap:
                    simplex = tuple(clusters[c].points[0] for c in combo)
                    simplices.append(simplex)
                    dimensions.append(k - 1)

        return simplices, dimensions

    def _get_intersection(
        self,
        cluster_indices: Tuple[int, ...],
        clusters: List[MapperCluster],
    ) -> set:
        """Get intersection of points in cluster collection."""
        if len(cluster_indices) == 0:
            return set()

        result = set(clusters[cluster_indices[0]].points)

        for idx in cluster_indices[1:]:
            result = result.intersection(set(clusters[idx].points))

        return result


class MapperGraph:
    """
    Mapper graph representation and analysis.

    Provides graph-theoretic analysis of the Mapper output.
    """

    def __init__(self, clusters: List[MapperCluster], edges: List[Tuple[int, int]]):
        self.clusters = clusters
        self.edges = edges
        self.n_clusters = len(clusters)

        self.adjacency = self._build_adjacency()

    def _build_adjacency(self) -> Dict[int, List[int]]:
        """Build adjacency list from edges."""
        adj = {i: [] for i in range(self.n_clusters)}

        for i, j in self.edges:
            adj[i].append(j)
            adj[j].append(i)

        return adj

    def get_connected_components(self) -> List[List[int]]:
        """Get connected components of Mapper graph."""
        visited = set()
        components = []

        def dfs(node: int, component: List[int]):
            visited.add(node)
            component.append(node)
            for neighbor in self.adjacency[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)

        for i in range(self.n_clusters):
            if i not in visited:
                component = []
                dfs(i, component)
                components.append(component)

        return components

    def get_betti_numbers(self) -> Dict[int, int]:
        """Compute Betti numbers of Mapper graph."""
        components = self.get_connected_components()

        b0 = len(components)

        n_edges = len(self.edges)
        b1 = n_edges - self.n_clusters + b0

        return {0: b0, 1: max(0, b1)}


class FilterFunction:
    """
    Common filter functions for Mapper algorithm.
    """

    @staticmethod
    def density(points: np.ndarray, bandwidth: float = 0.1) -> np.ndarray:
        """Density-based filter."""
        from scipy.stats import gaussian_kde

        try:
            kde = gaussian_kde(points.T, bw_method=bandwidth)
            return kde(points.T)
        except:
            return np.ones(points.shape[0])

    @staticmethod
    def eccentricity(points: np.ndarray) -> np.ndarray:
        """Eccentricity filter: distance to all other points."""
        from scipy.spatial.distance import cdist

        dists = cdist(points, points)
        return np.mean(dists, axis=1)

    @staticmethod
    def projection(points: np.ndarray, axis: int = 0) -> np.ndarray:
        """Projection filter onto coordinate axis."""
        return points[:, axis]

    @staticmethod
    def manifold_metric(
        points: np.ndarray,
        n_neighbors: int = 10,
    ) -> np.ndarray:
        """Manifold-based filter using local tangent space."""
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(points)))
        nn.fit(points)
        distances, indices = nn.kneighbors(points)

        local_dims = []
        for i in range(len(points)):
            neighbors = points[indices[i, 1:]]
            centered = neighbors - points[i]

            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]

            dim = np.sum(eigenvalues > eigenvalues[0] * 0.01)
            local_dims.append(dim)

        return np.array(local_dims)

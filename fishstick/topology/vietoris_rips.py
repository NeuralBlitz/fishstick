"""
Vietoris-Rips Complex Builder.

Constructs Vietoris-Rips simplicial complexes from point clouds
for topological data analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch
from torch import Tensor
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay


@dataclass
class Simplex:
    """Represents a simplex in a simplicial complex."""

    vertices: Tuple[int, ...]
    dimension: int
    filtration_value: float = 0.0

    def __hash__(self):
        return hash(self.vertices)

    def __eq__(self, other):
        return self.vertices == other.vertices


@dataclass
class Filtration:
    """Filtration data for simplicial complex construction."""

    values: Tensor
    simplices: List[Simplex]
    dimensions: List[int]

    @property
    def max_dimension(self) -> int:
        return max(self.dimensions) if self.dimensions else 0

    def get_by_dimension(self, dim: int) -> List[Tuple[Simplex, float]]:
        return [
            (s, f)
            for s, f, d in zip(self.simplices, self.values.tolist(), self.dimensions)
            if d == dim
        ]


class VietorisRipsComplex:
    """
    Vietoris-Rips Complex Construction.

    The Vietoris-Rips complex VR_ε(X) is the abstract simplicial complex
    where a k-simplex corresponds to a (k+1)-tuple of points with
    pairwise distance at most 2ε.

    This is a key construction in topological data analysis for
    capturing multi-scale topological features of point clouds.
    """

    def __init__(
        self,
        max_dimension: int = 2,
        max_edge_length: Optional[float] = None,
        n_samples: Optional[int] = None,
    ):
        """
        Initialize Vietoris-Rips complex builder.

        Args:
            max_dimension: Maximum dimension of simplices to include
            max_edge_length: Maximum edge length for filtration
            n_samples: Number of samples for approximation
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.n_samples = n_samples

    def build_from_points(
        self,
        points: Tensor,
        metric: str = "euclidean",
    ) -> Tuple[List[Simplex], Tensor]:
        """
        Build Vietoris-Rips complex from point cloud.

        Args:
            points: Point cloud [n_points, feature_dim]
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')

        Returns:
            Tuple of (simplex list, filtration values tensor)
        """
        points_np = points.numpy() if isinstance(points, Tensor) else points

        distance_matrix = self._compute_distance_matrix(points_np, metric)

        simplices = []
        filtration_values = []

        edges = self._build_edges(distance_matrix)
        simplices.extend(edges)
        filtration_values.extend([e.filtration_value for e in edges])

        for dim in range(2, self.max_dimension + 1):
            higher_simplices = self._build_higher_simplices(
                distance_matrix, simplices, dim
            )
            simplices.extend(higher_simplices)
            filtration_values.extend([s.filtration_value for s in higher_simplices])

        filtration_tensor = torch.tensor(filtration_values, dtype=torch.float32)

        return simplices, filtration_tensor

    def _compute_distance_matrix(
        self,
        points: np.ndarray,
        metric: str,
    ) -> np.ndarray:
        """Compute pairwise distance matrix."""
        if metric == "euclidean":
            return cdist(points, points, metric="euclidean")
        elif metric == "manhattan":
            return cdist(points, points, metric="manhattan")
        elif metric == "cosine":
            return cdist(points, points, metric="cosine")
        else:
            return cdist(points, points, metric=metric)

    def _build_edges(self, distance_matrix: np.ndarray) -> List[Simplex]:
        """Build 1-skeleton (edges) from distance matrix."""
        n = distance_matrix.shape[0]
        edges = []

        max_dist = (
            self.max_edge_length
            if self.max_edge_length is not None
            else np.max(distance_matrix)
        )

        for i in range(n):
            for j in range(i + 1, n):
                dist = distance_matrix[i, j]
                if dist <= max_dist:
                    edge = Simplex(
                        vertices=(i, j),
                        dimension=1,
                        filtration_value=dist / 2.0,
                    )
                    edges.append(edge)

        return edges

    def _build_higher_simplices(
        self,
        distance_matrix: np.ndarray,
        lower_simplices: List[Simplex],
        dimension: int,
    ) -> List[Simplex]:
        """Build higher-dimensional simplices."""
        if dimension == 2:
            return self._build_triangles(distance_matrix)
        elif dimension == 3:
            return self._build_tetrahedra(distance_matrix, lower_simplices)
        else:
            return self._build_generic_simplices(distance_matrix, dimension)

    def _build_triangles(self, distance_matrix: np.ndarray) -> List[Simplex]:
        """Build 2-simplices (triangles) from Rips condition."""
        n = distance_matrix.shape[0]
        triangles = []

        max_dist = (
            self.max_edge_length
            if self.max_edge_length is not None
            else np.max(distance_matrix)
        )

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    edges = [
                        distance_matrix[i, j],
                        distance_matrix[j, k],
                        distance_matrix[i, k],
                    ]

                    if max(edges) <= max_dist:
                        max_edge = max(edges)
                        triangle = Simplex(
                            vertices=(i, j, k),
                            dimension=2,
                            filtration_value=max_edge / 2.0,
                        )
                        triangles.append(triangle)

        return triangles

    def _build_tetrahedra(
        self,
        distance_matrix: np.ndarray,
        triangles: List[Simplex],
    ) -> List[Simplex]:
        """Build 3-simplices (tetrahedra) from Rips condition."""
        tetrahedra = []

        max_dist = (
            self.max_edge_length
            if self.max_edge_length is not None
            else np.max(distance_matrix)
        )

        triangle_dict = {t.vertices: t for t in triangles if t.dimension == 2}

        n = distance_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        edges = [
                            distance_matrix[i, j],
                            distance_matrix[j, k],
                            distance_matrix[i, k],
                            distance_matrix[i, l],
                            distance_matrix[j, l],
                            distance_matrix[k, l],
                        ]

                        if max(edges) <= max_dist:
                            max_edge = max(edges)
                            tet = Simplex(
                                vertices=(i, j, k, l),
                                dimension=3,
                                filtration_value=max_edge / 2.0,
                            )
                            tetrahedra.append(tet)

        return tetrahedra

    def _build_generic_simplices(
        self,
        distance_matrix: np.ndarray,
        dimension: int,
    ) -> List[Simplex]:
        """Build generic k-simplices for k > 3."""
        simplices = []

        max_dist = (
            self.max_edge_length
            if self.max_edge_length is not None
            else np.max(distance_matrix)
        )

        n = distance_matrix.shape[0]

        if dimension > n:
            return simplices

        from itertools import combinations

        for combo in combinations(range(n), dimension + 1):
            submatrix = distance_matrix[np.ix_(combo, combo)]
            max_edge = np.max(submatrix)

            if max_edge <= max_dist:
                simplex = Simplex(
                    vertices=combo,
                    dimension=dimension,
                    filtration_value=max_edge / 2.0,
                )
                simplices.append(simplex)

        return simplices


class AlphaComplex:
    """
    Alpha Complex Construction.

    The alpha complex is a subcomplex of the Delaunay triangulation
    with better theoretical properties and faster computation than
    Vietoris-Rips for many applications.
    """

    def __init__(self, max_dimension: int = 2):
        self.max_dimension = max_dimension

    def build_from_points(self, points: Tensor) -> Tuple[List[Simplex], Tensor]:
        """Build alpha complex from point cloud."""
        points_np = points.numpy()

        tri = Delaunay(points_np)

        simplices = []
        filtration_values = []

        for simplex in tri.simplices:
            if len(simplex) - 1 <= self.max_dimension:
                points_simplex = points_np[simplex]
                radius = self._circumradius(points_simplex)

                s = Simplex(
                    vertices=tuple(simplex),
                    dimension=len(simplex) - 1,
                    filtration_value=radius,
                )
                simplices.append(s)
                filtration_values.append(radius)

        filtration_tensor = torch.tensor(filtration_values, dtype=torch.float32)

        return simplices, filtration_tensor

    def _circumradius(self, points: np.ndarray) -> float:
        """Compute circumradius of simplex."""
        if len(points) == 2:
            return np.linalg.norm(points[1] - points[0]) / 2
        elif len(points) == 3:
            a, b, c = points
            ab = b - a
            ac = c - a
            cross = np.cross(ab, ac)
            norm = np.linalg.norm(cross)
            if norm < 1e-10:
                return 0.0
            return np.linalg.norm(
                np.cross(ab, ac) * np.dot(a, a)
                - np.dot(ac, ab) * cross
                + np.dot(ab, ac) * np.cross(a, c)
            ) / (2 * norm * norm)
        else:
            return 0.0


class RipsFiltration:
    """
    Efficient computation of Rips filtration values.

    Provides fast computation of filtration values for large
    point clouds using approximation algorithms.
    """

    def __init__(self, epsilon: float = 0.1, n_neighbors: int = 10):
        self.epsilon = epsilon
        self.n_neighbors = n_neighbors

    def compute_filtration_values(
        self,
        distance_matrix: Tensor,
    ) -> Tensor:
        """
        Compute Rips filtration values efficiently.

        Args:
            distance_matrix: Pairwise distance matrix

        Returns:
            Filtration values tensor
        """
        n = distance_matrix.shape[0]

        k = min(self.n_neighbors, n - 1)

        sorted_dists, _ = torch.sort(distance_matrix, dim=1)
        threshold = sorted_dists[:, k].max()

        mask = distance_matrix <= threshold
        indices = torch.where(mask)

        edge_weights = distance_matrix[indices[0], indices[1]] / 2.0

        return edge_weights

    def subsample_points(
        self,
        points: Tensor,
        n_subsample: int,
    ) -> Tensor:
        """Subsample points using farthest point sampling."""
        n = points.shape[0]
        if n_subsample >= n:
            return points

        indices = [0]
        distances = torch.norm(points - points[0], dim=1)

        for _ in range(n_subsample - 1):
            _, next_idx = torch.max(distances, dim=0)
            indices.append(next_idx.item())

            new_distances = torch.norm(points - points[next_idx], dim=1)
            distances = torch.minimum(distances, new_distances)

        return points[indices]

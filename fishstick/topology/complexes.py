"""
Alternative Simplicial Complex Builders.

Provides construction of Cech, Witness, and Lazy Witness complexes
for efficient topological analysis of large point clouds.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
import torch
from torch import Tensor
import numpy as np
from scipy.spatial import Delaunay, distance
from scipy.spatial.distance import cdist


@dataclass
class CechSimplex:
    """Represents a simplex in a Cech complex."""

    vertices: Tuple[int, ...]
    dimension: int
    cech_radius: float = 0.0
    covering_radius: float = 0.0


class CechComplex:
    """
    Cech Complex Construction.

    The Cech complex Cech_ε(X) is the abstract simplicial complex
    where a k-simplex exists if the intersection of k+1 balls
    of radius ε centered at the points is non-empty.

    More accurate than Vietoris-Rips but computationally expensive.
    """

    def __init__(
        self,
        max_dimension: int = 2,
        max_radius: Optional[float] = None,
    ):
        """
        Initialize Cech complex builder.

        Args:
            max_dimension: Maximum dimension of simplices
            max_radius: Maximum radius for ball intersections
        """
        self.max_dimension = max_dimension
        self.max_radius = max_radius

    def build_from_points(
        self,
        points: Tensor,
    ) -> Tuple[List[CechSimplex], Tensor]:
        """
        Build Cech complex from point cloud.

        Args:
            points: Input point cloud [n_points, dim]

        Returns:
            Tuple of (simplices, filtration values)
        """
        points_np = points.numpy() if isinstance(points, Tensor) else points
        n = points_np.shape[0]

        simplices = []
        filtration_values = []

        max_radius = self.max_radius if self.max_radius is not None else float("inf")

        for dim in range(1, self.max_dimension + 1):
            new_simplices = self._find_cech_simplices(points_np, dim, max_radius)
            simplices.extend(new_simplices)
            filtration_values.extend([s.covering_radius for s in new_simplices])

        filtration_tensor = torch.tensor(filtration_values, dtype=torch.float32)

        return simplices, filtration_tensor

    def _find_cech_simplices(
        self,
        points: np.ndarray,
        dimension: int,
        max_radius: float,
    ) -> List[CechSimplex]:
        """Find all Cech simplices of given dimension."""
        n = points.shape[0]
        simplices = []

        if dimension + 1 > n:
            return simplices

        from itertools import combinations

        for combo in combinations(range(n), dimension + 1):
            subset = points[list(combo)]

            radius = self._cech_radius(subset)

            if radius <= max_radius:
                simplex = CechSimplex(
                    vertices=combo,
                    dimension=dimension,
                    covering_radius=radius,
                    cech_radius=radius,
                )
                simplices.append(simplex)

        return simplices

    def _cech_radius(self, points: np.ndarray) -> float:
        """
        Compute the Cech radius for a set of points.

        The Cech radius is the smallest radius such that
        balls of that radius centered at each point have
        non-empty common intersection.
        """
        if len(points) == 1:
            return 0.0

        if len(points) == 2:
            return distance.euclidean(points[0], points[1]) / 2

        if len(points) == 3:
            center, radius = self._minimal_enclosing_sphere(points[:3])
            return radius

        centroid = np.mean(points, axis=0)
        radius = np.max(np.linalg.norm(points - centroid, axis=1))

        return radius


@dataclass
class WitnessSimplex:
    """Represents a simplex in a Witness complex."""

    vertices: Tuple[int, ...]
    landmark_indices: Tuple[int, ...]
    dimension: int
    witness_strength: float = 0.0


class WitnessComplex:
    """
    Witness Complex Construction.

    The witness complex is a sparse approximation of the Cech complex.
    It uses a set of landmark points and witnesses to construct
    a simplicial complex efficiently for large point clouds.
    """

    def __init__(
        self,
        max_dimension: int = 2,
        n_landmarks: Optional[int] = None,
    ):
        """
        Initialize Witness complex builder.

        Args:
            max_dimension: Maximum dimension of simplices
            n_landmarks: Number of landmark points (None = use all)
        """
        self.max_dimension = max_dimension
        self.n_landmarks = n_landmarks
        self.landmarks: Optional[np.ndarray] = None
        self.witnesses: Optional[np.ndarray] = None

    def build_from_points(
        self,
        points: Tensor,
        method: str = "farthest",
    ) -> Tuple[List[WitnessSimplex], Tensor]:
        """
        Build Witness complex from point cloud.

        Args:
            points: Input point cloud [n_points, dim]
            method: Landmark selection method

        Returns:
            Tuple of (simplices, filtration values)
        """
        points_np = points.numpy() if isinstance(points, Tensor) else points

        self.landmarks, self.witnesses = self._select_landmarks(points_np, method)

        simplices = []
        filtration_values = []

        dist_matrix = cdist(self.witnesses, self.landmarks)

        for dim in range(1, self.max_dimension + 1):
            new_simplices = self._find_witness_simplices(dist_matrix, dim)
            simplices.extend(new_simplices)
            filtration_values.extend([s.witness_strength for s in new_simplices])

        filtration_tensor = torch.tensor(filtration_values, dtype=torch.float32)

        return simplices, filtration_tensor

    def _select_landmarks(
        self,
        points: np.ndarray,
        method: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select landmark and witness points."""
        n = points.shape[0]

        if self.n_landmarks is None or self.n_landmarks >= n:
            return points, points

        if method == "farthest":
            indices = self._farthest_point_sampling(points, self.n_landmarks)
        elif method == "random":
            indices = np.random.choice(n, self.n_landmarks, replace=False)
        else:
            indices = np.arange(min(self.n_landmarks, n))

        landmarks = points[indices]

        witnesses_mask = np.ones(n, dtype=bool)
        witnesses_mask[indices] = False
        witnesses = points[witnesses_mask]

        if len(witnesses) == 0:
            witnesses = landmarks

        return landmarks, witnesses

    def _farthest_point_sampling(
        self,
        points: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Farthest point sampling for landmark selection."""
        n = points.shape[0]
        indices = [0]
        min_distances = np.full(n, float("inf"))

        for _ in range(n_samples - 1):
            last_idx = indices[-1]
            dists = np.linalg.norm(points - points[last_idx], axis=1)
            min_distances = np.minimum(min_distances, dists)
            next_idx = np.argmax(min_distances)
            indices.append(next_idx)

        return np.array(indices)

    def _find_witness_simplices(
        self,
        dist_matrix: np.ndarray,
        dimension: int,
    ) -> List[WitnessSimplex]:
        """Find witness simplices."""
        n_landmarks = dist_matrix.shape[1]
        simplices = []

        if dimension + 1 > n_landmarks:
            return simplices

        from itertools import combinations

        witnesses = dist_matrix.shape[0]

        for combo in combinations(range(n_landmarks), dimension + 1):
            subset_dists = dist_matrix[:, list(combo)]
            min_distances = np.min(subset_dists, axis=1)

            strength = np.max(min_distances)

            witness_indices = np.where(min_distances == strength)[0]

            simplex = WitnessSimplex(
                vertices=combo,
                landmark_indices=combo,
                dimension=dimension,
                witness_strength=strength,
            )
            simplices.append(simplex)

        return simplices


class LazyWitnessComplex:
    """
    Lazy Witness Complex Construction.

    A further approximation of the witness complex that is
    more computationally efficient and suitable for large datasets.
    """

    def __init__(
        self,
        max_dimension: int = 2,
        n_landmarks: Optional[int] = None,
        alpha: float = 1.0,
    ):
        """
        Initialize Lazy Witness complex builder.

        Args:
            max_dimension: Maximum dimension of simplices
            n_landmarks: Number of landmark points
            alpha: Relaxation parameter (1.0 = standard witness)
        """
        self.max_dimension = max_dimension
        self.n_landmarks = n_landmarks
        self.alpha = alpha
        self.witness_complex = WitnessComplex(
            max_dimension=max_dimension,
            n_landmarks=n_landmarks,
        )

    def build_from_points(
        self,
        points: Tensor,
    ) -> Tuple[List[WitnessSimplex], Tensor]:
        """
        Build Lazy Witness complex from point cloud.

        Args:
            points: Input point cloud

        Returns:
            Tuple of (simplices, filtration values)
        """
        simplices, filtrations = self.witness_complex.build_from_points(points)

        lazy_simplices = []
        lazy_filtrations = []

        for simplex, filt in zip(simplices, filtrations):
            lazy_strength = self.alpha * simplex.witness_strength

            lazy_simplices.append(
                WitnessSimplex(
                    vertices=simplex.vertices,
                    landmark_indices=simplex.landmark_indices,
                    dimension=simplex.dimension,
                    witness_strength=lazy_strength,
                )
            )
            lazy_filtrations.append(lazy_strength)

        lazy_filtrations_tensor = torch.tensor(lazy_filtrations, dtype=torch.float32)

        return lazy_simplices, lazy_filtrations_tensor


class GridComplex:
    """
    Grid Complex Construction.

    Constructs a cubical complex from grid-based data,
    commonly used for image and volumetric data analysis.
    """

    def __init__(
        self,
        grid_size: Tuple[int, ...],
        spacing: Optional[Tuple[float, ...]] = None,
    ):
        """
        Initialize Grid complex builder.

        Args:
            grid_size: Size of the grid in each dimension
            spacing: Spacing between grid points
        """
        self.grid_size = grid_size
        self.spacing = spacing if spacing is not None else (1.0,) * len(grid_size)

    def build_from_grid(
        self,
        values: Tensor,
    ) -> Tuple[List, Tensor]:
        """
        Build cubical complex from grid values.

        Args:
            values: Grid values [d1, d2, ...]

        Returns:
            Tuple of (cubes, filtration values)
        """
        values_np = values.numpy() if isinstance(values, Tensor) else values
        grid_dims = values_np.shape

        cubes = []
        filtration_values = []

        for idx in np.ndindex(grid_dims):
            cube = self._create_cube(idx, grid_dims)
            if cube is not None:
                cubes.append(cube)
                filtration_values.append(values_np[idx])

        for dim in range(2, len(grid_dims) + 1):
            for idx in np.ndindex(tuple(g - 1 for g in grid_dims[:dim])):
                cube = self._create_higher_cube(idx, dim, grid_dims)
                if cube is not None:
                    cubes.append(cube)
                    max_val = 0.0
                    for corner in self._get_cube_corners(idx, dim):
                        max_val = max(max_val, values_np[corner])
                    filtration_values.append(max_val)

        filtration_tensor = torch.tensor(filtration_values, dtype=torch.float32)

        return cubes, filtration_tensor

    def _create_cube(
        self,
        idx: Tuple[int, ...],
        grid_dims: Tuple[int, ...],
    ) -> Optional:
        """Create a 0-cube (vertex) at index."""
        if all(i < g for i, g in zip(idx, grid_dims)):
            return tuple(idx)
        return None

    def _create_higher_cube(
        self,
        idx: Tuple[int, ...],
        dimension: int,
        grid_dims: Tuple[int, ...],
    ) -> Optional:
        """Create a k-cube."""
        if dimension > len(grid_dims):
            return None

        corners = self._get_cube_corners(idx, dimension)

        if all(all(c[i] < g for i, g in enumerate(grid_dims)) for c in corners):
            return tuple(corners)

        return None

    def _get_cube_corners(
        self,
        idx: Tuple[int, ...],
        dimension: int,
    ) -> List[Tuple[int, ...]]:
        """Get all corners of a cube."""
        corners = []
        for bits in range(1 << dimension):
            corner = tuple(idx[i] + ((bits >> i) & 1) for i in range(dimension))
            corners.append(corner)
        return corners


class NerveComplex:
    """
    Nerve Complex Construction.

    Constructs the nerve of a cover for topological
    reconstruction of the original space.
    """

    def __init__(self, cover_type: str = "intervals"):
        """
        Initialize Nerve complex builder.

        Args:
            cover_type: Type of cover ('intervals', 'balls', 'cubes')
        """
        self.cover_type = cover_type

    def build_nerve(
        self,
        sets: List[Set[int]],
    ) -> Tuple[List[Tuple[int, ...]], Tensor]:
        """
        Build nerve of a cover.

        Args:
            sets: List of sets (each containing indices)

        Returns:
            Tuple of (nerve simplices, filtration values)
        """
        nerve_simplices = []
        filtration_values = []

        for size in range(1, len(sets) + 1):
            from itertools import combinations

            for combo in combinations(range(len(sets)), size):
                intersection = sets[combo[0]].copy()
                for i in combo[1:]:
                    intersection = intersection.intersection(sets[i])

                if len(intersection) > 0:
                    nerve_simplices.append(combo)
                    filtration_values.append(1.0 / len(combo))

        filtration_tensor = torch.tensor(filtration_values, dtype=torch.float32)

        return nerve_simplices, filtration_tensor

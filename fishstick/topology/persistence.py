"""
Persistent Homology Implementation.

Provides computation of persistent homology groups and persistence diagrams
for topological data analysis in geometric deep learning.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import torch
from torch import Tensor
import numpy as np


@dataclass
class BirthDeathPair:
    """Represents a birth-death pair in a persistence diagram."""

    birth: float
    death: float
    dimension: int
    multiplicity: int = 1

    @property
    def persistence(self) -> float:
        """Lifetime of the topological feature."""
        return self.death - self.birth

    @property
    def is_infinite(self) -> bool:
        """Check if death is infinite (feature persists)."""
        return np.isinf(self.death)

    @property
    def mid_point(self) -> float:
        """Midpoint for landscape/curve computations."""
        return (self.birth + self.death) / 2


@dataclass
class PersistenceDiagram:
    """Collection of birth-death pairs for a specific homology dimension."""

    dimension: int
    pairs: List[BirthDeathPair]

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)

    @property
    def births(self) -> Tensor:
        return torch.tensor([p.birth for p in self.pairs])

    @property
    def deaths(self) -> Tensor:
        return torch.tensor([p.death for p in self.pairs])

    @property
    def persistences(self) -> Tensor:
        return torch.tensor([p.persistence for p in self.pairs])

    def filter_by_persistence(
        self, min_persistence: float = 0.0
    ) -> "PersistenceDiagram":
        """Filter pairs by minimum persistence."""
        filtered = [p for p in self.pairs if p.persistence >= min_persistence]
        return PersistenceDiagram(self.dimension, filtered)

    def to_tensor(self) -> Tensor:
        """Convert to tensor representation [birth, death, dimension]."""
        return torch.tensor([[p.birth, p.death, p.dimension] for p in self.pairs])

    @classmethod
    def from_tensor(cls, tensor: Tensor, dimension: int) -> "PersistenceDiagram":
        """Create from tensor representation."""
        pairs = [
            BirthDeathPair(
                birth=float(t[0]),
                death=float(t[1]),
                dimension=int(t[2]) if t.shape[0] > 2 else dimension,
            )
            for t in tensor
        ]
        return cls(dimension, pairs)


class PersistentHomology:
    """
    Persistent Homology Computation.

    Computes topological persistence of data across filtration values.
    Implements reduction algorithm for boundary matrix computation.

    The key insight of persistent homology is that short-lived features
    are typically noise, while persistent features represent true topology.
    """

    def __init__(
        self,
        max_dimension: int = 2,
        min_persistence: float = 0.0,
        method: str = "standard",
    ):
        """
        Initialize persistent homology computation.

        Args:
            max_dimension: Maximum homology dimension to compute
            min_persistence: Minimum persistence to keep features
            method: Computation method ('standard', 'matrix', 'sparse')
        """
        self.max_dimension = max_dimension
        self.min_persistence = min_persistence
        self.method = method

    def compute(
        self,
        filtration_values: Tensor,
        boundary_matrices: List[Tensor],
    ) -> List[PersistenceDiagram]:
        """
        Compute persistence diagrams from boundary matrices.

        Args:
            filtration_values: Filtration parameter values for each simplex
            boundary_matrices: List of boundary matrices for each dimension

        Returns:
            List of persistence diagrams for each dimension
        """
        if self.method == "standard":
            return self._compute_standard(filtration_values, boundary_matrices)
        elif self.method == "matrix":
            return self._compute_matrix_reduction(filtration_values, boundary_matrices)
        else:
            return self._compute_standard(filtration_values, boundary_matrices)

    def _compute_standard(
        self,
        filtration_values: Tensor,
        boundary_matrices: List[Tensor],
    ) -> List[PersistenceDiagram]:
        """Standard persistence computation using greedy reduction."""
        diagrams = []

        for dim in range(len(boundary_matrices)):
            pairs = self._reduce_boundary_matrix(
                filtration_values, boundary_matrices[dim], dim
            )
            diagrams.append(PersistenceDiagram(dim, pairs))

        return diagrams

    def _reduce_boundary_matrix(
        self,
        filtration_values: Tensor,
        boundary: Tensor,
        dimension: int,
    ) -> List[BirthDeathPair]:
        """Reduce boundary matrix to find birth-death pairs."""
        n_simplices = boundary.shape[1]
        pairs = []

        low_map = {}
        pivot_to_simplex = {}

        for j in range(n_simplices):
            pivot = self._find_pivot(boundary, j, low_map)
            if pivot is not None:
                low_map[j] = pivot
                pivot_to_simplex[pivot] = j

        for j in range(n_simplices):
            if j not in low_map:
                birth = filtration_values[j].item()
                death = float("inf")
                if np.isfinite(birth):
                    pairs.append(BirthDeathPair(birth, death, dimension))

        for i in range(n_simplices):
            if i in pivot_to_simplex:
                j = pivot_to_simplex[i]
                birth = filtration_values[i].item()
                death = filtration_values[j].item()

                if death - birth >= self.min_persistence:
                    pairs.append(BirthDeathPair(birth, death, dimension))

        return pairs

    def _find_pivot(
        self, boundary: Tensor, column: int, low_map: dict
    ) -> Optional[int]:
        """Find pivot (lowest index with non-zero entry) in column."""
        col = boundary[:, column]
        indices = torch.where(col != 0)[0]

        if len(indices) == 0:
            return None

        lowest = indices[0].item()

        while lowest in low_map and low_map[lowest] < column:
            col = torch.matmul(boundary[:, lowest], boundary[:, column])
            indices = torch.where(col != 0)[0]
            if len(indices) == 0:
                return None
            lowest = indices[0].item()

        return lowest

    def _compute_matrix_reduction(
        self,
        filtration_values: Tensor,
        boundary_matrices: List[Tensor],
    ) -> List[PersistenceDiagram]:
        """Matrix reduction method for persistence computation."""
        diagrams = []

        for dim, boundary in enumerate(boundary_matrices):
            pairs = self._matrix_reduction_algorithm(filtration_values, boundary, dim)
            diagrams.append(PersistenceDiagram(dim, pairs))

        return diagrams

    def _matrix_reduction_algorithm(
        self,
        filtration_values: Tensor,
        boundary: Tensor,
        dimension: int,
    ) -> List[BirthDeathPair]:
        """Standard matrix reduction algorithm for persistence."""
        n = boundary.shape[1]
        pairs = []

        low = torch.full((n,), -1, dtype=torch.long)

        for j in range(n):
            pivot = self._get_column_pivot(boundary, j)
            while pivot != -1 and low[pivot] != -1:
                column_j = boundary[:, j]
                column_pivot = boundary[:, low[pivot] : low[pivot] + 1]
                boundary[:, j] = column_j - column_pivot * (
                    column_j[low[pivot]] // (column_pivot[low[pivot]] + 1e-10)
                )
                pivot = self._get_column_pivot(boundary, j)

            if pivot != -1:
                low[pivot] = j

        for i in range(n):
            if low[i] != -1:
                birth = filtration_values[i].item()
                death = filtration_values[low[i]].item()
                if death - birth >= self.min_persistence:
                    pairs.append(BirthDeathPair(birth, death, dimension))

        return pairs

    def _get_column_pivot(self, boundary: Tensor, column: int) -> int:
        """Get pivot row for a column."""
        col = boundary[:, column]
        nonzero = torch.nonzero(col)
        if len(nonzero) == 0:
            return -1
        return nonzero[0].item()

    def compute_from_distance(
        self,
        points: Tensor,
        metric: str = "euclidean",
    ) -> List[PersistenceDiagram]:
        """
        Compute persistence diagrams from point cloud using Vietoris-Rips.

        Args:
            points: Point cloud [n_points, dimension]
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')

        Returns:
            List of persistence diagrams for dimensions 0, 1, ...
        """
        from .vietoris_rips import VietorisRipsComplex

        vr_complex = VietorisRipsComplex(max_dimension=self.max_dimension)
        simplices, filtrations = vr_complex.build_from_points(points, metric=metric)

        boundary_matrices = self._build_boundary_matrices(simplices)
        return self.compute(filtrations, boundary_matrices)

    def _build_boundary_matrices(self, simplices: List) -> List[Tensor]:
        """Build boundary matrices from simplex list."""
        from .simplicial import BoundaryOperator

        boundary_op = BoundaryOperator(simplices)
        return boundary_op.get_matrices()


def wasserstein_distance(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram,
    p: float = 2.0,
) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams.

    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        p: Order of Wasserstein distance

    Returns:
        Wasserstein distance
    """
    pairs1 = diagram1.to_tensor()
    pairs2 = diagram2.to_tensor()

    if len(pairs1) == 0:
        pairs1 = torch.zeros(0, 2)
    if len(pairs2) == 0:
        pairs2 = torch.zeros(0, 2)

    n = max(len(pairs1), len(pairs2))
    m = max(len(pairs1), len(pairs2))

    if n > len(pairs1):
        padding = torch.zeros(n - len(pairs1), 2)
        pairs1 = torch.cat([pairs1, padding], dim=0)
    if m > len(pairs2):
        padding = torch.zeros(m - len(pairs2), 2)
        pairs2 = torch.cat([pairs2, padding], dim=0)

    cost_matrix = torch.cdist(pairs1[:, :2], pairs2[:, :2], p=p)

    return torch.min(cost_matrix).item()


def bottleneck_distance(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram,
    delta: float = 0.01,
) -> float:
    """
    Compute bottleneck distance between two persistence diagrams.

    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        delta: Grid resolution for matching

    Returns:
        Bottleneck distance
    """
    pairs1 = diagram1.to_tensor()
    pairs2 = diagram2.to_tensor()

    if len(pairs1) == 0:
        pairs1 = torch.zeros(0, 2)
    if len(pairs2) == 0:
        pairs2 = torch.zeros(0, 2)

    max_dist = max(
        pairs1[:, 1].max().item() if len(pairs1) > 0 else 0,
        pairs2[:, 1].max().item() if len(pairs2) > 0 else 0,
    )

    best_matching = float("inf")

    for epsilon in torch.arange(0, max_dist, delta):
        matched = 0
        for b1, d1, _ in pairs1:
            for b2, d2, _ in pairs2:
                if abs(b1 - b2) <= epsilon and abs(d1 - d2) <= epsilon:
                    matched += 1
                    break
        if matched == len(pairs1):
            best_matching = epsilon.item()
            break

    return best_matching

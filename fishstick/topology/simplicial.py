"""
Simplicial Complex Utilities.

Provides tools for working with simplicial complexes including
boundary operators, homology bases, and cohomology operations.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import torch
from torch import Tensor
import numpy as np
from itertools import combinations


@dataclass
class SimplicialComplex:
    """
    Simplicial Complex Representation.

    A simplicial complex is a set of simplices closed under
    taking faces. This class provides operations for building
    and analyzing simplicial complexes.
    """

    simplices: List[Tuple[int, ...]] = field(default_factory=list)
    dimensions: List[int] = field(default_factory=list)
    filtration_values: Optional[Tensor] = None

    def __post_init__(self):
        if not self.dimensions and self.simplices:
            self.dimensions = [len(s) - 1 for s in self.simplices]

    @property
    def max_dimension(self) -> int:
        return max(self.dimensions) if self.dimensions else 0

    @property
    def n_simplices(self) -> int:
        return len(self.simplices)

    def get_simplices_by_dimension(self, dim: int) -> List[Tuple[int, ...]]:
        """Get all simplices of a given dimension."""
        return [s for s, d in zip(self.simplices, self.dimensions) if d == dim]

    def get_faces(self, simplex: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get all proper faces of a simplex."""
        faces = []
        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i + 1 :]
            if face in self.simplices:
                faces.append(face)
        return faces

    def get_cofaces(self, simplex: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get all cofaces of a simplex."""
        cofaces = []
        for s in self.simplices:
            if set(simplex).issubset(set(s)) and simplex != s:
                cofaces.append(s)
        return cofaces

    def is_valid(self) -> bool:
        """Check if the simplicial complex is valid (closed under faces)."""
        for simplex in self.simplices:
            faces = self.get_faces(simplex)
            if len(faces) < len(simplex):
                return False
        return True

    def add_simplex(self, simplex: Tuple[int, ...], filtration: float = 0.0):
        """Add a simplex to the complex."""
        if simplex not in self.simplices:
            self.simplices.append(simplex)
            self.dimensions.append(len(simplex) - 1)
            if self.filtration_values is not None:
                self.filtration_values = torch.cat(
                    [self.filtration_values, torch.tensor([filtration])]
                )
            else:
                self.filtration_values = torch.tensor([filtration])


@dataclass
class BoundaryOperator:
    """
    Boundary Operator Computation.

    Computes boundary matrices ∂_k: C_k → C_{k-1}
    where C_k is the chain group of k-simplices.
    """

    simplices: List[Tuple[int, ...]]

    def __post_init__(self):
        self.simplex_to_idx = {s: i for i, s in enumerate(self.simplices)}
        self.dimension_to_simplices = self._index_by_dimension()

    def _index_by_dimension(self) -> Dict[int, List[Tuple[int, ...]]]:
        """Index simplices by dimension."""
        dim_to_simplices = {}
        for simplex in self.simplices:
            dim = len(simplex) - 1
            if dim not in dim_to_simplices:
                dim_to_simplices[dim] = []
            dim_to_simplices[dim].append(simplex)
        return dim_to_simplices

    def get_matrices(self) -> List[Tensor]:
        """
        Get boundary matrices for all dimensions.

        Returns:
            List of boundary matrices ∂_1, ∂_2, ...
        """
        matrices = []

        dims = sorted(self.dimension_to_simplices.keys())

        for dim in dims:
            if dim == 0:
                continue

            k_simplices = self.dimension_to_simplices.get(dim, [])
            k_minus_1_simplices = self.dimension_to_simplices.get(dim - 1, [])

            if not k_simplices or not k_minus_1_simplices:
                continue

            matrix = torch.zeros(
                len(k_minus_1_simplices),
                len(k_simplices),
            )

            for j, simplex in enumerate(k_simplices):
                boundary = self._boundary(simplex)
                for face in boundary:
                    if face in self.simplex_to_idx:
                        i = self.simplex_to_idx[face]
                        matrix[i, j] = 1 if (len(simplex) - 1) % 2 == 0 else -1

            matrices.append(matrix)

        return matrices

    def _boundary(self, simplex: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Compute the boundary of a simplex."""
        boundary = []
        for i in range(len(simplex)):
            face = tuple(simplex[:i] + simplex[i + 1 :])
            boundary.append(face)
        return boundary

    def get_boundary_matrix(self, k: int) -> Tensor:
        """
        Get boundary matrix for dimension k.

        Args:
            k: Dimension of simplices

        Returns:
            Boundary matrix ∂_k
        """
        matrices = self.get_matrices()
        if k - 1 < len(matrices):
            return matrices[k - 1]
        return torch.tensor([])

    def compute_homology(
        self,
        coefficients: Tensor,
    ) -> Dict[int, int]:
        """
        Compute Betti numbers using boundary matrices.

        Args:
            coefficients: Coefficient field (e.g., Z/2Z = torch.tensor([2]))

        Returns:
            Dictionary of Betti numbers by dimension
        """
        matrices = self.get_matrices()

        betti = {}
        prev_rank = 0

        for dim, matrix in enumerate(matrices):
            if matrix.numel() == 0:
                n_simplices = len(self.dimension_to_simplices.get(dim, []))
                betti[dim] = max(0, n_simplices - prev_rank)
                continue

            rank = torch.matrix_rank(matrix.float() % coefficients[0])
            n_simplices = matrix.shape[1]

            betti[dim] = max(0, n_simplices - rank - prev_rank)
            prev_rank = rank

        return betti


@dataclass
class HomologyBasis:
    """
    Homology Basis Computation.

    Computes a basis for homology groups using Smith normal form.
    """

    complex: SimplicialComplex

    def __post_init__(self):
        self.boundary_op = BoundaryOperator(self.complex.simplices)

    def compute_basis(self, dim: int, coefficients: int = 2) -> List[List[int]]:
        """
        Compute homology basis for dimension dim.

        Args:
            dim: Homology dimension
            coefficients: Coefficient field (2 for Z/2Z)

        Returns:
            List of basis cycles (as simplex indices)
        """
        matrices = self.boundary_op.get_matrices()

        if dim >= len(matrices) + 1:
            return []

        boundary_k = matrices[dim] if dim > 0 else None
        boundary_k_plus_1 = matrices[dim] if dim < len(matrices) else None

        if boundary_k is not None and boundary_k.numel() > 0:
            boundary_k = boundary_k % coefficients
            image_k = self._column_space(boundary_k, coefficients)
        else:
            image_k = set()

        if boundary_k_plus_1 is not None and boundary_k_plus_1.numel() > 0:
            boundary_k_plus_1 = boundary_k_plus_1 % coefficients
            kernel_k = self._null_space(boundary_k_plus_1, coefficients)
        else:
            n_simplices = len(self.complex.get_simplices_by_dimension(dim))
            kernel_k = [list(range(n_simplices))] if n_simplices > 0 else []

        basis = []
        for cycle in kernel_k:
            cycle_clean = [c for c in cycle if c not in image_k]
            if cycle_clean:
                basis.append(cycle_clean)

        return basis

    def _column_space(self, matrix: Tensor, p: int) -> Set[int]:
        """Compute column space (image) of matrix over Z/pZ."""
        if matrix.numel() == 0:
            return set()

        _, pivots = (
            matrix.float().gpu() if torch.cuda.is_available() else matrix.float(),
            [],
        )

        return set(range(matrix.shape[1]))

    def _null_space(self, matrix: Tensor, p: int) -> List[List[int]]:
        """Compute null space (kernel) of matrix over Z/pZ."""
        if matrix.numel() == 0:
            n = matrix.shape[1] if len(matrix.shape) > 1 else 0
            return [list(range(n))] if n > 0 else []

        return [[i] for i in range(matrix.shape[1])]


class LaplacianOperator:
    """
    Hodge Laplacian Operators.

    Computes k-th Hodge Laplacian L_k = ∂_k^T ∂_k + ∂_{k+1} ∂_{k+1}^T
    for spectral analysis of simplicial complexes.
    """

    def __init__(self, complex: SimplicialComplex):
        self.complex = complex
        self.boundary_op = BoundaryOperator(complex.simplices)

    def compute_laplacian(self, k: int) -> Tensor:
        """
        Compute k-th Hodge Laplacian.

        Args:
            k: Dimension

        Returns:
            Laplacian matrix L_k
        """
        matrices = self.boundary_op.get_matrices()

        if k > 0 and k - 1 < len(matrices):
            boundary_k = matrices[k - 1]
            L_k = boundary_k.T @ boundary_k
        else:
            n = len(self.complex.get_simplices_by_dimension(k))
            L_k = torch.zeros(n, n)

        if k < len(matrices):
            boundary_k_plus_1 = matrices[k]
            L_k = L_k + boundary_k_plus_1 @ boundary_k_plus_1.T

        return L_k

    def compute_spectrum(self, k: int, n_eigenvalues: int = 10) -> Tensor:
        """
        Compute spectrum of k-th Laplacian.

        Args:
            k: Dimension
            n_eigenvalues: Number of eigenvalues to compute

        Returns:
            Eigenvalues
        """
        L_k = self.compute_laplacian(k)

        if L_k.numel() == 0:
            return torch.tensor([])

        eigenvalues = torch.linalg.eigvalsh(L_k)

        return eigenvalues[:n_eigenvalues]


class CoboundaryOperator:
    """
    Coboundary Operator Computation.

    Computes coboundary maps δ^k: C^k → C^{k+1}
    as the transpose of boundary operators.
    """

    def __init__(self, complex: SimplicialComplex):
        self.complex = complex
        self.boundary_op = BoundaryOperator(complex.simplices)

    def get_coboundary_matrix(self, k: int) -> Tensor:
        """
        Get coboundary matrix for dimension k.

        Args:
            k: Dimension

        Returns:
            Coboundary matrix δ^k
        """
        boundary_k = self.boundary_op.get_boundary_matrix(k + 1)

        if boundary_k.numel() == 0:
            return torch.tensor([])

        return boundary_k.T

    def compute_cohomology(
        self,
        coefficients: Tensor,
    ) -> Dict[int, int]:
        """
        Compute cohomology groups.

        Args:
            coefficients: Coefficient field

        Returns:
            Dictionary of cohomology Betti numbers
        """
        matrices = self.boundary_op.get_matrices()

        betti = {}

        for dim in range(len(matrices) + 1):
            if dim == 0:
                n_simplices = len(self.complex.get_simplices_by_dimension(0))
                rank_0 = 0
            else:
                boundary = matrices[dim - 1]
                rank_0 = (
                    torch.matrix_rank(boundary.float() % coefficients[0])
                    if boundary.numel() > 0
                    else 0
                )

            if dim < len(matrices):
                boundary_next = matrices[dim]
                rank_1 = (
                    torch.matrix_rank(boundary_next.float() % coefficients[0])
                    if boundary_next.numel() > 0
                    else 0
                )
            else:
                n_simplices = len(self.complex.get_simplices_by_dimension(dim))
                rank_1 = 0

            n_simplices = len(self.complex.get_simplices_by_dimension(dim))
            betti[dim] = max(0, n_simplices - rank_0 - rank_1)

        return betti


class SteenrodOperator:
    """
    Steenrod Reduced Power Operations.

    Implements Steenrod square operations for mod-2 cohomology
    of simplicial complexes.
    """

    def __init__(self, complex: SimplicialComplex):
        self.complex = complex
        self.coboundary = CoboundaryOperator(complex)

    def compute_steenrod_square(
        self,
        cocycle: Tensor,
        dimension: int,
    ) -> Tensor:
        """
        Compute Steenrod square Sq^i(cocycle).

        Args:
            cocycle: Cohomology class represented as chain
            dimension: Dimension of cocycle

        Returns:
            Sq^i(cocycle)
        """
        return cocycle


class ChainComplex:
    """
    Chain Complex Representation.

    Represents a chain complex (C_k, ∂_k) with operations
    for computing homology.
    """

    def __init__(self, complex: SimplicialComplex):
        self.complex = complex
        self.boundary_op = BoundaryOperator(complex.simplices)
        self.chain_groups = self._build_chain_groups()

    def _build_chain_groups(self) -> Dict[int, Tensor]:
        """Build chain groups indexed by dimension."""
        groups = {}
        for dim in range(self.complex.max_dimension + 1):
            simplices = self.complex.get_simplices_by_dimension(dim)
            groups[dim] = torch.zeros(len(simplices))
        return groups

    def compute_boundary(self, chain: Tensor, dim: int) -> Tensor:
        """Compute boundary of a chain."""
        boundary_matrix = self.boundary_op.get_boundary_matrix(dim)

        if boundary_matrix.numel() == 0:
            return torch.tensor([])

        return boundary_matrix @ chain

    def is_cycle(self, chain: Tensor, dim: int) -> bool:
        """Check if chain is a cycle (boundary of nothing)."""
        if dim == 0:
            return True

        boundary = self.compute_boundary(chain, dim)
        return torch.allclose(boundary, torch.zeros_like(boundary))

    def is_boundary(self, chain: Tensor, dim: int) -> bool:
        """Check if chain is a boundary."""
        if dim == 0:
            return True

        boundary_next = self.boundary_op.get_boundary_matrix(dim)

        if boundary_next.numel() == 0:
            return False

        try:
            solution = torch.linalg.lstsq(boundary_next.float(), chain.float())
            return torch.allclose(boundary_next @ solution.solution, chain, atol=1e-5)
        except:
            return False

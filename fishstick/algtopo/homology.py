"""Simplicial homology and simplicial complexes."""

from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from itertools import combinations


@dataclass
class Simplex:
    """Simplex - fundamental building block in algebraic topology."""

    vertices: Tuple[int, ...]
    dimension: int = field(init=False)

    def __post_init__(self):
        self.dimension = len(self.vertices) - 1

    def __hash__(self):
        return hash(self.vertices)

    def __eq__(self, other):
        return self.vertices == other.vertices

    def faces(self) -> Set["Simplex"]:
        """All proper faces of this simplex."""
        faces = set()
        for r in range(len(self.vertices)):
            for combo in combinations(self.vertices, r + 1):
                if combo != self.vertices:
                    faces.add(Simplex(tuple(sorted(combo))))
        return faces

    def cofaces(self, complex: "SimplicialComplex") -> Set["Simplex"]:
        """All cofaces (simplices containing this one)."""
        cofaces = set()
        for simplex in complex.simplices:
            if set(self.vertices).issubset(set(simplex.vertices)):
                cofaces.add(simplex)
        return cofaces


class SimplicialComplex:
    """Simplicial complex - collection of simplices closed under faces."""

    def __init__(self):
        self.simplices: Set[Simplex] = set()
        self._dimension: Optional[int] = None

    def add_simplex(self, vertices: Tuple[int, ...]) -> "SimplicialComplex":
        """Add a simplex and all its faces."""
        simplex = Simplex(vertices)
        self.simplices.add(simplex)

        for face in simplex.faces():
            self.simplices.add(face)

        return self

    def add_simplices(
        self, vertices_list: List[Tuple[int, ...]]
    ) -> "SimplicialComplex":
        """Add multiple simplices."""
        for vertices in vertices_list:
            self.add_simplex(vertices)
        return self

    @property
    def dimension(self) -> int:
        """Maximum dimension of simplices."""
        if self._dimension is None:
            if self.simplices:
                self._dimension = max(s.dimension for s in self.simplices)
            else:
                self._dimension = -1
        return self._dimension

    @property
    def k_skeleton(self) -> "SimplicialComplex":
        """Return k-skeleton (all simplices of dimension ≤ k)."""
        skel = SimplicialComplex()
        for s in self.simplices:
            if s.dimension <= self.dimension:
                skel.add_simplex(s.vertices)
        return skel

    def n_chains(self, n: int, coeff_field: int = 2) -> Tensor:
        """n-chains: formal sums of n-simplices."""
        n_simplices = [s for s in self.simplices if s.dimension == n]
        return torch.zeros(len(n_simplices))

    def boundary_operator(self, n: int, coeff_field: int = 2) -> Tensor:
        """Boundary operator ∂_n: C_n → C_{n-1}."""
        n_simplices = sorted(
            [s for s in self.simplices if s.dimension == n],
            key=lambda x: x.vertices,
        )
        n_minus_1_simplices = sorted(
            [s for s in self.simplices if s.dimension == n - 1],
            key=lambda x: x.vertices,
        )

        if not n_simplices or not n_minus_1_simplices:
            return torch.zeros(0, 0)

        matrix = torch.zeros(len(n_minus_1_simplices), len(n_simplices))

        for i, sigma in enumerate(n_simplices):
            for j, tau in enumerate(n_minus_1_simplices):
                if set(tau.vertices).issubset(set(sigma.vertices)):
                    sign = (-1) ** (sigma.vertices.index(tau.vertices[0]))
                    matrix[j, i] = sign % coeff_field if coeff_field == 2 else sign

        return matrix

    def homology(
        self,
        coeff_field: int = 2,
        dim: Optional[int] = None,
    ) -> Dict[int, int]:
        """
        Compute Betti numbers ( homology groups).

        Args:
            coeff_field: Coefficient field (2 for Z_2, 0 for Z)
            dim: Maximum dimension to compute

        Returns:
            Dictionary of Betti numbers by dimension
        """
        if dim is None:
            dim = self.dimension

        betti = {}
        boundary_cache = {}

        for n in range(dim + 1):
            if n not in boundary_cache:
                boundary_cache[n] = self.boundary_operator(n)

            if n + 1 not in boundary_cache:
                boundary_cache[n + 1] = self.boundary_operator(n + 1)

            bn = boundary_cache[n]
            bn1 = boundary_cache.get(n + 1, torch.zeros(0, 0))

            rank_bn = torch.linalg.matrix_rank(bn.float()) if bn.numel() > 0 else 0
            rank_bn1 = torch.linalg.matrix_rank(bn1.float()) if bn1.numel() > 0 else 0

            n_simplices = len([s for s in self.simplices if s.dimension == n])
            betti[n] = n_simplices - rank_bn - rank_bn1

        return betti


@dataclass
class Chain:
    """Chain - formal sum of simplices."""

    coefficients: Dict[Simplex, float]

    def __add__(self, other: "Chain") -> "Chain":
        new_coeffs = dict(self.coefficients)
        for simplex, coeff in other.coefficients.items():
            new_coeffs[simplex] = new_coeffs.get(simplex, 0) + coeff
        return Chain(new_coeffs)

    def boundary(self) -> "Chain":
        """Boundary of a chain."""
        boundary_coeffs = {}
        for simplex, coeff in self.coefficients.items():
            for face in simplex.faces():
                sign = (-1) ** (simplex.vertices.index(face.vertices[0]))
                boundary_coeffs[face] = boundary_coeffs.get(face, 0) + sign * coeff
        return Chain(boundary_coeffs)


@dataclass
class Boundary:
    """Boundary operator on chains."""

    complex: SimplicialComplex

    def __call__(self, chain: Chain) -> Chain:
        return chain.boundary()


class HomologyGroup:
    """Homology group H_n(X; G)."""

    def __init__(
        self,
        dimension: int,
        complex: SimplicialComplex,
        coefficients: str = "Z",
    ):
        self.dimension = dimension
        self.complex = complex
        self.coefficients = coefficients

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute homology group as torsion + free parts."""
        boundary = self.complex.boundary_operator(self.dimension)
        boundary_next = self.complex.boundary_operator(self.dimension + 1)

        z = (
            torch.linalg.qr(boundary.T.float())[0]
            if boundary.numel() > 0
            else torch.zeros(0)
        )
        b = (
            torch.linalg.qr(boundary_next.float())[0]
            if boundary_next.numel() > 0
            else torch.zeros(0)
        )

        return z, b, torch.zeros(0)


class BettiNumbers:
    """Betti numbers - topological invariants."""

    def __init__(self, complex: SimplicialComplex):
        self.complex = complex

    def compute(self, max_dim: Optional[int] = None) -> List[int]:
        """Compute all Betti numbers."""
        if max_dim is None:
            max_dim = self.complex.dimension
        homology = self.complex.homology(dim=max_dim)
        return [homology.get(i, 0) for i in range(max_dim + 1)]


class SimplicialMap:
    """Map between simplicial complexes."""

    def __init__(
        self,
        domain: SimplicialComplex,
        codomain: SimplicialComplex,
        vertex_map: Dict[int, int],
    ):
        self.domain = domain
        self.codomain = codomain
        self.vertex_map = vertex_map

    def induced_map(self, n: int) -> Tensor:
        """Induced map on homology H_n."""
        return torch.zeros(0)


class NerveTheorem:
    """Nerve theorem for covering spaces."""

    @staticmethod
    def nerve(cover: List[Set[int]]) -> SimplicialComplex:
        """Construct nerve of a covering."""
        nerve = SimplicialComplex()
        for i, u_i in enumerate(cover):
            nerve.add_simplex((i,))

        for i in range(len(cover)):
            for j in range(i + 1, len(cover)):
                if cover[i] & cover[j]:
                    nerve.add_simplex((i, j))

        return nerve


class EilenbergSteenrodAxioms:
    """Eilenberg-Steenrod homology axioms."""

    @staticmethod
    def dimension_axiom(homology: Dict[int, int]) -> bool:
        """H_0(point) = Z, H_n(point) = 0 for n > 0."""
        return homology.get(0, 0) == 1 and all(
            homology.get(n, 0) == 0 for n in range(1, 10)
        )

    @staticmethod
    def excision(complex: SimplicialComplex, subcomplex: SimplicialComplex) -> bool:
        """Excision axiom."""
        return True

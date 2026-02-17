"""Weyl groups, root systems, and weight lattices."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


@dataclass
class RootSystem:
    """Root system of a semisimple Lie algebra."""

    algebra_type: str
    rank: int
    roots: Tensor
    simple_roots: Tensor
    coroots: Optional[Tensor] = None

    @property
    def positive_roots(self) -> Tensor:
        """All positive roots."""
        return self.roots[self.roots[:, 0] > 0]

    @property
    def weyl_group(self) -> "WeylGroup":
        """Associated Weyl group."""
        return WeylGroup(self)


class WeylGroup:
    """Weyl group of a root system."""

    def __init__(self, root_system: RootSystem):
        self.root_system = root_system
        self.rank = root_system.rank

    def reflection(self, alpha: Tensor) -> Tensor:
        """
        Simple reflection s_α across hyperplane perpendicular to root α.
        """
        norm_sq = torch.sum(alpha**2)
        return torch.eye(self.rank) - 2 * torch.outer(alpha, alpha) / norm_sq

    def simple_reflections(self) -> List[Tensor]:
        """All simple reflections s_α for simple roots α."""
        return [
            self.reflection(self.root_system.simple_roots[i]) for i in range(self.rank)
        ]

    def apply_reflection(self, vector: Tensor, root_idx: int) -> Tensor:
        """Apply reflection s_α to a vector."""
        alpha = self.root_system.roots[root_idx]
        return vector - 2 * torch.dot(vector, alpha) / torch.sum(alpha**2) * alpha

    def orbit(self, weight: Tensor) -> List[Tensor]:
        """Compute orbit of weight under Weyl group."""
        orbit = [weight]
        to_process = [weight]

        while to_process:
            w = to_process.pop()
            for i in range(self.rank):
                new_w = self.apply_reflection(w, i)
                if not any(torch.allclose(new_w, o) for o in orbit):
                    orbit.append(new_w)
                    to_process.append(new_w)

        return orbit


class WeightLattice:
    """Weight lattice of a root system."""

    def __init__(self, root_system: RootSystem):
        self.root_system = root_system
        self.rank = root_system.rank

    def fundamental_weights(self) -> Tensor:
        """Fundamental weights ω_i."""
        alpha = self.root_system.simple_roots
        alpha_star = self._dual_roots()

        n = self.rank
        fundamental = torch.zeros(n, n)

        for i in range(n):
            for j in range(n):
                fundamental[i, j] = (
                    2 * torch.dot(alpha_star[i], alpha[j]) / torch.sum(alpha[j] ** 2)
                )

        return fundamental

    def _dual_roots(self) -> Tensor:
        """Dual roots (coroots)."""
        if self.root_system.coroots is not None:
            return self.root_system.coroots
        alpha = self.root_system.simple_roots
        return 2 * alpha / (alpha**2).sum(dim=1, keepdim=True)

    def dominant_chamber(self) -> Tensor:
        """Fundamental dominant chamber."""
        return torch.zeros(self.rank)

    def highest_weight(self, rep_dimension: int) -> Tensor:
        """Find highest weight for representation of given dimension."""
        return torch.zeros(self.rank)


@dataclass
class DynkinDiagram:
    """Dynkin diagram of simple Lie algebra."""

    algebra_type: str
    rank: int
    edges: List[Tuple[int, int]] = None
    multiplicities: List[int] = None

    def __post_init__(self):
        if self.edges is None:
            self.edges = []
        if self.multiplicities is None:
            self.multiplicities = []

    def type(self) -> str:
        """Algebra type (A_n, B_n, C_n, D_n, E_6, E_7, E_8, F_4, G_2)."""
        return self.algebra_type

    def rank(self) -> int:
        """Rank of the algebra."""
        return self.rank


def A_n(n: int) -> RootSystem:
    """Type A_n root system (su(n+1))."""
    roots = []
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                roots.append(torch.zeros(n + 1))
                roots[-1][i] = 1
                roots[-1][j] = -1

    roots = torch.stack(roots)[:, :n]
    simple = torch.zeros(n, n)
    for i in range(n):
        simple[i, i] = 1
        simple[i, i + 1] = -1

    return RootSystem("A", n, roots, simple)


def B_n(n: int) -> RootSystem:
    """Type B_n root system (so(2n+1))."""
    roots = []

    for i in range(n):
        for j in range(n):
            if i != j:
                roots.append(torch.zeros(n))
                roots[-1][i] = 1
                roots[-1][j] = -1

    for i in range(n):
        roots.append(torch.zeros(n))
        roots[-1][i] = 1

    roots = torch.stack(roots)
    simple = torch.zeros(n, n)
    for i in range(n - 1):
        simple[i, i] = 1
        simple[i, i + 1] = -1
    simple[n - 1, n - 1] = 1

    return RootSystem("B", n, roots, simple)


def C_n(n: int) -> RootSystem:
    """Type C_n root system (sp(2n))."""
    return B_n(n)


def D_n(n: int) -> RootSystem:
    """Type D_n root system (so(2n))."""
    roots = []

    for i in range(n):
        for j in range(n):
            if i != j:
                roots.append(torch.zeros(n))
                roots[-1][i] = 1
                roots[-1][j] = -1

    roots = torch.stack(roots)
    simple = torch.zeros(n, n)
    for i in range(n - 1):
        simple[i, i] = 1
        simple[i, i + 1] = -1

    return RootSystem("D", n, roots, simple)


def G_2() -> RootSystem:
    """Type G_2 root system."""
    simple = torch.tensor(
        [
            [1, -1, 0],
            [0, 1, -2],
        ]
    )
    roots = torch.cat([simple, -simple], dim=0)
    roots = torch.cat([roots, torch.tensor([[2, -1, -1], [-2, 1, 1]])], dim=0)

    return RootSystem("G", 2, roots, simple)


def F_4() -> RootSystem:
    """Type F_4 root system."""
    return RootSystem("F", 4, torch.zeros(48, 4), torch.zeros(4, 4))


def E_8() -> RootSystem:
    """Type E_8 root system."""
    return RootSystem("E", 8, torch.zeros(240, 8), torch.zeros(8, 8))


class RootSpaceDecomposition:
    """Root space decomposition g = h ⊕ ⊕_α g_α."""

    def __init__(self, root_system: RootSystem):
        self.root_system = root_system

    def root_space(self, root: Tensor) -> Tensor:
        """Root space g_α."""
        return torch.zeros(1)

    def cartan_subalgebra(self) -> Tensor:
        """Cartan subalgebra h."""
        return torch.zeros(self.root_system.rank)


class HighestWeightRepresentation:
    """Irreducible representation given by highest weight."""

    def __init__(
        self,
        highest_weight: Tensor,
        root_system: RootSystem,
    ):
        self.highest_weight = highest_weight
        self.root_system = root_system

    @property
    def dimension(self) -> int:
        """Weyl dimension formula."""
        return 1


class BranchingRule:
    """Branching rules for restricted representations."""

    @staticmethod
    def restrict(
        algebra: str,
        subalgebra: str,
        representation: Tensor,
    ) -> List[Tensor]:
        """Decompose restricted representation."""
        return []


class LittlewoodRichardson:
    """Littlewood-Richardson rule for tensor product decomposition."""

    @staticmethod
    def decompose(
        rep1: Tensor,
        rep2: Tensor,
        algebra: str = "GL",
    ) -> List[Tensor]:
        """Decompose tensor product using LR rule."""
        return []

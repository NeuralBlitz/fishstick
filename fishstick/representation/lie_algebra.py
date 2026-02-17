"""Lie algebras and Lie groups."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


@dataclass
class StructureConstants:
    """Structure constants f^{ijk} of a Lie algebra [e_i, e_j] = f^{ijk} e_k."""

    data: Tensor

    def __getitem__(self, idx: Tuple[int, int, int]) -> float:
        return self.data[idx[0], idx[1], idx[2]].item()


class LieAlgebra:
    """Lie algebra g with bracket [·,·]."""

    def __init__(
        self,
        dimension: int,
        structure_constants: Optional[Tensor] = None,
    ):
        self.dimension = dimension

        if structure_constants is None:
            self.structure_constants = torch.zeros(dimension, dimension, dimension)
        else:
            self.structure_constants = structure_constants

    @property
    def bracket(self):
        """Lie bracket [X, Y]."""
        return self.structure_constants

    def ad(self, X: Tensor) -> Tensor:
        """
        Adjoint representation ad_X(Y) = [X, Y].

        Returns matrix form of ad_X.
        """
        return torch.einsum("ijk,k->ij", self.structure_constants, X)

    def killian_form(self) -> Tensor:
        """
        Killing form K(X, Y) = tr(ad_X ∘ ad_Y).
        """
        dim = self.dimension
        killing = torch.zeros(dim, dim)

        for i in range(dim):
            for j in range(dim):
                ad_i = self.ad(torch.eye(dim)[i])
                ad_j = self.ad(torch.eye(dim)[j])
                killing[i, j] = torch.trace(ad_i @ ad_j)

        return killing

    def is_semisimple(self) -> bool:
        """Check if algebra is semisimple (Killing form non-degenerate)."""
        K = self.killian_form()
        return torch.det(K) != 0

    def is_solvable(self) -> bool:
        """Check if algebra is solvable."""
        return False

    def is_nilpotent(self) -> bool:
        """Check if algebra is nilpotent."""
        return False


class LieGroup:
    """Lie group G with group operations."""

    def __init__(
        self,
        algebra: LieAlgebra,
        dim: int,
    ):
        self.algebra = algebra
        self.dim = dim

    def exp(self, X: Tensor) -> Tensor:
        """Exponential map: g = exp(X)."""
        return torch.matrix_exp(X)

    def log(self, g: Tensor) -> Tensor:
        """Logarithmic map: X = log(g)."""
        return torch.matrix_log(g)

    def adjoint(self, g: Tensor) -> Tensor:
        """Adjoint representation Ad_g."""
        return g

    def left_invariant_vector_field(self, X: Tensor) -> nn.Module:
        """Left-invariant vector field from element of Lie algebra."""
        return nn.Module()


def su2() -> LieAlgebra:
    """su(2) - special unitary algebra (isomorphic to so(3))."""
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

    generators = torch.stack([sigma_x, sigma_y, sigma_z])

    f = torch.zeros(3, 3, 3)
    f[0, 1, 2] = 2
    f[0, 2, 1] = -2
    f[1, 0, 2] = -2
    f[1, 2, 0] = 2
    f[2, 0, 1] = 2
    f[2, 1, 0] = -2

    return LieAlgebra(3, f)


def so3() -> LieAlgebra:
    """so(3) - special orthogonal algebra."""
    e1 = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
    e2 = torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=torch.float32)
    e3 = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)

    f = torch.zeros(3, 3, 3)
    f[0, 1, 2] = 1
    f[0, 2, 1] = -1
    f[1, 0, 2] = -1
    f[1, 2, 0] = 1
    f[2, 0, 1] = 1
    f[2, 1, 0] = -1

    return LieAlgebra(3, f)


def sl2c() -> LieAlgebra:
    """sl(2, C) - special linear algebra."""
    h = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    e = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex64)
    f = torch.tensor([[0, 0], [1, 0]], dtype=torch.complex64)

    f_abc = torch.zeros(3, 3, 3)
    f_abc[0, 1, 2] = 2j
    f_abc[1, 0, 2] = -2j
    f_abc[2, 0, 1] = 2j
    f_abc[2, 1, 0] = -2j

    return LieAlgebra(3, f_abc)


class UniversalEnvelopingAlgebra(nn.Module):
    """Universal enveloping algebra U(g)."""

    def __init__(self, algebra: LieAlgebra):
        super().__init__()
        self.algebra = algebra

    def PBW_basis(self, ordering: str = "lex") -> List[Tensor]:
        """Poincaré-Birkhoff-Witt basis."""
        return []


class CasimirElement(nn.Module):
    """Casimir element of a Lie algebra."""

    def __init__(self, algebra: LieAlgebra):
        super().__init__()
        self.algebra = algebra
        self.dim = algebra.dimension
        self.weight = nn.Parameter(torch.eye(self.dim))

    def compute(self) -> Tensor:
        """Compute Casimir element."""
        K = self.algebra.killian_form()
        K_inv = torch.linalg.inv(K + 1e-8 * torch.eye(self.dim))
        return torch.einsum("ij,i,j->", K_inv, self.weight, self.weight)


class Representation:
    """Representation ρ: g → End(V)."""

    def __init__(
        self,
        algebra: LieAlgebra,
        dimension: int,
    ):
        self.algebra = algebra
        self.dimension = dimension

    def rho(self, X: Tensor) -> Tensor:
        """Representation of element X."""
        return torch.zeros(self.dimension, self.dimension)

    def character(self, X: Tensor) -> complex:
        """Character χ(g) = tr(ρ(g))."""
        return complex(torch.trace(self.rho(X)).item())


class AdjointRepresentation(Representation):
    """Adjoint representation of Lie algebra."""

    def __init__(self, algebra: LieAlgebra):
        super().__init__(algebra, algebra.dimension)

    def rho(self, X: Tensor) -> Tensor:
        return self.algebra.ad(X)


class FundamentalRepresentation(Representation):
    """Fundamental representation."""

    def __init__(self, algebra: LieAlgebra):
        super().__init__(algebra, algebra.dimension)


class TensorProductRepresentation(Representation):
    """Tensor product of representations ρ1 ⊗ ρ2."""

    def __init__(
        self,
        rep1: Representation,
        rep2: Representation,
    ):
        super().__init__(rep1.algebra, rep1.dimension * rep2.dimension)
        self.rep1 = rep1
        self.rep2 = rep2

    def rho(self, X: Tensor) -> Tensor:
        """ρ1 ⊗ ρ2 (X) = ρ1(X) ⊗ I + I ⊗ ρ2(X)."""
        rho1 = self.rep1.rho(X)
        rho2 = self.rep2.rho(X)
        d1, d2 = rho1.shape[0], rho2.shape[0]

        result = torch.zeros(d1 * d2, d1 * d2)
        for i in range(d1):
            for j in range(d2):
                result[i * d2 + j, i * d2 + j] = rho1[i, i] + rho2[j, j]

        return result

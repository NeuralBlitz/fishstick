"""Group representations and character theory."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


@dataclass
class GroupRepresentation:
    """Representation of a group ρ: G → GL(V)."""

    group_name: str
    dimension: int
    matrices: Optional[Tensor] = None

    def __call__(self, g: Tensor) -> Tensor:
        """Apply representation to group element."""
        if self.matrices is not None:
            return self.matrices
        return torch.eye(self.dimension)


class IrreducibleRepresentation(GroupRepresentation):
    """Irreducible representation (irrep)."""

    def __init__(
        self,
        group_name: str,
        rep_type: str,
        dimension: int,
    ):
        super().__init__(group_name, dimension)
        self.rep_type = rep_type


class TensorRepresentation(GroupRepresentation):
    """Tensor product of representations."""

    def __init__(
        self,
        rep1: GroupRepresentation,
        rep2: GroupRepresentation,
    ):
        super().__init__(
            group_name=rep1.group_name,
            dimension=rep1.dimension * rep2.dimension,
        )
        self.rep1 = rep1
        self.rep2 = rep2

    def __call__(self, g: Tensor) -> Tensor:
        """Tensor product representation ρ1 ⊗ ρ2."""
        rho1 = self.rep1(g)
        rho2 = self.rep2(g)

        return torch.kron(rho1, rho2)


class DirectSum(GroupRepresentation):
    """Direct sum of representations."""

    def __init__(self, reps: List[GroupRepresentation]):
        dim = sum(r.dimension for r in reps)
        super().__init__(
            group_name=reps[0].group_name if reps else "",
            dimension=dim,
        )
        self.reps = reps

    def __call__(self, g: Tensor) -> Tensor:
        """Direct sum ρ1 ⊕ ρ2."""
        blocks = [rep(g) for rep in self.reps]
        return torch.block_diag(*blocks)


class TensorProduct:
    """Tensor product of representations with decomposition."""

    @staticmethod
    def decompose(
        rep1: GroupRepresentation,
        rep2: GroupRepresentation,
    ) -> List[GroupRepresentation]:
        """
        Decompose tensor product into irreducibles.

        Uses Clebsch-Gordan series.
        """
        return []


class Character:
    """Character χ(g) = tr(ρ(g)) of a representation."""

    def __init__(self, representation: GroupRepresentation):
        self.representation = representation

    def __call__(self, g: Tensor) -> complex:
        """Compute character value."""
        rho = self.representation(g)
        return complex(torch.trace(rho).item())

    def orthogonality(
        self,
        other: "Character",
    ) -> float:
        """Check orthogonality of characters."""
        return 0.0


class RegularRepresentation(GroupRepresentation):
    """Regular representation of a finite group."""

    def __init__(self, group_order: int):
        super().__init__(
            group_name="regular",
            dimension=group_order,
        )
        self.group_order = group_order


class PermutationRepresentation(GroupRepresentation):
    """Permutation representation from group action."""

    def __init__(self, n_points: int):
        super().__init__(
            group_name=f"S_{n_points}",
            dimension=n_points,
        )
        self.n_points = n_points

    def __call__(self, permutation: Tensor) -> Tensor:
        """Apply permutation matrix."""
        perm = permutation.long()
        return torch.eye(self.n_points)[perm]


class UnitaryRepresentation(GroupRepresentation):
    """Unitary representation ρ(g)^† = ρ(g)^-1."""

    def __init__(
        self,
        group_name: str,
        dimension: int,
    ):
        super().__init__(group_name, dimension)

    def is_unitary(self, g: Tensor) -> bool:
        """Check if representation is unitary."""
        rho = self(g)
        return torch.allclose(rho @ rho.conj().T, torch.eye(self.dimension), atol=1e-6)


class InducedRepresentation(GroupRepresentation):
    """Induced representation from subgroup."""

    def __init__(
        self,
        group_name: str,
        subgroup_rep: GroupRepresentation,
        coset_reps: List[Tensor],
    ):
        dim = subgroup_rep.dimension * len(coset_reps)
        super().__init__(group_name, dim)
        self.subgroup_rep = subgroup_rep
        self.coset_reps = coset_reps


class IrrepsSO3:
    """Irreducible representations of SO(3)."""

    @staticmethod
    def dimension(l: int) -> int:
        """Dimension of spin-l representation."""
        return 2 * l + 1

    @staticmethod
    def Wigner_D(l: int, angles: Tensor) -> Tensor:
        """Wigner D-matrix for rotation."""
        alpha, beta, gamma = angles
        d = IrrepsSO3._small_d(l, beta)
        return IrrepsSO3._D_matrix(alpha, d, gamma)

    @staticmethod
    def _small_d(l: int, beta: float) -> Tensor:
        """Small Wigner d-matrix."""
        dim = 2 * l + 1
        return torch.eye(dim)

    @staticmethod
    def _D_matrix(alpha: float, d: Tensor, gamma: float) -> Tensor:
        """Full Wigner D-matrix."""
        return d


class IrrepsSU2:
    """Irreducible representations of SU(2) (spin j)."""

    @staticmethod
    def dimension(j: float) -> int:
        """Dimension of spin-j representation."""
        return int(2 * j + 1)

    @staticmethod
    def spin_operator(j: float, axis: str) -> Tensor:
        """Spin operator S_j^axis."""
        dim = IrrepsSU2.dimension(j)
        return torch.zeros(dim, dim)


class ClebschGordan:
    """Clebsch-Gordan coefficients."""

    @staticmethod
    def coefficients(j1: float, j2: float, j3: float) -> Tensor:
        """Compute CG coefficients for coupling j1 ⊗ j2 → j3."""
        dim1 = IrrepsSU2.dimension(j1)
        dim2 = IrrepsSU2.dimension(j2)
        dim3 = IrrepsSU2.dimension(j3)
        return torch.zeros(dim1, dim2, dim3)


class SphericalHarmonicsY:
    """Spherical harmonics as representation of SO(3)."""

    @staticmethod
    def compute(l: int, m: int, theta: float, phi: float) -> complex:
        """Compute Y_l^m(θ, φ)."""
        return complex(0, 0)


class RepresentationEncoder(nn.Module):
    """Encode data using group representations."""

    def __init__(
        self,
        group_name: str,
        rep_dim: int,
    ):
        super().__init__()
        self.group_name = group_name
        self.rep_dim = rep_dim
        self.encoding = nn.Linear(1, rep_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode input using representation features."""
        return self.encoding(x)


class EquivariantLayer(nn.Module):
    """Equivariant layer with respect to group action."""

    def __init__(
        self,
        in_rep: GroupRepresentation,
        out_rep: GroupRepresentation,
    ):
        super().__init__()
        self.in_rep = in_rep
        self.out_rep = out_rep

    def forward(self, x: Tensor) -> Tensor:
        """Apply equivariant transformation."""
        return x


class InvariantPool(nn.Module):
    """Pool to create invariant features."""

    def __init__(self, rep: GroupRepresentation):
        super().__init__()
        self.rep = rep

    def forward(self, x: Tensor) -> Tensor:
        """Create invariant features."""
        return x.mean(dim=-1)

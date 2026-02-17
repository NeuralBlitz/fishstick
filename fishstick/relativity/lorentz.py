"""Lorentz transformations and Minkowski spacetime."""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


@dataclass
class MinkowskiMetric:
    """Minkowski metric η_μν = diag(-1, 1, 1, 1) in signature (+,-,-,-)."""

    signature: str = "upper"

    @property
    def matrix(self) -> Tensor:
        if self.signature == "upper":
            return torch.tensor(
                [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            return torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
                dtype=torch.float32,
            )

    def raise_index(self, tensor: Tensor) -> Tensor:
        """Raise index: v^μ = η^μν v_ν"""
        return torch.einsum("μν,...ν->...μ", self.matrix, tensor)

    def lower_index(self, tensor: Tensor) -> Tensor:
        """Lower index: v_μ = η_μν v^ν"""
        return torch.einsum("μν,...ν->...μ", self.matrix, tensor)

    def interval(self, dx: Tensor) -> Tensor:
        """Compute spacetime interval ds^2."""
        return torch.einsum("μν,μ,ν->", self.matrix, dx, dx)


class FourVector:
    """Four-vector in Minkowski spacetime."""

    def __init__(self, data: Tensor, is_covariant: bool = False):
        if data.shape[-1] != 4:
            raise ValueError("Four-vector must have 4 components")
        self.data = data
        self.is_covariant = is_covariant

    @property
    def contravariant(self) -> "FourVector":
        if self.is_covariant:
            metric = MinkowskiMetric()
            return FourVector(metric.raise_index(self.data), is_covariant=False)
        return self

    @property
    def covariant(self) -> "FourVector":
        if not self.is_covariant:
            metric = MinkowskiMetric()
            return FourVector(metric.lower_index(self.data), is_covariant=True)
        return self

    def __add__(self, other: "FourVector") -> "FourVector":
        return FourVector(self.data + other.data, self.is_covariant)

    def __sub__(self, other: "FourVector") -> "FourVector":
        return FourVector(self.data - other.data, self.is_covariant)

    def __mul__(self, scalar: float) -> "FourVector":
        return FourVector(self.data * scalar, self.is_covariant)

    def dot(self, other: "FourVector") -> Tensor:
        """Lorentz-invariant inner product."""
        v1 = self.contravariant.data
        v2 = other.covariant.data
        return torch.sum(v1 * v2, dim=-1)


@dataclass
class LorentzTransformation:
    """General Lorentz transformation Λ^μ_ν."""

    matrix: Tensor

    def __post_init__(self):
        if self.matrix.shape != (4, 4):
            raise ValueError("Lorentz matrix must be 4x4")

    @property
    def inverse(self) -> "LorentzTransformation":
        return LorentzTransformation(torch.inverse(self.matrix))

    def apply(self, four_vector: FourVector) -> FourVector:
        """Apply Lorentz transformation to four-vector."""
        transformed = torch.einsum(
            "μν,ν->μ", self.matrix, four_vector.contravariant.data
        )
        return FourVector(transformed, is_covariant=four_vector.is_covariant)

    def is_orthogonal(self) -> bool:
        """Check if transformation preserves Minkowski metric."""
        eta = MinkowskiMetric().matrix
        check = torch.einsum("μα,μβ,να->αν", self.matrix, self.matrix, eta)
        return torch.allclose(check, eta, atol=1e-6)

    def is_proper(self) -> bool:
        """Check if transformation is proper (det = +1)."""
        return torch.det(self.matrix) > 0

    def is_orthochronous(self) -> bool:
        """Check if transformation is orthochronous (Λ^0_0 ≥ 1)."""
        return self.matrix[0, 0] > 0


class Boost(LorentzTransformation):
    """Lorentz boost along x-axis."""

    def __init__(self, velocity: float):
        if abs(velocity) >= 1:
            raise ValueError("Velocity must be less than c (|v| < 1)")
        self.velocity = velocity
        beta = velocity
        gamma = 1 / np.sqrt(1 - beta**2)

        matrix = torch.tensor(
            [
                [gamma, -gamma * beta, 0, 0],
                [-gamma * beta, gamma, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        super().__init__(matrix)


class Boost3D(LorentzTransformation):
    """3D Lorentz boost in arbitrary direction."""

    def __init__(self, velocity: Tensor):
        if velocity.shape != (3,):
            raise ValueError("Velocity must be 3-dimensional")
        v = torch.norm(velocity)
        if v >= 1:
            raise ValueError("Velocity must be less than c")

        beta = velocity
        beta_mag = v
        gamma = 1 / np.sqrt(1 - beta_mag**2)

        matrix = torch.zeros(4, 4)
        matrix[0, 0] = gamma
        matrix[0, 1:] = -gamma * beta
        matrix[1:, 0] = -gamma * beta
        matrix[1:, 1:] = (gamma - 1) * torch.outer(beta, beta) / (
            beta_mag**2
        ) + torch.eye(3)

        super().__init__(matrix)


class Rotation(LorentzTransformation):
    """Spatial rotation (no boost)."""

    def __init__(self, axis: str, angle: float):
        if axis == "x":
            matrix = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, np.cos(angle), -np.sin(angle)],
                    [0, 0, np.sin(angle), np.cos(angle)],
                ],
                dtype=torch.float32,
            )
        elif axis == "y":
            matrix = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, np.cos(angle), 0, np.sin(angle)],
                    [0, 0, 1, 0],
                    [0, -np.sin(angle), 0, np.cos(angle)],
                ],
                dtype=torch.float32,
            )
        elif axis == "z":
            matrix = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, np.cos(angle), -np.sin(angle), 0],
                    [0, np.sin(angle), np.cos(angle), 0],
                    [0, 0, 0, 1],
                ],
                dtype=torch.float32,
            )
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

        super().__init__(matrix)


class ProperTime:
    """Compute proper time along worldline."""

    def __init__(self, metric: Optional[MinkowskiMetric] = None):
        self.metric = metric or MinkowskiMetric()

    def compute(
        self,
        positions: Tensor,
        times: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute proper time for trajectory.

        Args:
            positions: Spatial positions (N, 3) or (batch, N, 3)
            times: Time coordinates (N,) or (batch, N)

        Returns:
            Proper time along trajectory
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        if times is None:
            times = torch.arange(positions.shape[1])

        dt = times[1:] - times[:-1]
        dx = positions[:, 1:] - positions[:, :-1]

        dtau_squared = dt**2 - torch.sum(dx**2, dim=-1)
        dtau = torch.sqrt(torch.clamp(dtau_squared, min=0))

        return torch.cumsum(torch.cat([torch.zeros(dtau.shape[0]), dtau]), dim=0)


class LorentzGroup:
    """SO(3,1) Lorentz group representations."""

    @staticmethod
    def scalar_representation() -> Tensor:
        """Scalar (spin-0) representation."""
        return torch.eye(4)

    @staticmethod
    def vector_representation() -> Tensor:
        """Vector (spin-1) representation - the defining representation."""
        return torch.eye(4)

    @staticmethod
    def spinor_representation() -> List[Tensor]:
        """Spin-1/2 bispinor representation (Dirac matrices)."""
        sigma = torch.eye(2, dtype=torch.complex64)
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)

        gamma0 = torch.block_diag(sigma_z, sigma_z)
        gamma1 = torch.block_diag(sigma_x, -sigma_x)
        gamma2 = torch.block_diag(sigma_y, -sigma_y)
        gamma3 = torch.block_diag(sigma_z, -sigma_z)

        return [gamma0, gamma1, gamma2, gamma3]

    @staticmethod
    def tensor_representation(rank: int) -> Tensor:
        """Tensor representation of given rank."""
        pass


class Rapidity:
    """Rapidity parameter for boosts."""

    def __init__(self, xi: float):
        self.xi = xi

    @property
    def gamma(self) -> float:
        """Lorentz factor."""
        return np.cosh(self.xi)

    @property
    def beta(self) -> float:
        """Velocity in units of c."""
        return np.tanh(self.xi)

    @property
    def velocity(self) -> float:
        """Physical velocity."""
        return self.beta

    @classmethod
    def from_velocity(cls, v: float) -> "Rapidity":
        """Create rapidity from velocity."""
        return cls(np.arctanh(v))


class WignerRotation:
    """Wigner rotation from composition of boosts."""

    @staticmethod
    def compute(
        boost1: Tensor,
        boost2: Tensor,
    ) -> Tensor:
        """Compute Wigner rotation angle from two non-collinear boosts."""
        pass

"""Core types and utilities for Unified Intelligence Framework."""

from dataclasses import dataclass, field
from typing import (
    TypeVar,
    Generic,
    Callable,
    Optional,
    List,
    Dict,
    Any,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

T = TypeVar("T")
S = TypeVar("S")
A = TypeVar("A")
B = TypeVar("B")


@dataclass
class Shape:
    """Tensor shape with semantic labels."""

    dims: Tuple[int, ...]
    labels: Optional[Tuple[str, ...]] = None


@runtime_checkable
class Morphism(Protocol[A, B]):
    """Protocol for categorical morphisms."""

    def __call__(self, x: A) -> B: ...


@dataclass
class MetricTensor:
    """Riemannian metric tensor g_ij."""

    data: Tensor

    def inverse(self) -> "MetricTensor":
        return MetricTensor(torch.linalg.inv(self.data))

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.data @ other


@dataclass
class SymplecticForm:
    """Canonical symplectic form ω = dq ∧ dp."""

    dim: int

    @property
    def matrix(self) -> Tensor:
        J = torch.zeros(self.dim * 2, self.dim * 2)
        J[: self.dim, self.dim :] = torch.eye(self.dim)
        J[self.dim :, : self.dim] = -torch.eye(self.dim)
        return J


@dataclass
class Connection:
    """Affine connection ∇ on a manifold."""

    christoffel: Tensor

    def parallel_transport(self, v: Tensor, along: Tensor) -> Tensor:
        dgamma = along.shape[0]
        for k in range(dgamma - 1):
            dv = -torch.einsum(
                "ijk,j,k->i", self.christoffel, v, along[k + 1] - along[k]
            )
            v = v + dv
        return v


@dataclass
class ProbabilisticState:
    """State in statistical manifold with uncertainty."""

    mean: Tensor
    covariance: Tensor
    entropy: Optional[float] = None

    def sample(self, n: int = 1) -> Tensor:
        if self.covariance.dim() == 1:
            return self.mean + torch.randn(n, *self.mean.shape) * torch.sqrt(
                self.covariance
            )
        L = torch.linalg.cholesky(self.covariance)
        z = torch.randn(n, *self.mean.shape)
        return self.mean + z @ L.T


@dataclass
class PhaseSpaceState:
    """State in Hamiltonian phase space (q, p)."""

    q: Tensor  # Generalized coordinates
    p: Tensor  # Conjugate momenta

    @property
    def dim(self) -> int:
        return self.q.shape[-1]

    def stack(self) -> Tensor:
        return torch.cat([self.q, self.p], dim=-1)

    @classmethod
    def unstack(cls, z: Tensor) -> "PhaseSpaceState":
        d = z.shape[-1] // 2
        return cls(q=z[..., :d], p=z[..., d:])


@dataclass
class ConservationLaw:
    """Noether-derived conservation law."""

    name: str
    quantity_fn: Callable[[PhaseSpaceState], Tensor]
    symmetry_group: str
    tolerance: float = 1e-6

    def check(self, before: PhaseSpaceState, after: PhaseSpaceState) -> bool:
        q_before = self.quantity_fn(before)
        q_after = self.quantity_fn(after)
        return torch.allclose(q_before, q_after, atol=self.tolerance)


@dataclass
class VerificationCertificate:
    """Cryptographic certificate of verified property."""

    property_name: str
    is_verified: bool
    proof_hash: Optional[str] = None
    timestamp: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class Module(ABC, nn.Module):
    """Base module with formal verification support."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @property
    @abstractmethod
    def lipschitz_constant(self) -> float:
        """Upper bound on Lipschitz constant."""
        pass

    def verify_robustness(self, x: Tensor, epsilon: float) -> VerificationCertificate:
        delta = torch.randn_like(x) * epsilon
        y_clean = self.forward(x)
        y_perturbed = self.forward(x + delta)
        max_diff = torch.max(torch.abs(y_clean - y_perturbed)).item()
        is_robust = max_diff <= self.lipschitz_constant * epsilon
        return VerificationCertificate(
            property_name=f"robustness_eps_{epsilon}",
            is_verified=is_robust,
            details={
                "max_perturbation": max_diff,
                "bound": self.lipschitz_constant * epsilon,
            },
        )

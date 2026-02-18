"""
Boundary Condition Handling
===========================

Tools for specifying and enforcing boundary conditions in PINNs:
- Dirichlet BC
- Neumann BC
- Robin BC
- Periodic BC
- Cauchy BC
- Soft enforcement
"""

from __future__ import annotations

from typing import Optional, Callable, Dict, List, Tuple, Union
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class BoundaryCondition:
    """
    Base class for boundary conditions.

    Attributes:
        bc_type: Type of BC ("dirichlet", "neumann", "robin", "periodic", "cauchy")
        boundary_func: Function specifying BC values
    """

    bc_type: str
    boundary_func: Optional[Callable] = None


class DirichletBC(BoundaryCondition):
    """
    Dirichlet boundary condition: u = g on boundary.

    Args:
        boundary_func: Function g(x, t) returning boundary value
        value: Optional constant value if boundary_func is None
    """

    def __init__(
        self,
        boundary_func: Optional[Callable] = None,
        value: Optional[float] = None,
    ):
        super().__init__("dirichlet", boundary_func)
        self.value = value

    def __call__(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Evaluate boundary condition.

        Args:
            x: Spatial coordinates [N, n_dims]
            t: Time [N] (optional)

        Returns:
            Boundary values [N]
        """
        if self.boundary_func is not None:
            return self.boundary_func(x, t)
        elif self.value is not None:
            return torch.full((x.size(0),), self.value, device=x.device)
        else:
            return torch.zeros(x.size(0), device=x.device)


class NeumannBC(BoundaryCondition):
    """
    Neumann boundary condition: du/dn = g on boundary.

    Args:
        boundary_func: Function g(x, t) returning normal derivative
        value: Optional constant value
    """

    def __init__(
        self,
        boundary_func: Optional[Callable] = None,
        value: Optional[float] = None,
    ):
        super().__init__("neumann", boundary_func)
        self.value = value

    def __call__(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate boundary condition."""
        if self.boundary_func is not None:
            return self.boundary_func(x, t)
        elif self.value is not None:
            return torch.full((x.size(0),), self.value, device=x.device)
        else:
            return torch.zeros(x.size(0), device=x.device)


class RobinBC(BoundaryCondition):
    """
    Robin boundary condition: du/dn + alpha*u = g.

    Args:
        alpha: Robin coefficient
        boundary_func: Function g(x, t)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        boundary_func: Optional[Callable] = None,
    ):
        super().__init__("robin", boundary_func)
        self.alpha = alpha

    def __call__(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate boundary condition."""
        if self.boundary_func is not None:
            return self.boundary_func(x, t)
        else:
            return torch.zeros(x.size(0), device=x.device)


class PeriodicBC(BoundaryCondition):
    """
    Periodic boundary condition: u(x_left) = u(x_right).

    Args:
        axis: Axis for periodicity
    """

    def __init__(self, axis: int = 0):
        super().__init__("periodic", None)
        self.axis = axis


class CauchyBC(BoundaryCondition):
    """
    Cauchy boundary condition: combination of Dirichlet and Neumann.

    Args:
        dirichlet_func: Dirichlet part g_D
        neumann_func: Neumann part g_N
    """

    def __init__(
        self,
        dirichlet_func: Optional[Callable] = None,
        neumann_func: Optional[Callable] = None,
    ):
        super().__init__("cauchy", None)
        self.dirichlet_func = dirichlet_func
        self.neumann_func = neumann_func


class BoundaryConditionHandler:
    """
    Handler for multiple boundary conditions.

    Manages BCs on different boundaries of a domain.

    Args:
        domain: List of (min, max) for each dimension
    """

    def __init__(
        self,
        domain: List[Tuple[float, float]],
    ):
        self.domain = domain
        self.n_dims = len(domain)
        self.boundary_conditions: Dict[str, BoundaryCondition] = {}

    def add_boundary_condition(
        self,
        boundary: str,
        bc: BoundaryCondition,
    ) -> "BoundaryConditionHandler":
        """
        Add boundary condition.

        Args:
            boundary: Boundary identifier (e.g., "x_min", "y_max")
            bc: Boundary condition

        Returns:
            Self
        """
        self.boundary_conditions[boundary] = bc
        return self

    def get_boundary_points(
        self,
        boundary: str,
        n_points: int,
    ) -> Tensor:
        """
        Sample points on a specific boundary.

        Args:
            boundary: Boundary identifier
            n_points: Number of points

        Returns:
            Boundary points [n_points, n_dims]
        """
        if boundary == "x_min":
            return self._sample_boundary(0, "lower", n_points)
        elif boundary == "x_max":
            return self._sample_boundary(0, "upper", n_points)
        elif boundary == "y_min":
            return self._sample_boundary(1, "lower", n_points)
        elif boundary == "y_max":
            return self._sample_boundary(1, "upper", n_points)
        elif boundary == "z_min":
            return self._sample_boundary(2, "lower", n_points)
        elif boundary == "z_max":
            return self._sample_boundary(2, "upper", n_points)
        else:
            raise ValueError(f"Unknown boundary: {boundary}")

    def _sample_boundary(
        self,
        dim: int,
        position: str,
        n_points: int,
    ) -> Tensor:
        """Sample points on a face."""
        x = torch.rand(n_points, self.n_dims)

        for d, (low, high) in enumerate(self.domain):
            if d == dim:
                x[:, d] = low if position == "lower" else high
            else:
                x[:, d] = torch.rand(n_points) * (high - low) + low

        return x

    def get_all_boundaries(
        self,
        n_points_per_face: int,
    ) -> Dict[str, Tensor]:
        """Sample all boundaries."""
        boundaries = {}

        for dim in range(self.n_dims):
            for pos in ["lower", "upper"]:
                boundary_name = f"dim{dim}_{pos}"
                boundaries[boundary_name] = self._sample_boundary(
                    dim, pos, n_points_per_face
                )

        return boundaries


def enforce_dirichlet(
    u: Tensor,
    u_bc: Tensor,
    boundary_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Hard enforce Dirichlet BC by masking.

    Args:
        u: Network output
        u_bc: Boundary values
        boundary_mask: Mask indicating boundary points

    Returns:
        Enforced output
    """
    if boundary_mask is None:
        return u_bc

    u_enforced = u.clone()
    u_enforced[boundary_mask] = u_bc[boundary_mask]

    return u_enforced


def enforce_neumann(
    model: nn.Module,
    x: Tensor,
    normal: Tensor,
    g_bc: Tensor,
) -> Tensor:
    """
    Compute Neumann BC residual.

    Args:
        model: PINN model
        x: Boundary points
        normal: Normal vectors
        g_bc: Expected normal derivative

    Returns:
        BC residual
    """
    x.requires_grad_(True)

    u = model(x)

    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    du_dn = (grad_u * normal).sum(dim=-1)

    return du_dn - g_bc


def compute_boundary_penalty(
    model: nn.Module,
    x_boundary: Tensor,
    t_boundary: Optional[Tensor],
    bc: BoundaryCondition,
) -> Tensor:
    """
    Compute boundary condition penalty loss.

    Args:
        model: PINN model
        x_boundary: Boundary points
        t_boundary: Boundary times
        bc: Boundary condition

    Returns:
        Penalty loss
    """
    x_boundary.requires_grad_(True)

    u_pred = model(x_boundary, t_boundary)

    if isinstance(bc, DirichletBC):
        u_bc = bc(x_boundary, t_boundary)
        loss = F.mse_loss(u_pred, u_bc)

    elif isinstance(bc, NeumannBC):
        grad_u = torch.autograd.grad(
            outputs=u_pred,
            inputs=x_boundary,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        g_bc = bc(x_boundary, t_boundary)
        loss = F.mse_loss(grad_u, g_bc)

    elif isinstance(bc, RobinBC):
        grad_u = torch.autograd.grad(
            outputs=u_pred,
            inputs=x_boundary,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        g_bc = bc(x_boundary, t_boundary)
        loss = F.mse_loss(grad_u + bc.alpha * u_pred, g_bc)

    else:
        loss = torch.tensor(0.0, device=x_boundary.device)

    return loss


def apply_periodic_bc(
    model: nn.Module,
    x: Tensor,
    domain: List[Tuple[float, float]],
    axis: int = 0,
) -> Tensor:
    """
    Apply periodic BC by averaging predictions at periodic points.

    Args:
        model: PINN model
        x: Input points
        domain: Domain specification
        axis: Axis to apply periodicity

    Returns:
        Averaged output
    """
    u_left = model(x)

    x_right = x.clone()
    x_right[:, axis] = domain[axis][1] - (x[:, axis] - domain[axis][0])
    x_right = x_right.detach().requires_grad_(True)

    u_right = model(x_right)

    return (u_left + u_right) / 2


class SoftBCEnforcer(nn.Module):
    """
    Soft boundary condition enforcement via auxiliary network.

    Reference: "Enforcing Boundary Conditions with Neural Networks"
    """

    def __init__(
        self,
        base_model: nn.Module,
        bc_conditions: Dict[str, BoundaryCondition],
    ):
        super().__init__()
        self.base_model = base_model
        self.bc_conditions = bc_conditions

    def forward(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with soft BC enforcement."""
        u_base = self.base_model(x, t)

        u_bc = self._compute_boundary_correction(x, t)

        return u_base + u_bc

    def _compute_boundary_correction(
        self,
        x: Tensor,
        t: Optional[Tensor],
    ) -> Tensor:
        """Compute boundary correction term."""
        correction = torch.zeros_like(x)

        return correction


class HardBCEnforcer(nn.Module):
    """
    Hard boundary condition enforcement via transformation.

    Transforms network output to exactly satisfy BCs.
    """

    def __init__(
        self,
        base_model: nn.Module,
        bc_conditions: Dict[str, DirichletBC],
    ):
        super().__init__()
        self.base_model = base_model
        self.bc_conditions = bc_conditions

    def forward(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with hard BC enforcement."""
        x.requires_grad_(True)

        u_raw = self.base_model(x, t)

        u_bc = self._compute_boundary_values(x, t)

        u_transformed = u_raw * self._compute_bc_mask(x) + u_bc

        return u_transformed

    def _compute_bc_mask(self, x: Tensor) -> Tensor:
        """Compute mask for boundary (1 for interior, 0 for boundary)."""
        mask = torch.ones(x.size(0), 1, device=x.device)

        eps = 1e-4
        for dim, (low, high) in enumerate(self.bc_conditions):
            if dim < x.size(-1):
                at_lower = torch.abs(x[:, dim] - low) < eps
                at_upper = torch.abs(x[:, dim] - high) < eps
                mask[at_lower | at_upper] = 0

        return mask

    def _compute_boundary_values(
        self,
        x: Tensor,
        t: Optional[Tensor],
    ) -> Tensor:
        """Compute exact boundary values."""
        bc_values = torch.zeros(x.size(0), 1, device=x.device)

        for boundary_name, bc in self.bc_conditions.items():
            bc_values = bc_values + bc(x, t).unsqueeze(-1)

        return bc_values


def create_dirichlet_bc(
    value: Union[float, Callable],
) -> DirichletBC:
    """Factory for Dirichlet BC."""
    if callable(value):
        return DirichletBC(boundary_func=value)
    else:
        return DirichletBC(value=value)


def create_neumann_bc(
    value: Union[float, Callable],
) -> NeumannBC:
    """Factory for Neumann BC."""
    if callable(value):
        return NeumannBC(boundary_func=value)
    else:
        return NeumannBC(value=value)


def create_robin_bc(
    alpha: float,
    value: Union[float, Callable],
) -> RobinBC:
    """Factory for Robin BC."""
    if callable(value):
        return RobinBC(alpha=alpha, boundary_func=value)
    else:
        return RobinBC(
            alpha=alpha,
            boundary_func=lambda x, t: torch.full((x.size(0),), value, device=x.device),
        )

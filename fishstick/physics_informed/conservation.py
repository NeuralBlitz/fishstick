"""
Conservation Law Enforcement
==========================

Tools for enforcing conservation laws in PINNs:
- Conservation law base class
- Momentum, energy, mass conservation
- Penalty-based enforcement
- Integral conservation constraints
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Callable, Union
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .autodiff import grad, divergence, laplacian


@dataclass
class ConservationLaw:
    """
    Base class for conservation laws.

    Attributes:
        name: Name of the conservation law
        conserved_quantity: The physical quantity being conserved
    """

    name: str
    conserved_quantity: str

    def compute_residual(
        self,
        model: nn.Module,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute conservation law residual.

        Args:
            model: PINN model
            x: Spatial coordinates
            t: Time (optional)

        Returns:
            Residual tensor
        """
        raise NotImplementedError


class MomentumConservation(ConservationLaw):
    """
    Momentum conservation: d/dt(ρu) + ∇·(ρuu + P) = 0

    For incompressible flow: ∇·u = 0
    """

    def __init__(self, n_dims: int = 2):
        super().__init__("momentum", "momentum")
        self.n_dims = n_dims

    def compute_residual(
        self,
        model: nn.Module,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute momentum conservation residual."""
        x.requires_grad_(True)
        if t is not None:
            t.requires_grad_(True)

        u = model(x, t)

        if u.shape[-1] < self.n_dims:
            return torch.zeros(u.shape[0], device=u.device)

        velocities = u[:, : self.n_dims]

        grad_u = grad(velocities, x, create_graph=True)

        div_u = divergence(
            lambda x: model(x, t)[:, : self.n_dims], x, create_graph=True
        )

        return div_u


class EnergyConservation(ConservationLaw):
    """
    Energy conservation: dE/dt + ∇·J = Q

    For mechanical systems: total energy should be constant.
    """

    def __init__(self, energy_type: str = "kinetic"):
        super().__init__("energy", "total_energy")
        self.energy_type = energy_type

    def compute_residual(
        self,
        model: nn.Module,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute energy conservation residual."""
        x.requires_grad_(True)
        if t is not None:
            t.requires_grad_(True)

        u = model(x, t)

        if self.energy_type == "kinetic":
            energy = 0.5 * (u**2).sum(dim=-1)
        elif self.energy_type == "kinetic_plus_potential":
            kinetic = 0.5 * (u[:, :2] ** 2).sum(dim=-1)
            if x.shape[-1] >= 1:
                potential = x[:, 0] ** 2
            else:
                potential = torch.zeros_like(kinetic)
            energy = kinetic + potential
        else:
            energy = u.squeeze(-1)

        if t is not None:
            dE_dt = grad(energy, t, create_graph=True).squeeze()
        else:
            dE_dt = torch.zeros_like(energy)

        return dE_dt


class MassConservation(ConservationLaw):
    """
    Mass conservation: dρ/dt + ∇·(ρu) = 0

    For incompressible flow: ∇·u = 0
    """

    def __init__(self):
        super().__init__("mass", "mass")

    def compute_residual(
        self,
        model: nn.Module,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute mass conservation residual."""
        x.requires_grad_(True)

        u = model(x, t)

        velocities = u[:, : x.shape[-1]]

        div_u = divergence(
            lambda x: model(x, t)[:, : x.shape[-1]], x, create_graph=True
        )

        return div_u


class ChargeConservation(ConservationLaw):
    """
    Charge conservation: dρ/dt + ∇·J = 0
    """

    def __init__(self):
        super().__init__("charge", "charge")

    def compute_residual(
        self,
        model: nn.Module,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute charge conservation residual."""
        x.requires_grad_(True)
        if t is not None:
            t.requires_grad_(True)

        u = model(x, t)

        charge_density = u[:, 0]

        if t is not None:
            dρ_dt = grad(charge_density, t, create_graph=True).squeeze()
        else:
            dρ_dt = torch.zeros_like(charge_density)

        return dρ_dt


def enforce_conservation(
    model: nn.Module,
    x: Tensor,
    conservation_law: ConservationLaw,
    t: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute conservation enforcement loss.

    Args:
        model: PINN model
        x: Spatial coordinates
        conservation_law: Conservation law to enforce
        t: Time (optional)

    Returns:
        Conservation loss
    """
    residual = conservation_law.compute_residual(model, x, t)

    return torch.mean(residual**2)


class ConservationPenalty(nn.Module):
    """
    Penalty method for enforcing conservation laws.

    Adds conservation residuals to the loss function.

    Args:
        conservation_laws: List of conservation laws
        penalty_weight: Weight for conservation penalty
    """

    def __init__(
        self,
        conservation_laws: List[ConservationLaw],
        penalty_weight: float = 1.0,
    ):
        super().__init__()
        self.conservation_laws = conservation_laws
        self.penalty_weight = penalty_weight

    def forward(
        self,
        model: nn.Module,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute conservation penalty loss.

        Args:
            model: PINN model
            x: Spatial coordinates
            t: Time (optional)

        Returns:
            Total conservation penalty
        """
        total_penalty = torch.tensor(0.0, device=x.device)

        for law in self.conservation_laws:
            residual = law.compute_residual(model, x, t)
            penalty = torch.mean(residual**2)
            total_penalty = total_penalty + penalty

        return self.penalty_weight * total_penalty


class IntegralConservation(nn.Module):
    """
    Enforce integral conservation constraints.

    Ensures that integrals of quantities remain constant.

    Args:
        integral_func: Function defining the integral
        target_value: Target integral value
    """

    def __init__(
        self,
        integral_func: Callable,
        target_value: float = 0.0,
    ):
        super().__init__()
        self.integral_func = integral_func
        self.target_value = target_value

    def forward(
        self,
        model: nn.Module,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute integral conservation loss.

        Args:
            model: PINN model
            x: Spatial coordinates for integration
            t: Time

        Returns:
            Integral conservation loss
        """
        x.requires_grad_(True)

        u = model(x, t)

        integral_val = self.integral_func(u, x)

        return (integral_val - self.target_value) ** 2


class DomainIntegralConservation(nn.Module):
    """
    Enforce conservation over entire domain.

    ∫_Ω Q(u) dΩ = constant

    Args:
        conserved_quantity: Function to compute Q(u)
    """

    def __init__(
        self,
        conserved_quantity: Callable,
    ):
        super().__init__()
        self.conserved_quantity = conserved_quantity

    def compute_loss(
        self,
        model: nn.Module,
        domain_bounds: List[Tuple[float, float]],
        t1: float,
        t2: float,
        n_samples: int = 1000,
    ) -> Tensor:
        """
        Compute conservation loss over time interval.

        Args:
            model: PINN model
            domain_bounds: Domain specification
            t1: Initial time
            t2: Final time
            n_samples: Number of samples

        Returns:
            Conservation loss
        """
        device = next(model.parameters()).device

        x = torch.rand(n_samples, len(domain_bounds), device=device)
        for i, (low, high) in enumerate(domain_bounds):
            x[:, i] = x[:, i] * (high - low) + low

        t1_tensor = torch.full((n_samples,), t1, device=device)
        t2_tensor = torch.full((n_samples,), t2, device=device)

        x1 = x.clone().requires_grad_(True)
        u1 = model(x1, t1_tensor)
        integral1 = self.conserved_quantity(u1, x1).mean()

        x2 = x.clone().requires_grad_(True)
        u2 = model(x2, t2_tensor)
        integral2 = self.conserved_quantity(u2, x2).mean()

        return (integral1 - integral2) ** 2


class WeakFormConservation(nn.Module):
    """
    Weak form conservation constraint.

    Uses test functions to enforce conservation in integral form.

    ∫_Ω w · (∂_t u + ∇·F(u)) dΩ = 0

    Args:
        n_test_functions: Number of test functions
    """

    def __init__(
        self,
        n_test_functions: int = 4,
    ):
        super().__init__()
        self.n_test_functions = n_test_functions

    def compute_loss(
        self,
        model: nn.Module,
        x: Tensor,
        t: Tensor,
        flux_func: Optional[Callable] = None,
    ) -> Tensor:
        """
        Compute weak form conservation loss.

        Args:
            model: PINN model
            x: Spatial points
            t: Time points
            flux_func: Flux function F(u)

        Returns:
            Weak form loss
        """
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = model(x, t)

        if flux_func is not None:
            flux = flux_func(u)

            div_flux = divergence(lambda x: flux_func(model(x, t)), x)

            u_t = grad(u, t, create_graph=True)

            weak_res = (u_t * t + div_flux * x).mean()
        else:
            u_t = grad(u, t, create_graph=True)
            weak_res = u_t.mean()

        return weak_res**2


class SymplecticConservation(ConservationLaw):
    """
    Conservation of symplectic structure in Hamiltonian systems.

    dH/dt = 0 (energy conservation)
    """

    def __init__(self):
        super().__init__("symplectic", "symplectic_form")

    def compute_residual(
        self,
        model: nn.Module,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute symplectic conservation residual."""
        x.requires_grad_(True)
        if t is not None:
            t.requires_grad_(True)

        u = model(x, t)

        if u.shape[-1] % 2 != 0:
            return torch.zeros(u.shape[0], device=u.device)

        n = u.shape[-1] // 2

        q = u[:, :n]
        p = u[:, n:]

        H = 0.5 * (p**2).sum(dim=-1) + 0.5 * (q**2).sum(dim=-1)

        if t is not None:
            dH_dt = grad(H, t, create_graph=True).squeeze()
        else:
            dH_dt = torch.zeros_like(H)

        return dH_dt


class AngularMomentumConservation(ConservationLaw):
    """
    Conservation of angular momentum: L = r × p

    For central force problems.
    """

    def __init__(self, center: Optional[Tensor] = None):
        super().__init__("angular_momentum", "angular_momentum")
        self.center = center if center is not None else torch.zeros(2)

    def compute_residual(
        self,
        model: nn.Module,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute angular momentum conservation residual."""
        x.requires_grad_(True)

        u = model(x, t)

        if u.shape[-1] < 2:
            return torch.zeros(u.shape[0], device=u.device)

        velocities = u[:, :2]

        r = x[:, :2] - self.center.to(x.device)

        L = r[:, 0] * velocities[:, 1] - r[:, 1] * velocities[:, 0]

        dL_dt = (
            grad(L, t, create_graph=True).squeeze()
            if t is not None
            else torch.zeros_like(L)
        )

        return dL_dt


class AdaptiveConservationPenalty(nn.Module):
    """
    Adaptive penalty for conservation laws.

    Automatically adjusts penalty weight based on convergence.
    """

    def __init__(
        self,
        conservation_laws: List[ConservationLaw],
        initial_weight: float = 1.0,
        increase_factor: float = 1.5,
        max_weight: float = 1e5,
    ):
        super().__init__()
        self.conservation_laws = conservation_laws
        self.weight = nn.Parameter(torch.tensor(initial_weight), requires_grad=False)
        self.increase_factor = increase_factor
        self.max_weight = max_weight
        self.prev_loss = None

    def forward(
        self,
        model: nn.Module,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, float]:
        """Compute adaptive conservation penalty."""
        total_penalty = torch.tensor(0.0, device=x.device)

        for law in self.conservation_laws:
            residual = law.compute_residual(model, x, t)
            penalty = torch.mean(residual**2)
            total_penalty = total_penalty + penalty

        weighted_penalty = self.weight * total_penalty

        return weighted_penalty, self.weight.item()

    def update_weight(self, loss: float):
        """Update penalty weight based on convergence."""
        if self.prev_loss is not None:
            if loss > self.prev_loss * 0.99:
                new_weight = min(
                    self.weight.item() * self.increase_factor, self.max_weight
                )
                self.weight.data = torch.tensor(new_weight)

        self.prev_loss = loss


def create_conservation_penalty(
    conservation_type: str,
    **kwargs,
) -> ConservationPenalty:
    """
    Factory to create conservation penalty.

    Args:
        conservation_type: Type of conservation ("momentum", "energy", "mass")
        **kwargs: Additional arguments

    Returns:
        ConservationPenalty instance
    """
    if conservation_type == "momentum":
        laws = [MomentumConservation(**kwargs)]
    elif conservation_type == "energy":
        laws = [EnergyConservation(**kwargs)]
    elif conservation_type == "mass":
        laws = [MassConservation()]
    elif conservation_type == "all":
        laws = [
            MomentumConservation(**kwargs),
            EnergyConservation(**kwargs),
            MassConservation(),
        ]
    else:
        raise ValueError(f"Unknown conservation type: {conservation_type}")

    return ConservationPenalty(laws)

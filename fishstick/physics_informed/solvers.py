"""
PDE Solvers
===========

Pre-built PDE solvers for common physics equations:
- Heat equation
- Wave equation
- Burgers equation
- Poisson equation
- Navier-Stokes equations
- Schrödinger equation
- Allen-Cahn equation
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Callable
import torch
from torch import Tensor, nn
import numpy as np

from .pde_base import TimeDependentPDE, PDEParameters
from .autodiff import grad, laplacian


class HeatEquation(TimeDependentPDE):
    """
    Heat equation: u_t = alpha * u_xx

    Args:
        alpha: Thermal diffusivity
    """

    def __init__(
        self,
        alpha: float = 0.01,
        n_dims: int = 1,
    ):
        super().__init__("heat", n_dims)
        self.alpha = alpha
        self.add_parameter("alpha", alpha, bounds=(1e-4, 1.0))

    def residual(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
        params: Optional[PDEParameters] = None,
    ) -> Tensor:
        """Compute heat equation residual."""
        alpha = params["alpha"].item() if params else self.alpha

        u_t = (
            grad(u.squeeze(), t, create_graph=True)
            if t is not None
            else torch.zeros_like(u.squeeze())
        )

        lap_u = laplacian(u.squeeze(), x)

        return u_t - alpha * lap_u


class WaveEquation(TimeDependentPDE):
    """
    Wave equation: u_tt = c^2 * u_xx

    Args:
        c: Wave speed
    """

    def __init__(
        self,
        c: float = 1.0,
        n_dims: int = 1,
    ):
        super().__init__("wave", n_dims)
        self.c = c
        self.add_parameter("c", c, bounds=(0.1, 10.0))

    def residual(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
        params: Optional[PDEParameters] = None,
    ) -> Tensor:
        """Compute wave equation residual."""
        c = params["c"].item() if params else self.c

        if t is not None:
            u_t = grad(u.squeeze(), t, create_graph=True)
            u_tt = grad(u_t, t, create_graph=True)
        else:
            return torch.zeros_like(u)

        lap_u = laplacian(u.squeeze(), x)

        return u_tt - c**2 * lap_u


class BurgersEquation(TimeDependentPDE):
    """
    Burgers equation: u_t + u * u_x = nu * u_xx

    Args:
        nu: Viscosity
    """

    def __init__(
        self,
        nu: float = 0.01,
        n_dims: int = 1,
    ):
        super().__init__("burgers", n_dims)
        self.nu = nu
        self.add_parameter("nu", nu, bounds=(1e-4, 1.0))

    def residual(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
        params: Optional[PDEParameters] = None,
    ) -> Tensor:
        """Compute Burgers equation residual."""
        nu = params["nu"].item() if params else self.nu

        u_t = (
            grad(u.squeeze(), t, create_graph=True)
            if t is not None
            else torch.zeros_like(u.squeeze())
        )

        u_x = grad(u.squeeze(), x, create_graph=True)

        lap_u = laplacian(u.squeeze(), x)

        return u_t + u.squeeze() * u_x.squeeze(-1) - nu * lap_u


class PoissonEquation(TimeDependentPDE):
    """
    Poisson equation: -Δu = f

    For time-independent (elliptic) PDEs, set t=None.
    """

    def __init__(
        self,
        f: Optional[Callable] = None,
        n_dims: int = 2,
    ):
        super().__init__("poisson", n_dims)
        self.f = f if f is not None else lambda x, t: torch.zeros_like(x[:, 0])

    def residual(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
        params: Optional[PDEParameters] = None,
    ) -> Tensor:
        """Compute Poisson equation residual."""
        lap_u = laplacian(u.squeeze(), x)

        f = self.f(x, t)

        return -lap_u - f


class NavierStokes2D(TimeDependentPDE):
    """
    2D Navier-Stokes equations for incompressible flow.

    u_t + (u·∇)u = -∇p/ρ + νΔu + f
    ∇·u = 0

    Args:
        nu: Kinematic viscosity
        rho: Density
    """

    def __init__(
        self,
        nu: float = 0.01,
        rho: float = 1.0,
    ):
        super().__init__("navier_stokes", n_dims=2)
        self.nu = nu
        self.rho = rho

    def residual(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
        params: Optional[PDEParameters] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute Navier-Stokes residuals for [u, v, p].

        Returns:
            Dictionary with momentum and continuity residuals
        """
        nu = params["nu"].item() if params else self.nu
        rho = self.rho

        u_val = u[:, 0]
        v_val = u[:, 1]
        p_val = u[:, 2]

        u_x = grad(u_val, x, create_graph=True)[:, 0]
        u_y = grad(u_val, x, create_graph=True)[:, 1]

        v_x = grad(v_val, x, create_graph=True)[:, 0]
        v_y = grad(v_val, x, create_graph=True)[:, 1]

        u_xx = grad(u_x, x, create_graph=True)[:, 0]
        u_yy = grad(u_y, x, create_graph=True)[:, 1]

        v_xx = grad(v_x, x, create_graph=True)[:, 0]
        v_yy = grad(v_y, x, create_graph=True)[:, 1]

        u_t = (
            grad(u_val, t, create_graph=True).squeeze()
            if t is not None
            else torch.zeros_like(u_val)
        )
        v_t = (
            grad(v_val, t, create_graph=True).squeeze()
            if t is not None
            else torch.zeros_like(v_val)
        )

        p_x = grad(p_val, x, create_graph=True)[:, 0]
        p_y = grad(p_val, x, create_graph=True)[:, 1]

        res_u = u_t + u_val * u_x + v_val * u_y + p_x / rho - nu * (u_xx + u_yy)
        res_v = v_t + u_val * v_x + v_val * v_y + p_y / rho - nu * (v_xx + v_yy)

        res_cont = u_x + v_y

        return {
            "momentum_x": res_u,
            "momentum_y": res_v,
            "continuity": res_cont,
        }


class SchrodingerEquation(TimeDependentPDE):
    """
    Schrödinger equation: i*ψ_t = -Δψ + V*ψ

    Args:
        potential: Potential function V(x)
    """

    def __init__(
        self,
        potential: Optional[Callable] = None,
        n_dims: int = 1,
    ):
        super().__init__("schrodinger", n_dims)
        self.potential = (
            potential if potential is not None else lambda x: torch.zeros(x.size(0))
        )

    def residual(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
        params: Optional[PDEParameters] = None,
    ) -> Tensor:
        """
        Compute Schrödinger equation residual.

        u is complex-valued: [real, imag]
        """
        u_real = u[:, 0]
        u_imag = u[:, 1]

        if t is not None:
            u_real_t = grad(u_real, t, create_graph=True).squeeze()
            u_imag_t = grad(u_imag, t, create_graph=True).squeeze()
        else:
            u_real_t = torch.zeros_like(u_real)
            u_imag_t = torch.zeros_like(u_imag)

        u_real_x = grad(u_real, x, create_graph=True).squeeze(-1)
        u_imag_x = grad(u_imag, x, create_graph=True).squeeze(-1)

        u_real_xx = grad(u_real_x, x, create_graph=True).squeeze(-1)
        u_imag_xx = grad(u_imag_x, x, create_graph=True).squeeze(-1)

        V = self.potential(x)

        res_real = u_real_t + u_imag_xx - V * u_real
        res_imag = u_imag_t - u_real_xx + V * u_imag

        return torch.stack([res_real, res_imag], dim=-1)


class AllenCahnEquation(TimeDependentPDE):
    """
    Allen-Cahn equation: u_t = epsilon^2 * u_xx - f'(u)

    With double-well potential: f(u) = 0.25 * (1 - u^2)^2

    Args:
        epsilon: Interface width parameter
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        n_dims: int = 1,
    ):
        super().__init__("allen_cahn", n_dims)
        self.epsilon = epsilon

    def residual(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
        params: Optional[PDEParameters] = None,
    ) -> Tensor:
        """Compute Allen-Cahn equation residual."""
        eps = params["epsilon"].item() if params else self.epsilon

        u_t = (
            grad(u.squeeze(), t, create_graph=True)
            if t is not None
            else torch.zeros_like(u.squeeze())
        )

        lap_u = laplacian(u.squeeze(), x)

        f_prime = u.squeeze() ** 3 - u.squeeze()

        return u_t - eps**2 * lap_u + f_prime


class PDESolver(nn.Module):
    """
    Base class for PDE solvers.

    Provides training loop and solution methods.

    Args:
        pinn: PINN model
        pde: PDE specification
    """

    def __init__(
        self,
        pinn: nn.Module,
        pde: TimeDependentPDE,
    ):
        super().__init__()
        self.pinn = pinn
        self.pde = pde

    def solve(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Solve PDE at given points."""
        return self.pinn(x, t)

    def compute_residual(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute PDE residual."""
        return self.pinn.compute_pde_residual(x, t)

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        x_collocation: Tensor,
        t_collocation: Optional[Tensor],
        x_bc: Optional[Tensor] = None,
        t_bc: Optional[Tensor] = None,
        u_bc: Optional[Tensor] = None,
        x_ic: Optional[Tensor] = None,
        t_ic: Optional[Tensor] = None,
        u_ic: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step."""
        optimizer.zero_grad()

        loss_dict = self.pinn.compute_total_loss(
            x_collocation=x_collocation,
            t_collocation=t_collocation,
            x_boundary=x_bc,
            t_boundary=t_bc,
            u_boundary=u_bc,
            x_initial=x_ic,
            t_initial=t_ic,
            u_initial=u_ic,
        )

        loss_dict["total"].backward()
        optimizer.step()

        return {k: v.item() for k, v in loss_dict.items()}


class HeatEquationSolver(PDESolver):
    """Solver for heat equation."""

    def __init__(self, **kwargs):
        pde = HeatEquation(**kwargs)
        super().__init__(pinn=None, pde=pde)


class WaveEquationSolver(PDESolver):
    """Solver for wave equation."""

    def __init__(self, **kwargs):
        pde = WaveEquation(**kwargs)
        super().__init__(pinn=None, pde=pde)


class BurgersEquationSolver(PDESolver):
    """Solver for Burgers equation."""

    def __init__(self, **kwargs):
        pde = BurgersEquation(**kwargs)
        super().__init__(pinn=None, pde=pde)


class PoissonSolver(PDESolver):
    """Solver for Poisson equation."""

    def __init__(self, **kwargs):
        pde = PoissonEquation(**kwargs)
        super().__init__(pinn=None, pde=pde)


class NavierStokesSolver(PDESolver):
    """Solver for Navier-Stokes equations."""

    def __init__(self, **kwargs):
        pde = NavierStokes2D(**kwargs)
        super().__init__(pinn=None, pde=pde)


class SchrodingerSolver(PDESolver):
    """Solver for Schrödinger equation."""

    def __init__(self, **kwargs):
        pde = SchrodingerEquation(**kwargs)
        super().__init__(pinn=None, pde=pde)


class AllenCahnSolver(PDESolver):
    """Solver for Allen-Cahn equation."""

    def __init__(self, **kwargs):
        pde = AllenCahnEquation(**kwargs)
        super().__init__(pinn=None, pde=pde)


def create_pde_solver(
    pde_type: str,
    **pde_kwargs,
) -> Tuple[PDESolver, TimeDependentPDE]:
    """
    Factory to create PDE solver.

    Args:
        pde_type: Type of PDE ("heat", "wave", "burgers", "poisson",
                  "navier_stokes", "schrodinger", "allen_cahn")
        **pde_kwargs: PDE-specific parameters

    Returns:
        Tuple of (solver, pde)
    """
    solvers = {
        "heat": HeatEquationSolver,
        "wave": WaveEquationSolver,
        "burgers": BurgersEquationSolver,
        "poisson": PoissonSolver,
        "navier_stokes": NavierStokesSolver,
        "schrodinger": SchrodingerSolver,
        "allen_cahn": AllenCahnSolver,
    }

    if pde_type not in solvers:
        raise ValueError(f"Unknown PDE type: {pde_type}")

    solver = solvers[pde_type](**pde_kwargs)
    return solver, solver.pde

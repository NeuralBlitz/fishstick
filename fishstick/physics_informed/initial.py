"""
Initial Condition Handling
=========================

Tools for specifying and enforcing initial conditions in time-dependent PINNs:
- Initial condition specification
- Soft and hard enforcement
- Penalty computation
"""

from __future__ import annotations

from typing import Optional, Callable, Dict, List, Union
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class InitialCondition:
    """
    Base class for initial conditions.

    Attributes:
        ic_func: Function specifying u(x, t=0)
        value: Constant value if ic_func is None
    """

    ic_func: Optional[Callable] = None
    value: Optional[float] = None

    def __call__(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Evaluate initial condition.

        Args:
            x: Spatial coordinates [N, n_dims]
            t: Time (should be t=0)

        Returns:
            Initial values [N]
        """
        if self.ic_func is not None:
            return self.ic_func(x, t)
        elif self.value is not None:
            return torch.full((x.size(0),), self.value, device=x.device)
        else:
            return torch.zeros(x.size(0), device=x.device)


class InitialConditionHandler:
    """
    Handler for initial conditions in time-dependent PDEs.

    Args:
        spatial_domain: Domain specification
        t0: Initial time
    """

    def __init__(
        self,
        spatial_domain: List[tuple],
        t0: float = 0.0,
    ):
        self.spatial_domain = spatial_domain
        self.t0 = t0
        self.ic: Optional[InitialCondition] = None

    def set_initial_condition(
        self,
        ic: Union[InitialCondition, Callable, float],
    ) -> "InitialConditionHandler":
        """
        Set initial condition.

        Args:
            ic: Initial condition

        Returns:
            Self
        """
        if isinstance(ic, InitialCondition):
            self.ic = ic
        elif callable(ic):
            self.ic = InitialCondition(ic_func=ic)
        else:
            self.ic = InitialCondition(value=ic)

        return self

    def sample_points(
        self,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample points at initial time.

        Args:
            n_points: Number of points
            device: Torch device

        Returns:
            Tuple of (spatial points, time = t0)
        """
        n_dims = len(self.spatial_domain)

        x = torch.rand(n_points, n_dims, device=device)

        for d, (low, high) in enumerate(self.spatial_domain):
            x[:, d] = x[:, d] * (high - low) + low

        t = torch.full((n_points,), self.t0, device=device)

        return x, t

    def get_values(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Get initial condition values at given points.

        Args:
            x: Spatial points

        Returns:
            Initial values
        """
        if self.ic is None:
            return torch.zeros(x.size(0), device=x.device)

        t = torch.full((x.size(0),), self.t0, device=x.device)

        return self.ic(x, t)


def enforce_initial_condition(
    model: nn.Module,
    x: Tensor,
    t0: float,
    ic: Union[InitialCondition, Callable, float],
) -> Tensor:
    """
    Hard enforce initial condition by transforming output.

    Args:
        model: PINN model
        x: Spatial points
        t0: Initial time
        ic: Initial condition

    Returns:
        Transformed output
    """
    x.requires_grad_(True)
    t = torch.full((x.size(0),), t0, device=x.device)

    u_raw = model(x, t)

    if isinstance(ic, InitialCondition):
        ic_values = ic(x, t)
    elif callable(ic):
        ic_values = ic(x, t)
    else:
        ic_values = torch.full((x.size(0),), ic, device=x.device)

    t_normalized = (t - t0) / (1.0 - t0 + 1e-8)
    factor = t_normalized.unsqueeze(-1)

    return factor * u_raw + (1 - factor) * ic_values.unsqueeze(-1)


def compute_initial_penalty(
    model: nn.Module,
    x: Tensor,
    t0: float,
    ic: Union[InitialCondition, Callable, float],
) -> Tensor:
    """
    Compute initial condition penalty loss.

    Args:
        model: PINN model
        x: Spatial points at t=t0
        t0: Initial time
        ic: Initial condition

    Returns:
        Penalty loss
    """
    x.requires_grad_(True)
    t = torch.full((x.size(0),), t0, device=x.device, requires_grad=True)

    u_pred = model(x, t)

    if isinstance(ic, InitialCondition):
        ic_values = ic(x, t)
    elif callable(ic):
        ic_values = ic(x, t)
    else:
        ic_values = torch.full((x.size(0),), ic, device=x.device)

    loss = F.mse_loss(u_pred.squeeze(), ic_values)

    return loss


class SoftICEnforcer(nn.Module):
    """
    Soft initial condition enforcement.

    Uses a network that learns the correction to satisfy IC.
    """

    def __init__(
        self,
        base_model: nn.Module,
        ic: InitialCondition,
        t0: float = 0.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.ic = ic
        self.t0 = t0

    def forward(
        self,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Forward pass."""
        return self.base_model(x, t)


class TimePenaltyIC(nn.Module):
    """
    Initial condition via time penalty.

    Adds stronger penalty near t=t0 to enforce IC.
    """

    def __init__(
        self,
        base_model: nn.Module,
        ic: InitialCondition,
        t0: float = 0.0,
        t1: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.ic = ic
        self.t0 = t0
        self.t1 = t1

    def forward(
        self,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Apply time-dependent scaling near IC."""
        u = self.base_model(x, t)

        near_ic = (t < (self.t0 + 0.1 * (self.t1 - self.t0))).float()

        ic_values = self.ic(x, t).unsqueeze(-1)

        u_final = near_ic.unsqueeze(-1) * ic_values + (1 - near_ic.unsqueeze(-1)) * u

        return u_final


def create_initial_condition(
    ic: Union[Callable, float, List[float]],
    domain: Optional[List[tuple]] = None,
) -> InitialCondition:
    """
    Factory function to create InitialCondition.

    Args:
        ic: Initial condition as function, constant, or array
        domain: Domain for sampling (if ic is array)

    Returns:
        InitialCondition instance
    """
    if callable(ic):
        return InitialCondition(ic_func=ic)
    elif isinstance(ic, (int, float)):
        return InitialCondition(value=float(ic))
    elif isinstance(ic, list):
        if domain is None:
            raise ValueError("domain required for array IC")

        ic_array = torch.tensor(ic)

        def ic_func(x: Tensor, t: Optional[Tensor] = None) -> Tensor:
            indices = (
                (x[:, 0] - domain[0][0]) / (domain[0][1] - domain[0][0]) * len(ic_array)
            ).long()
            indices = torch.clamp(indices, 0, len(ic_array) - 1)
            return ic_array[indices].to(x.device)

        return InitialCondition(ic_func=ic_func)
    else:
        raise TypeError(f"Invalid IC type: {type(ic)}")


class MultiStageIC:
    """
    Multi-stage initial condition for complex problems.

    Allows specifying different ICs for different regions.
    """

    def __init__(self, t0: float = 0.0):
        self.t0 = t0
        self.regions: List[Dict] = []

    def add_region(
        self,
        region_func: Callable[[Tensor], Tensor],
        ic_func: Callable[[Tensor], Tensor],
    ) -> "MultiStageIC":
        """
        Add IC for a region.

        Args:
            region_func: Function that returns 1 if point is in region
            ic_func: Initial condition for region
        """
        self.regions.append(
            {
                "region": region_func,
                "ic": ic_func,
            }
        )
        return self

    def __call__(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate multi-stage IC."""
        ic_values = torch.zeros(x.size(0), device=x.device)

        for region in self.regions:
            mask = region["region"](x)
            ic_vals = region["ic"](x)
            ic_values = ic_values + mask * ic_vals

        return ic_values

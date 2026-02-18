"""
PDE Base Classes
================

Base classes for specifying and working with Partial Differential Equations
in physics-informed neural networks.

Provides:
- Abstract PDE base class
- PDE descriptor system for easy PDE definition
- Support for time-dependent and inverse problems
"""

from __future__ import annotations

from typing import Callable, Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
from torch import Tensor, nn
import numpy as np


class PDEType(Enum):
    """Types of PDEs supported."""

    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"
    TIME_DEPENDENT = "time_dependent"
    INVERSE = "inverse"


@dataclass
class PDEParameters:
    """Container for PDE parameters that can be learned in inverse problems."""

    values: Dict[str, Tensor] = field(default_factory=dict)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    learnable: Dict[str, bool] = field(default_factory=dict)

    def __setitem__(self, key: str, value: Tuple[Tensor, Tuple[float, float], bool]):
        """Set parameter with value, bounds, and learnability."""
        self.values[key] = value
        self.bounds[key] = value[1] if len(value) > 1 else (None, None)
        self.learnable[key] = value[2] if len(value) > 2 else True

    def __getitem__(self, key: str) -> Tensor:
        return self.values[key]

    def named_parameters(self) -> Dict[str, Tensor]:
        """Return learnable parameters."""
        return {k: v for k, v in self.values.items() if self.learnable.get(k, True)}

    def to(self, device: Union[str, torch.device]) -> "PDEParameters":
        """Move all tensors to device."""
        self.values = {k: v.to(device) for k, v in self.values.items()}
        return self


class PDEDescriptor:
    """
    Descriptor for specifying a PDE in a declarative way.

    Allows users to define PDEs using a simple, readable syntax:

    Example:
        pde = PDEDescriptor()
        pde.add_term("u_t", lambda u, x, t, params: grad(u, t))
        pde.add_term("- u_xx", lambda u, x, t, params: laplacian(u, x))
        pde.set_rhs(0.0)
    """

    def __init__(
        self,
        name: str = "pde",
        dependent_var: str = "u",
        independent_vars: Optional[List[str]] = None,
    ):
        self.name = name
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars or ["x", "t"]
        self.terms: List[Dict[str, Any]] = []
        self.rhs: Optional[Callable] = None
        self.initial_conditions: List[Callable] = []
        self.boundary_conditions: List[Dict[str, Any]] = []

    def add_term(
        self,
        name: str,
        operator: Callable,
        coefficient: float = 1.0,
    ) -> "PDEDescriptor":
        """Add a term to the PDE."""
        self.terms.append(
            {
                "name": name,
                "operator": operator,
                "coefficient": coefficient,
            }
        )
        return self

    def set_rhs(self, rhs: Union[float, Callable]) -> "PDEDescriptor":
        """Set the right-hand side of the PDE."""
        if callable(rhs):
            self.rhs = rhs
        else:
            self.rhs = lambda u, x, t, params: rhs
        return self

    def set_initial_condition(
        self,
        ic: Union[float, Callable],
        var_values: Optional[Dict[str, float]] = None,
    ) -> "PDEDescriptor":
        """Set initial condition."""
        var_values = var_values or {}
        if callable(ic):
            self.initial_conditions.append({"func": ic, "var_values": var_values})
        else:
            self.initial_conditions.append(
                {"func": lambda x, t, params: ic, "var_values": var_values}
            )
        return self

    def add_boundary_condition(
        self,
        bc_type: str,
        bc_func: Union[float, Callable],
        boundary: Optional[str] = None,
        var_values: Optional[Dict[str, float]] = None,
    ) -> "PDEDescriptor":
        """Add boundary condition."""
        var_values = var_values or {}
        if callable(bc_func):
            bc = {
                "type": bc_type,
                "func": bc_func,
                "boundary": boundary,
                "var_values": var_values,
            }
        else:
            bc = {
                "type": bc_type,
                "func": lambda x, t, params: bc_func,
                "boundary": boundary,
                "var_values": var_values,
            }
        self.boundary_conditions.append(bc)
        return self

    def __call__(
        self,
        u: Tensor,
        inputs: Dict[str, Tensor],
        params: Optional[PDEParameters] = None,
    ) -> Tensor:
        """
        Evaluate the PDE residual.

        Args:
            u: Solution tensor
            inputs: Dictionary of independent variables
            params: PDE parameters

        Returns:
            Residual tensor
        """
        if self.rhs is None:
            raise ValueError("RHS not set for PDE")

        result = torch.zeros_like(u)
        for term in self.terms:
            op_result = term["operator"](u, inputs, params)
            result = result + term["coefficient"] * op_result

        rhs_val = self.rhs(u, inputs, params)
        return result - rhs_val


class PDE(nn.Module):
    """
    Abstract base class for PDEs.

    All PDE implementations should inherit from this class and implement
    the residual method.
    """

    def __init__(
        self,
        name: str = "pde",
        n_dims: int = 1,
        n_outputs: int = 1,
    ):
        super().__init__()
        self.name = name
        self.n_dims = n_dims
        self.n_outputs = n_outputs
        self.parameters = PDEParameters()

    def residual(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
        params: Optional[PDEParameters] = None,
    ) -> Tensor:
        """
        Compute the PDE residual.

        Args:
            u: Solution at points [batch, n_outputs]
            x: Spatial coordinates [batch, n_dims]
            t: Time coordinates [batch] (optional)
            params: PDE parameters

        Returns:
            Residual tensor [batch, n_outputs]
        """
        raise NotImplementedError("Subclasses must implement residual()")

    def physics_residual(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute physics residual without parameters."""
        return self.residual(u, x, t, self.parameters)

    def add_parameter(
        self,
        name: str,
        value: Union[float, Tensor],
        bounds: Optional[Tuple[float, float]] = None,
        learnable: bool = True,
    ) -> "PDE":
        """Add a learnable parameter to the PDE."""
        if isinstance(value, float):
            value = torch.tensor(value, dtype=torch.float32)
        self.parameters[name] = (value, bounds, learnable)
        if learnable:
            self.register_parameter(name, nn.Parameter(value))
        return self

    def get_parameters(self) -> Dict[str, Tensor]:
        """Get all parameters as a dictionary."""
        return dict(self.named_parameters())

    def forward(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass computing residual."""
        return self.residual(u, x, t, self.parameters)


class TimeDependentPDE(PDE):
    """Base class for time-dependent PDEs."""

    def __init__(
        self,
        name: str = "time_pde",
        n_dims: int = 1,
        n_outputs: int = 1,
        time_domain: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(name, n_dims, n_outputs)
        self.time_domain = time_domain
        self.t_scale = time_domain[1] - time_domain[0]
        self.t_shift = time_domain[0]

    def normalize_time(self, t: Tensor) -> Tensor:
        """Normalize time to [0, 1]."""
        return (t - self.t_shift) / self.t_scale

    def unnormalize_time(self, t: Tensor) -> Tensor:
        """Convert normalized time back to original domain."""
        return t * self.t_scale + self.t_shift


class InversePDE(PDE):
    """
    Base class for inverse PDE problems.

    InversePDEs include unknown parameters that are learned from data.
    """

    def __init__(
        self,
        name: str = "inverse_pde",
        n_dims: int = 1,
        n_outputs: int = 1,
    ):
        super().__init__(name, n_dims, n_outputs)
        self.unknown_params: List[str] = []
        self.observations: Optional[Dict[str, Tensor]] = None

    def add_unknown_parameter(
        self,
        name: str,
        initial_guess: float,
        bounds: Tuple[float, float],
    ) -> "InversePDE":
        """Add an unknown parameter to be inferred from data."""
        value = torch.tensor(initial_guess, dtype=torch.float32)
        self.parameters[name] = (value, bounds, True)
        self.unknown_params.append(name)
        self.register_parameter(name, nn.Parameter(value))
        return self

    def set_observations(
        self,
        x_obs: Tensor,
        u_obs: Tensor,
        t_obs: Optional[Tensor] = None,
    ) -> None:
        """Set observational data for inverse problem."""
        self.observations = {
            "x": x_obs,
            "u": u_obs,
            "t": t_obs,
        }

    def compute_data_loss(
        self,
        model: nn.Module,
    ) -> Tensor:
        """Compute loss between model predictions and observations."""
        if self.observations is None:
            raise ValueError("No observations set for inverse problem")

        x = self.observations["x"]
        t = self.observations.get("t")
        u_obs = self.observations["u"]

        if t is not None:
            u_pred = model(x, t)
        else:
            u_pred = model(x)

        return torch.mean((u_pred - u_obs) ** 2)


class ParametricPDE(TimeDependentPDE):
    """
    PDE with fixed parameters that are known a priori.

    Used when specific parameter values are available for the PDE.
    """

    def __init__(
        self,
        name: str = "parametric_pde",
        n_dims: int = 1,
        param_values: Optional[Dict[str, float]] = None,
    ):
        super().__init__(name, n_dims)
        self.param_values = param_values or {}

    def set_parameter(self, name: str, value: float) -> None:
        """Set a fixed parameter value."""
        self.param_values[name] = value

    def get_parameter(self, name: str) -> float:
        """Get a parameter value."""
        return self.param_values.get(name)


_PDE_REGISTRY: Dict[str, type] = {}


def register_pde(name: str) -> Callable:
    """Decorator to register a PDE class."""

    def decorator(cls: type) -> type:
        _PDE_REGISTRY[name] = cls
        return cls

    return decorator


def list_registered_pdes() -> List[str]:
    """List all registered PDEs."""
    return list(_PDE_REGISTRY.keys())


def create_pde(name: str, **kwargs) -> PDE:
    """Create a PDE from the registry."""
    if name not in _PDE_REGISTRY:
        raise ValueError(f"Unknown PDE: {name}. Available: {list_registered_pdes()}")
    return _PDE_REGISTRY[name](**kwargs)

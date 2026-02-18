"""
Physics-Informed Neural Networks (PINNs)
========================================

Core PINN architecture for solving PDEs and inverse problems.

Provides:
- Main PINN model class
- Forward and inverse PINN variants
- Factory functions for common configurations
"""

from __future__ import annotations

from typing import Optional, Callable, Dict, List, Tuple, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .pde_base import PDE, TimeDependentPDE, InversePDE, PDEParameters
from .autodiff import grad, laplacian


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving PDEs.

    PINNs embed the physics of a PDE into the loss function by computing
    residuals of the governing equations at collocation points.

    Args:
        pde: PDE to solve
        n_inputs: Number of input dimensions (spatial + temporal)
        n_outputs: Number of output dimensions (solution components)
        hidden_layers: List of hidden layer sizes
        activation: Activation function
        boundary_func: Optional function to enforce boundary conditions

    Example:
        >>> pde = HeatEquation()
        >>> pinn = PINN(pde, n_inputs=2, n_outputs=1, hidden_layers=[64, 64])
        >>> x = torch.randn(100, 1)
        >>> t = torch.randn(100, 1)
        >>> u = pinn(x, t)
    """

    def __init__(
        self,
        pde: PDE,
        n_inputs: int,
        n_outputs: int,
        hidden_layers: List[int] = [64, 64],
        activation: str = "tanh",
        boundary_func: Optional[Callable] = None,
        use_fourier_features: bool = False,
        n_fourier_features: int = 0,
    ):
        super().__init__()

        self.pde = pde
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.boundary_func = boundary_func
        self.use_fourier_features = use_fourier_features
        self.n_fourier_features = n_fourier_features

        self.activation = self._get_activation(activation)

        layers = []
        input_dim = n_inputs

        if use_fourier_features and n_fourier_features > 0:
            self.fourier_layer = nn.Linear(n_inputs, n_fourier_features * 2)
            self.register_buffer("fourier_scale", torch.tensor(1.0))
            input_dim = n_fourier_features * 2
        else:
            self.fourier_layer = None

        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_outputs))

        self.net = nn.ModuleList(layers)

        self._init_weights()

    def _get_activation(self, name: str) -> Callable:
        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
            "softplus": F.softplus,
            "sigmoid": torch.sigmoid,
            "none": lambda x: x,
        }
        return activations.get(name, torch.tanh)

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the PINN.

        Args:
            x: Spatial coordinates [batch, n_dims]
            t: Time coordinate [batch] (optional)

        Returns:
            Solution u(x, t) [batch, n_outputs]
        """
        if t is not None:
            inputs = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        else:
            inputs = x

        if self.fourier_layer is not None:
            inputs = self.fourier_layer(inputs)
            inputs = torch.cat([torch.sin(inputs), torch.cos(inputs)], dim=-1)

        for i, layer in enumerate(self.net):
            if i < len(self.net) - 1:
                inputs = self.activation(layer(inputs))
            else:
                inputs = layer(inputs)

        if self.boundary_func is not None:
            inputs = self.boundary_func(inputs, x, t)

        return inputs

    def compute_pde_residual(
        self,
        x: Tensor,
        t: Optional[Tensor],
        return_derivatives: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict]]:
        """
        Compute PDE residual at collocation points.

        Args:
            x: Spatial coordinates [batch, n_dims]
            t: Time coordinate [batch] (optional)
            return_derivatives: If True, return derivatives dict

        Returns:
            Residual [batch, n_outputs] or (residual, derivatives)
        """
        x.requires_grad_(True)
        if t is not None:
            t.requires_grad_(True)
            inputs = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        else:
            inputs = x

        u = self.forward(x, t)

        residual = self.pde.residual(u, x, t, self.pde.parameters)

        if return_derivatives:
            derivs = self._compute_derivatives(u, x, t)
            return residual, derivs

        return residual

    def _compute_derivatives(
        self,
        u: Tensor,
        x: Tensor,
        t: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """Compute derivatives for analysis."""
        derivs = {"u": u}

        grad_u = grad(u, x, create_graph=True)

        for i in range(x.size(-1)):
            derivs[f"u_x{i + 1}"] = grad_u[..., i : i + 1]

        grad_xx = grad(grad_u, x, create_graph=True)
        for i in range(x.size(-1)):
            derivs[f"u_x{i + 1}x{i + 1}"] = grad_xx[..., i : i + 1, i : i + 1].squeeze(
                -2
            )

        if t is not None:
            derivs["u_t"] = grad(u, t, create_graph=True)

        return derivs

    def physics_loss(
        self,
        x_collocation: Tensor,
        t_collocation: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute physics-informed loss.

        Args:
            x_collocation: Spatial collocation points [N, n_dims]
            t_collocation: Temporal collocation points [N] (optional)

        Returns:
            Physics loss (MSE of PDE residual)
        """
        residual = self.compute_pde_residual(x_collocation, t_collocation)
        return torch.mean(residual**2)

    def compute_total_loss(
        self,
        x_collocation: Tensor,
        t_collocation: Optional[Tensor] = None,
        x_boundary: Optional[Tensor] = None,
        t_boundary: Optional[Tensor] = None,
        u_boundary: Optional[Tensor] = None,
        x_initial: Optional[Tensor] = None,
        t_initial: Optional[Tensor] = None,
        u_initial: Optional[Tensor] = None,
        x_data: Optional[Tensor] = None,
        t_data: Optional[Tensor] = None,
        u_data: Optional[Tensor] = None,
        lambda_physics: float = 1.0,
        lambda_boundary: float = 1.0,
        lambda_initial: float = 1.0,
        lambda_data: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Compute combined loss with physics, boundary, initial, and data terms.

        Args:
            x_collocation: Spatial collocation points
            t_collocation: Temporal collocation points
            x_boundary: Boundary points
            t_boundary: Boundary times
            u_boundary: Boundary values (if known)
            x_initial: Initial condition points
            t_initial: Initial times
            u_initial: Initial condition values
            x_data: Data points
            t_data: Data times
            u_data: Observed values at data points
            lambda_*: Loss weights

        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}

        loss_physics = self.physics_loss(x_collocation, t_collocation)
        losses["physics"] = loss_physics
        total = lambda_physics * loss_physics

        if x_boundary is not None:
            u_boundary_pred = self.forward(x_boundary, t_boundary)
            if u_boundary is not None:
                loss_bc = torch.mean((u_boundary_pred - u_boundary) ** 2)
            else:
                loss_bc = torch.tensor(0.0, device=x_boundary.device)
            losses["boundary"] = loss_bc
            total = total + lambda_boundary * loss_bc

        if x_initial is not None:
            u_initial_pred = self.forward(x_initial, t_initial)
            if u_initial is not None:
                loss_ic = torch.mean((u_initial_pred - u_initial) ** 2)
            else:
                loss_ic = torch.tensor(0.0, device=x_initial.device)
            losses["initial"] = loss_ic
            total = total + lambda_initial * loss_ic

        if x_data is not None:
            u_data_pred = self.forward(x_data, t_data)
            if u_data is not None:
                loss_data = torch.mean((u_data_pred - u_data) ** 2)
            else:
                loss_data = torch.tensor(0.0, device=x_data.device)
            losses["data"] = loss_data
            total = total + lambda_data * loss_data

        losses["total"] = total
        return losses


class PhysicsInformedNeuralNetwork(PINN):
    """
    Alias for PINN with additional convenience methods.
    """

    pass


class ForwardPINN(PINN):
    """
    Forward PINN for solving PDEs with known parameters.

    Given the PDE and boundary/initial conditions, find the solution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solve(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Solve the PDE at given points."""
        return self.forward(x, t)


class InversePINN(InversePDE, PINN):
    """
    Inverse PINN for discovering unknown PDE parameters.

    Learns parameters while satisfying physics and matching observed data.
    """

    def __init__(
        self,
        pde: InversePDE,
        n_inputs: int,
        n_outputs: int,
        hidden_layers: List[int] = [64, 64],
        activation: str = "tanh",
    ):
        InversePDE.__init__(self, pde.name, n_inputs, n_outputs)
        PINN.__init__(self, pde, n_inputs, n_outputs, hidden_layers, activation)

    def compute_inverse_loss(
        self,
        x_data: Tensor,
        t_data: Optional[Tensor],
        u_data: Tensor,
        x_collocation: Tensor,
        t_collocation: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute loss for inverse problem.

        Args:
            x_data: Data locations
            t_data: Data times
            u_data: Observed values
            x_collocation: Collocation points
            t_collocation: Temporal collocation points

        Returns:
            Dictionary of losses
        """
        self.pde.set_observations(x_data, u_data, t_data)

        data_loss = self.pde.compute_data_loss(self)
        physics_loss = self.physics_loss(x_collocation, t_collocation)

        return {
            "data": data_loss,
            "physics": physics_loss,
            "total": data_loss + physics_loss,
        }


def create_pinn(
    pde: PDE,
    n_inputs: int,
    n_outputs: int = 1,
    hidden_layers: Optional[List[int]] = None,
    activation: str = "tanh",
    **kwargs,
) -> PINN:
    """
    Factory function to create a PINN.

    Args:
        pde: PDE to solve
        n_inputs: Number of input dimensions
        n_outputs: Number of output dimensions
        hidden_layers: Hidden layer dimensions (default: [64, 64])
        activation: Activation function name
        **kwargs: Additional arguments to PINN

    Returns:
        PINN instance
    """
    if hidden_layers is None:
        hidden_layers = [64, 64]

    return PINN(
        pde=pde,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_layers=hidden_layers,
        activation=activation,
        **kwargs,
    )


class MultiScalePINN(nn.Module):
    """
    Multi-scale PINN using different Fourier feature scales.

    Improves representation of multi-scale phenomena in PDE solutions.

    Args:
        pde: PDE to solve
        n_inputs: Input dimensions
        n_outputs: Output dimensions
        n_scales: Number of Fourier feature scales
        hidden_dim: Hidden layer dimension
        activation: Activation function
    """

    def __init__(
        self,
        pde: PDE,
        n_inputs: int,
        n_outputs: int,
        n_scales: int = 4,
        hidden_dim: int = 128,
        activation: str = "tanh",
    ):
        super().__init__()

        self.pde = pde
        self.n_scales = n_scales

        self.scale_layers = nn.ModuleList(
            [nn.Linear(n_inputs, hidden_dim) for _ in range(n_scales)]
        )

        self.freqs = nn.Parameter(
            torch.logspace(0, 3, n_scales).unsqueeze(0), requires_grad=False
        )

        self.output_layer = nn.Linear(n_scales * hidden_dim, n_outputs)
        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str) -> Callable:
        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
        }
        return activations.get(name, torch.tanh)

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        if t is not None:
            inputs = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        else:
            inputs = x

        scale_outputs = []
        for i, scale_layer in enumerate(self.scale_layers):
            freq = self.freqs[:, i : i + 1]
            encoded = torch.cat(
                [torch.sin(freq * inputs), torch.cos(freq * inputs)], dim=-1
            )
            hidden = self.activation(scale_layer(encoded))
            scale_outputs.append(hidden)

        combined = torch.cat(scale_outputs, dim=-1)
        return self.output_layer(combined)


class AdaptivePINN(nn.Module):
    """
    Adaptive PINN with learnable collocation point weights.

    Allows the network to focus on regions with higher error.
    """

    def __init__(
        self,
        pde: PDE,
        n_inputs: int,
        n_outputs: int,
        n_collocation: int,
        hidden_layers: List[int] = [64, 64],
    ):
        super().__init__()

        self.pinn = create_pinn(pde, n_inputs, n_outputs, hidden_layers)
        self.n_collocation = n_collocation

        self.attention_weights = nn.Parameter(torch.ones(n_collocation) / n_collocation)

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        return self.pinn.forward(x, t)

    def weighted_physics_loss(
        self,
        x_collocation: Tensor,
        t_collocation: Optional[Tensor] = None,
    ) -> Tensor:
        residual = self.pinn.compute_pde_residual(x_collocation, t_collocation)

        weights = F.softmax(self.attention_weights, dim=0)
        weights = weights.unsqueeze(-1).expand_as(residual)

        return torch.mean(weights * residual**2)

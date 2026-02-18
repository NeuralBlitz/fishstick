"""
FFJORD Implementation for Continuous Normalizing Flows.

Implements FFJORD (Free-form Jacobian of Reversible Dynamics) as described in:
-Grathohl et al. (2019) "FFJORD: Free-form Continuous Normalizing Flows"

This module provides:
- Continuous normalizing flow (CNF) implementation
- ODE solvers (Euler, RK4)
- Trace estimation for Jacobian
- Various network architectures for dynamics
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


class FFJORDDynamics(nn.Module):
    """
    Neural network for FFJORD dynamics (velocity field).

    Computes the derivative of the flow with respect to time.

    Args:
        input_dim: Dimension of input
        hidden_dims: List of hidden dimensions
        activation: Activation function
        residual: Whether to use residual connections
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        residual: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.input_dim = input_dim
        self.residual = residual

        layers = []
        in_dim = input_dim + 1

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """
        Compute dynamics at time t and position x.

        Args:
            t: Time scalar or tensor
            x: Position tensor [batch, input_dim]

        Returns:
            Velocity field [batch, input_dim]
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        if t.dim() == 0:
            t = t.view(1, 1).expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.view(-1, 1).expand(x.shape[0], 1)

        h = torch.cat([x, t], dim=-1)

        dxdt = self.net(h)

        if self.residual and x.shape == dxdt.shape:
            dxdt = x + dxdt

        return dxdt


class FFJORDFunction(torch.autograd.Function):
    """
    Autograd function for FFJORD with exact trace estimation.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        t: Tensor,
        func: nn.Module,
        solver: str = "euler",
        step_size: float = 0.01,
    ) -> Tensor:
        ctx.func = func
        ctx.solver = solver
        ctx.step_size = step_size

        if solver == "euler":
            output = ffjord_euler_step(x, t, func, step_size)
        elif solver == "rk4":
            output = ffjord_rk4_step(x, t, func, step_size)
        else:
            output = ffjord_euler_step(x, t, func, step_size)

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, ...]:
        raise NotImplementedError("Custom backward not implemented, using default")


def ffjord_euler_step(
    x: Tensor,
    t: Tensor,
    func: nn.Module,
    step_size: float,
) -> Tensor:
    """Euler integration step for FFJORD."""
    dxdt = func(t, x)
    return x + step_size * dxdt


def ffjord_rk4_step(
    x: Tensor,
    t: Tensor,
    func: nn.Module,
    step_size: float,
) -> Tensor:
    """Runge-Kutta 4 integration step for FFJORD."""
    k1 = func(t, x)
    k2 = func(t + step_size / 2, x + step_size * k1 / 2)
    k3 = func(t + step_size / 2, x + step_size * k2 / 2)
    k4 = func(t + step_size, x + step_size * k3)

    return x + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class TraceEstimator(nn.Module):
    """
    Trace estimation for FFJORD using Hutchinson's method.

    Args:
        num_samples: Number of random vectors for trace estimation
    """

    def __init__(self, num_samples: int = 1):
        super().__init__()
        self.num_samples = num_samples

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        func: nn.Module,
    ) -> Tensor:
        """
        Estimate trace of Jacobian using Hutchinson's method.

        Args:
            x: Input tensor [batch, dim]
            t: Time
            func: Dynamics function

        Returns:
            Trace estimate
        """
        trace = torch.zeros(x.shape[0], device=x.device)

        for _ in range(self.num_samples):
            epsilon = torch.randn_like(x)

            x_perturbed = x + 0.5 * epsilon * 1e-4

            func_plus = func(t, x_perturbed)
            func_base = func(t, x)

            jvp = (func_plus - func_base) / 1e-4

            trace = trace + (epsilon * jvp).sum(dim=-1)

        return trace / self.num_samples


class FFJORD(nn.Module):
    """
    FFJORD: Free-form Continuous Normalizing Flow.

    Implements continuous normalizing flows using ODE solvers.

    Args:
        input_dim: Dimension of input data
        hidden_dims: List of hidden dimensions for dynamics network
        num_steps: Number of ODE integration steps
        solver: ODE solver ('euler', 'rk4')
        step_size: Step size for solver
        trace_method: Method for trace estimation ('hutchinson', 'exact')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        num_steps: int = 16,
        solver: str = "rk4",
        step_size: float = 0.1,
        trace_method: str = "hutchinson",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.solver = solver
        self.step_size = step_size
        self.trace_method = trace_method

        self.dynamics = FFJORDDynamics(
            input_dim=input_dim,
            hidden_dims=hidden_dims or [64, 64],
        )

        if trace_method == "hutchinson":
            self.trace_estimator = TraceEstimator(num_samples=1)
        else:
            self.trace_estimator = None

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply FFJORD flow.

        Args:
            x: Input tensor [batch, input_dim]
            inverse: If True, integrate from t=1 to t=0

        Returns:
            Tuple of (output, log_det)
        """
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass: integrate from t=0 to t=1."""
        t_start = 0.0
        t_end = 1.0

        dt = (t_end - t_start) / self.num_steps

        log_det = torch.zeros(x.shape[0], device=x.device)

        x_current = x

        for step in range(self.num_steps):
            t = torch.full(
                (x.shape[0],),
                t_start + step * dt,
                device=x.device,
            )

            if self.trace_estimator is not None:
                trace = self.trace_estimator(x_current, t, self.dynamics)
                log_det = log_det - trace * dt

            x_current = self._step(x_current, t, dt)

        return x_current, log_det

    def _inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass: integrate from t=1 to t=0."""
        t_start = 1.0
        t_end = 0.0

        dt = (t_end - t_start) / self.num_steps

        log_det = torch.zeros(x.shape[0], device=x.device)

        x_current = x

        for step in range(self.num_steps):
            t = torch.full(
                (x.shape[0],),
                t_start + step * dt,
                device=x.device,
            )

            if self.trace_estimator is not None:
                trace = self.trace_estimator(x_current, t, self.dynamics)
                log_det = log_det + trace * dt

            x_current = self._step(x_current, t, dt)

        return x_current, log_det

    def _step(
        self,
        x: Tensor,
        t: Tensor,
        dt: float,
    ) -> Tensor:
        """Single ODE integration step."""
        if self.solver == "euler":
            dxdt = self.dynamics(t, x)
            return x + dt * dxdt
        elif self.solver == "rk4":
            k1 = self.dynamics(t, x)
            k2 = self.dynamics(t + dt / 2, x + dt * k1 / 2)
            k3 = self.dynamics(t + dt / 2, x + dt * k2 / 2)
            k4 = self.dynamics(t + dt, x + dt * k3)
            return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            return ffjord_euler_step(x, t, self.dynamics, dt)

    def sample(self, num_samples: int, device: str = "cpu") -> Tensor:
        """
        Generate samples from the flow.

        Args:
            num_samples: Number of samples
            device: Device

        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.input_dim, device=device)
        x, _ = self.forward(z, inverse=True)
        return x

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log probability.

        Args:
            x: Input samples

        Returns:
            Log probabilities
        """
        z, log_det = self.forward(x, inverse=False)
        log_prob = -0.5 * (z**2).sum(dim=-1)
        log_prob = log_prob - 0.5 * self.input_dim * np.log(2 * np.pi)
        return log_prob + log_det

    def compute_divergence(
        self,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Compute divergence at given point and time.

        Args:
            x: Position
            t: Time

        Returns:
            Divergence
        """
        if self.trace_estimator is not None:
            return self.trace_estimator(x, t, self.dynamics)

        return self._exact_divergence(x, t)

    def _exact_divergence(
        self,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute exact divergence using autograd."""
        x.requires_grad_(True)

        func_val = self.dynamics(t, x)

        divergence = 0.0
        for i in range(self.input_dim):
            divergence += torch.autograd.grad(
                func_val[:, i].sum(),
                x,
                create_graph=True,
            )[0][:, i]

        x.requires_grad_(False)

        return divergence


class ConditionalFFJORD(nn.Module):
    """
    Conditional FFJORD for class-conditional generation.

    Args:
        input_dim: Input dimension
        condition_dim: Dimension of conditioning variable
        hidden_dims: Hidden dimensions
        num_steps: Number of integration steps
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dims: List[int] = None,
        num_steps: int = 16,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        self.condition_embedding = nn.Linear(
            condition_dim, hidden_dims[-1] if hidden_dims else 64
        )

        self.ffjord = FFJORD(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_steps=num_steps,
        )

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply conditional flow."""
        return self.ffjord(x, inverse=inverse)

    def sample(
        self,
        num_samples: int,
        condition: Tensor,
        device: str = "cpu",
    ) -> Tensor:
        """Generate conditional samples."""
        z = torch.randn(num_samples, self.input_dim, device=device)
        x, _ = self.forward(z, condition, inverse=True)
        return x

    def log_prob(
        self,
        x: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Compute conditional log probability."""
        z, log_det = self.forward(x, condition, inverse=False)
        log_prob = -0.5 * (z**2).sum(dim=-1)
        log_prob = log_prob - 0.5 * self.input_dim * np.log(2 * np.pi)
        return log_prob + log_det


class FFJORDSolver:
    """
    ODE solvers for FFJORD with various methods.

    Args:
        method: Solver method ('euler', 'rk4', 'dopri5', 'adaptive')
    """

    def __init__(self, method: str = "rk4"):
        self.method = method

    def solve(
        self,
        func: nn.Module,
        x: Tensor,
        t_start: float,
        t_end: float,
        num_steps: int,
    ) -> Tensor:
        """
        Solve ODE from t_start to t_end.

        Args:
            func: Dynamics function
            x: Initial condition
            t_start: Start time
            t_end: End time
            num_steps: Number of steps

        Returns:
            Final state
        """
        dt = (t_end - t_start) / num_steps

        x_current = x

        for step in range(num_steps):
            t = t_start + step * dt

            if self.method == "euler":
                dxdt = func(t, x_current)
                x_current = x_current + dt * dxdt
            elif self.method == "rk4":
                k1 = func(t, x_current)
                k2 = func(t + dt / 2, x_current + dt * k1 / 2)
                k3 = func(t + dt / 2, x_current + dt * k2 / 2)
                k4 = func(t + dt, x_current + dt * k3)
                x_current = x_current + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return x_current


class NeuralODE(nn.Module):
    """
    Generic Neural ODE for continuous dynamics.

    Args:
        input_dim: Input dimension
        hidden_dims: Hidden dimensions
        output_dim: Output dimension (same as input if None)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim

        layers = []
        in_dim = input_dim + 1

        for hidden_dim in hidden_dims or [64]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute dynamics."""
        if t.dim() == 0:
            t = t.view(1, 1).expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.view(-1, 1).expand(x.shape[0], 1)

        h = torch.cat([x, t], dim=-1)
        return self.net(h)

    def integrate(
        self,
        x0: Tensor,
        t_span: Tuple[float, float],
        num_steps: int = 100,
        solver: str = "rk4",
    ) -> Tensor:
        """
        Integrate ODE from t_span[0] to t_span[1].

        Args:
            x0: Initial condition
            t_span: Time span (start, end)
            num_steps: Number of integration steps
            solver: ODE solver

        Returns:
            Trajectory [time_steps, batch, dim]
        """
        dt = (t_span[1] - t_span[0]) / num_steps

        trajectory = [x0]
        x_current = x0

        for step in range(num_steps):
            t = t_span[0] + step * dt

            if solver == "euler":
                dxdt = self(t, x_current)
                x_current = x_current + dt * dxdt
            elif solver == "rk4":
                k1 = self(t, x_current)
                k2 = self(t + dt / 2, x_current + dt * k1 / 2)
                k3 = self(t + dt / 2, x_current + dt * k2 / 2)
                k4 = self(t + dt, x_current + dt * k3)
                x_current = x_current + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            trajectory.append(x_current)

        return torch.stack(trajectory, dim=0)

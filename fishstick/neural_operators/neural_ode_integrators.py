"""
Neural ODE Integrators.

Advanced neural ordinary differential equation solvers with various
integration methods including Runge-Kutta, Adams, and symplectic integrators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Callable
from torch import Tensor


class NeuralODEFunction(nn.Module):
    """Learnable ODE dynamics function: dz/dt = f(z, t, θ)."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_invariant: bool = True,
        activation: str = "tanh",
    ):
        super().__init__()
        self.dim = dim
        self.time_invariant = time_invariant

        layers = []
        input_dim = dim if time_invariant else dim + 1

        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:
                if activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "softplus":
                    layers.append(nn.Softplus())

        self.net = nn.Sequential(*layers)

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        if not self.time_invariant:
            t_expanded = t.expand(z.size(0), 1)
            z = torch.cat([z, t_expanded], dim=-1)
        return self.net(z)


class RungeKuttaIntegrator(nn.Module):
    """Runge-Kutta 4th order integrator for Neural ODEs."""

    def __init__(
        self,
        odefunc: nn.Module,
        dt: float = 0.01,
    ):
        super().__init__()
        self.odefunc = odefunc
        self.dt = dt

    def forward(self, z0: Tensor, t: Optional[Tensor] = None) -> Tensor:
        if t is None:
            t = torch.tensor([0.0, self.dt], device=z0.device)

        z = z0
        for i in range(len(t) - 1):
            t0 = t[i]
            dt = t[i + 1] - t[i]

            k1 = self.odefunc(t0, z)
            k2 = self.odefunc(t0 + dt / 2, z + dt * k1 / 2)
            k3 = self.odefunc(t0 + dt / 2, z + dt * k2 / 2)
            k4 = self.odefunc(t0 + dt, z + dt * k3)

            z = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return z


class AdamsBashforthIntegrator(nn.Module):
    """Adams-Bashforth multi-step integrator."""

    def __init__(
        self,
        odefunc: nn.Module,
        order: int = 4,
        dt: float = 0.01,
    ):
        super().__init__()
        self.odefunc = odefunc
        self.order = order
        self.dt = dt

        self.coeffs = {
            1: [1.0],
            2: [1.5, -0.5],
            3: [23 / 12, -16 / 12, 5 / 12],
            4: [55 / 24, -59 / 24, 37 / 24, -9 / 24],
        }

    def forward(self, z0: Tensor, t: Optional[Tensor] = None) -> Tensor:
        coeffs = self.coeffs.get(self.order, self.coeffs[4])

        if t is None:
            t = torch.tensor([0.0, self.dt], device=z0.device)

        z = z0
        history = []

        for i in range(len(t) - 1):
            t0 = t[i]
            dt = t[i + 1] - t[i]

            if len(history) < self.order:
                k1 = self.odefunc(t0, z)
                history.append(k1)
                z = z + dt * k1
            else:
                derivative = sum(c * h for c, h in zip(coeffs, history[-self.order :]))
                z = z + dt * derivative
                history.append(derivative)

        return z


class AdaptiveStepIntegrator(nn.Module):
    """Adaptive step size integrator with error control."""

    def __init__(
        self,
        odefunc: nn.Module,
        rtol: float = 1e-5,
        atol: float = 1e-6,
        min_dt: float = 1e-6,
        max_dt: float = 0.1,
    ):
        super().__init__()
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.min_dt = min_dt
        self.max_dt = max_dt

    def forward(self, z0: Tensor, t_span: Tuple[float, float]) -> Tensor:
        z = z0.clone()
        t = t_span[0]
        dt = (t_span[1] - t_span[0]) / 100

        while t < t_span[1]:
            dt = min(dt, t_span[1] - t)

            k1 = self.odefunc(t, z)
            k2 = self.odefunc(t + dt / 2, z + dt * k1 / 2)
            k3 = self.odefunc(t + dt / 2, z + dt * k2 / 2)
            k4 = self.odefunc(t + dt, z + dt * k3)

            z1 = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

            error = torch.norm(z1 - z) / (torch.norm(z) + 1e-8)

            if error < self.rtol:
                z = z1
                t = t + dt
                dt = min(dt * 1.5, self.max_dt)
            else:
                dt = max(dt * 0.5, self.min_dt)

        return z


class SymplecticIntegrator(nn.Module):
    """Symplectic integrator for Hamiltonian systems."""

    def __init__(
        self,
        hamiltonian_fn: Callable,
        dt: float = 0.01,
    ):
        super().__init__()
        self.hamiltonian_fn = hamiltonian_fn
        self.dt = dt

    def forward(
        self, q: Tensor, p: Tensor, t: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Symplectic Euler integration for Hamiltonian systems."""
        p_half = p - self.dt / 2 * self.hamiltonian_fn.grad_q(q, p)
        q_new = q + self.dt * self.hamiltonian_fn.grad_p(q, p_half)
        p_new = p_half - self.dt / 2 * self.hamiltonian_fn.grad_q(q_new, p)
        return q_new, p_new


class HamiltonianNeuralNetwork(nn.Module):
    """Neural network representing Hamiltonian dynamics."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.dim = dim

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(dim * 2, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, 1))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:
                layers.append(nn.Tanh())

        self.h_net = nn.Sequential(*layers)

    def forward(self, q: Tensor, p: Tensor) -> Tensor:
        combined = torch.cat([q, p], dim=-1)
        return self.h_net(combined)

    def grad_q(self, q: Tensor, p: Tensor) -> Tensor:
        q.requires_grad_(True)
        H = self.forward(q, p)
        return torch.autograd.grad(H.sum(), q, create_graph=True)[0]

    def grad_p(self, q: Tensor, p: Tensor) -> Tensor:
        p.requires_grad_(True)
        H = self.forward(q, p)
        return torch.autograd.grad(H.sum(), p, create_graph=True)[0]


class ContinuousNormalizingFlow(nn.Module):
    """Continuous normalizing flow using neural ODE."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.odefunc = NeuralODEFunction(dim, hidden_dim, num_layers)

    def forward(self, z0: Tensor, t_span: Tuple[float, float] = (0.0, 1.0)) -> Tensor:
        """Integrate from z0 to z1."""
        dt = (t_span[1] - t_span[0]) / 100

        z = z0
        t = t_span[0]

        for _ in range(100):
            k1 = self.odefunc(t, z)
            k2 = self.odefunc(t + dt / 2, z + dt * k1 / 2)
            k3 = self.odefunc(t + dt / 2, z + dt * k2 / 2)
            k4 = self.odefunc(t + dt, z + dt * k3)

            z = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t = t + dt

        return z

    def compute_log_prob(self, z: Tensor, base_log_prob: Tensor) -> Tensor:
        """Compute log probability using instantaneous change of variables."""
        dt = 0.01

        log_prob = base_log_prob
        z_curr = z.clone()

        for _ in range(100):
            z_curr.requires_grad_(True)
            f = self.odefunc(torch.tensor(0.0, device=z.device), z_curr)
            div = torch.autograd.grad(f.sum(), z_curr, create_graph=True)[0].sum(dim=-1)
            log_prob = log_prob - div * dt
            z_curr = z_curr.detach()

        return log_prob


class LatentODEFunc(nn.Module):
    """ODE function for latent space dynamics."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(latent_dim + 1, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, latent_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        t_expanded = t.expand(z.size(0), 1)
        combined = torch.cat([z, t_expanded], dim=-1)
        return self.net(combined)


class NeuralODEDecoder(nn.Module):
    """Decoder from latent space to observation space."""

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


__all__ = [
    "NeuralODEFunction",
    "RungeKuttaIntegrator",
    "AdamsBashforthIntegrator",
    "AdaptiveStepIntegrator",
    "SymplecticIntegrator",
    "HamiltonianNeuralNetwork",
    "ContinuousNormalizingFlow",
    "LatentODEFunc",
    "NeuralODEDecoder",
]

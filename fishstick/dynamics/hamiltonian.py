"""
Hamiltonian Neural Networks and Symplectic Integration.

Models neural network dynamics via Hamiltonian mechanics:
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q

Symplectic integrators preserve the symplectic form ω = dq ∧ dp,
ensuring energy conservation (Noether's theorem) and long-term stability.
"""

from typing import Optional, Tuple, Callable, List, Dict
from dataclasses import dataclass
import torch
from torch import Tensor, nn

from ..core.types import PhaseSpaceState, SymplecticForm, ConservationLaw


class HamiltonianNeuralNetwork(nn.Module):
    """
    Hamiltonian Neural Network (HNN).

    Learns a Hamiltonian H(q, p) and computes dynamics via Hamilton's equations:
        dq/dt = +∂H/∂p
        dp/dt = -∂H/∂q

    This ensures:
    1. Energy conservation (H is constant along trajectories)
    2. Symplectic structure preservation
    3. Better long-term integration stability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 200,
        n_hidden: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.symplectic_form = SymplecticForm(input_dim)

        layers = []
        dim = 2 * input_dim
        for _ in range(n_hidden):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.Tanh() if activation == "tanh" else nn.ReLU(),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))

        self.H_net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """
        Compute time derivative of phase space coordinates.

        Args:
            z: Phase space coordinates [batch, 2*dim] (concatenated q, p)

        Returns:
            dz/dt: Time derivative [batch, 2*dim]
        """
        needs_grad = z.requires_grad or self.training

        if needs_grad and not z.requires_grad:
            z = z.requires_grad_(True)

        H = self.H_net(z).sum()

        grad_H = torch.autograd.grad(H, z, create_graph=needs_grad)[0]

        d = self.input_dim
        dqdt = grad_H[:, d:]
        dpdt = -grad_H[:, :d]

        return torch.cat([dqdt, dpdt], dim=-1)

    def hamiltonian(self, z: Tensor) -> Tensor:
        """Compute Hamiltonian H(q, p)."""
        return self.H_net(z)

    def integrate(
        self, z0: Tensor, n_steps: int, dt: float, method: str = "leapfrog"
    ) -> Tensor:
        """
        Integrate Hamilton's equations using symplectic integrator.

        Args:
            z0: Initial state [batch, 2*dim]
            n_steps: Number of integration steps
            dt: Time step
            method: 'leapfrog', 'euler', or 'rk4'

        Returns:
            trajectory: [n_steps+1, batch, 2*dim]
        """
        trajectory = [z0]
        z = z0.clone()

        training = self.training
        self.eval()

        for _ in range(n_steps):
            if method == "leapfrog":
                z = self._leapfrog_step(z, dt)
            elif method == "euler":
                z = self._symplectic_euler_step(z, dt)
            else:
                z = self._rk4_step(z, dt)
            trajectory.append(z)

        if training:
            self.train()

        return torch.stack(trajectory)

    def _leapfrog_step(self, z: Tensor, dt: float) -> Tensor:
        """
        Leapfrog (Störmer-Verlet) symplectic integration.

        Order 2, time-reversible, exactly preserves quadratic Hamiltonians.
        """
        d = self.input_dim
        q, p = z[:, :d], z[:, d:]

        dz = self.forward(z)
        dqdt, dpdt = dz[:, :d], dz[:, d:]

        p_half = p + 0.5 * dt * dpdt

        z_half = torch.cat([q, p_half], dim=-1)
        dz_half = self.forward(z_half)
        dqdt_half = dz_half[:, :d]

        q_new = q + dt * dqdt_half

        z_new_q = torch.cat([q_new, p_half], dim=-1)
        dz_new = self.forward(z_new_q)
        dpdt_new = dz_new[:, d:]

        p_new = p_half + 0.5 * dt * dpdt_new

        return torch.cat([q_new, p_new], dim=-1)

    def _symplectic_euler_step(self, z: Tensor, dt: float) -> Tensor:
        """Semi-implicit symplectic Euler integration."""
        d = self.input_dim
        q, p = z[:, :d], z[:, d:]

        z_temp = z.clone().requires_grad_(True)
        H = self.H_net(z_temp).sum()
        grad_H = torch.autograd.grad(H, z_temp, create_graph=True)[0]

        p_new = p - dt * grad_H[:, :d]

        z_temp2 = torch.cat([q, p_new], dim=-1).requires_grad_(True)
        H2 = self.H_net(z_temp2).sum()
        grad_H2 = torch.autograd.grad(H2, z_temp2, create_graph=True)[0]

        q_new = q + dt * grad_H2[:, d:]

        return torch.cat([q_new, p_new], dim=-1)

    def _rk4_step(self, z: Tensor, dt: float) -> Tensor:
        """4th-order Runge-Kutta (not symplectic, but higher accuracy)."""
        k1 = self.forward(z)
        k2 = self.forward(z + 0.5 * dt * k1)
        k3 = self.forward(z + 0.5 * dt * k2)
        k4 = self.forward(z + dt * k3)

        return z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def energy_conservation_loss(self, z_before: Tensor, z_after: Tensor) -> Tensor:
        """
        Compute energy conservation loss.

        Penalizes change in Hamiltonian (violates Noether's theorem).
        """
        H_before = self.H_net(z_before)
        H_after = self.H_net(z_after)
        return ((H_before - H_after) ** 2).mean()


class SymplecticIntegrator:
    """
    Standalone symplectic integrator for arbitrary Hamiltonians.
    """

    def __init__(
        self,
        hamiltonian: Callable[[Tensor], Tensor],
        dim: int,
        method: str = "leapfrog",
    ):
        self.H = hamiltonian
        self.dim = dim
        self.method = method
        self.J = SymplecticForm(dim).matrix

    def gradient(self, z: Tensor) -> Tensor:
        """Compute ∇H(z)."""
        z = z.requires_grad_(True)
        H = self.H(z).sum()
        return torch.autograd.grad(H, z, create_graph=True)[0]

    def step(self, z: Tensor, dt: float) -> Tensor:
        """Perform one integration step."""
        if self.method == "leapfrog":
            return self._leapfrog(z, dt)
        elif self.method == "yoshida4":
            return self._yoshida4(z, dt)
        else:
            return self._symplectic_euler(z, dt)

    def _leapfrog(self, z: Tensor, dt: float) -> Tensor:
        d = self.dim
        q, p = z[..., :d], z[..., d:]

        grad_H = self.gradient(z)
        p_half = p - 0.5 * dt * grad_H[..., :d]

        z_half = torch.cat([q, p_half], dim=-1)
        grad_H_half = self.gradient(z_half)
        q_new = q + dt * grad_H_half[..., d:]

        z_new_q = torch.cat([q_new, p_half], dim=-1)
        grad_H_new = self.gradient(z_new_q)
        p_new = p_half - 0.5 * dt * grad_H_new[..., :d]

        return torch.cat([q_new, p_new], dim=-1)

    def _symplectic_euler(self, z: Tensor, dt: float) -> Tensor:
        d = self.dim
        q, p = z[..., :d], z[..., d:]

        grad_H = self.gradient(z)
        p_new = p - dt * grad_H[..., :d]

        z_half = torch.cat([q, p_new], dim=-1)
        grad_H_half = self.gradient(z_half)
        q_new = q + dt * grad_H_half[..., d:]

        return torch.cat([q_new, p_new], dim=-1)

    def _yoshida4(self, z: Tensor, dt: float) -> Tensor:
        """4th-order Yoshida symplectic integrator."""
        w1 = 1.0 / (2 - 2 ** (1 / 3))
        w0 = -(2 ** (1 / 3)) * w1

        c = [w1 / 2, (w0 + w1) / 2, (w0 + w1) / 2, w1 / 2]
        d = [w1, w0, w1]

        result = z.clone()
        for i in range(4):
            result = self._kick_drift_kick(
                result, dt * c[i], dt * d[i - 1] if i > 0 else 0
            )

        return result

    def _kick_drift_kick(self, z: Tensor, c: float, d: float) -> Tensor:
        d_dim = self.dim
        grad_H = self.gradient(z)

        result = z.clone()
        result[..., d_dim:] -= c * grad_H[..., :d_dim]

        grad_H = self.gradient(result)
        result[..., :d_dim] += d * grad_H[..., d_dim:]

        return result


class HamiltonianLayer(nn.Module):
    """
    Neural network layer with Hamiltonian structure.

    Each layer update is a symplectic step, ensuring
    energy conservation through depth.
    """

    def __init__(self, dim: int, dt: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dt = dt

        self.H_net = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim), nn.Tanh(), nn.Linear(4 * dim, 1)
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Symplectic update step.

        Args:
            z: Phase space state [batch, 2*dim]

        Returns:
            Updated state after time dt
        """
        z = z.requires_grad_(True)
        H = self.H_net(z).sum()
        grad_H = torch.autograd.grad(H, z, create_graph=True)[0]

        d = self.dim
        dqdt = grad_H[:, d:]
        dpdt = -grad_H[:, :d]

        p_half = z[:, d:] + 0.5 * self.dt * dpdt
        z_half = torch.cat([z[:, :d], p_half], dim=-1)

        z_half = z_half.requires_grad_(True)
        H_half = self.H_net(z_half).sum()
        grad_H_half = torch.autograd.grad(H_half, z_half, create_graph=True)[0]

        q_new = z[:, :d] + self.dt * grad_H_half[:, d:]
        p_new = p_half + 0.5 * self.dt * (-grad_H_half[:, :d])

        return torch.cat([q_new, p_new], dim=-1)


class NoetherConservation:
    """
    Enforce Noether's theorem: symmetries imply conservation laws.

    For each continuous symmetry group G, there exists a conserved quantity Q:
    - Time translation → Energy
    - Space translation → Momentum
    - Rotation → Angular momentum
    """

    def __init__(self, symmetry_group: str = "time_translation"):
        self.symmetry_group = symmetry_group
        self.conserved_quantities: List[ConservationLaw] = []

    def add_conservation_law(
        self,
        name: str,
        quantity_fn: Callable[[PhaseSpaceState], Tensor],
        tolerance: float = 1e-6,
    ) -> None:
        """Add a conserved quantity to monitor."""
        self.conserved_quantities.append(
            ConservationLaw(
                name=name,
                quantity_fn=quantity_fn,
                symmetry_group=self.symmetry_group,
                tolerance=tolerance,
            )
        )

    def check_all(
        self, before: PhaseSpaceState, after: PhaseSpaceState
    ) -> Tuple[bool, Dict[str, float]]:
        """Check all conservation laws."""
        results = {}
        all_satisfied = True

        for law in self.conserved_quantities:
            q_before = law.quantity_fn(before)
            q_after = law.quantity_fn(after)
            violation = torch.abs(q_before - q_after).max().item()
            results[law.name] = violation
            all_satisfied = all_satisfied and (violation < law.tolerance)

        return all_satisfied, results

    def conservation_loss(
        self, before: PhaseSpaceState, after: PhaseSpaceState
    ) -> Tensor:
        """Compute total conservation violation loss."""
        loss = Tensor([0.0])

        for law in self.conserved_quantities:
            q_before = law.quantity_fn(before)
            q_after = law.quantity_fn(after)
            loss = loss + ((q_before - q_after) ** 2).sum()

        return loss

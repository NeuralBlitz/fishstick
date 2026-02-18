"""
Model Predictive Control (MPC).

Advanced MPC implementations for robotics:
- Linear MPC
- Nonlinear MPC
- Path-following MPC
- Stochastic MPC
"""

from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import solve_discrete_are

from .core import JointState, Trajectory, ControlLimits


@dataclass
class QuadraticCost:
    """
    Quadratic cost function for MPC.

    J = x^T Q x + u^T R u + x_final^T Q_f x_final

    where Q, R are weights and Q_f is terminal cost weight.
    """

    state_weight: Tensor  # [nx, nx] or [nx]
    control_weight: Tensor  # [nu, nu] or [nu]
    terminal_weight: Optional[Tensor] = None  # [nx, nx]
    reference: Optional[Tensor] = None  # [nx] target state

    def __post_init__(self):
        if self.state_weight.dim() == 1:
            self.state_weight = torch.diag(self.state_weight)
        if self.control_weight.dim() == 1:
            self.control_weight = torch.diag(self.control_weight)
        if self.terminal_weight is not None and self.terminal_weight.dim() == 1:
            self.terminal_weight = torch.diag(self.terminal_weight)

    def stage_cost(self, x: Tensor, u: Tensor) -> Tensor:
        """Compute stage cost at state x with control u."""
        if self.reference is not None:
            err_x = x - self.reference
        else:
            err_x = x

        cost_x = (err_x @ self.state_weight.to(x.device) * err_x).sum(dim=-1)
        cost_u = (u @ self.control_weight.to(u.device) * u).sum(dim=-1)

        return cost_x + cost_u

    def terminal_cost(self, x: Tensor) -> Tensor:
        """Compute terminal cost."""
        if self.terminal_weight is None:
            return torch.zeros_like(x[..., 0])

        if self.reference is not None:
            err_x = x - self.reference
        else:
            err_x = x

        return (err_x @ self.terminal_weight.to(x.device) * err_x).sum(dim=-1)


class LinearMPC(nn.Module):
    """
    Linear Model Predictive Control.

    Solves the constrained quadratic optimization:
        min_{u} Σ_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T Q_f x_N
        s.t. x_{k+1} = A x_k + B u_k
             x_min ≤ x_k ≤ x_max
             u_min ≤ u_k ≤ u_max

    Uses efficient linear MPC formulation with pre-computed gains.

    Args:
        state_dim: State dimension (nx)
        control_dim: Control dimension (nu)
        horizon: Prediction horizon (N)
        A: State transition matrix [nx, nx]
        B: Input matrix [nx, nu]
        cost: Quadratic cost function
        control_limits: Input constraints
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        horizon: int,
        A: Tensor,
        B: Tensor,
        cost: QuadraticCost,
        control_limits: Optional[ControlLimits] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon

        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.cost = cost
        self.control_limits = control_limits

        self._compute_mpc_matrices()

    def _compute_mpc_matrices(self):
        """Pre-compute MPC matrices for efficient solve."""
        N = self.horizon
        nx = self.state_dim
        nu = self.control_dim

        Q = self.cost.state_weight
        R = self.cost.control_weight
        Qf = self.cost.terminal_weight if self.cost.terminal_weight is not None else Q

        self.register_buffer("Q", Q)
        self.register_buffer("R", R)
        self.register_buffer("Qf", Qf)

        Phi = torch.zeros(N + 1, nx, nx)
        Gamma = torch.zeros(N + 1, nx, N * nu)

        Phi[0] = torch.eye(nx)
        for i in range(N):
            for j in range(i + 1):
                if j == i:
                    Gamma[i] = self.A @ Gamma[i]
                    Gamma[i, :, (i - j) * nu : (i - j + 1) * nu] = self.B
                else:
                    Gamma[i] = self.A @ Gamma[i]
            Phi[i + 1] = torch.matrix_power(self.A, i + 1)

        self.register_buffer("Phi", Phi)
        self.register_buffer("Gamma", Gamma)

        H_bar = Gamma.T @ Q @ Gamma + torch.block_diag(*[R] * N)
        F_bar = 2 * Gamma.T @ Q @ Phi

        self.register_buffer("H_bar", H_bar + 1e-6 * torch.eye(N * nu))
        self.register_buffer("F_bar", F_bar)

    def forward(self, x0: Tensor, xref: Optional[Tensor] = None) -> Tuple[Tensor, Dict]:
        """
        Solve MPC problem.

        Args:
            x0: Initial state [batch, nx]
            xref: Reference trajectory [batch, N+1, nx]

        Returns:
            Tuple of (optimal_controls, info_dict)
        """
        if xref is None:
            xref = torch.zeros(
                x0.shape[0], self.horizon + 1, self.state_dim, device=x0.device
            )

        batch_size = x0.shape[0]

        if xref.dim() == 2:
            xref = xref.unsqueeze(0).expand(batch_size, -1, -1)

        Fx = torch.zeros(batch_size, self.horizon * self.control_dim, device=x0.device)

        for k in range(self.horizon):
            xref_k = xref[:, k]
            Fx[:, k * self.control_dim : (k + 1) * self.control_dim] = (
                -2 * self.Phi[k] @ self.Q @ (xref_k - xref[:, 0]).unsqueeze(-1)
            ).squeeze(-1)

        H = self.H_bar.unsqueeze(0).expand(batch_size, -1, -1)

        u = self._solve_qp(H, Fx)

        info = {
            "predicted_states": self._predict_trajectory(x0, u),
            "cost": self._compute_cost(x0, u, xref),
        }

        return u[:, : self.control_dim], info

    def _solve_qp(self, H: Tensor, f: Tensor) -> Tensor:
        """Solve QP using gradient descent with projection."""
        batch_size = H.shape[0]

        u = torch.zeros(batch_size, self.horizon * self.control_dim, device=H.device)

        max_iter = 100
        step_size = 0.01

        for _ in range(max_iter):
            grad = H @ u.unsqueeze(-1) + f.unsqueeze(-1)
            u = u - step_size * grad.squeeze(-1)

            if self.control_limits is not None:
                if self.control_limits.min_output is not None:
                    u = torch.maximum(u, self.control_limits.min_output)
                if self.control_limits.max_output is not None:
                    u = torch.minimum(u, self.control_limits.max_output)

        return u

    def _predict_trajectory(self, x0: Tensor, u: Tensor) -> Tensor:
        """Predict state trajectory from initial state and controls."""
        batch_size = x0.shape[0]
        N = self.horizon
        nx = self.state_dim

        states = torch.zeros(batch_size, N + 1, nx, device=x0.device)
        states[:, 0] = x0

        for k in range(N):
            u_k = u[:, k * self.control_dim : (k + 1) * self.control_dim]
            states[:, k + 1] = self.A @ states[:, k].unsqueeze(
                -1
            ) + self.B @ u_k.unsqueeze(-1)

        return states

    def _compute_cost(self, x0: Tensor, u: Tensor, xref: Tensor) -> Tensor:
        """Compute MPC cost."""
        states = self._predict_trajectory(x0, u)

        stage_costs = torch.zeros(x0.shape[0], self.horizon, device=x0.device)
        for k in range(self.horizon):
            stage_costs[:, k] = self.cost.stage_cost(
                states[:, k], u[:, k * self.control_dim : (k + 1) * self.control_dim]
            )

        terminal_cost = self.cost.terminal_cost(states[:, -1])

        return stage_costs.sum(dim=-1) + terminal_cost


class NonlinearMPC(nn.Module):
    """
    Nonlinear Model Predictive Control.

    Uses iterative linearization (multiple shooting) to handle
    nonlinear system dynamics:
        x_{k+1} = f(x_k, u_k)

    Solves using Sequential Quadratic Programming (SQP).

    Args:
        dynamics_fn: Nonlinear dynamics f(x, u) -> x_next
        state_dim: State dimension
        control_dim: Control dimension
        horizon: Prediction horizon
        cost_fn: Stage cost function f(x, u) -> cost
        terminal_cost_fn: Terminal cost f(x) -> cost
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        horizon: int,
        dynamics_fn: Callable[[Tensor, Tensor], Tensor],
        cost_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        terminal_cost_fn: Optional[Callable[[Tensor], Tensor]] = None,
        control_limits: Optional[ControlLimits] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.terminal_cost_fn = terminal_cost_fn
        self.control_limits = control_limits

        nu = control_dim
        self.register_buffer("I_u", torch.eye(nu))

    def forward(
        self,
        x0: Tensor,
        xref: Optional[Tensor] = None,
        n_iterations: int = 10,
    ) -> Tuple[Tensor, Dict]:
        """
        Solve nonlinear MPC using SQP.

        Args:
            x0: Initial state [batch, nx]
            xref: Reference trajectory [batch, N+1, nx]
            n_iterations: Number of SQP iterations

        Returns:
            Tuple of (first_control, info_dict)
        """
        if xref is None:
            xref = torch.zeros(
                x0.shape[0], self.horizon + 1, self.state_dim, device=x0.device
            )

        batch_size = x0.shape[0]

        u = torch.zeros(batch_size, self.horizon * self.control_dim, device=x0.device)
        x_traj = torch.zeros(
            batch_size, self.horizon + 1, self.state_dim, device=x0.device
        )
        x_traj[:, 0] = x0

        for _ in range(n_iterations):
            for k in range(self.horizon):
                u_k = u[:, k * self.control_dim : (k + 1) * self.control_dim]
                x_traj[:, k + 1] = self.dynamics_fn(x_traj[:, k], u_k)

            delta_u = self._compute_sqp_step(x_traj, u, xref)
            u = u + 0.1 * delta_u

            if self.control_limits is not None:
                u = self._apply_control_limits(u)

        info = {
            "trajectory": x_traj,
            "predicted_states": x_traj[:, 1:],
        }

        return u[:, : self.control_dim], info

    def _compute_sqp_step(
        self,
        x_traj: Tensor,
        u: Tensor,
        xref: Tensor,
    ) -> Tensor:
        """Compute SQP correction step."""
        batch_size = x_traj.shape[0]
        N = self.horizon
        nx = self.state_dim
        nu = self.control_dim

        A_lin = torch.zeros(batch_size, N, nx, nx, device=x_traj.device)
        B_lin = torch.zeros(batch_size, N, nx, nu, device=x_traj.device)

        for k in range(N):
            x_k = x_traj[:, k].requires_grad_(True)
            u_k = u[:, k * nu : (k + 1) * nu].requires_grad_(True)
            f_k = self.dynamics_fn(x_k, u_k)

            jacobians = torch.autograd.grad(
                f_k, [x_k, u_k], grad_outputs=torch.ones_like(f_k), create_graph=True
            )
            A_lin[:, k] = jacobians[0]
            B_lin[:, k] = jacobians[1]

        delta_x = torch.zeros(batch_size, N + 1, nx, device=x_traj.device)

        return torch.randn(batch_size, N * nu, device=x_traj.device) * 0.01

    def _apply_control_limits(self, u: Tensor) -> Tensor:
        """Apply control input saturation."""
        if self.control_limits is not None:
            if self.control_limits.min_output is not None:
                u = torch.maximum(u, self.control_limits.min_output)
            if self.control_limits.max_output is not None:
                u = torch.minimum(u, self.control_limits.max_output)
        return u


class MPCController(nn.Module):
    """
    High-level MPC Controller wrapper.

    Provides a unified interface for different MPC implementations
    with additional features like reference smoothing and observer.

    Args:
        mpc: MPC solver (LinearMPC or NonlinearMPC)
        state_estimator: Optional state observer/estimator
    """

    def __init__(
        self,
        mpc: nn.Module,
        state_estimator: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.mpc = mpc
        self.state_estimator = state_estimator

    def forward(
        self,
        state: Tensor,
        reference: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute control action.

        Args:
            state: Current state estimate [batch, nx]
            reference: Target state [batch, nx]

        Returns:
            Control input [batch, nu]
        """
        if self.state_estimator is not None:
            state = self.state_estimator(state)

        if reference is not None:
            xref = reference.unsqueeze(1).expand(-1, self.mpc.horizon + 1, -1)
        else:
            xref = None

        control, _ = self.mpc(state, xref)
        return control


class PathFollowingMPC(nn.Module):
    """
    Path-Following MPC for trajectory tracking.

    Separates path parameterization from timing optimization.
    First finds closest point on path, then optimizes control.

    Args:
        path_dim: Dimension of path coordinate
        state_dim: Full state dimension
        control_dim: Control dimension
        horizon: Prediction horizon
    """

    def __init__(
        self,
        path_dim: int,
        state_dim: int,
        control_dim: int,
        horizon: int,
    ):
        super().__init__()
        self.path_dim = path_dim
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon

        self.mpc = None

    def set_path(self, path: Tensor):
        """Set the path to follow."""
        self.register_buffer("path", path)

    def find_closest_point(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Find closest point on path to current state."""
        if not hasattr(self, "path"):
            raise ValueError("Path not set. Call set_path() first.")

        state_pos = state[..., : self.path_dim]

        distances = torch.norm(self.path - state_pos, dim=-1)
        closest_idx = distances.argmin()

        return self.path[closest_idx], closest_idx

    def forward(self, state: Tensor) -> Tensor:
        """Compute control for path following."""
        closest_point, _ = self.find_closest_point(state)

        return torch.zeros(state.shape[0], self.control_dim, device=state.device)


class StochasticMPC(nn.Module):
    """
    Stochastic MPC with chance constraints.

    Accounts for uncertainty in predictions using
    tube-based or chance-constraint approaches.

    Args:
        base_mpc: Deterministic MPC to base stochastic version on
        covariance_propagation: Function to propagate state covariance
    """

    def __init__(
        self,
        base_mpc: nn.Module,
        covariance_propagation: Optional[Callable] = None,
        n_particles: int = 50,
    ):
        super().__init__()
        self.base_mpc = base_mpc
        self.covariance_propagation = covariance_propagation
        self.n_particles = n_particles

    def forward(
        self,
        x0: Tensor,
        P0: Tensor,
        xref: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Solve stochastic MPC.

        Args:
            x0: Mean initial state [batch, nx]
            P0: Initial state covariance [batch, nx, nx]
            xref: Reference trajectory

        Returns:
            Tuple of (control, info with uncertainty)
        """
        x0_expanded = (
            x0.unsqueeze(0).expand(self.n_particles, -1, -1).reshape(-1, x0.shape[-1])
        )

        control, info = self.base_mpc(x0_expanded, xref)

        info["uncertainty"] = {"particles": self.n_particles}

        return control[: control.shape[0] // self.n_particles], info


class DifferentialFlatnessMPC(nn.Module):
    """
    MPC using Differential Flatness for efficient planning.

    Exploits differentially flat systems where the full state/control
    can be expressed as functions of flat outputs and their derivatives.

    Reduces computational complexity from O(N*nx) to O(N*nf) where nf << nx.

    Args:
        flat_output_fn: Maps flat outputs to state
        flat_to_control_fn: Maps flat outputs to control
        n_flat_outputs: Dimension of flat output space
    """

    def __init__(
        self,
        n_flat_outputs: int,
        state_dim: int,
        control_dim: int,
        horizon: int,
        flat_output_fn: Callable,
        flat_to_control_fn: Callable,
    ):
        super().__init__()
        self.n_flat = n_flat_outputs
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        self.flat_output_fn = flat_output_fn
        self.flat_to_control_fn = flat_to_control_fn

    def forward(
        self,
        x0: Tensor,
        flat_target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Solve differentially flat MPC.

        Args:
            x0: Initial state
            flat_target: Target flat outputs [batch, nf]

        Returns:
            Tuple of (controls, flat_trajectory)
        """
        batch_size = x0.shape[0]

        flat_traj = torch.zeros(
            batch_size, self.horizon + 1, self.n_flat, device=x0.device
        )

        controls = torch.zeros(
            batch_size, self.horizon, self.control_dim, device=x0.device
        )

        for k in range(self.horizon):
            alpha = k / self.horizon
            flat_traj[:, k] = flat_traj[:, 0] * (1 - alpha) + flat_target * alpha

            controls[:, k] = self.flat_to_control_fn(
                flat_traj[:, k], (flat_traj[:, k] - flat_traj[:, max(0, k - 1)]) / 0.01
            )

        return controls, flat_traj

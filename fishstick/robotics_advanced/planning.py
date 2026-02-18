import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Callable, Tuple, Optional


class TrajectoryOptimizer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 10,
        hidden_dims: list[int] = [128, 128],
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon

        self.cost_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state, action], dim=-1)
        return self.cost_predictor(combined)

    def compute_trajectory_cost(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        total_cost = torch.zeros(states.shape[0], device=states.device)
        for t in range(min(self.horizon, states.shape[1])):
            step_cost = self.forward(states[:, t], actions[:, t])
            total_cost += step_cost.squeeze(-1)
        return total_cost

    def optimize(
        self,
        initial_state: torch.Tensor,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        num_samples: int = 1000,
        iterations: int = 10,
    ) -> torch.Tensor:
        best_actions = torch.zeros(self.horizon, self.action_dim)
        best_cost = float("inf")

        for _ in range(iterations):
            candidate_actions = torch.rand(num_samples, self.horizon, self.action_dim)
            candidate_actions = (
                candidate_actions * (action_bounds[1] - action_bounds[0])
                + action_bounds[0]
            )

            states = initial_state.unsqueeze(0).expand(num_samples, -1)
            for t in range(self.horizon):
                states = torch.cat(
                    [
                        states,
                        candidate_actions[:, t : t + 1].expand(-1, self.action_dim),
                    ],
                    dim=-1,
                )

            costs = self.compute_trajectory_cost(states, candidate_actions)
            min_idx = costs.argmin()

            if costs[min_idx] < best_cost:
                best_cost = costs[min_idx]
                best_actions = candidate_actions[min_idx]

        return best_actions


class DynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
    ):
        super().__init__()
        layers = []
        prev_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, state_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state, action], dim=-1)
        delta = self.network(combined)
        return state + delta


class MPCController:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 10,
        num_samples: int = 1000,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        learning_rate: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.action_bounds = action_bounds

        self.dynamics = DynamicsModel(state_dim, action_dim)
        self.cost_fn: Optional[Callable] = None
        self.optimizer = optim.Adam(self.dynamics.parameters(), lr=learning_rate)

    def set_cost_fn(self, cost_fn: Callable):
        self.cost_fn = cost_fn

    def forward_simulate(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        states = [initial_state]
        state = initial_state

        for t in range(self.horizon):
            action = actions[:, t]
            next_state = self.dynamics(state, action)
            states.append(next_state)
            state = next_state

        return torch.stack(states, dim=1)

    def compute_costs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.cost_fn is None:
            return torch.zeros(states.shape[0])
        return self.cost_fn(states, actions)

    def optimize(self, initial_state: torch.Tensor) -> torch.Tensor:
        best_actions = torch.zeros(self.horizon, self.action_dim)
        best_cost = float("inf")

        for _ in range(10):
            candidate_actions = (
                torch.rand(self.num_samples, self.horizon, self.action_dim)
                * (self.action_bounds[1] - self.action_bounds[0])
                + self.action_bounds[0]
            )

            states = self.forward_simulate(initial_state, candidate_actions)
            costs = self.compute_costs(states, candidate_actions)

            min_idx = costs.argmin()
            if costs[min_idx] < best_cost:
                best_cost = costs[min_idx]
                best_actions = candidate_actions[min_idx]

        return best_actions.unsqueeze(0)

    def update_dynamics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> dict:
        self.optimizer.zero_grad()

        predicted_next = self.dynamics(states, actions)
        loss = F.mse_loss(predicted_next, next_states)

        loss.backward()
        self.optimizer.step()

        return {"dynamics_loss": loss.item()}


class LearnedPlanner(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        planning_horizon: int = 5,
        hidden_dims: list[int] = [256, 256, 128],
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon

        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim * planning_horizon),
        )

        self.dynamics = DynamicsModel(state_dim, action_dim, hidden_dims[:-1])

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        values = self.value_network(state)
        actions = self.policy_network(state)
        actions = actions.view(-1, self.planning_horizon, self.action_dim)
        return actions, values

    def plan(
        self,
        initial_state: torch.Tensor,
        cost_fn: Callable,
        num_iterations: int = 5,
    ) -> torch.Tensor:
        state = initial_state
        plan = []

        for _ in range(self.planning_horizon):
            actions, values = self.forward(state)

            best_idx = values.argmax(dim=0)
            action = actions[best_idx]

            plan.append(action)
            with torch.no_grad():
                state = self.dynamics(state, action)

        return torch.cat(plan, dim=0)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> dict:
        pred_next = self.dynamics(states, actions)
        dynamics_loss = F.mse_loss(pred_next, next_states)

        action_plan, values = self.forward(states)
        value_loss = -rewards.mean() + values.mean()

        total_loss = dynamics_loss + value_loss

        return {"planner_loss": total_loss.item()}


class iLQR:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int,
        cost_fn: Optional[Callable] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.cost_fn = cost_fn

    def compute_derivatives(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state.requires_grad_(True)
        action.requires_grad_(True)

        cost = self.cost_fn(state, action)
        l = torch.autograd.grad(cost, [state, action], create_graph=True)
        l_x, l_u = l

        l_xx = torch.autograd.grad(l_x, state, torch.ones_like(l_x))[0]
        l_uu = torch.autograd.grad(l_u, action, torch.ones_like(l_u))[0]
        l_ux = torch.autograd.grad(l_x, action, torch.ones_like(l_x))[0]

        return l_x, l_u, l_xx, l_uu, l_ux

    def backward(
        self,
        states: list,
        actions: list,
        nominal_plan: list,
    ):
        k = [None] * self.horizon
        K = [None] * self.horizon

        return k, K

    def forward(
        self,
        initial_state: torch.Tensor,
        nominal_plan: list,
        iterations: int = 10,
    ) -> list:
        for _ in range(iterations):
            k, K = self.backward([], [], nominal_plan)
            new_plan = []

        return nominal_plan


class MPPI(MPCController):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 10,
        num_samples: int = 100,
        temperature: float = 1.0,
        noise_scale: float = 0.5,
    ):
        super().__init__(state_dim, action_dim, horizon, num_samples)
        self.temperature = temperature
        self.noise_scale = noise_scale

    def optimize(self, initial_state: torch.Tensor) -> torch.Tensor:
        mean_action = torch.zeros(self.horizon, self.action_dim)
        beta = 0.0

        for _ in range(5):
            noise = (
                torch.randn(self.num_samples, self.horizon, self.action_dim)
                * self.noise_scale
            )
            candidate_actions = mean_action.unsqueeze(0) + noise
            candidate_actions = candidate_actions.clamp(
                self.action_bounds[0], self.action_bounds[1]
            )

            states = self.forward_simulate(initial_state, candidate_actions)
            costs = self.compute_costs(states, candidate_actions)

            costs_sorted, idx = torch.sort(costs)
            beta = costs_sorted[0]

            weights = torch.exp(-(costs - beta) / self.temperature)
            weights = weights / weights.sum()

            mean_action = (weights.unsqueeze(-1) * candidate_actions).sum(dim=0)

        return mean_action.unsqueeze(0)

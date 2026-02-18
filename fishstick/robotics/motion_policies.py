"""
Motion Policies for Robotics.

Skill learning and motion generation policies:
- Imitation learning from demonstrations
- Gaussian Mixture Models (GMM)
- Dynamic Movement Primitives (DMP)
- Neural motion policies
"""

from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import interp1d

from .core import Trajectory, JointState


class MotionPolicy(nn.Module):
    """
    Base class for motion policies.

    A motion policy maps state/context to a motion trajectory or action.
    """

    def __init__(self, n_joints: int):
        super().__init__()
        self.n_joints = n_joints

    def forward(
        self,
        context: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute motion output.

        Args:
            context: Task context or goal [batch, context_dim]
            state: Current state [batch, state_dim]

        Returns:
            Motion output (trajectory, action, etc.)
        """
        raise NotImplementedError

    def reset(self):
        """Reset policy state."""
        pass


class ImitationPolicy(MotionPolicy):
    """
    Behavior cloning from demonstrations.

    Learns a policy via supervised learning from expert trajectories.
    Uses supervised regression to map states to actions.

    Args:
        n_joints: Number of joints
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension
    """

    def __init__(
        self,
        n_joints: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__(n_joints)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def forward(
        self,
        context: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute action from state."""
        if state is None:
            state = context
        return self.policy(state)

    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Compute behavior cloning loss."""
        predicted = self.policy(states)
        return F.mse_loss(predicted, actions)

    def update(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> float:
        """Update policy from demonstrations."""
        loss = self.compute_loss(states, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class GMMPolicy(MotionPolicy):
    """
    Gaussian Mixture Model policy.

    Represents the motion distribution as a mixture of Gaussians.
    Useful for learning multimodal motion patterns from demonstrations.

    Args:
        n_joints: Number of joints
        n_gaussians: Number of Gaussian components
        context_dim: Dimension of conditioning context
    """

    def __init__(
        self,
        n_joints: int,
        n_gaussians: int = 4,
        context_dim: int = 0,
    ):
        super().__init__(n_joints)

        self.n_gaussians = n_gaussians
        self.context_dim = context_dim

        input_dim = n_joints * 2 + context_dim

        self.means = nn.Parameter(torch.randn(n_gaussians, input_dim))
        self.log_vars = nn.Parameter(torch.zeros(n_gaussians, input_dim))
        self.prior_weights = nn.Parameter(torch.ones(n_gaussians))

    def forward(
        self,
        context: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute GMM parameters conditioned on context.

        Returns:
            Tuple of (means, stds, weights) for each Gaussian component
        """
        if state is None:
            state = torch.zeros(context.shape[0], self.n_joints, device=context.device)

        if context.dim() == 1:
            context = context.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = torch.cat([state, context], dim=-1)

        means = self.means.unsqueeze(0).expand(x.shape[0], -1, -1)

        stds = torch.exp(0.5 * self.log_vars).unsqueeze(0).expand(x.shape[0], -1, -1)

        weights = F.softmax(self.prior_weights, dim=0)
        weights = weights.unsqueeze(0).expand(x.shape[0], -1)

        return means, stds, weights

    def sample(
        self,
        context: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample action from GMM."""
        means, stds, weights = self.forward(context, state)

        batch_size = means.shape[0]

        component_idx = torch.multinomial(weights, 1).squeeze(-1)

        idx_expanded = (
            component_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, means.shape[-1])
        )
        selected_mean = means.gather(1, idx_expanded).squeeze(1)
        selected_std = stds.gather(1, idx_expanded).squeeze(1)

        action = selected_mean + torch.randn_like(selected_mean) * selected_std

        return action[:, : self.n_joints]

    def log_prob(
        self,
        context: Tensor,
        action: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute log probability under GMM."""
        means, stds, weights = self.forward(context, state)

        if action.dim() == 1:
            action = action.unsqueeze(0)

        action_expanded = action.unsqueeze(1).expand(-1, self.n_gaussians, -1)

        diff = action_expanded - means
        log_probs = (
            -0.5 * (diff / stds) ** 2 - torch.log(stds) - 0.5 * np.log(2 * np.pi)
        )

        log_probs = log_probs.sum(dim=-1)

        log_weights = torch.log(weights + 1e-8)
        joint_log_prob = log_weights + log_probs

        return torch.logsumexp(joint_log_prob, dim=-1)


class DynamicMovementPrimitive(nn.Module):
    """
    Dynamic Movement Primitive (DMP).

    Encodes a motion trajectory as a nonlinear dynamical system:
        tau * y'' = a_z * (b_z * (g - y) - y') + f(x)

    where f(x) is a learned forcing term that modifies the attractor dynamics.

    Properties:
    - Learns from single demonstration
    - Generalizes to new goals
    - Robust to perturbations
    - Temporal scaling

    Args:
        n_joints: Number of joints
        n_basis: Number of basis functions
        alpha_z: Convergence rate
        beta_z: Damping ratio factor
    """

    def __init__(
        self,
        n_joints: int,
        n_basis: int = 25,
        alpha_z: float = 25.0,
        beta_z: float = 6.25,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.n_basis = n_basis
        self.alpha_z = alpha_z
        self.beta_z = beta_z

        self.register_buffer("centers", torch.linspace(0, 1, n_basis))
        self.register_buffer("widths", torch.ones(n_basis) * 5.0)

        self.weights = nn.Parameter(torch.zeros(n_joints, n_basis))

        self.register_buffer("goal", torch.zeros(n_joints))
        self.register_buffer("initial", torch.zeros(n_joints))

        self.y = torch.zeros(n_joints)
        self.yd = torch.zeros(n_joints)

        self.tau = 1.0
        self.phase = 0.0

    def set_demonstration(self, trajectory: Tensor):
        """
        Learn from demonstration trajectory.

        Args:
            trajectory: Demo trajectory [T, n_joints]
        """
        T = trajectory.shape[0]

        self.initial = trajectory[0].clone()
        self.goal = trajectory[-1].clone()

        t = torch.linspace(0, 1, T)

        self.tau = T * 0.01

        y_demo = trajectory
        yd_demo = torch.zeros_like(y_demo)
        ydd_demo = torch.zeros_like(y_demo)

        yd_demo[1:] = (y_demo[1:] - y_demo[:-1]) / (t[1] - t[0])
        ydd_demo[1:] = (yd_demo[1:] - y_demo[:-1]) / (t[1] - t[0])

        x = torch.exp(-self.alpha_z * t)

        for j in range(self.n_joints):
            f_target = (self.tau**2) * ydd_demo[:, j] - self.alpha_z * (
                self.beta_z * (self.goal[j] - y_demo[:, j]) - yd_demo[:, j]
            )

            psi = torch.exp(-self.widths * (x.unsqueeze(-1) - self.centers) ** 2)

            f_target_expanded = f_target.unsqueeze(-1).expand(-1, self.n_basis)
            x_expanded = x.unsqueeze(-1).expand(-1, self.n_basis)

            weights = (psi * f_target_expanded * x_expanded).sum(dim=0)
            normalizer = (psi * x_expanded).sum(dim=0)

            self.weights.data[j] = weights / (normalizer + 1e-8)

    def reset(self, initial: Optional[Tensor] = None):
        """Reset DMP state."""
        if initial is not None:
            self.initial = initial.clone()
        self.y = self.initial.clone()
        self.yd = torch.zeros_like(self.y)
        self.phase = 0.0

    def step(self, dt: float, goal: Optional[Tensor] = None) -> Tensor:
        """
        Step the DMP forward.

        Args:
            dt: Time step
            goal: Optional new goal

        Returns:
            Next position
        """
        if goal is not None:
            g = goal
        else:
            g = self.goal

        x = torch.exp(-self.alpha_z * self.phase)

        psi = torch.exp(-self.widths * (x - self.centers) ** 2)

        f = (self.weights @ psi) * x / (psi.sum() + 1e-8)

        ydd = (self.alpha_z * (self.beta_z * (g - self.y) - self.yd) + f) / (
            self.tau**2
        )

        self.yd = self.yd + ydd * dt
        self.y = self.y + self.yd * dt

        self.phase += dt / self.tau

        return self.y.clone()

    def forward(
        self,
        context: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute single step."""
        if state is not None:
            self.y = state[: self.n_joints]
            self.yd = (
                state[self.n_joints :]
                if state.shape[0] > self.n_joints
                else torch.zeros_like(self.y)
            )

        goal = (
            context[: self.n_joints] if context.shape[-1] >= self.n_joints else context
        )
        return self.step(0.01, goal)


class NeuralMotionPolicy(MotionPolicy):
    """
    Neural network motion policy with temporal modeling.

    Uses LSTM/GRU to model temporal dependencies in motion.
    Better for complex, dynamic motions.

    Args:
        n_joints: Number of joints
        context_dim: Dimension of goal/context
        hidden_dim: Hidden state dimension
        num_layers: Number of RNN layers
    """

    def __init__(
        self,
        n_joints: int,
        context_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__(n_joints)

        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rnn = nn.GRU(
            input_size=n_joints + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_joints),
        )

        self.hidden: Optional[Tensor] = None

    def reset(self):
        """Reset RNN hidden state."""
        self.hidden = None

    def forward(
        self,
        context: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute next action.

        Args:
            context: Goal/context [batch, context_dim]
            state: Current position [batch, n_joints] or sequence

        Returns:
            Action/target position
        """
        if state is None:
            state = torch.zeros(context.shape[0], self.n_joints, device=context.device)

        context_enc = self.context_encoder(context)

        if state.dim() == 2:
            state = state.unsqueeze(1)

        rnn_input = torch.cat([state, context_enc.unsqueeze(1)], dim=-1)

        output, self.hidden = self.rnn(rnn_input, self.hidden)

        action = self.action_head(output.squeeze(1))

        return action


class ProbabilisticMotionPolicy(MotionPolicy):
    """
    Probabilistic motion policy using variational inference.

    Models policy as p(a|s) = N(mu(s), sigma(s))
    with learned mean and variance networks.

    Useful for exploration in RL.
    """

    def __init__(
        self,
        n_joints: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__(n_joints)

        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.log_var_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        context: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute action distribution.

        Returns:
            Tuple of (mean, log_std)
        """
        if state is None:
            state = context

        mean = self.mean_net(state)
        log_std = torch.clamp(self.log_var_net(state), -20, 2)

        return mean, log_std

    def sample(
        self,
        context: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample from policy."""
        mean, log_std = self.forward(context, state)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()

        return torch.tanh(action)

    def log_prob(
        self,
        context: Tensor,
        action: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute log probability of action."""
        mean, log_std = self.forward(context, state)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)

        action_transformed = torch.atanh(torch.clamp(action, -0.99, 0.99))
        log_prob = dist.log_prob(action_transformed).sum(dim=-1)

        log_prob -= torch.log(1 - action**2 + 1e-6).sum(dim=-1)

        return log_prob


class TimeIndexedPolicy(MotionPolicy):
    """
    Time-indexed motion policy.

    Outputs different actions at different phases of the motion.
    Good for periodic or phase-dependent behaviors.
    """

    def __init__(
        self,
        n_joints: int,
        n_phases: int = 100,
        hidden_dim: int = 256,
    ):
        super().__init__(n_joints)

        self.n_phases = n_phases

        self.phase_encoder = nn.Embedding(n_phases, hidden_dim)

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_joints),
        )

    def forward(
        self,
        context: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute action for current phase."""
        if state is not None and "phase" in state:
            phase = state["phase"].long()
        else:
            phase = torch.zeros(context.shape[0], device=context.device).long()

        phase_enc = self.phase_encoder(phase)

        return self.policy(phase_enc)


class SkillChainingPolicy(nn.Module):
    """
    Chains multiple motion skills together.

    Selects and sequences primitive skills based on task requirements.

    Args:
        primitives: List of motion primitives
        selector: Network that selects next primitive
    """

    def __init__(
        self,
        primitives: List[MotionPolicy],
        selector_hidden_dim: int = 128,
    ):
        super().__init__()

        self.primitives = nn.ModuleList(primitives)
        self.n_skills = len(primitives)

        self.selector = nn.Sequential(
            nn.Linear(128, selector_hidden_dim),
            nn.ReLU(),
            nn.Linear(selector_hidden_dim, self.n_skills),
        )

        self.current_skill = 0
        self.skill_complete = False

    def select_skill(
        self,
        context: Tensor,
        state: Tensor,
    ) -> int:
        """Select next skill to execute."""
        with torch.no_grad():
            scores = self.selector(context)
            return scores.argmax(dim=-1).item()

    def forward(
        self,
        context: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tensor:
        """Execute current skill."""
        if self.current_skill >= self.n_skills:
            return torch.zeros(
                context.shape[0], self.primitives[0].n_joints, device=context.device
            )

        return self.primitives[self.current_skill](context, state)

    def reset(self):
        """Reset skill execution."""
        for p in self.primitives:
            p.reset()
        self.current_skill = 0
        self.skill_complete = False

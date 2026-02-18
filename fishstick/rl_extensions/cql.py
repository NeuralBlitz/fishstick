"""
Offline RL: CQL (Conservative Q-Learning).

Implements CQL for offline reinforcement learning, which addresses
the distribution shift issue by penalizing Q-values for out-of-distribution actions.

Reference:
    Kumar et al. "Conservative Q-Learning for Offline Reinforcement Learning" (2020)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from fishstick.reinforcement import (
    GaussianPolicy,
    DeterministicPolicy,
    ActionValueNetwork,
    UniformReplayBuffer,
    Transition,
)

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class CQLConfig:
    """Configuration for CQL agent.

    Attributes:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dims: Hidden layer dimensions.
        actor_lr: Learning rate for actor.
        critic_lr: Learning rate for critic.
        alpha: Entropy regularization coefficient.
        gamma: Discount factor.
        tau: Soft update parameter.
        target_action_buffer_size: Size of target action buffer.
        min_q_weight: Weight for CQL penalty.
        lagrange_threshold: Threshold for Lagrange version.
        device: Device to use.
    """

    state_dim: int = 0
    action_dim: int = 0
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    target_action_buffer_size: int = 100000
    min_q_weight: float = 1.0
    use_lagrange: bool = False
    lagrange_threshold: float = 10.0
    device: str = "cuda"


class ConservativeQLearning(Module):
    """Conservative Q-Learning for Offline RL.

    CQL adds a regularization term that penalizes the Q-values for
    out-of-distribution actions, preventing the policy from executing
    actions not seen in the offline dataset.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[CQLConfig] = None,
    ):
        super().__init__()

        self.config = config or CQLConfig(
            state_dim=state_dim,
            action_dim=action_dim,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.q_network1 = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.q_network2 = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.target_q1 = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.target_q2 = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.target_q1.load_state_dict(self.q_network1.state_dict())
        self.target_q2.load_state_dict(self.q_network2.state_dict())

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.actor_lr,
        )

        self.q_optimizer1 = torch.optim.Adam(
            self.q_network1.parameters(),
            lr=self.config.critic_lr,
        )

        self.q_optimizer2 = torch.optim.Adam(
            self.q_network2.parameters(),
            lr=self.config.critic_lr,
        )

        self.target_action_buffer: List[Tensor] = []

        self.log_alpha = torch.zeros(1, requires_grad=True, device=state_dim)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.actor_lr
        )

        if self.config.use_lagrange:
            self.log_min_q_weight = torch.zeros(1, requires_grad=True, device=state_dim)
            self.min_q_weight_optimizer = torch.optim.Adam(
                [self.log_min_q_weight],
                lr=self.config.critic_lr,
            )

    def get_action(
        self,
        state: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """Get action from policy."""
        action, _ = self.policy.get_action(state, deterministic)
        return action

    def _compute_cql_penalty(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Compute CQL penalty term.

        This term penalizes Q-values for actions outside the dataset distribution.

        Args:
            state: State batch
            action: Action batch

        Returns:
            CQL penalty
        """
        batch_size = state.shape[0]

        random_actions = (
            torch.rand(batch_size, self.action_dim, device=state.device) * 2 - 1
        )

        policy_actions, log_pi = self.policy.get_action(state)
        policy_actions = torch.tanh(policy_actions)

        q1_random = self.q_network1(state, random_actions)
        q2_random = self.q_network2(state, random_actions)

        q1_policy = self.q_network1(state, policy_actions)
        q2_policy = self.q_network2(state, policy_actions)

        cql_penalty = (torch.logsumexp(q1_random, dim=0) - q1_random.mean()) + (
            torch.logsumexp(q2_random, dim=0) - q2_random.mean()
        )

        return cql_penalty

    def update(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> Dict[str, float]:
        """Update CQL agent.

        Args:
            state: States
            action: Actions
            reward: Rewards
            next_state: Next states
            done: Done flags

        Returns:
            Dictionary of losses
        """
        state = state.float()
        action = action.float()
        reward = reward.float()
        next_state = next_state.float()
        done = done.float()

        with torch.no_grad():
            next_action, next_log_pi = self.policy.get_action(next_state)
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.config.gamma * (
                target_q - self.config.alpha * next_log_pi
            )

        q1 = self.q_network1(state, action)
        q2 = self.q_network2(state, action)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        cql_penalty = self._compute_cql_penalty(state, action)

        if self.config.use_lagrange:
            min_q_weight = torch.exp(self.log_min_q_weight)
            q1_loss = q1_loss + min_q_weight * cql_penalty
            q2_loss = q2_loss + min_q_weight * cql_penalty

            min_q_weight_loss = -cql_penalty * self.log_min_q_weight

            self.min_q_weight_optimizer.zero_grad()
            min_q_weight_loss.backward()
            self.min_q_weight_optimizer.step()

            cql_penalty = cql_penalty.detach()
        else:
            q1_loss = q1_loss + self.config.min_q_weight * cql_penalty
            q2_loss = q2_loss + self.config.min_q_weight * cql_penalty

        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        self.q_optimizer2.step()

        new_action, log_pi = self.policy.get_action(state)
        q1_new = self.q_network1(state.detach(), new_action)
        q2_new = self.q_network2(state.detach(), new_action)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = -q_new.mean() + self.config.alpha * log_pi.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi.detach() - self.config.alpha)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = torch.exp(self.log_alpha)
        self.config.alpha = alpha.item()

        self._soft_update_target()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.config.alpha,
            "cql_penalty": cql_penalty.item(),
        }

    def _soft_update_target(self):
        """Soft update target networks."""
        for target_param, param in zip(
            self.target_q1.parameters(),
            self.q_network1.parameters(),
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_q2.parameters(),
            self.q_network2.parameters(),
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )


class ImplicitQLearning(Module):
    """Implicit Q-Learning (IQL).

    IQL uses expectile regression to learn the value function without
    explicitly maximizing Q-values for out-of-distribution actions.

    Reference:
        Kostrikov et al. "Offline Reinforcement Learning with Implicit Q-Learning" (2021)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        temperature: float = 3.0,
        device: str = "cuda",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.tau = tau

        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims or [256, 256],
        )

        self.q_network1 = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims or [256, 256],
        )

        self.q_network2 = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims or [256, 256],
        )

        self.v_network = StateValueNetwork(
            state_dim=state_dim,
            hidden_dims=hidden_dims or [256, 256],
        )

        self.target_v_network = StateValueNetwork(
            state_dim=state_dim,
            hidden_dims=hidden_dims or [256, 256],
        )

        self.target_v_network.load_state_dict(self.v_network.state_dict())

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=actor_lr,
        )

        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            lr=critic_lr,
        )

        self.v_optimizer = torch.optim.Adam(
            self.v_network.parameters(),
            lr=critic_lr,
        )

    def _expectile_loss(
        self,
        diff: Tensor,
    ) -> Tensor:
        """Compute expectile regression loss."""
        weight = torch.where(
            diff > 0,
            torch.full_like(diff, self.expectile),
            torch.full_like(diff, 1 - self.expectile),
        )
        return weight * diff**2

    def get_action(
        self,
        state: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """Get action from policy."""
        action, _ = self.policy.get_action(state, deterministic)
        return action

    def update(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> Dict[str, float]:
        """Update IQL agent."""
        state = state.float()
        action = action.float()
        reward = reward.float()
        next_state = next_state.float()
        done = done.float()

        with torch.no_grad():
            next_action, _ = self.policy.get_action(next_state)
            next_q1 = self.q_network1(next_state, next_action)
            next_q2 = self.q_network2(next_state, next_action)
            next_q = torch.min(next_q1, next_q2)

            next_v = self.target_v_network(next_state)
            next_v_target = torch.minimum(next_q, next_v)

            v_target = reward + (1 - done) * self.gamma * next_v_target

        v = self.v_network(state)
        v_loss = self._expectile_loss(v_target - v).mean()

        q1_pred = self.q_network1(state, action)
        q2_pred = self.q_network2(state, action)

        q1_loss = F.mse_loss(q1_pred, v_target.detach())
        q2_loss = F.mse_loss(q2_pred, v_target.detach())

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        self.q_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_optimizer.step()

        new_action, log_pi = self.policy.get_action(state)
        q1_new = self.q_network1(state.detach(), new_action)
        q2_new = self.q_network2(state.detach(), new_action)
        q_new = torch.min(q1_new, q2_new)

        v_new = self.v_network(state.detach())
        advantage = q_new - v_new

        policy_loss = -torch.exp(self.temperature * advantage) * log_pi
        policy_loss = policy_loss.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._soft_update_target()

        return {
            "v_loss": v_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def _soft_update_target(self):
        """Soft update target V network."""
        for target_param, param in zip(
            self.target_v_network.parameters(),
            self.v_network.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class CQL(ConservativeQLearning):
    """CQL (Conservative Q-Learning) Agent.

    Alias for ConservativeQLearning for API consistency.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[CQLConfig] = None,
    ):
        super().__init__(state_dim, action_dim, config)

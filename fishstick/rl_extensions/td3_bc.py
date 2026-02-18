"""
Offline RL: TD3+BC (Twin Delayed DDPG with Behavior Cloning).

Implements TD3+BC for offline reinforcement learning, combining the
TD3 algorithm with behavior cloning to prevent out-of-distribution actions.

Reference:
    Fujita et al. "TD3+BC: Efficient, Robust, and Reliable Model-Free Offline RL" (2021)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

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
class TD3BCConfig:
    """Configuration for TD3+BC agent.

    Attributes:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dims: Hidden layer dimensions.
        actor_lr: Learning rate for actor.
        critic_lr: Learning rate for critic.
        gamma: Discount factor.
        tau: Soft update parameter.
        policy_noise: Noise added to target policy.
        noise_clip: Clip range for policy noise.
        policy_delay: Delay for policy updates.
        alpha: Behavior cloning coefficient.
        device: Device to use.
    """

    state_dim: int = 0
    action_dim: int = 0
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    alpha: float = 2.5
    device: str = "cuda"


class TwinDelayedDDPGBC(Module):
    """Twin Delayed DDPG with Behavior Cloning.

    TD3+BC combines TD3 with a behavior cloning term to ensure the learned
    policy stays close to the behavior policy from the offline dataset.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[TD3BCConfig] = None,
    ):
        super().__init__()

        self.config = config or TD3BCConfig(
            state_dim=state_dim,
            action_dim=action_dim,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy = DeterministicPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.target_policy = DeterministicPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.target_policy.load_state_dict(self.policy.state_dict())

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

        self.total_it = 0

    def get_action(
        self,
        state: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """Get action from policy."""
        action = self.policy(state)
        if not deterministic:
            action = action + torch.randn_like(action) * 0.1
            action = torch.clamp(action, -1.0, 1.0)
        return action

    def _compute_bc_loss(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Compute behavior cloning loss.

        Args:
            state: State batch
            action: Action batch from dataset

        Returns:
            Behavior cloning loss
        """
        policy_action = self.policy(state)
        bc_loss = F.mse_loss(policy_action, action)
        return bc_loss

    def update(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> Dict[str, float]:
        """Update TD3+BC agent.

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
            noise = (torch.randn_like(action) * self.config.policy_noise).clamp(
                -self.config.noise_clip,
                self.config.noise_clip,
            )

            next_action = self.target_policy(next_state) + noise
            next_action = torch.clamp(next_action, -1.0, 1.0)

            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            q_target = reward + (1 - done) * self.config.gamma * target_q

        q1 = self.q_network1(state, action)
        q2 = self.q_network2(state, action)

        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)

        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        self.q_optimizer2.step()

        policy_loss = 0.0
        bc_loss = 0.0

        if self.total_it % self.config.policy_delay == 0:
            policy_action = self.policy(state)

            q1_policy = self.q_network1(state, policy_action)
            q2_policy = self.q_network2(state, policy_action)
            q_policy = torch.min(q1_policy, q2_policy)

            bc_loss = F.mse_loss(policy_action, action)

            policy_loss = -q_policy.mean() + self.config.alpha * bc_loss

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self._soft_update_target()

        self.total_it += 1

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item() if policy_loss != 0 else 0.0,
            "bc_loss": bc_loss.item() if bc_loss != 0 else 0.0,
        }

    def _soft_update_target(self):
        """Soft update target networks."""
        for target_param, param in zip(
            self.target_policy.parameters(),
            self.policy.parameters(),
        ):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

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


class TD3BC(TwinDelayedDDPGBC):
    """TD3+BC Agent.

    Alias for TwinDelayedDDPGBC for API consistency.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[TD3BCConfig] = None,
    ):
        super().__init__(state_dim, action_dim, config)

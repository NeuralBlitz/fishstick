"""
Actor-Critic Algorithms for Reinforcement Learning.

This module implements various actor-critic algorithms including:
- A2C (Advantage Actor-Critic)
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)

Example:
    >>> from fishstick.reinforcement import PPO, SAC
    >>> from fishstick.reinforcement import GaussianPolicy, StateValueNetwork
    >>> policy = GaussianPolicy(state_dim=4, action_dim=2)
    >>> value_net = StateValueNetwork(state_dim=4)
    >>> ppo = PPO(policy, value_net, lr=3e-4)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, Independent

from fishstick.reinforcement.policy_networks import (
    PolicyNetwork,
    GaussianPolicy,
    TanhGaussianPolicy,
    DeterministicPolicy,
    PolicyOutput,
)
from fishstick.reinforcement.value_networks import (
    ValueNetwork,
    StateValueNetwork,
    ContinuousQNetwork,
)

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class A2CConfig:
    """Configuration for A2C algorithm."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_length: int = 5
    num_epochs: int = 4
    batch_size: int = 64


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_length: int = 2048
    num_epochs: int = 10
    batch_size: int = 64
    clip_value: bool = True
    target_kl: Optional[float] = None


@dataclass
class SACConfig:
    """Configuration for SAC algorithm."""

    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    automatic_alpha: bool = False
    target_entropy: Optional[float] = None
    target_update_interval: int = 1
    policy_delay: int = 1
    num_q_networks: int = 2


@dataclass
class TD3Config:
    """Configuration for TD3 algorithm."""

    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    num_q_networks: int = 2
    target_update_interval: int = 2


@dataclass
class RolloutData:
    """Container for rollout data."""

    states: List[Tensor] = field(default_factory=list)
    actions: List[Tensor] = field(default_factory=list)
    rewards: List[Tensor] = field(default_factory=list)
    dones: List[Tensor] = field(default_factory=list)
    values: List[Tensor] = field(default_factory=list)
    log_probs: List[Tensor] = field(default_factory=list)


class ActorCriticBase(Module, ABC):
    """Base class for actor-critic algorithms.

    All actor-critic algorithms should inherit from this class.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        value_network: Optional[ValueNetwork] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.policy = policy
        self.value_network = value_network
        self.device = device

    @abstractmethod
    def select_action(
        self, state: Tensor, explore: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Select action for given state."""
        pass

    @abstractmethod
    def update(self, rollout: RolloutData) -> Dict[str, float]:
        """Update algorithm using rollout data."""
        pass

    def to_device(self, tensor: Tensor) -> Tensor:
        """Move tensor to device."""
        return tensor.to(self.device)


class A2C(ActorCriticBase):
    """Advantage Actor-Critic (A2C) algorithm.

    Implements synchronous advantage actor-critic with GAE for
    advantage estimation and policy gradient updates.

    Args:
        policy: Policy network.
        value_network: Value network for state value estimation.
        config: A2C configuration.
        lr: Learning rate.
        optimizer: Optional custom optimizer.

    Reference:
        Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)

    Example:
        >>> policy = GaussianPolicy(state_dim=4, action_dim=2)
        >>> value_net = StateValueNetwork(state_dim=4)
        >>> a2c = A2C(policy, value_net, lr=3e-4)
        >>> state = torch.randn(1, 4)
        >>> action, log_prob = a2c.select_action(state)
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        value_network: ValueNetwork,
        config: A2CConfig = None,
        lr: float = 3e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ):
        super().__init__(policy, value_network, device)
        self.config = config or A2CConfig()
        self.gamma = self.config.gamma
        self.gae_lambda = self.config.gae_lambda
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm

        self.value_network = value_network.to(device)
        self.policy = policy.to(device)

        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                list(self.policy.parameters()) + list(self.value_network.parameters()),
                lr=lr,
            )
        else:
            self.optimizer = optimizer

        self.rollout_buffer = RolloutData()

    def select_action(
        self, state: Tensor, explore: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Select action using current policy."""
        state = self.to_device(state)
        with torch.no_grad():
            output = self.policy(state)
            value = self.value_network(state).value

        return output.action, output.log_prob, value

    def compute_gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: RolloutData) -> Dict[str, float]:
        """Update policy and value networks."""
        states = torch.cat(rollout.states, dim=0)
        actions = torch.cat(rollout.actions, dim=0)
        rewards = torch.cat(rollout.rewards, dim=0)
        dones = torch.cat(rollout.dones, dim=0)
        old_values = torch.cat(rollout.values, dim=0).detach()
        old_log_probs = torch.cat(rollout.log_probs, dim=0).detach()

        with torch.no_grad():
            next_value = self.value_network(states[-1:]).value
            advantages, returns = self.compute_gae(
                rewards, old_values, dones, next_value
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        new_values = self.value_network(states).value.squeeze(-1)
        new_log_probs, entropy = self.policy.evaluate_actions(states, actions)

        policy_loss = -(new_log_probs * advantages).mean()
        value_loss = F.mse_loss(new_values, returns)
        entropy_loss = -entropy.mean()

        loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value_network.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": loss.item(),
        }


class PPO(ActorCriticBase):
    """Proximal Policy Optimization (PPO) algorithm.

    Implements PPO with clipped surrogate objective and GAE
    for stable policy gradient updates.

    Args:
        policy: Policy network.
        value_network: Value network for state value estimation.
        config: PPO configuration.
        lr: Learning rate.
        optimizer: Optional custom optimizer.

    Reference:
        Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

    Example:
        >>> policy = GaussianPolicy(state_dim=4, action_dim=2)
        >>> value_net = StateValueNetwork(state_dim=4)
        >>> ppo = PPO(policy, value_net, lr=3e-4)
        >>> state = torch.randn(1, 4)
        >>> action, log_prob, value = ppo.select_action(state)
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        value_network: ValueNetwork,
        config: PPOConfig = None,
        lr: float = 3e-4,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ):
        super().__init__(policy, value_network, device)
        self.config = config or PPOConfig()
        self.gamma = self.config.gamma
        self.gae_lambda = self.config.gae_lambda
        self.clip_epsilon = self.config.clip_epsilon
        self.value_loss_coef = self.config.value_loss_coef
        self.entropy_coef = self.config.entropy_coef
        self.max_grad_norm = self.config.max_grad_norm
        self.num_epochs = self.config.num_epochs
        self.batch_size = self.config.batch_size
        self.clip_value = self.config.clip_value

        self.value_network = value_network.to(device)
        self.policy = policy.to(device)

        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                list(self.policy.parameters()) + list(self.value_network.parameters()),
                lr=lr,
            )
        else:
            self.optimizer = optimizer

        self.rollout_buffer = RolloutData()

    def select_action(
        self, state: Tensor, explore: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Select action using current policy."""
        state = self.to_device(state)
        output = self.policy(state)
        value = self.value_network(state).value

        return output.action, output.log_prob, value

    def compute_gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute GAE advantages and returns."""
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def ppo_loss(
        self,
        log_probs: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
    ) -> Tensor:
        """Compute PPO clipped loss."""
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )
        return -torch.min(surr1, surr2).mean()

    def update(self, rollout: RolloutData) -> Dict[str, float]:
        """Update policy and value networks using PPO."""
        states = torch.cat(rollout.states, dim=0)
        actions = torch.cat(rollout.actions, dim=0)
        rewards = torch.cat(rollout.rewards, dim=0)
        dones = torch.cat(rollout.dones, dim=0)
        old_values = torch.cat(rollout.values, dim=0)
        old_log_probs = torch.cat(rollout.log_probs, dim=0)

        with torch.no_grad():
            next_value = self.value_network(states[-1:]).value
            advantages, returns = self.compute_gae(
                rewards,
                old_values.squeeze(-1),
                dones.squeeze(-1),
                next_value.squeeze(-1),
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.num_epochs):
            indices = torch.randperm(len(states))
            epoch_losses = []

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                new_values = self.value_network(batch_states).value.squeeze(-1)
                new_log_probs, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                policy_loss = self.ppo_loss(
                    new_log_probs, batch_old_log_probs, batch_advantages
                )

                if self.clip_value:
                    old_values_clipped = batch_old_log_probs.new_empty_like(
                        batch_old_log_probs
                    )
                    torch.clamp(
                        new_values,
                        batch_returns - self.clip_epsilon,
                        batch_returns + self.clip_epsilon,
                        out=old_values_clipped,
                    )
                    value_loss1 = F.mse_loss(new_values, batch_returns)
                    value_loss2 = F.mse_loss(old_values_clipped, batch_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(new_values, batch_returns)

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters())
                    + list(self.value_network.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                epoch_losses.append(loss.item())

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": np.mean(epoch_losses),
        }


class SAC(ActorCriticBase):
    """Soft Actor-Critic (SAC) algorithm.

    Implements maximum entropy actor-critic with automatic
    temperature adjustment and twin Q-networks.

    Args:
        policy: Policy network (TanhGaussianPolicy recommended).
        q_network: Q-network for state-action value estimation.
        config: SAC configuration.
        lr: Learning rate.

    Reference:
        Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy RL" (2018)

    Example:
        >>> policy = TanhGaussianPolicy(state_dim=4, action_dim=2)
        >>> q_net = ContinuousQNetwork(state_dim=4, action_dim=2)
        >>> sac = SAC(policy, q_net, lr=3e-4)
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        q_network: ContinuousQNetwork,
        config: SACConfig = None,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        super().__init__(policy, None, device)
        self.config = config or SACConfig()
        self.gamma = self.config.gamma
        self.tau = self.config.tau
        self.alpha = self.config.alpha
        self.automatic_alpha = self.config.automatic_alpha
        self.target_update_interval = self.config.target_update_interval
        self.policy_delay = self.config.policy_delay
        self.num_q_networks = self.config.num_q_networks
        self.update_count = 0

        self.policy = policy.to(device)
        self.q_network = q_network.to(device)

        self.target_q_network = ContinuousQNetwork(
            q_network.state_dim, q_network.action_dim, q_network.hidden_dims
        ).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.log_alpha = nn.Parameter(torch.tensor(float(np.log(self.alpha))))
        if self.automatic_alpha:
            target_entropy = self.config.target_entropy
            if target_entropy is None:
                target_entropy = -policy.action_dim
            self.target_entropy = target_entropy
        else:
            self.target_entropy = None

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + [self.log_alpha], lr=lr
        )

    @property
    def alpha(self) -> float:
        """Get current temperature."""
        return torch.exp(self.log_alpha).item()

    def select_action(
        self, state: Tensor, explore: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Select action using current policy."""
        state = self.to_device(state)
        output = self.policy(state)
        return output.action, output.log_prob

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Update SAC networks."""
        states = self.to_device(batch["states"])
        actions = self.to_device(batch["actions"])
        rewards = self.to_device(batch["rewards"]).squeeze(-1)
        next_states = self.to_device(batch["next_states"])
        dones = self.to_device(batch["dones"]).squeeze(-1)

        with torch.no_grad():
            next_output = self.policy(next_states)
            next_log_prob = next_output.log_prob
            next_q1 = self.target_q_network(next_states, next_output.action).value
            next_q2 = self.target_q_network(next_states, next_output.action).value
            next_q = torch.min(next_q1, next_q2)
            next_value = next_q - self.alpha * next_log_prob
            target_q = rewards + self.gamma * (1 - dones) * next_value.squeeze(-1)

        q1 = self.q_network(states, actions).value.squeeze(-1)
        q2 = self.q_network(states, actions).value.squeeze(-1)

        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.update_count += 1

        if self.update_count % self.policy_delay == 0:
            output = self.policy(states)
            new_q1 = self.q_network(states, output.action).value
            new_q2 = self.q_network(states, output.action).value
            new_q = torch.min(new_q1, new_q2)

            policy_loss = (self.alpha * output.log_prob - new_q.squeeze(-1)).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            if self.automatic_alpha:
                alpha_loss = -(
                    self.log_alpha * (output.log_prob + self.target_entropy).detach()
                ).mean()
                self.q_optimizer.zero_grad()
                alpha_loss.backward()
                self.q_optimizer.step()

        if self.update_count % self.target_update_interval == 0:
            self._soft_update()

        if self.automatic_alpha:
            alpha_value = self.alpha
        else:
            alpha_value = self.config.alpha

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item()
            if self.update_count % self.policy_delay == 0
            else 0,
            "alpha": alpha_value,
        }

    def _soft_update(self):
        """Soft update target network."""
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class TD3(ActorCriticBase):
    """Twin Delayed DDPG (TD3) algorithm.

    Implements twin Q-networks with delayed policy updates,
    target policy smoothing, and clipped double Q-learning.

    Args:
        policy: Deterministic policy network.
        q_network: Q-network for state-action value.
        config: TD3 configuration.
        lr: Learning rate.

    Reference:
        Fujita et al., "Addressing Function Approximation Error in Actor-Critic Methods" (2018)

    Example:
        >>> policy = DeterministicPolicy(state_dim=4, action_dim=2)
        >>> q_net = ContinuousQNetwork(state_dim=4, action_dim=2)
        >>> td3 = TD3(policy, q_net, lr=3e-4)
    """

    def __init__(
        self,
        policy: DeterministicPolicy,
        q_network: ContinuousQNetwork,
        config: TD3Config = None,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        super().__init__(policy, None, device)
        self.config = config or TD3Config()
        self.gamma = self.config.gamma
        self.tau = self.config.tau
        self.policy_noise = self.config.policy_noise
        self.noise_clip = self.config.noise_clip
        self.policy_delay = self.config.policy_delay
        self.target_update_interval = self.config.target_update_interval
        self.update_count = 0

        self.policy = policy.to(device)
        self.q_network = q_network.to(device)

        self.target_policy = DeterministicPolicy(
            policy.state_dim,
            policy.action_dim,
            policy.hidden_dims,
            policy.activation_fn,
            policy.action_scale,
            policy.action_bias,
        ).to(device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.eval()

        self.target_q_network = ContinuousQNetwork(
            q_network.state_dim, q_network.action_dim, q_network.hidden_dims
        ).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(
        self, state: Tensor, explore: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Select action using current policy."""
        state = self.to_device(state)
        with torch.no_grad():
            output = self.policy(state)
            if explore and self.policy_noise > 0:
                noise = torch.randn_like(output.action) * self.policy_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                output_action = torch.clamp(
                    output.action + noise,
                    -self.policy.action_scale,
                    self.policy.action_scale,
                )
            else:
                output_action = output.action
        return output_action, torch.tensor(0.0)

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Update TD3 networks."""
        states = self.to_device(batch["states"])
        actions = self.to_device(batch["actions"])
        rewards = self.to_device(batch["rewards"]).squeeze(-1)
        next_states = self.to_device(batch["next_states"])
        dones = self.to_device(batch["dones"]).squeeze(-1)

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = self.target_policy(next_states).action + noise
            next_actions = torch.clamp(
                next_actions, -self.policy.action_scale, self.policy.action_scale
            )

            next_q1 = self.target_q_network(next_states, next_actions).value
            next_q2 = self.target_q_network(next_states, next_actions).value
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + self.gamma * (1 - dones) * next_q.squeeze(-1)

        q1 = self.q_network(states, actions).value.squeeze(-1)
        q2 = self.q_network(states, actions).value.squeeze(-1)

        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.update_count += 1

        policy_loss = 0.0

        if self.update_count % self.policy_delay == 0:
            policy_output = self.policy(states)
            q_value = self.q_network(states, policy_output.action).value
            policy_loss = -q_value.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        if self.update_count % self.target_update_interval == 0:
            self._soft_update_target()
            self._soft_update_policy()

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss,
        }

    def _soft_update_target(self):
        """Soft update target Q-network."""
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _soft_update_policy(self):
        """Soft update target policy."""
        for target_param, param in zip(
            self.target_policy.parameters(), self.policy.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: Tensor of rewards.
        values: Tensor of value estimates.
        dones: Tensor of done flags.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        Tuple of (advantages, returns).

    Example:
        >>> rewards = torch.tensor([0.0, 0.0, 1.0])
        >>> values = torch.tensor([0.5, 0.4, 0.3])
        >>> dones = torch.tensor([0.0, 0.0, 1.0])
        >>> advantages, returns = compute_gae(rewards, values, dones)
    """
    advantages = torch.zeros_like(rewards)
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = torch.tensor(0.0)
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


class RolloutBuffer:
    """Buffer for storing rollouts during training.

    Stores states, actions, rewards, dones, values, and log probs
    for use in policy gradient algorithms.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.position = 0
        self.size = 0

        self.states = torch.zeros(capacity, state_dim)
        self.actions = torch.zeros(capacity, action_dim)
        self.rewards = torch.zeros(capacity, 1)
        self.dones = torch.zeros(capacity, 1)
        self.values = torch.zeros(capacity, 1)
        self.log_probs = torch.zeros(capacity, 1)

    def add(
        self,
        state: NDArray[np.float64],
        action: NDArray[np.float64],
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """Add transition to buffer."""
        self.states[self.position] = torch.from_numpy(state)
        self.actions[self.position] = torch.from_numpy(action)
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self) -> RolloutData:
        """Get all data as RolloutData."""
        return RolloutData(
            states=[self.states[: self.size]],
            actions=[self.actions[: self.size]],
            rewards=[self.rewards[: self.size]],
            dones=[self.dones[: self.size]],
            values=[self.values[: self.size]],
            log_probs=[self.log_probs[: self.size]],
        )

    def reset(self):
        """Reset buffer."""
        self.position = 0
        self.size = 0

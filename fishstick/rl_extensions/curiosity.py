"""
Curiosity-Driven Exploration for RL.

Implements intrinsic motivation techniques:
- ICM (Intrinsic Curiosity Module)
- RND (Random Network Distillation)

These methods encourage exploration by adding intrinsic rewards based on
novelty or surprise.

References:
    Pathak et al. "Curiosity-driven Exploration by Self-supervised Prediction" (2017)
    Burda et al. "Exploration by Random Network Distillation" (2018)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from fishstick.reinforcement import (
    GaussianPolicy,
    ActionValueNetwork,
)

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class ICMConfig:
    """Configuration for Intrinsic Curiosity Module.

    Attributes:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        embedding_dim: Dimension of state embedding.
        hidden_dim: Hidden dimension for networks.
        forward_scale: Scale for forward loss.
        inverse_scale: Scale for inverse loss.
        reward_scale: Scale for intrinsic reward.
        device: Device to use.
    """

    state_dim: int = 0
    action_dim: int = 0
    embedding_dim: int = 64
    hidden_dim: int = 256
    forward_scale: float = 0.1
    inverse_scale: float = 0.01
    reward_scale: float = 1.0
    device: str = "cuda"


@dataclass
class RNDConfig:
    """Configuration for Random Network Distillation.

    Attributes:
        state_dim: Dimension of state space.
        hidden_dim: Hidden dimension for networks.
        reward_scale: Scale for intrinsic reward.
        device: Device to use.
    """

    state_dim: int = 0
    hidden_dim: int = 256
    reward_scale: float = 1.0
    device: str = "cuda"


class IntrinsicCuriosityModule(Module):
    """Intrinsic Curiosity Module (ICM).

    Computes intrinsic rewards based on prediction error:
    - Forward model: predicts next state from current state and action
    - Inverse model: predicts action from current and next state

    The intrinsic reward is the prediction error of the forward model.

    Reference:
        Pathak et al. "Curiosity-driven Exploration by Self-supervised Prediction" (2017)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        forward_scale: float = 0.1,
        inverse_scale: float = 0.01,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.forward_scale = forward_scale
        self.inverse_scale = inverse_scale

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

        self.forward_model = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        state: Tensor,
        next_state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute intrinsic reward and losses.

        Args:
            state: Current states
            next_state: Next states
            action: Actions taken

        Returns:
            Tuple of (intrinsic_reward, forward_loss, inverse_loss)
        """
        state_emb = self.encoder(state)
        next_state_emb = self.encoder(next_state)

        inverse_input = torch.cat([state_emb, next_state_emb], dim=-1)
        predicted_action = self.inverse_model(inverse_input)

        forward_input = torch.cat([state_emb, action], dim=-1)
        predicted_next_emb = self.predictor(forward_input)

        forward_loss = F.mse_loss(predicted_next_emb, next_state_emb)
        inverse_loss = F.mse_loss(predicted_action, action)

        intrinsic_reward = forward_loss

        return intrinsic_reward, forward_loss, inverse_loss

    def compute_reward(
        self,
        state: Tensor,
        next_state: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Compute intrinsic reward only (no gradients).

        Args:
            state: Current states
            next_state: Next states
            action: Actions taken

        Returns:
            Intrinsic rewards
        """
        with torch.no_grad():
            intrinsic_reward, _, _ = self.forward(state, next_state, action)
        return intrinsic_reward


class RandomNetworkDistillation(Module):
    """Random Network Distillation (RND).

    Uses a random fixed network as target and trains a predictor to
    match its outputs. The prediction error serves as intrinsic reward.

    Reference:
        Burda et al. "Exploration by Random Network Distillation" (2018)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.state_dim = state_dim

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        for param in self.target_network.parameters():
            param.requires_grad = False

        self.predictor_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.optimizer = torch.optim.Adam(
            self.predictor_network.parameters(),
            lr=3e-4,
        )

    def compute_reward(
        self,
        state: Tensor,
    ) -> Tensor:
        """Compute intrinsic reward from state novelty.

        Args:
            state: States

        Returns:
            Intrinsic rewards
        """
        with torch.no_grad():
            target_output = self.target_network(state)
            predicted_output = self.predictor_network(state)
            reward = (target_output - predicted_output).pow(2).mean(dim=-1)
        return reward

    def update(
        self,
        state: Tensor,
    ) -> Dict[str, float]:
        """Update predictor network.

        Args:
            state: States

        Returns:
            Loss dictionary
        """
        target_output = self.target_network(state)
        predicted_output = self.predictor_network(state)

        loss = F.mse_loss(predicted_output, target_output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"rnd_loss": loss.item()}


class CuriosityRewardWrapper:
    """Wrapper for adding curiosity rewards to environments.

    Combines extrinsic and intrinsic rewards for exploration.
    """

    def __init__(
        self,
        icm: Optional[IntrinsicCuriosityModule] = None,
        rnd: Optional[RandomNetworkDistillation] = None,
        reward_scale: float = 1.0,
        combine_method: str = "add",
    ):
        self.icm = icm
        self.rnd = rnd
        self.reward_scale = reward_scale
        self.combine_method = combine_method

    def compute_total_reward(
        self,
        extrinsic_reward: Tensor,
        state: Tensor,
        next_state: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Compute total reward combining extrinsic and intrinsic.

        Args:
            extrinsic_reward: Environment rewards
            state: Current states
            next_state: Next states
            action: Actions taken

        Returns:
            Combined rewards
        """
        if self.icm is not None:
            intrinsic_reward = self.icm.compute_reward(state, next_state, action)
        elif self.rnd is not None:
            intrinsic_reward = self.rnd.compute_reward(state)
        else:
            return extrinsic_reward

        intrinsic_reward = intrinsic_reward * self.reward_scale

        if self.combine_method == "add":
            return extrinsic_reward + intrinsic_reward
        elif self.combine_method == "multiply":
            return extrinsic_reward * (1 + intrinsic_reward)
        else:
            return extrinsic_reward + intrinsic_reward

    def update(
        self,
        state: Tensor,
        next_state: Tensor,
        action: Tensor,
    ) -> Dict[str, float]:
        """Update intrinsic motivation module.

        Args:
            state: Current states
            next_state: Next states
            action: Actions taken

        Returns:
            Loss dictionary
        """
        if self.icm is not None:
            intrinsic_reward, forward_loss, inverse_loss = self.icm(
                state, next_state, action
            )
            return {
                "intrinsic_reward": intrinsic_reward.mean().item(),
                "forward_loss": forward_loss.item(),
                "inverse_loss": inverse_loss.item(),
            }
        elif self.rnd is not None:
            rnd_loss_dict = self.rnd.update(state)
            return rnd_loss_dict
        return {}


class CuriosityAgent(Module):
    """RL Agent with Curiosity-Driven Exploration.

    Combines a base RL algorithm with curiosity modules for exploration.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        use_icm: bool = True,
        use_rnd: bool = False,
        reward_scale: float = 1.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_scale = reward_scale

        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
        )

        self.q_network1 = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
        )

        self.q_network2 = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
        )

        self.target_q1 = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
        )

        self.target_q2 = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256],
        )

        self.target_q1.load_state_dict(self.q_network1.state_dict())
        self.target_q2.load_state_dict(self.q_network2.state_dict())

        if use_icm:
            self.curiosity = IntrinsicCuriosityModule(
                state_dim=state_dim,
                action_dim=action_dim,
            )
        elif use_rnd:
            self.curiosity = RandomNetworkDistillation(
                state_dim=state_dim,
            )
        else:
            self.curiosity = None

        self.reward_wrapper = CuriosityRewardWrapper(
            icm=self.curiosity if use_icm else None,
            rnd=self.curiosity if use_rnd else None,
            reward_scale=reward_scale,
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=3e-4,
        )

        self.q_optimizer1 = torch.optim.Adam(
            self.q_network1.parameters(),
            lr=3e-4,
        )

        self.q_optimizer2 = torch.optim.Adam(
            self.q_network2.parameters(),
            lr=3e-4,
        )

        self.gamma = 0.99
        self.tau = 0.005

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
        extrinsic_reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> Dict[str, float]:
        """Update agent with curiosity rewards."""
        state = state.float()
        action = action.float()
        extrinsic_reward = extrinsic_reward.float()
        next_state = next_state.float()
        done = done.float()

        intrinsic_reward_dict = {}
        if self.curiosity is not None:
            intrinsic_reward = self.reward_wrapper.compute_total_reward(
                extrinsic_reward, state, next_state, action
            ).detach()
            intrinsic_reward_dict = {"intrinsic_reward": intrinsic_reward.mean().item()}
            total_reward = extrinsic_reward + intrinsic_reward * self.reward_scale
        else:
            total_reward = extrinsic_reward

        with torch.no_grad():
            next_action, next_log_pi = self.policy.get_action(next_state)
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            q_target = total_reward + (1 - done) * self.gamma * (target_q - next_log_pi)

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

        new_action, log_pi = self.policy.get_action(state)
        q1_new = self.q_network1(state.detach(), new_action)
        q2_new = self.q_network2(state.detach(), new_action)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = -q_new.mean() + log_pi.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.curiosity is not None:
            curiosity_loss_dict = self.reward_wrapper.update(state, next_state, action)
        else:
            curiosity_loss_dict = {}

        self._soft_update_target()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            **intrinsic_reward_dict,
            **curiosity_loss_dict,
        }

    def _soft_update_target(self):
        """Soft update target networks."""
        for target_param, param in zip(
            self.target_q1.parameters(),
            self.q_network1.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_q2.parameters(),
            self.q_network2.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

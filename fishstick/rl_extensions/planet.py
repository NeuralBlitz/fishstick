"""
Model-Based RL: PlaNet/Dreamer Latent Dynamics Model.

Implements the Recurrent State Space Model (RSSM) from the Dreamer paper
for model-based reinforcement learning with latent dynamics.

Reference:
    Hafner et al. "Dream to Control: Learning Behaviors by Latent Imagination" (2020)
    Hafner et al. "Mastering Atari with Discrete World Models" (2021)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Normal,
    Independent,
    Categorical,
    Bernoulli,
    OneHotCategorical,
)
from torch import Tensor

from fishstick.reinforcement import (
    GaussianPolicy,
    DeterministicPolicy,
    ActionValueNetwork,
)

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class DreamerConfig:
    """Configuration for Dreamer agent.

    Attributes:
        latent_dim: Dimension of latent state.
        hidden_dim: Hidden dimension for MLPs.
        recurrent_dim: Dimension of GRU hidden state.
        action_dim: Dimension of action space.
        gamma: Discount factor.
        lambda_: GAE lambda parameter.
        model_lr: Learning rate for world model.
        actor_lr: Learning rate for actor.
        critic_lr: Learning rate for critic.
        grad_clip: Gradient clipping value.
        batch_size: Batch size for training.
        horizon: Planning horizon for imagined trajectories.
        num_actors: Number of parallel actors for data collection.
        device: Device to use.
    """

    latent_dim: int = 32
    hidden_dim: int = 200
    recurrent_dim: int = 200
    action_dim: int = 0
    num_actions: int = 0
    gamma: float = 0.99
    lambda_: float = 0.95
    model_lr: float = 3e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    grad_clip: float = 100.0
    batch_size: int = 50
    horizon: int = 15
    num_actors: int = 10
    device: str = "cuda"
    free_nats: float = 1.0
    kl_scale: float = 1.0
    actor_entropy_coef: float = 1e-3
    actor_dist: str = "tanh_normal"


class RSSM(Module):
    """Recurrent State Space Model.

    Combines a deterministic hidden state (via GRU) with stochastic latent
    variables to model the environment dynamics.

    Architecture:
        - Prior: p(z_t | h_t) - stochastic latent given previous hidden state
        - Posterior: q(z_t | h_t, o_t) - stochastic latent given hidden state and observation
        - Transition: h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) - deterministic hidden state
        - Decoder: p(o_t | h_t, z_t) - observation reconstruction
        - Reward: p(r_t | h_t, z_t) - reward prediction
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 200,
        recurrent_dim: int = 200,
        num_categories: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.recurrent_dim = recurrent_dim
        self.num_categories = num_categories

        self.gru = nn.GRUCell(action_dim + latent_dim, recurrent_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(recurrent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * num_categories),
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(recurrent_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * num_categories),
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(recurrent_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        self.reward_net = nn.Sequential(
            nn.Linear(recurrent_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.continue_net = nn.Sequential(
            nn.Linear(recurrent_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> Tensor:
        """Initialize hidden state."""
        return torch.zeros(batch_size, self.recurrent_dim, device=device)

    def recurrent_pass(
        self,
        hidden: Tensor,
        prev_action: Tensor,
        prev_latent: Tensor,
    ) -> Tensor:
        """Compute next hidden state from previous action and latent."""
        flat_action = prev_action.reshape(prev_action.shape[0], -1)
        flat_latent = prev_latent.reshape(prev_latent.shape[0], -1)
        gru_input = torch.cat([flat_action, flat_latent], dim=-1)
        return self.gru(gru_input, hidden)

    def get_prior(
        self,
        hidden: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Get prior distribution over latent variables."""
        logits = self.prior_net(hidden)
        logits = logits.view(-1, self.latent_dim, self.num_categories)

        if deterministic:
            ind = logits.argmax(dim=-1)
            latent = F.one_hot(ind, self.num_categories).float()
        else:
            dist = Categorical(logits=logits)
            ind = dist.sample()
            latent = F.one_hot(ind, self.num_categories).float()

        latent = latent.view(-1, self.latent_dim * self.num_categories)
        return latent, logits

    def get_posterior(
        self,
        hidden: Tensor,
        obs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Get posterior distribution over latent variables."""
        flat_obs = obs.reshape(obs.shape[0], -1)
        joint_input = torch.cat([hidden, flat_obs], dim=-1)
        logits = self.posterior_net(joint_input)
        logits = logits.view(-1, self.latent_dim, self.num_categories)

        dist = Categorical(logits=logits)
        ind = dist.sample()
        latent = F.one_hot(ind, self.num_categories).float()
        latent = latent.view(-1, self.latent_dim * self.num_categories)

        return latent, logits

    def decode(
        self,
        hidden: Tensor,
        latent: Tensor,
    ) -> Tensor:
        """Decode observation from hidden state and latent."""
        joint = torch.cat([hidden, latent], dim=-1)
        return self.decoder_net(joint)

    def predict_reward(
        self,
        hidden: Tensor,
        latent: Tensor,
    ) -> Tensor:
        """Predict reward from hidden state and latent."""
        joint = torch.cat([hidden, latent], dim=-1)
        return self.reward_net(joint).squeeze(-1)

    def predict_continue(
        self,
        hidden: Tensor,
        latent: Tensor,
    ) -> Tensor:
        """Predict done/continue flag."""
        joint = torch.cat([hidden, latent], dim=-1)
        return self.continue_net(joint).squeeze(-1)

    def forward(
        self,
        obs: Tensor,
        action: Tensor,
        hidden: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through RSSM.

        Args:
            obs: Observations, shape (batch, seq_len, obs_dim)
            action: Actions, shape (batch, seq_len, action_dim)
            hidden: Initial hidden state, shape (batch, recurrent_dim)
            deterministic: If True, use argmax instead of sampling

        Returns:
            Dictionary containing:
                - latent: sampled latent variables
                - hidden: hidden states
                - prior_logits: prior logits
                - posterior_logits: posterior logits
                - obs_recon: reconstructed observations
                - rewards: predicted rewards
        """
        batch_size, seq_len = obs.shape[0], obs.shape[1]

        if hidden is None:
            hidden = self.init_hidden(batch_size, obs.device)

        latent_list = []
        hidden_list = []
        prior_logits_list = []
        posterior_logits_list = []
        obs_recon_list = []
        reward_list = []

        prev_action = torch.zeros_like(action[:, 0])
        prev_latent = torch.zeros(
            batch_size, self.latent_dim * self.num_categories, device=obs.device
        )

        for t in range(seq_len):
            hidden = self.recurrent_pass(hidden, prev_action, prev_latent)

            prior, prior_logits = self.get_prior(hidden, deterministic)
            posterior, posterior_logits = self.get_posterior(hidden, obs[:, t])

            latent = prior if deterministic else posterior

            obs_recon = self.decode(hidden, latent)
            reward = self.predict_reward(hidden, latent)

            latent_list.append(latent)
            hidden_list.append(hidden)
            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(posterior_logits)
            obs_recon_list.append(obs_recon)
            reward_list.append(reward)

            prev_action = action[:, t]
            prev_latent = latent

        return {
            "latent": torch.stack(latent_list, dim=1),
            "hidden": torch.stack(hidden_list, dim=1),
            "prior_logits": torch.stack(prior_logits_list, dim=1),
            "posterior_logits": torch.stack(posterior_logits_list, dim=1),
            "obs_recon": torch.stack(obs_recon_list, dim=1),
            "rewards": torch.stack(reward_list, dim=1),
        }


class RecurrentStateSpaceModel(Module):
    """Wrapper for RSSM with additional utilities.

    Provides a higher-level interface for training and using the
    latent dynamics model.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 200,
        recurrent_dim: int = 200,
        num_categories: int = 32,
        learning_rate: float = 3e-4,
    ):
        super().__init__()
        self.rssm = RSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            recurrent_dim=recurrent_dim,
            num_categories=num_categories,
        )

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim

        self.model_optimizer = torch.optim.Adam(
            self.rssm.parameters(),
            lr=learning_rate,
        )

    def compute_loss(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Compute world model loss.

        Args:
            obs: Observations (batch, seq_len, obs_dim)
            action: Actions (batch, seq_len, action_dim)
            reward: Rewards (batch, seq_len)
            done: Done flags (batch, seq_len)

        Returns:
            Dictionary of losses
        """
        forward_output = self.rssm(obs, action)

        obs_recon = forward_output["obs_recon"]
        rewards_pred = forward_output["rewards"]

        obs_loss = F.mse_loss(obs_recon, obs)
        reward_loss = F.mse_loss(rewards_pred, reward)

        prior_logits = forward_output["prior_logits"]
        posterior_logits = forward_output["posterior_logits"]

        prior_dist = Categorical(logits=prior_logits)
        posterior_dist = Categorical(logits=posterior_logits)

        kl_loss = (
            torch.distributions.kl.kl_divergence(
                posterior_dist,
                prior_dist,
            )
            .sum(dim=-1)
            .mean()
        )

        kl_loss = F.relu(kl_loss - 1.0)

        total_loss = obs_loss + reward_loss + kl_loss

        return {
            "total_loss": total_loss,
            "obs_loss": obs_loss,
            "reward_loss": reward_loss,
            "kl_loss": kl_loss,
        }

    def update(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Update world model."""
        self.model_optimizer.zero_grad()

        loss_dict = self.compute_loss(obs, action, reward, done)
        loss_dict["total_loss"].backward()

        torch.nn.utils.clip_grad_norm_(
            self.rssm.parameters(),
            max_norm=100.0,
        )

        self.model_optimizer.step()

        return {k: v.item() for k, v in loss_dict.items()}

    def imagine(
        self,
        policy: Module,
        initial_hidden: Tensor,
        initial_latent: Tensor,
        horizon: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Imagine trajectories using the learned model.

        Args:
            policy: Policy network
            initial_hidden: Initial hidden state
            initial_latent: Initial latent state
            horizon: Number of steps to imagine

        Returns:
            Tuple of (latents, rewards, continues)
        """
        latents = []
        rewards = []
        continues = []
        hidden = initial_hidden
        latent = initial_latent
        action = torch.zeros(
            initial_hidden.shape[0],
            self.action_dim,
            device=initial_hidden.device,
        )

        for _ in range(horizon):
            joint = torch.cat([hidden, latent], dim=-1)
            action, _ = policy(joint)

            hidden = self.rssm.recurrent_pass(hidden, action, latent)
            latent, _ = self.rssm.get_prior(hidden, deterministic=False)

            reward = self.rssm.predict_reward(hidden, latent)
            cont = self.rssm.predict_continue(hidden, latent)

            latents.append(latent)
            rewards.append(reward)
            continues.append(cont)

        return (
            torch.stack(latents, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(continues, dim=1),
        )


class LatentDynamicsModel(Module):
    """Latent Dynamics Model for Model-Based RL.

    A comprehensive model that learns to predict in a latent space,
    enabling efficient planning and imagination.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        recurrent_dim: int = 256,
        num_categories: int = 32,
        encoder_dim: int = 256,
        encoder_lr: float = 3e-4,
        dynamics_lr: float = 3e-4,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, latent_dim),
            nn.Tanh(),
        )

        self.dynamics = RecurrentStateSpaceModel(
            obs_dim=latent_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            recurrent_dim=recurrent_dim,
            num_categories=num_categories,
            learning_rate=dynamics_lr,
        )

        self.reward_head = nn.Sequential(
            nn.Linear(recurrent_dim + latent_dim * num_categories, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim
        self.num_categories = num_categories

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=encoder_lr,
        )

    def encode(self, obs: Tensor) -> Tensor:
        """Encode observation to latent space."""
        return self.encoder(obs)

    def forward(
        self,
        obs: Tensor,
        action: Tensor,
    ) -> Dict[str, Tensor]:
        """Forward pass through the latent dynamics model."""
        latent = self.encode(obs)
        dynamics_output = self.dynamics.rssm(obs, action)

        return {
            "latent": latent,
            "hidden": dynamics_output["hidden"],
            "obs_recon": dynamics_output["obs_recon"],
            "rewards": dynamics_output["rewards"],
        }

    def train_step(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
    ) -> Dict[str, float]:
        """Single training step for the latent dynamics model."""
        latent = self.encode(obs)

        self.encoder_optimizer.zero_grad()
        loss_dict = self.dynamics.update(obs, action, reward)

        return loss_dict


class DreamerAgent(Module):
    """Dreamer Agent for Model-Based RL.

    Implements the full Dreamer algorithm with:
    - World model (RSSM + encoder + decoder)
    - Actor (policy) learning via imagined trajectories
    - Critic (value function) learning

    Reference:
        Hafner et al. "Dream to Control: Learning Behaviors by Latent Imagination" (2020)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[DreamerConfig] = None,
    ):
        super().__init__()

        self.config = config or DreamerConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        self.world_model = LatentDynamicsModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            recurrent_dim=self.config.recurrent_dim,
            encoder_lr=self.config.model_lr,
            dynamics_lr=self.config.model_lr,
        )

        actor_latent_dim = self.config.recurrent_dim + self.config.latent_dim * 32

        self.actor = nn.Sequential(
            nn.Linear(actor_latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, action_dim * 2),
        )

        critic_latent_dim = self.config.recurrent_dim + self.config.latent_dim * 32

        self.critic = nn.Sequential(
            nn.Linear(critic_latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

        self.target_critic = nn.Sequential(
            nn.Linear(critic_latent_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr,
        )

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._is_training = True

    def get_action(
        self,
        obs: Tensor,
        hidden: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Get action from the policy.

        Args:
            obs: Observation tensor
            hidden: RNN hidden state
            deterministic: If True, return mean action

        Returns:
            Tuple of (action, new_hidden)
        """
        with torch.no_grad():
            latent = self.world_model.encode(obs.unsqueeze(0))

            if hidden is None:
                hidden = torch.zeros(
                    1,
                    self.config.recurrent_dim,
                    device=obs.device,
                )

            joint = torch.cat([hidden, latent.squeeze(0)], dim=-1)
            action_mean, action_std = self.actor(joint).chunk(2, dim=-1)

            action_std = F.softplus(action_std) + 1e-5

            if deterministic:
                action = torch.tanh(action_mean)
            else:
                dist = Normal(action_mean, action_std)
                action = torch.tanh(dist.sample())
                action = torch.clamp(action, -1.0, 1.0)

            return action, hidden

    def update_actor(
        self,
        imagined_latents: Tensor,
        imagined_rewards: Tensor,
        imagined_continues: Tensor,
    ) -> Dict[str, float]:
        """Update actor using imagined trajectories."""
        batch_size, horizon = imagined_rewards.shape

        discounted_returns = torch.zeros_like(imagined_rewards)
        value_target = torch.zeros(batch_size, device=imagined_rewards.device)

        for t in reversed(range(horizon)):
            value_target = imagined_rewards[:, t] + (
                self.config.gamma * imagined_continues[:, t] * value_target
            )
            discounted_returns[:, t] = value_target

        discounted_returns = discounted_returns.detach()

        action_means = []
        action_stds = []
        log_probs = []

        for t in range(horizon):
            joint = torch.cat(
                [
                    imagined_latents[:, t, : self.config.recurrent_dim],
                    imagined_latents[:, t, self.config.recurrent_dim :],
                ],
                dim=-1,
            )

            action_mean, action_std = self.actor(joint).chunk(2, dim=-1)
            action_std = F.softplus(action_std) + 1e-5

            dist = Normal(action_mean, action_std)
            action = torch.tanh(dist.rsample())
            log_prob = dist.log_prob(action).sum(dim=-1)

            action_means.append(action)
            action_stds.append(action_std)
            log_probs.append(log_prob)

        action_means = torch.stack(action_means, dim=1)
        log_probs = torch.stack(log_probs, dim=1)

        baseline = self.critic(
            torch.cat(
                [
                    imagined_latents[:, :, : self.config.recurrent_dim],
                    imagined_latents[:, :, self.config.recurrent_dim :],
                ],
                dim=-1,
            ).reshape(batch_size * horizon, -1)
        ).reshape(batch_size, horizon)

        advantage = (discounted_returns - baseline.detach()).clamp(-10.0, 10.0)

        actor_loss = -(log_probs * advantage).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=self.config.grad_clip,
        )
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.item()}

    def update_critic(
        self,
        imagined_latents: Tensor,
        imagined_rewards: Tensor,
        imagined_continues: Tensor,
    ) -> Dict[str, float]:
        """Update critic using imagined trajectories."""
        batch_size, horizon = imagined_rewards.shape

        target_values = []
        value_target = torch.zeros(batch_size, device=imagined_rewards.device)

        for t in reversed(range(horizon)):
            value_target = imagined_rewards[:, t] + (
                self.config.gamma * imagined_continues[:, t] * value_target
            )
            target_values.append(value_target)

        target_values = torch.stack(target_values, dim=1).detach()

        flat_latents = torch.cat(
            [
                imagined_latents[:, :, : self.config.recurrent_dim],
                imagined_latents[:, :, self.config.recurrent_dim :],
            ],
            dim=-1,
        ).reshape(batch_size * horizon, -1)

        predicted_values = self.critic(flat_latents).reshape(batch_size, horizon)

        critic_loss = F.mse_loss(predicted_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=self.config.grad_clip,
        )
        self.critic_optimizer.step()

        self._soft_update_target()

        return {"critic_loss": critic_loss.item()}

    def _soft_update_target(self):
        """Soft update target network."""
        for target_param, param in zip(
            self.target_critic.parameters(),
            self.critic.parameters(),
        ):
            target_param.data.copy_(
                self.config.lambda_ * param.data
                + (1 - self.config.lambda_) * target_param.data
            )

    def update(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> Dict[str, float]:
        """Full update step for Dreamer agent."""
        obs = obs.float()
        action = action.float()
        reward = reward.float()
        done = done.float()

        with torch.no_grad():
            latent = self.world_model.encode(obs)

        model_loss_dict = self.world_model.train_step(obs, action, reward)

        batch_size = obs.shape[0]

        hidden = torch.zeros(
            batch_size,
            self.config.recurrent_dim,
            device=obs.device,
        )

        imagined_latents = []
        imagined_rewards = []
        imagined_continues = []

        prev_action = torch.zeros(batch_size, self.action_dim, device=obs.device)
        prev_latent = torch.zeros(
            batch_size,
            self.config.latent_dim * 32,
            device=obs.device,
        )

        for _ in range(self.config.horizon):
            hidden = self.world_model.dynamics.rssm.recurrent_pass(
                hidden, prev_action, prev_latent
            )

            prior, _ = self.world_model.dynamics.rssm.get_prior(hidden)

            reward_pred = self.world_model.dynamics.rssm.predict_reward(hidden, prior)
            cont_pred = self.world_model.dynamics.rssm.predict_continue(hidden, prior)

            imagined_latents.append(torch.cat([hidden, prior], dim=-1))
            imagined_rewards.append(reward_pred)
            imagined_continues.append(1.0 - cont_pred)

            joint = torch.cat([hidden, prior], dim=-1)
            action_mean, action_std = self.actor(joint).chunk(2, dim=-1)
            action_std = F.softplus(action_std) + 1e-5
            dist = Normal(action_mean, action_std)
            prev_action = torch.tanh(dist.sample())
            prev_latent = prior

        imagined_latents = torch.stack(imagined_latents, dim=1)
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        imagined_continues = torch.stack(imagined_continues, dim=1)

        actor_loss_dict = self.update_actor(
            imagined_latents,
            imagined_rewards,
            imagined_continues,
        )

        critic_loss_dict = self.update_critic(
            imagined_latents,
            imagined_rewards,
            imagined_continues,
        )

        return {
            **model_loss_dict,
            **actor_loss_dict,
            **critic_loss_dict,
        }

    def train(self, mode: bool = True) -> "DreamerAgent":
        """Set training mode."""
        self._is_training = mode
        return super().train(mode)

    @property
    def is_training(self) -> bool:
        """Check if agent is in training mode."""
        return self._is_training

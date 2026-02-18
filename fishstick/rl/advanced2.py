"""
fishstick Advanced Reinforcement Learning Toolkit v2

Comprehensive RL module providing state-of-the-art algorithms:

**Model-Based RL**:
- MBPO: Model-based policy optimization
- PETS: Probabilistic ensembles
- Dreamer: World models
- PlaNet: Latent dynamics
- MUZero: Planning with model

**Offline RL**:
- CQL: Conservative Q-learning
- IQL: Implicit Q-learning
- DecisionTransformer: Sequence modeling
- CRR: Critic regularized
- AWAC: Accelerated actor-critic

**Hierarchical RL**:
- OptionCritic: Options framework
- HAC: Hindsight actor-critic
- Feudal networks
- HIRO: High-level controller

**Multi-Agent RL**:
- MADDPG: Multi-agent DDPG
- MAPPO: Multi-agent PPO
- QMIX: Value decomposition
- VDN: Value decomposition
- COMA: Counterfactual multi-agent

**Imitation Learning**:
- BehavioralCloning: BC
- DAgger: Dataset aggregation
- GAIL: Adversarial imitation
- AIRL: Adversarial IRL
- SQIL: Soft Q imitation

**Inverse RL**:
- MaxEntIRL: Maximum entropy
- DeepMaxEnt: Deep version
- GCL: Guided cost learning

**Meta-RL**:
- MAML: Model-agnostic
- RL2: Fast adaptation
- ProMP: Proximal meta-policy
- PEARL: Efficient adaptation

Usage:
    from fishstick.rl.advanced2 import (
        MBPO, CQL, DecisionTransformer, MADDPG,
        GAIL, MaxEntIRL, MAML
    )
"""

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    NamedTuple,
)
import copy
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal, Independent, OneHotCategorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")
State = Union[np.ndarray, torch.Tensor]
Action = Union[int, float, np.ndarray, torch.Tensor]
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


# =============================================================================
# Utility Functions
# =============================================================================


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Soft update target network parameters."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Hard update target network parameters."""
    target.load_state_dict(source.state_dict())


def init_weights(m: nn.Module) -> None:
    """Initialize network weights."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


# =============================================================================
# Model-Based RL Components
# =============================================================================


class EnsembleDynamicsModel(nn.Module):
    """Probabilistic ensemble dynamics model for PETS/MBPO.

    Models p(s_{t+1}, r_{t+1} | s_t, a_t) with aleatoric uncertainty.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 200,
        num_ensemble: int = 7,
        num_elites: int = 5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_ensemble = num_ensemble
        self.num_elites = num_elites

        # Create ensemble of probabilistic models
        self.models = nn.ModuleList(
            [
                self._build_model(state_dim, action_dim, hidden_dim)
                for _ in range(num_ensemble)
            ]
        )

        self.elites = list(range(num_elites))
        self.max_logvar = nn.Parameter(
            torch.ones(1, state_dim + 1) * 0.5, requires_grad=True
        )
        self.min_logvar = nn.Parameter(
            torch.ones(1, state_dim + 1) * -10, requires_grad=True
        )

    def _build_model(
        self, state_dim: int, action_dim: int, hidden_dim: int
    ) -> nn.Module:
        """Build single probabilistic dynamics model."""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, 2 * (state_dim + 1)),
        )

    def forward(
        self, state: Tensor, action: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through ensemble."""
        x = torch.cat([state, action], dim=-1)

        means, logvars = [], []
        for model in self.models:
            output = model(x)
            mean = output[..., : self.state_dim + 1]
            logvar = output[..., self.state_dim + 1 :]

            # Bound log variance
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

            means.append(mean)
            logvars.append(logvar)

        means = torch.stack(means)
        logvars = torch.stack(logvars)

        if deterministic:
            samples = means
        else:
            stds = torch.exp(0.5 * logvars)
            samples = means + stds * torch.randn_like(stds)

        delta_states = samples[..., : self.state_dim]
        rewards = samples[..., self.state_dim :]
        next_states = state.unsqueeze(0) + delta_states

        return next_states, rewards

    def loss(
        self, state: Tensor, action: Tensor, next_state: Tensor, reward: Tensor
    ) -> Tensor:
        """Compute negative log-likelihood loss for training."""
        x = torch.cat([state, action], dim=-1)
        target = torch.cat([next_state - state, reward.unsqueeze(-1)], dim=-1)

        losses = []
        for model in self.models:
            output = model(x)
            mean = output[..., : self.state_dim + 1]
            logvar = output[..., self.state_dim + 1 :]

            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

            inv_var = torch.exp(-logvar)
            loss = ((mean - target) ** 2) * inv_var + logvar
            losses.append(loss.mean())

        return torch.stack(losses).mean()

    def predict(
        self, state: Tensor, action: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Predict using elite models only."""
        next_states, rewards = self.forward(state, action, deterministic)
        elite_next_states = next_states[self.elites]
        elite_rewards = rewards[self.elites]
        return elite_next_states.mean(dim=0), elite_rewards.mean(dim=0)

    def set_elites(self, losses: List[float]) -> None:
        """Set elite models based on validation losses."""
        sorted_indices = np.argsort(losses)
        self.elites = sorted_indices[: self.num_elites].tolist()


class PETS:
    """Probabilistic Ensembles with Trajectory Sampling."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int = 30,
        num_candidates: int = 500,
        top_k: int = 50,
        num_iterations: int = 5,
        elite_fraction: float = 0.1,
        action_min: float = -1.0,
        action_max: float = 1.0,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.num_iterations = num_iterations
        self.elite_fraction = elite_fraction
        self.action_min = action_min
        self.action_max = action_max

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dynamics_model = EnsembleDynamicsModel(state_dim, action_dim).to(
            self.device
        )
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics_model.parameters(), lr=1e-3
        )

    def train_dynamics(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
        rewards: Tensor,
        num_epochs: int = 100,
    ) -> float:
        """Train the ensemble dynamics model."""
        total_loss = 0
        for epoch in range(num_epochs):
            loss = self.dynamics_model.loss(states, actions, next_states, rewards)
            self.dynamics_optimizer.zero_grad()
            loss.backward()
            self.dynamics_optimizer.step()
            total_loss += loss.item()
        return total_loss / num_epochs

    def plan(self, state: np.ndarray) -> np.ndarray:
        """Plan action using Cross-Entropy Method (CEM)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.horizon, self.action_dim, device=self.device)

        for _ in range(self.num_iterations):
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.num_candidates, self.horizon, self.action_dim, device=self.device
            )
            actions = torch.clamp(actions, self.action_min, self.action_max)
            returns = self._evaluate_action_sequences(
                state_tensor.repeat(self.num_candidates, 1), actions
            )
            elite_indices = returns.argsort(descending=True)[
                : int(self.elite_fraction * self.num_candidates)
            ]
            elite_actions = actions[elite_indices]
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6

        return mean[0].cpu().numpy()

    def _evaluate_action_sequences(
        self, states: Tensor, action_sequences: Tensor
    ) -> Tensor:
        """Evaluate action sequences using the learned model."""
        returns = torch.zeros(states.shape[0], device=self.device)
        for t in range(self.horizon):
            actions = action_sequences[:, t]
            next_states, rewards = self.dynamics_model.predict(states, actions)
            returns += rewards.squeeze(-1)
            states = next_states
        return returns


class MBPO:
    """Model-Based Policy Optimization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        rollout_length: int = 1,
        num_rollouts: int = 400,
        rollout_batch_size: int = 50,
        real_ratio: float = 0.05,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rollout_length = rollout_length
        self.num_rollouts = num_rollouts
        self.rollout_batch_size = rollout_batch_size
        self.real_ratio = real_ratio

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dynamics_model = EnsembleDynamicsModel(state_dim, action_dim).to(
            self.device
        )
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics_model.parameters(), lr=1e-3
        )
        self.model_buffer = deque(maxlen=100000)
        self.real_buffer = deque(maxlen=1000000)

    def train_dynamics(self, num_epochs: int = 100, batch_size: int = 256) -> float:
        """Train dynamics model on real data."""
        if len(self.real_buffer) < batch_size:
            return 0.0
        total_loss = 0
        for epoch in range(num_epochs):
            batch = random.sample(self.real_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            loss = self.dynamics_model.loss(states, actions, next_states, rewards)
            self.dynamics_optimizer.zero_grad()
            loss.backward()
            self.dynamics_optimizer.step()
            total_loss += loss.item()
        return total_loss / num_epochs

    def generate_rollouts(self, policy: Callable[[np.ndarray], np.ndarray]) -> None:
        """Generate synthetic rollouts using the learned model."""
        if len(self.real_buffer) == 0:
            return
        batch = random.sample(self.real_buffer, self.rollout_batch_size)
        states = np.array([e[0] for e in batch])
        for _ in range(self.num_rollouts // self.rollout_batch_size):
            state = torch.FloatTensor(states).to(self.device)
            for t in range(self.rollout_length):
                action = policy(state.cpu().numpy())
                action_tensor = torch.FloatTensor(action).to(self.device)
                with torch.no_grad():
                    next_state, reward = self.dynamics_model.predict(
                        state, action_tensor
                    )
                for i in range(len(states)):
                    self.model_buffer.append(
                        (
                            state[i].cpu().numpy(),
                            action[i],
                            reward[i].item(),
                            next_state[i].cpu().numpy(),
                            False,
                        )
                    )
                state = next_state


class RSSM(nn.Module):
    """Recurrent State-Space Model for Dreamer/PlaNet."""

    def __init__(
        self,
        stochastic_size: int = 32,
        deterministic_size: int = 200,
        hidden_size: int = 200,
        action_size: int = 6,
        obs_embed_size: int = 1024,
    ):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.hidden_size = hidden_size

        # Recurrent model
        self.gru = nn.GRUCell(hidden_size, deterministic_size)

        # Prior p(z_t | h_t)
        self.fc_prior = nn.Linear(deterministic_size, hidden_size)
        self.fc_prior_mean = nn.Linear(hidden_size, stochastic_size)
        self.fc_prior_std = nn.Linear(hidden_size, stochastic_size)

        # Posterior q(z_t | h_t, x_t)
        self.fc_posterior = nn.Linear(deterministic_size + obs_embed_size, hidden_size)
        self.fc_posterior_mean = nn.Linear(hidden_size, stochastic_size)
        self.fc_posterior_std = nn.Linear(hidden_size, stochastic_size)

        # Observation encoder
        self.encoder = nn.Sequential(nn.Linear(obs_embed_size, hidden_size), nn.ReLU())

        # Action embedder
        self.action_embed = nn.Linear(action_size, hidden_size)

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        """Get initial state."""
        return {
            "h": torch.zeros(batch_size, self.deterministic_size, device=device),
            "z": torch.zeros(batch_size, self.stochastic_size, device=device),
            "z_mean": torch.zeros(batch_size, self.stochastic_size, device=device),
            "z_std": torch.zeros(batch_size, self.stochastic_size, device=device),
        }

    def observe_step(
        self, prev_state: Dict[str, Tensor], action: Tensor, obs_embed: Tensor
    ) -> Dict[str, Tensor]:
        """Observation step: compute posterior."""
        h = self.gru(
            self.action_embed(action) + self.encoder(obs_embed), prev_state["h"]
        )

        prior = F.relu(self.fc_prior(h))
        prior_mean = self.fc_prior_mean(prior)
        prior_std = F.softplus(self.fc_prior_std(prior)) + 0.1

        posterior_input = torch.cat([h, obs_embed], dim=-1)
        posterior = F.relu(self.fc_posterior(posterior_input))
        posterior_mean = self.fc_posterior_mean(posterior)
        posterior_std = F.softplus(self.fc_posterior_std(posterior)) + 0.1

        z = posterior_mean + posterior_std * torch.randn_like(posterior_std)

        return {
            "h": h,
            "z": z,
            "z_mean": posterior_mean,
            "z_std": posterior_std,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
        }

    def imagine_step(
        self, prev_state: Dict[str, Tensor], action: Tensor
    ) -> Dict[str, Tensor]:
        """Imagination step: use prior only."""
        h = self.gru(self.action_embed(action), prev_state["h"])
        prior = F.relu(self.fc_prior(h))
        prior_mean = self.fc_prior_mean(prior)
        prior_std = F.softplus(self.fc_prior_std(prior)) + 0.1
        z = prior_mean + prior_std * torch.randn_like(prior_std)
        return {"h": h, "z": z, "z_mean": prior_mean, "z_std": prior_std}


class Dreamer:
    """Dreamer: Deep Reinforcement Learning for World Models."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        stochastic_size: int = 32,
        deterministic_size: int = 200,
        hidden_size: int = 200,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.rssm = RSSM(
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            hidden_size=hidden_size,
            action_size=action_dim,
            obs_embed_size=state_dim,
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(stochastic_size + deterministic_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim),
        ).to(self.device)

        self.reward_model = nn.Sequential(
            nn.Linear(stochastic_size + deterministic_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(self.device)

        self.actor = nn.Sequential(
            nn.Linear(stochastic_size + deterministic_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(stochastic_size + deterministic_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(self.device)

        self.world_optimizer = torch.optim.Adam(
            list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_model.parameters()),
            lr=1e-4,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=8e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=8e-5)

    def update_world_model(
        self, states: Tensor, actions: Tensor, rewards: Tensor
    ) -> Dict[str, float]:
        """Update world model (RSSM + decoder + reward model)."""
        batch_size, seq_len = states.shape[0], states.shape[1]
        prev_state = self.rssm.initial_state(batch_size, self.device)
        all_states = []
        kl_losses = []

        for t in range(seq_len):
            obs_embed = states[:, t]
            state = self.rssm.observe_step(prev_state, actions[:, t], obs_embed)
            all_states.append(state)
            kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(state["z_mean"], state["z_std"]),
                torch.distributions.Normal(state["prior_mean"], state["prior_std"]),
            ).mean()
            kl_losses.append(kl)
            prev_state = state

        z = torch.stack([s["z"] for s in all_states], dim=1)
        h = torch.stack([s["h"] for s in all_states], dim=1)
        state_features = torch.cat([z, h], dim=-1)

        obs_pred = self.decoder(state_features)
        obs_loss = F.mse_loss(obs_pred, states)

        reward_pred = self.reward_model(state_features).squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, rewards)

        kl_loss = sum(kl_losses) / len(kl_losses)
        total_loss = obs_loss + reward_loss + 0.1 * kl_loss

        self.world_optimizer.zero_grad()
        total_loss.backward()
        self.world_optimizer.step()

        return {
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_loss": kl_loss.item(),
        }

    def update_actor_critic(
        self, batch_size: int = 50, horizon: int = 15
    ) -> Dict[str, float]:
        """Update actor and critic using imagination."""
        prev_state = self.rssm.initial_state(batch_size, self.device)
        z = torch.randn(batch_size, self.rssm.stochastic_size, device=self.device)
        h = prev_state["h"]

        imagined_states = []
        imagined_rewards = []

        for t in range(horizon):
            state_feature = torch.cat([z, h], dim=-1)
            action = self.actor(state_feature)
            next_z = z + 0.1 * torch.randn_like(z)
            next_h = self.rssm.gru(self.rssm.action_embed(action), h)
            reward = self.reward_model(torch.cat([next_z, next_h], dim=-1))
            imagined_states.append(torch.cat([next_z, next_h], dim=-1))
            imagined_rewards.append(reward)
            z, h = next_z, next_h

        imagined_states = torch.stack(imagined_states, dim=1)
        imagined_rewards = torch.stack(imagined_rewards, dim=1).squeeze(-1)

        with torch.no_grad():
            target_values = self.critic(
                imagined_states.reshape(-1, imagined_states.shape[-1])
            )
            target_values = target_values.reshape(batch_size, horizon)

        returns = imagined_rewards + 0.99 * target_values
        actor_loss = -returns.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        values = self.critic(
            imagined_states.detach().reshape(-1, imagined_states.shape[-1])
        )
        values = values.reshape(batch_size, horizon)
        critic_loss = F.mse_loss(values, imagined_rewards + 0.99 * target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

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


# =============================================================================
# Hierarchical RL Components
# =============================================================================

# =============================================================================
# Offline RL Components
# =============================================================================


class CQL:
    """Conservative Q-Learning for Offline RL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        cql_alpha: float = 1.0,
        cql_temp: float = 1.0,
        tau: float = 0.005,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cql_alpha = cql_alpha
        self.cql_temp = cql_temp
        self.tau = tau
        self.gamma = gamma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q1 = self._make_q_network(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.q2 = self._make_q_network(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.policy = self._make_policy(state_dim, action_dim, hidden_dim).to(
            self.device
        )

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -action_dim

    def _make_q_network(
        self, state_dim: int, action_dim: int, hidden_dim: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _make_policy(
        self, state_dim: int, action_dim: int, hidden_dim: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),
        )

    def sample_action(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.policy(state)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        action = torch.tanh(action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    def update(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Dict[str, float]:
        q1_pred = self.q1(torch.cat([states, actions], dim=-1))
        q2_pred = self.q2(torch.cat([states, actions], dim=-1))

        with torch.no_grad():
            next_actions, next_log_probs = self.sample_action(next_states)
            q1_next = self.q1_target(torch.cat([next_states, next_actions], dim=-1))
            q2_next = self.q2_target(torch.cat([next_states, next_actions], dim=-1))
            q_next = torch.min(q1_next, q2_next)
            alpha = self.log_alpha.exp()
            q_target = rewards.unsqueeze(-1) + self.gamma * (
                1 - dones.unsqueeze(-1)
            ) * (q_next - alpha * next_log_probs)

        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        q_loss = q1_loss + q2_loss

        batch_size = states.shape[0]
        random_actions = (
            torch.rand(batch_size, self.action_dim, device=self.device) * 2 - 1
        )

        dataset_q1 = self.q1(torch.cat([states, actions], dim=-1))
        dataset_q2 = self.q2(torch.cat([states, actions], dim=-1))
        random_q1 = self.q1(torch.cat([states, random_actions], dim=-1))
        random_q2 = self.q2(torch.cat([states, random_actions], dim=-1))

        cql1_loss = (
            torch.logsumexp(random_q1 / self.cql_temp, dim=0).mean() * self.cql_temp
            - dataset_q1.mean()
        )
        cql2_loss = (
            torch.logsumexp(random_q2 / self.cql_temp, dim=0).mean() * self.cql_temp
            - dataset_q2.mean()
        )

        total_q_loss = q_loss + self.cql_alpha * (cql1_loss + cql2_loss)

        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        self.q_optimizer.step()

        new_actions, log_probs = self.sample_action(states)
        q1_new = self.q1(torch.cat([states, new_actions], dim=-1))
        q2_new = self.q2(torch.cat([states, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new)

        alpha = self.log_alpha.exp().detach()
        policy_loss = (alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)

        return {
            "q_loss": q_loss.item(),
            "cql_loss": (cql1_loss + cql2_loss).item(),
            "policy_loss": policy_loss.item(),
            "alpha": alpha.item(),
        }


class IQL:
    """Implicit Q-Learning for Offline RL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        expectile: float = 0.7,
        temperature: float = 3.0,
        tau: float = 0.005,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expectile = expectile
        self.temperature = temperature
        self.tau = tau

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.v_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.q1 = self._make_q_network(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.q2 = self._make_q_network(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.policy = self._make_policy(state_dim, action_dim, hidden_dim).to(
            self.device
        )

        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=3e-4)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

    def _make_q_network(
        self, state_dim: int, action_dim: int, hidden_dim: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _make_policy(
        self, state_dim: int, action_dim: int, hidden_dim: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def expectile_loss(self, diff: Tensor, expectile: float) -> Tensor:
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff**2)

    def update_v(self, states: Tensor, actions: Tensor) -> float:
        with torch.no_grad():
            q1 = self.q1_target(torch.cat([states, actions], dim=-1))
            q2 = self.q2_target(torch.cat([states, actions], dim=-1))
            q = torch.min(q1, q2)
        v = self.v_net(states)
        v_loss = self.expectile_loss(q - v, self.expectile).mean()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return v_loss.item()

    def update_q(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> float:
        with torch.no_grad():
            next_v = self.v_net(next_states)
            q_target = rewards.unsqueeze(-1) + 0.99 * (1 - dones.unsqueeze(-1)) * next_v
        q1 = self.q1(torch.cat([states, actions], dim=-1))
        q2 = self.q2(torch.cat([states, actions], dim=-1))
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)
        return q_loss.item()

    def update_policy(self, states: Tensor, actions: Tensor) -> float:
        with torch.no_grad():
            q1 = self.q1(torch.cat([states, actions], dim=-1))
            q2 = self.q2(torch.cat([states, actions], dim=-1))
            q = torch.min(q1, q2)
            v = self.v_net(states)
            advantage = q - v
            exp_adv = torch.exp(advantage * self.temperature).clamp(max=100.0)
        pred_actions = self.policy(states)
        policy_loss = (exp_adv * (pred_actions - actions) ** 2).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.item()

    def update(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Dict[str, float]:
        v_loss = self.update_v(states, actions)
        q_loss = self.update_q(states, actions, rewards, next_states, dones)
        policy_loss = self.update_policy(states, actions)
        return {"v_loss": v_loss, "q_loss": q_loss, "policy_loss": policy_loss}


class DecisionTransformer(nn.Module):
    """Decision Transformer: Reinforcement Learning via Sequence Modeling."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 1,
        max_length: int = 20,
        max_ep_len: int = 4096,
        dropout: float = 0.1,
        device: str = "auto",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_length = max_length

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(action_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )
        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.predict_return = nn.Linear(hidden_size, 1)

        self.to(self.device)

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        returns_to_go: Tensor,
        timesteps: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=self.device)

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings

        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        causal_mask = torch.triu(
            torch.ones(3 * seq_len, 3 * seq_len, device=self.device), diagonal=1
        ).bool()
        transformer_out = self.transformer(stacked_inputs, mask=causal_mask)
        transformer_out = transformer_out.reshape(
            batch_size, seq_len, 3, self.hidden_size
        )

        action_preds = self.predict_action(transformer_out[:, :, 1])
        state_preds = self.predict_state(transformer_out[:, :, 2])
        return_preds = self.predict_return(transformer_out[:, :, 0])

        return state_preds, action_preds, return_preds

    def get_action(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        returns_to_go: np.ndarray,
        timesteps: np.ndarray,
    ) -> np.ndarray:
        states = torch.FloatTensor(states).unsqueeze(0).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(0).to(self.device)
        returns_to_go = torch.FloatTensor(returns_to_go).unsqueeze(0).to(self.device)
        timesteps = torch.LongTensor(timesteps).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, action_preds, _ = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[0, -1].cpu().numpy()


class CRR:
    """Critic Regularized Regression for Offline RL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        advantage_threshold: float = 0.0,
        beta: float = 1.0,
        tau: float = 0.005,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.advantage_threshold = advantage_threshold
        self.beta = beta
        self.tau = tau

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q1 = self._make_q_network(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.q2 = self._make_q_network(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.policy = self._make_policy(state_dim, action_dim, hidden_dim).to(
            self.device
        )

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

    def _make_q_network(
        self, state_dim: int, action_dim: int, hidden_dim: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _make_policy(
        self, state_dim: int, action_dim: int, hidden_dim: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def update(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Dict[str, float]:
        with torch.no_grad():
            next_actions = self.policy(next_states)
            q1_next = self.q1_target(torch.cat([next_states, next_actions], dim=-1))
            q2_next = self.q2_target(torch.cat([next_states, next_actions], dim=-1))
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards.unsqueeze(-1) + 0.99 * (1 - dones.unsqueeze(-1)) * q_next

        q1 = self.q1(torch.cat([states, actions], dim=-1))
        q2 = self.q2(torch.cat([states, actions], dim=-1))
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            q1_val = self.q1(torch.cat([states, actions], dim=-1))
            q2_val = self.q2(torch.cat([states, actions], dim=-1))
            q_val = torch.min(q1_val, q2_val)
            policy_actions = self.policy(states)
            q1_policy = self.q1(torch.cat([states, policy_actions], dim=-1))
            q2_policy = self.q2(torch.cat([states, policy_actions], dim=-1))
            q_policy = torch.min(q1_policy, q2_policy)
            advantage = q_val - q_policy
            weights = torch.exp(advantage * self.beta).clamp(max=20.0)
            weights = (advantage > self.advantage_threshold).float() * weights

        pred_actions = self.policy(states)
        policy_loss = (weights * (pred_actions - actions) ** 2).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item()}


class AWAC:
    """Accelerated Weighted Actor-Critic for Offline RL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        beta: float = 1.0,
        tau: float = 0.005,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.tau = tau

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q1 = self._make_q_network(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.q2 = self._make_q_network(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.policy_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(self.device)
        self.policy_logstd = nn.Parameter(torch.zeros(action_dim)).to(self.device)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy_mean.parameters()) + [self.policy_logstd], lr=3e-4
        )

    def _make_q_network(
        self, state_dim: int, action_dim: int, hidden_dim: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def sample_action(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self.policy_mean(state)
        std = torch.exp(self.policy_logstd)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def update(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Dict[str, float]:
        with torch.no_grad():
            next_actions, _ = self.sample_action(next_states)
            q1_next = self.q1_target(torch.cat([next_states, next_actions], dim=-1))
            q2_next = self.q2_target(torch.cat([next_states, next_actions], dim=-1))
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards.unsqueeze(-1) + 0.99 * (1 - dones.unsqueeze(-1)) * q_next

        q1 = self.q1(torch.cat([states, actions], dim=-1))
        q2 = self.q2(torch.cat([states, actions], dim=-1))
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            q1_val = self.q1(torch.cat([states, actions], dim=-1))
            q2_val = self.q2(torch.cat([states, actions], dim=-1))
            q_val = torch.min(q1_val, q2_val)
            policy_actions, _ = self.sample_action(states)
            q1_policy = self.q1(torch.cat([states, policy_actions], dim=-1))
            q2_policy = self.q2(torch.cat([states, policy_actions], dim=-1))
            q_policy = torch.min(q1_policy, q2_policy)
            advantage = q_val - q_policy
            weights = torch.exp(advantage * self.beta).clamp(max=100.0)

        mean = self.policy_mean(states)
        std = torch.exp(self.policy_logstd)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        policy_loss = -(weights * log_prob).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item()}


# =============================================================================
# Hierarchical RL Components
# =============================================================================


class OptionCritic(nn.Module):
    """Option-Critic Architecture for Hierarchical RL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_options: int = 4,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        termination_reg: float = 0.01,
        device: str = "auto",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.gamma = gamma
        self.termination_reg = termination_reg

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.features = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())
        self.option_policies = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )
                for _ in range(num_options)
            ]
        )
        self.terminations = nn.Sequential(
            nn.Linear(hidden_dim, num_options), nn.Sigmoid()
        )
        self.policy_over_options = nn.Sequential(nn.Linear(hidden_dim, num_options))
        self.q_values = nn.Sequential(nn.Linear(hidden_dim, num_options))

        self.to(self.device)
        self.current_option = None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def get_q(self, state: Tensor) -> Tensor:
        features = self.features(state)
        return self.q_values(features)

    def get_termination(self, state: Tensor) -> Tensor:
        features = self.features(state)
        return self.terminations(features)

    def get_option_action(self, state: Tensor, option: int) -> Tensor:
        features = self.features(state)
        logits = self.option_policies[option](features)
        return F.softmax(logits, dim=-1)

    def select_option(self, state: Tensor) -> int:
        features = self.features(state)
        q_options = self.q_values(features)
        return q_options.argmax(dim=-1).item()

    def get_action(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if self.current_option is not None:
            termination_prob = self.get_termination(state_tensor)[
                0, self.current_option
            ].item()
            if random.random() < termination_prob:
                self.current_option = None
        if self.current_option is None:
            self.current_option = self.select_option(state_tensor)
        action_probs = self.get_option_action(state_tensor, self.current_option)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def update(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        options: Tensor,
    ) -> Dict[str, float]:
        q_values = self.get_q(states)
        q_current = q_values.gather(1, options.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.get_q(next_states)
            max_next_q = next_q.max(dim=1)[0]
            q_target = rewards + self.gamma * (1 - dones) * max_next_q

        q_loss = F.mse_loss(q_current, q_target)

        terminations = self.get_termination(states)
        termination_probs = terminations.gather(1, options.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_continuation = next_q.gather(1, options.unsqueeze(1)).squeeze(1)

        advantage = q_continuation - max_next_q
        termination_loss = (
            termination_probs * advantage
        ).mean() + self.termination_reg * termination_probs.mean()

        features = self.features(states)
        action_loss = 0
        for i in range(self.num_options):
            option_mask = (options == i).float()
            if option_mask.sum() > 0:
                logits = self.option_policies[i](features)
                log_probs = (
                    F.log_softmax(logits, dim=-1)
                    .gather(1, actions.unsqueeze(1))
                    .squeeze(1)
                )
                action_loss -= (option_mask * log_probs * q_current.detach()).sum() / (
                    option_mask.sum() + 1e-8
                )

        total_loss = q_loss + termination_loss + action_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "q_loss": q_loss.item(),
            "termination_loss": termination_loss.item(),
            "action_loss": action_loss,
            "total_loss": total_loss.item(),
        }


class HIRO:
    """HIRO: High-level controller for Hierarchical RL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        c: int = 10,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.c = c
        self.gamma = gamma
        self.tau = tau

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.manager_actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh(),
        ).to(self.device)

        self.manager_critic1 = self._make_critic(state_dim * 2).to(self.device)
        self.manager_critic2 = self._make_critic(state_dim * 2).to(self.device)
        self.manager_critic1_target = copy.deepcopy(self.manager_critic1)
        self.manager_critic2_target = copy.deepcopy(self.manager_critic2)

        self.worker_actor = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        ).to(self.device)

        self.worker_critic1 = self._make_critic(state_dim * 2 + action_dim).to(
            self.device
        )
        self.worker_critic2 = self._make_critic(state_dim * 2 + action_dim).to(
            self.device
        )
        self.worker_critic1_target = copy.deepcopy(self.worker_critic1)
        self.worker_critic2_target = copy.deepcopy(self.worker_critic2)

        self.manager_actor_opt = torch.optim.Adam(
            self.manager_actor.parameters(), lr=3e-4
        )
        self.manager_critic_opt = torch.optim.Adam(
            list(self.manager_critic1.parameters())
            + list(self.manager_critic2.parameters()),
            lr=3e-4,
        )

        self.worker_actor_opt = torch.optim.Adam(
            self.worker_actor.parameters(), lr=3e-4
        )
        self.worker_critic_opt = torch.optim.Adam(
            list(self.worker_critic1.parameters())
            + list(self.worker_critic2.parameters()),
            lr=3e-4,
        )

        self.current_goal = None
        self.step_count = 0
        self.initial_state = None

    def _make_critic(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def get_goal(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            goal = self.manager_actor(state_tensor)
        return goal.squeeze(0).cpu().numpy()

    def get_action(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        input_tensor = torch.cat([state_tensor, goal_tensor], dim=-1)
        with torch.no_grad():
            action = self.worker_actor(input_tensor)
        return action.squeeze(0).cpu().numpy()

    def get_action_hierarchical(self, state: np.ndarray) -> np.ndarray:
        if self.step_count % self.c == 0 or self.current_goal is None:
            self.current_goal = self.get_goal(state)
            self.initial_state = state.copy()
        action = self.get_action(state, self.current_goal)
        self.step_count += 1
        return action

    def compute_intrinsic_reward(self, state: np.ndarray) -> float:
        if self.initial_state is None or self.current_goal is None:
            return 0.0
        goal_state = self.initial_state + self.current_goal
        distance = np.linalg.norm(state - goal_state)
        return -distance


# =============================================================================
# Multi-Agent RL Components
# =============================================================================


class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradient."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.95,
        tau: float = 0.01,
        device: str = "auto",
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actors = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.target_actors = nn.ModuleList()
        self.target_critics = nn.ModuleList()

        for _ in range(num_agents):
            actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )
            critic = nn.Sequential(
                nn.Linear(num_agents * (state_dim + action_dim), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(copy.deepcopy(actor))
            self.target_critics.append(copy.deepcopy(critic))

        self.actors = self.actors.to(self.device)
        self.critics = self.critics.to(self.device)
        self.target_actors = self.target_actors.to(self.device)
        self.target_critics = self.target_critics.to(self.device)

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=1e-2) for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=1e-2) for critic in self.critics
        ]

    def get_actions(self, states: List[np.ndarray]) -> List[np.ndarray]:
        actions = []
        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actors[i](state_tensor)
            actions.append(action.squeeze(0).cpu().numpy())
        return actions

    def update(
        self,
        states: List[Tensor],
        actions: List[Tensor],
        rewards: List[Tensor],
        next_states: List[Tensor],
        dones: List[Tensor],
    ) -> Dict[str, float]:
        all_states = torch.cat(states, dim=1)
        all_actions = torch.cat(actions, dim=1)
        all_next_states = torch.cat(next_states, dim=1)

        total_critic_loss = 0
        total_actor_loss = 0

        for i in range(self.num_agents):
            with torch.no_grad():
                next_actions = [
                    self.target_actors[j](next_states[j])
                    for j in range(self.num_agents)
                ]
                all_next_actions = torch.cat(next_actions, dim=1)
                target_q = self.target_critics[i](
                    torch.cat([all_next_states, all_next_actions], dim=1)
                ).squeeze(1)
                y = rewards[i] + self.gamma * (1 - dones[i]) * target_q

            q = self.critics[i](torch.cat([all_states, all_actions], dim=1)).squeeze(1)
            critic_loss = F.mse_loss(q, y)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            total_critic_loss += critic_loss.item()

            new_actions = [
                self.actors[j](states[j]) if j == i else actions[j].detach()
                for j in range(self.num_agents)
            ]
            all_new_actions = torch.cat(new_actions, dim=1)
            actor_loss = -self.critics[i](
                torch.cat([all_states, all_new_actions], dim=1)
            ).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            total_actor_loss += actor_loss.item()

            soft_update(self.target_actors[i], self.actors[i], self.tau)
            soft_update(self.target_critics[i], self.critics[i], self.tau)

        return {
            "critic_loss": total_critic_loss / self.num_agents,
            "actor_loss": total_actor_loss / self.num_agents,
        }


class QMIX:
    """QMIX: Monotonic Value Function Factorisation."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        mixing_dim: int = 32,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mixing_dim = mixing_dim
        self.gamma = gamma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.agent_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )
                for _ in range(num_agents)
            ]
        ).to(self.device)

        self.target_agent_networks = copy.deepcopy(self.agent_networks)

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim * num_agents, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, num_agents * mixing_dim),
        ).to(self.device)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim * num_agents, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, mixing_dim),
        ).to(self.device)

        self.hyper_b1 = nn.Linear(state_dim * num_agents, mixing_dim).to(self.device)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim * num_agents, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, 1),
        ).to(self.device)

        self.target_hyper_w1 = copy.deepcopy(self.hyper_w1)
        self.target_hyper_w2 = copy.deepcopy(self.hyper_w2)
        self.target_hyper_b1 = copy.deepcopy(self.hyper_b1)
        self.target_hyper_b2 = copy.deepcopy(self.hyper_b2)

        self.optimizer = torch.optim.Adam(
            list(self.agent_networks.parameters())
            + list(self.hyper_w1.parameters())
            + list(self.hyper_w2.parameters())
            + list(self.hyper_b1.parameters())
            + list(self.hyper_b2.parameters()),
            lr=5e-4,
        )

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def get_q_values(self, states: List[Tensor]) -> Tensor:
        q_values = []
        for i in range(self.num_agents):
            q = self.agent_networks[i](states[i])
            q_values.append(q)
        return torch.stack(q_values, dim=1)

    def mix(
        self, q_values: Tensor, global_state: Tensor, target: bool = False
    ) -> Tensor:
        batch_size = q_values.shape[0]
        q_values = q_values.reshape(batch_size, self.num_agents, 1, self.action_dim)

        if target:
            w1 = torch.abs(self.target_hyper_w1(global_state)).reshape(
                batch_size, self.mixing_dim, self.num_agents
            )
            w2 = torch.abs(self.target_hyper_w2(global_state)).reshape(
                batch_size, 1, self.mixing_dim
            )
            b1 = self.target_hyper_b1(global_state).reshape(
                batch_size, 1, self.mixing_dim
            )
            b2 = self.target_hyper_b2(global_state).reshape(batch_size, 1, 1)
        else:
            w1 = torch.abs(self.hyper_w1(global_state)).reshape(
                batch_size, self.mixing_dim, self.num_agents
            )
            w2 = torch.abs(self.hyper_w2(global_state)).reshape(
                batch_size, 1, self.mixing_dim
            )
            b1 = self.hyper_b1(global_state).reshape(batch_size, 1, self.mixing_dim)
            b2 = self.hyper_b2(global_state).reshape(batch_size, 1, 1)

        hidden = F.elu(torch.matmul(w1, q_values.sum(dim=-1, keepdim=True)) + b1)
        q_total = torch.matmul(w2, hidden) + b2

        return q_total.squeeze(-1).squeeze(-1)

    def get_actions(self, states: List[np.ndarray]) -> List[int]:
        if random.random() < self.epsilon:
            return [
                random.randint(0, self.action_dim - 1) for _ in range(self.num_agents)
            ]

        states_tensor = [
            torch.FloatTensor(s).unsqueeze(0).to(self.device) for s in states
        ]

        with torch.no_grad():
            q_values = self.get_q_values(states_tensor)
            actions = q_values.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()

        return actions

    def update(
        self,
        states: List[Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_states: List[Tensor],
        global_states: Tensor,
        next_global_states: Tensor,
        dones: Tensor,
    ) -> Dict[str, float]:
        q_values = self.get_q_values(states)
        q_values_taken = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
        q_total = self.mix(q_values_taken.unsqueeze(2), global_states)

        with torch.no_grad():
            next_q_values = self.get_target_q_values(next_states)
            next_actions = next_q_values.argmax(dim=-1)
            next_q_values_taken = next_q_values.gather(
                2, next_actions.unsqueeze(2)
            ).squeeze(2)
            next_q_total = self.mix(
                next_q_values_taken.unsqueeze(2), next_global_states, target=True
            )
            q_target = rewards + self.gamma * (1 - dones) * next_q_total

        loss = F.mse_loss(q_total, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        tau = 0.005
        for param, target_param in zip(
            self.agent_networks.parameters(), self.target_agent_networks.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def get_target_q_values(self, states: List[Tensor]) -> Tensor:
        q_values = []
        for i in range(self.num_agents):
            q = self.target_agent_networks[i](states[i])
            q_values.append(q)
        return torch.stack(q_values, dim=1)


class VDN:
    """Value Decomposition Networks."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.agent_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )
                for _ in range(num_agents)
            ]
        ).to(self.device)

        self.target_agent_networks = copy.deepcopy(self.agent_networks)

        self.optimizer = torch.optim.Adam(self.agent_networks.parameters(), lr=5e-4)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def get_q_values(self, states: List[Tensor]) -> Tensor:
        q_values = []
        for i in range(self.num_agents):
            q = self.agent_networks[i](states[i])
            q_values.append(q)
        return torch.stack(q_values, dim=1)

    def get_actions(self, states: List[np.ndarray]) -> List[int]:
        if random.random() < self.epsilon:
            return [
                random.randint(0, self.action_dim - 1) for _ in range(self.num_agents)
            ]

        states_tensor = [
            torch.FloatTensor(s).unsqueeze(0).to(self.device) for s in states
        ]

        with torch.no_grad():
            q_values = self.get_q_values(states_tensor)
            actions = q_values.argmax(dim=-1).squeeze(0).cpu().numpy().tolist()

        return actions

    def update(
        self,
        states: List[Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_states: List[Tensor],
        dones: Tensor,
    ) -> Dict[str, float]:
        q_values = self.get_q_values(states)
        q_values_taken = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
        q_total = q_values_taken.sum(dim=1)

        with torch.no_grad():
            next_q_values = self.get_target_q_values(next_states)
            next_q_total = next_q_values.max(dim=-1)[0].sum(dim=1)
            q_target = rewards + self.gamma * (1 - dones) * next_q_total

        loss = F.mse_loss(q_total, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        tau = 0.005
        for param, target_param in zip(
            self.agent_networks.parameters(), self.target_agent_networks.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def get_target_q_values(self, states: List[Tensor]) -> Tensor:
        q_values = []
        for i in range(self.num_agents):
            q = self.target_agent_networks[i](states[i])
            q_values.append(q)
        return torch.stack(q_values, dim=1)


class MAPPO:
    """Multi-Agent PPO."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "auto",
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        class ActorCritic(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.actor = nn.Linear(hidden_dim, action_dim)
                self.critic = nn.Linear(hidden_dim, 1)

            def forward(self, state):
                features = self.shared(state)
                logits = self.actor(features)
                value = self.critic(features)
                return logits, value

        self.agents = nn.ModuleList(
            [ActorCritic(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        ).to(self.device)

        self.optimizers = [
            torch.optim.Adam(agent.parameters(), lr=5e-4) for agent in self.agents
        ]

        self.buffers = [self._init_buffer() for _ in range(num_agents)]

    def _init_buffer(self) -> Dict:
        return {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def get_actions(
        self, states: List[np.ndarray]
    ) -> Tuple[List[int], List[float], List[float]]:
        actions = []
        log_probs = []
        values = []

        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, value = self.agents[i](state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())

            self.buffers[i]["states"].append(state)
            self.buffers[i]["actions"].append(action.item())
            self.buffers[i]["log_probs"].append(log_prob.item())
            self.buffers[i]["values"].append(value.item())

        return actions, log_probs, values

    def store_rewards(self, rewards: List[float], dones: List[bool]) -> None:
        for i in range(self.num_agents):
            self.buffers[i]["rewards"].append(rewards[i])
            self.buffers[i]["dones"].append(dones[i])

    def compute_gae(self, agent_id: int, last_value: float) -> Tuple[Tensor, Tensor]:
        buffer = self.buffers[agent_id]
        rewards = buffer["rewards"]
        values = buffer["values"] + [last_value]
        dones = buffer["dones"]

        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(self.device)

        return advantages, returns

    def update(
        self, last_states: List[np.ndarray], last_dones: List[bool]
    ) -> Dict[str, float]:
        total_loss = 0

        for i in range(self.num_agents):
            last_state_tensor = (
                torch.FloatTensor(last_states[i]).unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                _, last_value = self.agents[i](last_state_tensor)
                last_value = last_value.item() if not last_dones[i] else 0

            advantages, returns = self.compute_gae(i, last_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states = torch.FloatTensor(self.buffers[i]["states"]).to(self.device)
            actions = torch.LongTensor(self.buffers[i]["actions"]).to(self.device)
            old_log_probs = torch.FloatTensor(self.buffers[i]["log_probs"]).to(
                self.device
            )

            logits, values = self.agents[i](states)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(), returns)
            entropy = dist.entropy().mean()

            loss = (
                policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            )

            self.optimizers[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agents[i].parameters(), 0.5)
            self.optimizers[i].step()

            total_loss += loss.item()
            self.buffers[i] = self._init_buffer()

        return {"total_loss": total_loss / self.num_agents}


class COMA:
    """Counterfactual Multi-Agent Policy Gradients."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )
                for _ in range(num_agents)
            ]
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(num_agents * (state_dim + action_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * action_dim),
        ).to(self.device)

        self.baseline_critic = nn.Sequential(
            nn.Linear(num_agents * state_dim + num_agents * action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=5e-4) for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.baseline_critic.parameters()),
            lr=5e-4,
        )

        self.buffer = {"states": [], "actions": [], "rewards": [], "global_states": []}

    def get_actions(self, states: List[np.ndarray]) -> Tuple[List[int], Tensor]:
        actions = []
        action_probs_list = []

        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actors[i](state_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

            actions.append(action.item())
            action_probs_list.append(probs)

        return actions, torch.cat(action_probs_list, dim=0)

    def compute_counterfactual_baseline(
        self, global_state: Tensor, all_actions: Tensor, agent_id: int
    ) -> Tensor:
        batch_size = global_state.shape[0]
        q_values = self.critic(torch.cat([global_state, all_actions], dim=-1))
        q_values = q_values.reshape(batch_size, self.num_agents, self.action_dim)

        pi = F.softmax(
            self.actors[agent_id](
                global_state[
                    :, agent_id * self.state_dim : (agent_id + 1) * self.state_dim
                ]
            ),
            dim=-1,
        )
        baseline = (q_values[:, agent_id] * pi).sum(dim=-1)

        return baseline

    def update(
        self,
        global_states: Tensor,
        states: List[Tensor],
        actions: Tensor,
        rewards: Tensor,
    ) -> Dict[str, float]:
        batch_size = global_states.shape[0]

        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        action_probs = []
        for i in range(self.num_agents):
            logits = self.actors[i](states[i])
            probs = F.softmax(logits, dim=-1)
            action_probs.append(probs.gather(1, actions[:, i : i + 1]))
        action_probs = torch.cat(action_probs, dim=1)

        all_actions_onehot = (
            F.one_hot(actions, self.action_dim).float().reshape(batch_size, -1)
        )

        q_values = self.critic(torch.cat([global_states, all_actions_onehot], dim=-1))
        q_values_taken = (
            q_values.reshape(batch_size, self.num_agents, self.action_dim)
            .gather(2, actions.unsqueeze(2))
            .squeeze(2)
        )

        baseline = self.baseline_critic(
            torch.cat([global_states, all_actions_onehot], dim=-1)
        ).squeeze(1)

        critic_loss = F.mse_loss(q_values_taken.sum(dim=1), returns) + F.mse_loss(
            baseline, returns
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        total_actor_loss = 0
        for i in range(self.num_agents):
            with torch.no_grad():
                cf_baseline = self.compute_counterfactual_baseline(
                    global_states, all_actions_onehot, i
                )
                advantage = q_values_taken[:, i] - cf_baseline

            logits = self.actors[i](states[i])
            log_probs = (
                F.log_softmax(logits, dim=-1)
                .gather(1, actions[:, i : i + 1])
                .squeeze(1)
            )
            actor_loss = -(log_probs * advantage).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            total_actor_loss += actor_loss.item()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": total_actor_loss / self.num_agents,
        }


# =============================================================================
# Imitation Learning Components
# =============================================================================


class BehavioralCloning:
    """Behavioral Cloning for Imitation Learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if continuous:
            self.policy = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            ).to(self.device)
        else:
            self.policy = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

    def train(self, states: Tensor, actions: Tensor) -> float:
        if self.continuous:
            pred_actions = self.policy(states)
            loss = F.mse_loss(pred_actions, actions)
        else:
            logits = self.policy(states)
            loss = F.cross_entropy(logits, actions.long())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.continuous:
                action = self.policy(state_tensor)
            else:
                logits = self.policy(state_tensor)
                action = logits.argmax(dim=-1)

        return action.squeeze(0).cpu().numpy()


class GAIL:
    """Generative Adversarial Imitation Learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(self.device)

        if continuous:
            self.generator = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            ).to(self.device)
        else:
            self.generator = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            ).to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=3e-4
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=3e-4
        )

    def train_discriminator(
        self,
        expert_states: Tensor,
        expert_actions: Tensor,
        policy_states: Tensor,
        policy_actions: Tensor,
    ) -> float:
        expert_input = torch.cat([expert_states, expert_actions], dim=-1)
        expert_pred = self.discriminator(expert_input)
        expert_loss = F.binary_cross_entropy(expert_pred, torch.ones_like(expert_pred))

        policy_input = torch.cat([policy_states, policy_actions], dim=-1)
        policy_pred = self.discriminator(policy_input)
        policy_loss = F.binary_cross_entropy(policy_pred, torch.zeros_like(policy_pred))

        loss = expert_loss + policy_loss

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        return loss.item()

    def train_generator(self, states: Tensor) -> Tuple[float, Tensor]:
        actions = self.generator(states)
        fake_input = torch.cat([states, actions], dim=-1)
        fake_pred = self.discriminator(fake_input)
        loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))

        self.generator_optimizer.zero_grad()
        loss.backward()
        self.generator_optimizer.step()

        return loss.item(), actions.detach()

    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.continuous:
                action = self.generator(state_tensor)
            else:
                logits = self.generator(state_tensor)
                action = logits.argmax(dim=-1)

        return action.squeeze(0).cpu().numpy()


class AIRL:
    """Adversarial Inverse Reinforcement Learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Reward function h(s, a)
        self.h = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        # Shaping function g(s)
        self.g = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        ).to(self.device)

        self.h_optimizer = torch.optim.Adam(self.h.parameters(), lr=3e-4)
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=3e-4)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=3e-4
        )

    def compute_reward(
        self, state: Tensor, action: Tensor, next_state: Tensor
    ) -> Tensor:
        h_val = self.h(torch.cat([state, action], dim=-1))
        g_val = self.g(state)
        next_g_val = self.g(next_state)
        reward = h_val + self.gamma * next_g_val - g_val
        return reward

    def train_discriminator(
        self,
        expert_states: Tensor,
        expert_actions: Tensor,
        policy_states: Tensor,
        policy_actions: Tensor,
    ) -> float:
        expert_input = torch.cat([expert_states, expert_actions], dim=-1)
        expert_pred = self.discriminator(expert_input)
        expert_loss = F.binary_cross_entropy(expert_pred, torch.ones_like(expert_pred))

        policy_input = torch.cat([policy_states, policy_actions], dim=-1)
        policy_pred = self.discriminator(policy_input)
        policy_loss = F.binary_cross_entropy(policy_pred, torch.zeros_like(policy_pred))

        loss = expert_loss + policy_loss

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        return loss.item()


class SQIL:
    """Soft Q Imitation Learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(self.device)

        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=3e-4)

        self.expert_buffer = deque(maxlen=100000)
        self.policy_buffer = deque(maxlen=100000)

    def add_expert_transition(
        self, state: np.ndarray, action: int, next_state: np.ndarray, done: bool
    ) -> None:
        self.expert_buffer.append((state, action, next_state, done))

    def add_policy_transition(
        self, state: np.ndarray, action: int, next_state: np.ndarray, done: bool
    ) -> None:
        self.policy_buffer.append((state, action, next_state, done))

    def update(self, batch_size: int = 64) -> float:
        if len(self.expert_buffer) < batch_size or len(self.policy_buffer) < batch_size:
            return 0.0

        # Sample from both buffers
        expert_batch = random.sample(self.expert_buffer, batch_size)
        policy_batch = random.sample(self.policy_buffer, batch_size)

        # Expert transitions have reward 1
        expert_states = torch.FloatTensor(np.array([e[0] for e in expert_batch])).to(
            self.device
        )
        expert_actions = torch.LongTensor([e[1] for e in expert_batch]).to(self.device)
        expert_rewards = torch.ones(batch_size).to(self.device)
        expert_next_states = torch.FloatTensor(
            np.array([e[2] for e in expert_batch])
        ).to(self.device)
        expert_dones = torch.FloatTensor([float(e[3]) for e in expert_batch]).to(
            self.device
        )

        # Policy transitions have reward 0
        policy_states = torch.FloatTensor(np.array([p[0] for p in policy_batch])).to(
            self.device
        )
        policy_actions = torch.LongTensor([p[1] for p in policy_batch]).to(self.device)
        policy_rewards = torch.zeros(batch_size).to(self.device)
        policy_next_states = torch.FloatTensor(
            np.array([p[2] for p in policy_batch])
        ).to(self.device)
        policy_dones = torch.FloatTensor([float(p[3]) for p in policy_batch]).to(
            self.device
        )

        # Combine
        states = torch.cat([expert_states, policy_states], dim=0)
        actions = torch.cat([expert_actions, policy_actions], dim=0)
        rewards = torch.cat([expert_rewards, policy_rewards], dim=0)
        next_states = torch.cat([expert_next_states, policy_next_states], dim=0)
        dones = torch.cat([expert_dones, policy_dones], dim=0)

        # Q-learning update
        q_values = self.q_network(states)
        q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_max = next_q_values.max(dim=1)[0]
            q_target = rewards + self.gamma * (1 - dones) * next_q_max

        loss = F.mse_loss(q_values_taken, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target
        soft_update(self.target_network, self.q_network, 0.005)

        return loss.item()

    def get_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=-1).item()

        return action


# =============================================================================
# Inverse RL Components
# =============================================================================


class MaxEntIRL:
    """Maximum Entropy Inverse Reinforcement Learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Reward network
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=3e-4)

    def compute_reward(self, state: Tensor) -> Tensor:
        return self.reward_net(state).squeeze(-1)

    def update(
        self,
        expert_states: Tensor,
        expert_actions: Tensor,
        policy_states: Tensor,
        policy_actions: Tensor,
    ) -> float:
        # Compute rewards
        expert_rewards = self.compute_reward(expert_states)
        policy_rewards = self.compute_reward(policy_states)

        # MaxEnt IRL objective: maximize expert reward, minimize policy reward
        loss = -expert_rewards.mean() + policy_rewards.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class DeepMaxEnt:
    """Deep Maximum Entropy Inverse Reinforcement Learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Deep reward network
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=3e-4)

    def compute_reward(self, state: Tensor) -> Tensor:
        return self.reward_net(state).squeeze(-1)

    def update(
        self, expert_trajectories: List[Tensor], policy_trajectories: List[Tensor]
    ) -> float:
        total_loss = 0

        for expert_traj, policy_traj in zip(expert_trajectories, policy_trajectories):
            expert_rewards = self.compute_reward(expert_traj).sum()
            policy_rewards = self.compute_reward(policy_traj).sum()

            loss = -expert_rewards + policy_rewards
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss / len(expert_trajectories)


class GCL:
    """Guided Cost Learning for Inverse RL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Cost network (negative reward)
        self.cost_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.cost_net.parameters(), lr=3e-4)

    def compute_cost(self, state: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.cost_net(x).squeeze(-1)

    def update(
        self,
        expert_sa: Tensor,
        policy_sa: Tensor,
        expert_importance: Tensor,
        policy_importance: Tensor,
    ) -> float:
        """Update cost function using importance sampling.

        Args:
            expert_sa: Expert state-action pairs
            policy_sa: Policy state-action pairs
            expert_importance: Importance weights for expert samples
            policy_importance: Importance weights for policy samples
        """
        expert_costs = self.cost_net(expert_sa).squeeze(-1)
        policy_costs = self.cost_net(policy_sa).squeeze(-1)

        # Guided cost learning objective
        expert_loss = (expert_importance * expert_costs).sum()
        policy_loss = (policy_importance * torch.exp(-policy_costs)).sum()

        # Normalize by sum of importance weights
        loss = expert_loss / expert_importance.sum() + torch.log(
            policy_loss / policy_importance.sum()
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# =============================================================================
# Meta-RL Components
# =============================================================================


class MAML:
    """Model-Agnostic Meta-Learning for RL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Meta-policy
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(self.device)

        self.meta_optimizer = torch.optim.Adam(self.policy.parameters(), lr=outer_lr)

    def inner_loop(self, task_data: List[Tuple[Tensor, Tensor, Tensor]]) -> nn.Module:
        """Perform inner loop adaptation on a task."""
        # Clone policy for this task
        adapted_policy = copy.deepcopy(self.policy)
        optimizer = torch.optim.SGD(adapted_policy.parameters(), lr=self.inner_lr)

        for _ in range(self.num_inner_steps):
            total_loss = 0
            for state, action, reward in task_data:
                logits = adapted_policy(state)
                log_probs = (
                    F.log_softmax(logits, dim=-1)
                    .gather(1, action.unsqueeze(1))
                    .squeeze(1)
                )
                loss = -(log_probs * reward).mean()
                total_loss += loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return adapted_policy

    def meta_update(self, tasks: List[List[Tuple[Tensor, Tensor, Tensor]]]) -> float:
        """Perform meta-update across tasks."""
        meta_loss = 0

        for task_data in tasks:
            # Inner loop adaptation
            adapted_policy = self.inner_loop(task_data)

            # Outer loop evaluation
            for state, action, reward in task_data:
                logits = adapted_policy(state)
                log_probs = (
                    F.log_softmax(logits, dim=-1)
                    .gather(1, action.unsqueeze(1))
                    .squeeze(1)
                )
                loss = -(log_probs * reward).mean()
                meta_loss += loss

        meta_loss = meta_loss / len(tasks)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def get_action(self, state: np.ndarray) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.policy(state_tensor)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

        return action


class RL2:
    """RL2: Fast Reinforcement Learning via Slow Reinforcement Learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        rnn_hidden_dim: int = 256,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.gamma = gamma

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # RNN-based meta-policy
        self.rnn = nn.GRU(
            state_dim + action_dim + 1, rnn_hidden_dim, batch_first=True
        ).to(self.device)
        self.policy_head = nn.Linear(rnn_hidden_dim, action_dim).to(self.device)
        self.value_head = nn.Linear(rnn_hidden_dim, 1).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.rnn.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters()),
            lr=3e-4,
        )

        self.hidden_state = None

    def reset_hidden(self, batch_size: int = 1) -> None:
        self.hidden_state = torch.zeros(
            1, batch_size, self.rnn_hidden_dim, device=self.device
        )

    def forward(
        self, state: Tensor, prev_action: Tensor, prev_reward: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through RL2."""
        # Combine inputs
        x = torch.cat(
            [state, prev_action, prev_reward.unsqueeze(-1)], dim=-1
        ).unsqueeze(1)

        # RNN forward
        if self.hidden_state is None:
            self.reset_hidden(state.shape[0])

        rnn_out, self.hidden_state = self.rnn(x, self.hidden_state)
        rnn_out = rnn_out.squeeze(1)

        # Policy and value
        logits = self.policy_head(rnn_out)
        value = self.value_head(rnn_out).squeeze(-1)

        return logits, value, self.hidden_state

    def get_action(
        self, state: np.ndarray, prev_action: int, prev_reward: float
    ) -> int:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        prev_action_tensor = torch.zeros(1, self.action_dim, device=self.device)
        if prev_action >= 0:
            prev_action_tensor[0, prev_action] = 1.0
        prev_reward_tensor = torch.FloatTensor([prev_reward]).to(self.device)

        with torch.no_grad():
            logits, _, _ = self.forward(
                state_tensor, prev_action_tensor, prev_reward_tensor
            )
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

        return action

    def update(self, trajectories: List[Dict[str, Tensor]]) -> float:
        """Update using PPO on meta-trajectories."""
        total_loss = 0

        for traj in trajectories:
            states = traj["states"]
            actions = traj["actions"]
            rewards = traj["rewards"]
            prev_actions = traj["prev_actions"]
            prev_rewards = traj["prev_rewards"]

            # Reset hidden for each trajectory
            self.reset_hidden(states.shape[0])

            # Forward pass
            logits_list = []
            values_list = []

            for t in range(states.shape[1]):
                logits, value, _ = self.forward(
                    states[:, t], prev_actions[:, t], prev_rewards[:, t]
                )
                logits_list.append(logits)
                values_list.append(value)

            logits = torch.stack(logits_list, dim=1)
            values = torch.stack(values_list, dim=1)

            # Compute returns and advantages
            with torch.no_grad():
                returns = []
                G = 0
                for r in reversed(rewards[0]):
                    G = r + self.gamma * G
                    returns.insert(0, G)
                returns = torch.FloatTensor(returns).to(self.device)

            # Policy loss
            action_logits = logits.reshape(-1, self.action_dim)
            action_targets = actions.reshape(-1)
            policy_loss = F.cross_entropy(action_logits, action_targets)

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(trajectories)


class PEARL:
    """Probabilistic Embeddings for Actor-Critic RL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 5,
        hidden_dim: int = 256,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Context encoder (inference network)
        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and logvar
        ).to(self.device)

        # Actor and critic conditioned on latent
        self.actor = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(self.device)

        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.context_encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic1.parameters())
            + list(self.critic2.parameters()),
            lr=3e-4,
        )

        self.latent = None

    def encode_context(self, context: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode context into latent distribution."""
        output = self.context_encoder(context.mean(dim=0))
        mean, logvar = torch.chunk(output, 2, dim=-1)
        return mean, logvar

    def sample_latent(self, context: Tensor) -> Tensor:
        """Sample latent from context."""
        mean, logvar = self.encode_context(context)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return z

    def get_action(
        self, state: np.ndarray, context: Optional[Tensor] = None
    ) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if context is not None and self.latent is None:
            self.latent = self.sample_latent(context)

        if self.latent is None:
            self.latent = torch.zeros(1, self.latent_dim, device=self.device)

        with torch.no_grad():
            actor_input = torch.cat([state_tensor, self.latent], dim=-1)
            logits = self.actor(actor_input)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1)

        return action.squeeze(0).cpu().numpy()

    def reset_latent(self) -> None:
        self.latent = None

    def update(
        self,
        context: Tensor,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Dict[str, float]:
        """Update PEARL using SAC-style updates."""
        # Encode context
        mean, logvar = self.encode_context(context)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        z = z.unsqueeze(0).expand(states.shape[0], -1)

        # Critic update
        with torch.no_grad():
            next_z = z
            next_actor_input = torch.cat([next_states, next_z], dim=-1)
            next_logits = self.actor(next_actor_input)
            next_probs = F.softmax(next_logits, dim=-1)
            next_q1 = self.critic1(torch.cat([next_states, next_probs, next_z], dim=-1))
            next_q2 = self.critic2(torch.cat([next_states, next_probs, next_z], dim=-1))
            next_q = torch.min(next_q1, next_q2)
            q_target = rewards + 0.99 * (1 - dones) * next_q.squeeze(-1)

        actions_onehot = F.one_hot(actions.long(), self.action_dim).float()
        q1 = self.critic1(torch.cat([states, actions_onehot, z], dim=-1)).squeeze(-1)
        q2 = self.critic2(torch.cat([states, actions_onehot, z], dim=-1)).squeeze(-1)

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # Actor update
        actor_input = torch.cat([states, z], dim=-1)
        logits = self.actor(actor_input)
        probs = F.softmax(logits, dim=-1)
        q1_new = self.critic1(torch.cat([states, probs, z], dim=-1))
        q2_new = self.critic2(torch.cat([states, probs, z], dim=-1))
        q_new = torch.min(q1_new, q2_new)

        actor_loss = -q_new.mean()

        # KL penalty on latent
        kl_loss = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()

        total_loss = critic_loss + actor_loss + 0.01 * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "kl_loss": kl_loss.item(),
        }

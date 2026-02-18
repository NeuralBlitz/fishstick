"""
Model-Based RL: MBPO (Model-Based RL with Policy Optimization).

Implements ensemble-based dynamics models for model-based RL with
policy optimization on imagined rollouts.

Reference:
    Janner et al. "When to Trust Your Model: Model-Based Policy Optimization" (2019)
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
    SAC,
    SACConfig,
)

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class MBPOConfig:
    """Configuration for MBPO agent.

    Attributes:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dims: Hidden layer dimensions for networks.
        num_ensemble: Number of models in ensemble.
        num_elites: Number of elite models to use.
        rollout_batch_size: Batch size for model rollouts.
        rollout_length: Length of model rollouts.
        model_lr: Learning rate for dynamics model.
        actor_lr: Learning rate for actor.
        critic_lr: Learning rate for critic.
        gamma: Discount factor.
        tau: Soft update parameter.
        device: Device to use.
    """

    state_dim: int = 0
    action_dim: int = 0
    hidden_dims: List[int] = field(default_factory=lambda: [200, 200, 200])
    num_ensemble: int = 7
    num_elites: int = 5
    rollout_batch_size: int = 100000
    rollout_length: int = 5
    model_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    device: str = "cuda"
    model_loss: str = "mse"


class EnsembleDynamicsModel(Module):
    """Ensemble of Probabilistic Dynamics Models.

    Uses an ensemble of neural networks to model environment dynamics
    with uncertainty quantification.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        num_ensemble: int = 7,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_ensemble = num_ensemble
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.models = nn.ModuleList()
        for _ in range(num_ensemble):
            model = self._build_model(
                state_dim, action_dim, hidden_dims or [200, 200, 200]
            )
            self.models.append(model)

        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=3e-4) for model in self.models
        ]

    def _build_model(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
    ) -> nn.Sequential:
        """Build a single dynamics model."""
        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, state_dim * 2))

        return nn.Sequential(*layers)

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass through ensemble.

        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
            deterministic: If True, return mean without uncertainty

        Returns:
            Tuple of (next_state_mean, next_state_std)
        """
        state_action = torch.cat([state, action], dim=-1)

        if deterministic:
            outputs = []
            for model in self.models[: self.num_ensemble]:
                output = model(state_action)
                outputs.append(output)
            output = torch.stack(outputs).mean(dim=0)
            mean, log_std = output.chunk(2, dim=-1)
            return mean, torch.zeros_like(mean)

        outputs = []
        for model in self.models:
            output = model(state_action)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        mean = outputs.mean(dim=0)
        log_std = outputs.std(dim=0)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def predict(
        self,
        state: Tensor,
        action: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """Predict next state with ensemble.

        Args:
            state: State tensor
            action: Action tensor
            deterministic: If True, use mean prediction

        Returns:
            Predicted next states
        """
        mean, log_std = self.forward(state, action, deterministic)

        if deterministic:
            return mean

        std = torch.exp(log_std)
        dist = Normal(mean, std)
        next_state = dist.rsample()

        return next_state

    def get_error(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Tensor:
        """Calculate prediction error for model selection.

        Args:
            state: Current states
            action: Actions taken
            next_state: Resulting next states

        Returns:
            Mean squared error for each model
        """
        state_action = torch.cat([state, action], dim=-1)
        errors = []

        for model in self.models:
            pred = model(state_action)
            pred_next_state, _ = pred.chunk(2, dim=-1)
            error = F.mse_loss(pred_next_state, next_state, reduction="none").mean(
                dim=-1
            )
            errors.append(error)

        return torch.stack(errors, dim=0)

    def update(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Dict[str, float]:
        """Update ensemble dynamics models.

        Args:
            state: States
            action: Actions
            next_state: Next states

        Returns:
            Dictionary of losses
        """
        state_action = torch.cat([state, action], dim=-1)
        target = next_state

        losses = []
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            optimizer.zero_grad()

            output = model(state_action)
            mean, log_std = output.chunk(2, dim=-1)

            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)

            dist = Normal(mean, std)
            log_prob = dist.log_prob(target).sum(dim=-1)
            loss = -log_prob.mean()

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return {
            "model_loss": sum(losses) / len(losses),
            "model_losses": losses,
        }

    def sample_elites(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> List[int]:
        """Select elite models based on validation loss.

        Args:
            state: Validation states
            action: Validation actions
            next_state: Validation next states

        Returns:
            Indices of elite models
        """
        errors = self.get_error(state, action, next_state)
        elite_indices = torch.argsort(errors)[: self.num_elites]
        return elite_indices.tolist()


class RolloutWorker:
    """Worker for generating rollouts from dynamics model.

    Generates imagined trajectories using the learned dynamics model
    for policy training.
    """

    def __init__(
        self,
        model: EnsembleDynamicsModel,
        policy: Module,
        env: Any,
        rollout_length: int = 5,
        num_rollouts: int = 10000,
    ):
        self.model = model
        self.policy = policy
        self.env = env
        self.rollout_length = rollout_length
        self.num_rollouts = num_rollouts

    def generate_rollouts(
        self,
        initial_states: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate rollouts using the model.

        Args:
            initial_states: Starting states for rollouts

        Returns:
            Tuple of (states, actions, next_states, rewards)
        """
        states_list = []
        actions_list = []
        next_states_list = []
        rewards_list = []

        batch_size = initial_states.shape[0]
        state = initial_states

        for _ in range(self.rollout_length):
            with torch.no_grad():
                action, _ = self.policy.get_action(state)

            next_state = self.model.predict(state, action)

            reward = torch.zeros(next_state.shape[0], device=next_state.device)

            states_list.append(state)
            actions_list.append(action)
            next_states_list.append(next_state)
            rewards_list.append(reward)

            state = next_state

        return (
            torch.cat(states_list, dim=0),
            torch.cat(actions_list, dim=0),
            torch.cat(next_states_list, dim=0),
            torch.cat(rewards_list, dim=0),
        )


class ModelBasedSampler:
    """Sampler for model-based RL.

    Combines real environment interactions with model rollouts.
    """

    def __init__(
        self,
        model: EnsembleDynamicsModel,
        policy: Module,
        env: Any,
        state_dim: int,
        action_dim: int,
        buffer_size: int = 100000,
        rollout_length: int = 5,
        rollout_batch_size: int = 1000,
    ):
        self.model = model
        self.policy = policy
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rollout_length = rollout_length
        self.rollout_batch_size = rollout_batch_size

        self.replay_buffer = {
            "state": [],
            "action": [],
            "next_state": [],
            "reward": [],
            "done": [],
        }
        self.buffer_size = buffer_size

    def add_to_buffer(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward: Tensor,
        done: Tensor,
    ):
        """Add transition to replay buffer."""
        self.replay_buffer["state"].append(state)
        self.replay_buffer["action"].append(action)
        self.replay_buffer["next_state"].append(next_state)
        self.replay_buffer["reward"].append(reward)
        self.replay_buffer["done"].append(done)

        total_size = len(self.replay_buffer["state"])
        if total_size > self.buffer_size:
            for key in self.replay_buffer:
                self.replay_buffer[key] = self.replay_buffer[key][-self.buffer_size :]

    def sample_real_data(self, batch_size: int) -> Dict[str, Tensor]:
        """Sample from real experience buffer."""
        total_size = len(self.replay_buffer["state"])
        if total_size < batch_size:
            return None

        indices = torch.randint(0, total_size, (batch_size,))

        return {
            "state": torch.stack([self.replay_buffer["state"][i] for i in indices]),
            "action": torch.stack([self.replay_buffer["action"][i] for i in indices]),
            "next_state": torch.stack(
                [self.replay_buffer["next_state"][i] for i in indices]
            ),
            "reward": torch.stack([self.replay_buffer["reward"][i] for i in indices]),
            "done": torch.stack([self.replay_buffer["done"][i] for i in indices]),
        }

    def sample_imagined_data(
        self,
        initial_states: Tensor,
    ) -> Optional[Dict[str, Tensor]]:
        """Generate and sample imagined rollouts."""
        if initial_states.shape[0] < self.rollout_batch_size:
            return None

        indices = torch.randint(0, initial_states.shape[0], (self.rollout_batch_size,))
        init_states = initial_states[indices]

        states_list = []
        actions_list = []
        next_states_list = []
        rewards_list = []
        dones_list = []

        state = init_states

        for _ in range(self.rollout_length):
            with torch.no_grad():
                action, _ = self.policy.get_action(state)

            next_state = self.model.predict(state, action)
            reward = torch.zeros(next_state.shape[0], device=next_state.device)
            done = torch.zeros(next_state.shape[0], device=next_state.device)

            states_list.append(state)
            actions_list.append(action)
            next_states_list.append(next_state)
            rewards_list.append(reward)
            dones_list.append(done)

            state = next_state

        return {
            "state": torch.cat(states_list, dim=0),
            "action": torch.cat(actions_list, dim=0),
            "next_state": torch.cat(next_states_list, dim=0),
            "reward": torch.cat(rewards_list, dim=0),
            "done": torch.cat(dones_list, dim=0),
        }


class MBPO:
    """Model-Based RL with Policy Optimization (MBPO).

    Implements the MBPO algorithm combining:
    - Ensemble dynamics models
    - Model rollouts for policy training
    - SAC for policy optimization

    Reference:
        Janner et al. "When to Trust Your Model: Model-Based Policy Optimization" (2019)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[MBPOConfig] = None,
    ):
        self.config = config or MBPOConfig(
            state_dim=state_dim,
            action_dim=action_dim,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dynamics_model = EnsembleDynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
            num_ensemble=self.config.num_ensemble,
        )

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

        self.sampler = ModelBasedSampler(
            model=self.dynamics_model,
            policy=self.policy,
            env=None,
            state_dim=state_dim,
            action_dim=action_dim,
            rollout_length=self.config.rollout_length,
            rollout_batch_size=self.config.rollout_batch_size,
        )

    def get_action(
        self,
        state: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """Get action from policy."""
        action, _ = self.policy.get_action(state, deterministic)
        return action

    def update_models(
        self,
        real_batch: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """Update dynamics models with real data."""
        state = real_batch["state"]
        action = real_batch["action"]
        next_state = real_batch["next_state"]

        loss_dict = self.dynamics_model.update(state, action, next_state)

        return loss_dict

    def update_policy(
        self,
        batch: Dict[str, Tensor],
        imaginary_weight: float = 1.0,
    ) -> Dict[str, float]:
        """Update policy with combined real and imaginary data."""
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            next_action, next_log_pi = self.policy.get_action(next_state)
            target_q1 = self.target_q1(next_state, next_action)
            target_q2 = self.target_q2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.config.gamma * (
                target_q - next_log_pi * imaginary_weight
            )

        q1 = self.q_network1(state, action)
        q2 = self.q_network2(state, action)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

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

        policy_loss = -q_new.mean() + log_pi.mean() * imaginary_weight

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._soft_update_target()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
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

    def update(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> Dict[str, float]:
        """Full update step.

        Args:
            state: States from environment
            action: Actions taken
            reward: Rewards received
            next_state: Next states
            done: Done flags

        Returns:
            Dictionary of losses
        """
        self.sampler.add_to_buffer(state, action, next_state, reward, done)

        model_loss_dict = self.update_models(
            {"state": state, "action": action, "next_state": next_state}
        )

        real_batch = self.sampler.sample_real_data(self.config.rollout_batch_size)

        if real_batch is not None:
            policy_loss_dict = self.update_policy(real_batch)

            init_states = next_state[: min(100, next_state.shape[0])]
            imaginary_batch = self.sampler.sample_imagined_data(init_states)

            if imaginary_batch is not None:
                imag_policy_loss_dict = self.update_policy(
                    imaginary_batch, imaginary_weight=0.1
                )

                return {
                    **model_loss_dict,
                    **policy_loss_dict,
                    "imag_q1_loss": imag_policy_loss_dict["q1_loss"],
                    "imag_q2_loss": imag_policy_loss_dict["q2_loss"],
                    "imag_policy_loss": imag_policy_loss_dict["policy_loss"],
                }

            return {**model_loss_dict, **policy_loss_dict}

        return model_loss_dict

    def to(self, device: str) -> "MBPO":
        """Move to device."""
        self.dynamics_model = self.dynamics_model.to(device)
        self.policy = self.policy.to(device)
        self.q_network1 = self.q_network1.to(device)
        self.q_network2 = self.q_network2.to(device)
        self.target_q1 = self.target_q1.to(device)
        self.target_q2 = self.target_q2.to(device)
        return self

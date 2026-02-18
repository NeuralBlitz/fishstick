"""Exploration strategies for reinforcement learning."""

import torch
import numpy as np
from typing import Optional, Callable
from abc import ABC, abstractmethod


class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""

    @abstractmethod
    def get_action(self, q_values: torch.Tensor, step: int) -> torch.Tensor:
        """Get action with exploration."""
        pass

    @abstractmethod
    def reset(self):
        """Reset exploration state."""
        pass


class EpsilonGreedy(ExplorationStrategy):
    """Epsilon-greedy exploration with optional decay."""

    def __init__(
        self,
        n_actions: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device

    def get_action(self, q_values: torch.Tensor, step: int = 0) -> torch.Tensor:
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(
                0, self.n_actions, (q_values.size(0),), device=self.device
            )
        return q_values.argmax(dim=-1)

    def reset(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def step(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class GaussianNoise(ExplorationStrategy):
    """Gaussian noise exploration for continuous action spaces."""

    def __init__(
        self,
        action_dim: int,
        sigma_start: float = 1.0,
        sigma_end: float = 0.01,
        sigma_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.sigma = sigma_start
        self.sigma_end = sigma_end
        self.sigma_decay = sigma_decay
        self.device = device

    def get_action(self, action: torch.Tensor, step: int = 0) -> torch.Tensor:
        noise = torch.randn_like(action) * self.sigma
        return (action + noise).clamp(-1, 1)

    def reset(self):
        self.sigma = max(self.sigma_end, self.sigma * self.sigma_decay)

    def step(self):
        self.sigma = max(self.sigma_end, self.sigma * self.sigma_decay)


class OrnsteinUhlenbeckNoise(ExplorationStrategy):
    """Ornstein-Uhlenbeck process for continuous control."""

    def __init__(
        self,
        action_dim: int,
        theta: float = 0.15,
        sigma: float = 0.3,
        sigma_end: float = 0.01,
        sigma_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.sigma_end = sigma_end
        self.sigma_decay = sigma_decay
        self.device = device
        self.state = torch.zeros(action_dim, device=device)

    def get_action(self, action: torch.Tensor, step: int = 0) -> torch.Tensor:
        noise = torch.randn(self.action_dim, device=self.device) * self.sigma
        self.state = self.state - self.theta * self.state + noise
        return (action + self.state).clamp(-1, 1)

    def reset(self):
        self.state = torch.zeros(self.action_dim, device=self.device)
        self.sigma = max(self.sigma_end, self.sigma * self.sigma_decay)

    def step(self):
        self.sigma = max(self.sigma_end, self.sigma * self.sigma_decay)


class BoltzmannExploration(ExplorationStrategy):
    """Boltzmann (softmax) exploration."""

    def __init__(
        self,
        temperature: float = 1.0,
        temperature_decay: float = 0.99,
        temperature_min: float = 0.1,
        device: str = "cpu",
    ):
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min
        self.device = device

    def get_action(self, q_values: torch.Tensor, step: int = 0) -> torch.Tensor:
        probs = torch.softmax(q_values / self.temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def reset(self):
        self.temperature = max(
            self.temperature_min, self.temperature * self.temperature_decay
        )

    def step(self):
        self.temperature = max(
            self.temperature_min, self.temperature * self.temperature_decay
        )


class EntropyRegularization:
    """Entropy regularization for policy improvement."""

    @staticmethod
    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of policy distribution."""
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

    @staticmethod
    def compute_bonus(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute entropy bonus for taken actions."""
        entropy = EntropyRegularization.compute_entropy(logits)
        return entropy.gather(1, action.unsqueeze(-1)).squeeze(-1)


class ParameterSpaceNoise:
    """Parameter space noise for exploration (similar to OpenAI's PPO)."""

    def __init__(
        self,
        model: torch.nn.Module,
        std_init: float = 0.1,
        adapt_scale: float = 0.02,
        device: str = "cpu",
    ):
        self.model = model
        self.std_init = std_init
        self.adapt_scale = adapt_scale
        self.device = device
        self.noise_std = std_init
        self.adaptive_std = True
        self._original_params = None
        self._noisy_params = None

    def add_noise(self):
        """Add noise to model parameters."""
        self._original_params = {}
        for name, param in self.model.named_parameters():
            self._original_params[name] = param.data.clone()
            noise = torch.randn_like(param) * self.noise_std
            param.data.add_(noise)

    def remove_noise(self):
        """Remove noise from model parameters."""
        if self._original_params is not None:
            for name, param in self.model.named_parameters():
                param.data.copy_(self._original_params[name])

    def adapt(self, noise_to_action_distance: float):
        """Adapt noise magnitude based on distance."""
        target_distance = self.adapt_scale
        if noise_to_action_distance > target_distance:
            self.noise_std *= 0.9
        else:
            self.noise_std *= 1.1
        self.noise_std = max(0.01, min(1.0, self.noise_std))


class CuriosityDrivenExploration:
    """Intrinsic curiosity module for exploration."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        reward_scale: float = 0.01,
        device: str = "cpu",
    ):
        self.reward_scale = reward_scale
        self.device = device

        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feature_dim),
        ).to(device)

        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feature_dim),
        ).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.forward_model.parameters())
            + list(self.inverse_model.parameters()),
            lr=learning_rate,
        )

    def compute_intrinsic_reward(
        self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Compute curiosity reward."""
        state_action = torch.cat([state, action], dim=-1)
        predicted_next_state = self.forward_model(state_action)

        predicted_action = self.inverse_model(torch.cat([state, next_state], dim=-1))

        forward_loss = (predicted_next_state - next_state.detach()).pow(2).mean()
        inverse_loss = (predicted_action - action.detach()).pow(2).mean()

        loss = forward_loss + inverse_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        intrinsic_reward = forward_loss.detach() * self.reward_scale
        return intrinsic_reward


def create_exploration_strategy(strategy: str, **kwargs) -> ExplorationStrategy:
    """Factory function to create exploration strategies."""
    strategies = {
        "epsilon_greedy": EpsilonGreedy,
        "gaussian": GaussianNoise,
        "ou_noise": OrnsteinUhlenbeckNoise,
        "boltzmann": BoltzmannExploration,
    }
    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}"
        )
    return strategies[strategy](**kwargs)

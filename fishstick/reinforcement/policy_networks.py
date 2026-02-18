"""
Policy Networks for Reinforcement Learning.

This module provides various policy network architectures for
discrete and continuous action spaces, including categorical,
gaussian, and deterministic policies with action scaling.

Example:
    >>> from fishstick.reinforcement import CategoricalPolicy, GaussianPolicy
    >>> policy = CategoricalPolicy(state_dim=4, action_dim=2)
    >>> output = policy(torch.randn(1, 4))
    >>> action = output.sample()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Categorical,
    Normal,
    Independent,
    TanhTransform,
    TransformedDistribution,
    MultivariateNormal,
)

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class PolicyOutput:
    """Output from a policy network.

    Attributes:
        action: Sampled action(s).
        log_prob: Log probability of sampled action(s).
        entropy: Entropy of the action distribution.
        distribution: The underlying distribution object.
        mean: Mean of the distribution (for continuous policies).
        std: Standard deviation (for continuous policies).
    """

    action: Tensor
    log_prob: Tensor
    entropy: Tensor
    distribution: Optional[object] = None
    mean: Optional[Tensor] = None
    std: Optional[Tensor] = None

    def detach(self) -> PolicyOutput:
        """Detach all tensors from computation graph."""
        return PolicyOutput(
            action=self.action.detach(),
            log_prob=self.log_prob.detach(),
            entropy=self.entropy.detach(),
            distribution=None,
            mean=self.mean.detach() if self.mean is not None else None,
            std=self.std.detach() if self.std is not None else None,
        )


class PolicyNetwork(Module, ABC):
    """Base class for policy networks.

    All policy networks should inherit from this class and implement
    the forward method.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name ('relu', 'tanh', 'elu', 'gelu').
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [64, 64]

        self.activation_fn = self._get_activation(activation)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "leaky_relu": nn.LeakyReLU,
        }
        return activations.get(name, nn.ReLU)()

    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """Build MLP network."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    self.activation_fn,
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    @abstractmethod
    def forward(self, state: Tensor) -> PolicyOutput:
        """Forward pass through the policy.

        Args:
            state: State tensor of shape (batch_size, state_dim).

        Returns:
            PolicyOutput containing action, log_prob, entropy, etc.
        """
        pass

    @abstractmethod
    def evaluate_actions(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate log probability and entropy of given actions.

        Args:
            state: State tensor.
            action: Action tensor to evaluate.

        Returns:
            Tuple of (log_prob, entropy).
        """
        pass

    def get_action(self, state: Tensor, deterministic: bool = False) -> Tensor:
        """Get action for given state.

        Args:
            state: State tensor.
            deterministic: If True, return deterministic action.

        Returns:
            Action tensor.
        """
        output = self.forward(state)
        if deterministic:
            if output.mean is not None:
                return output.mean
            return output.action
        return output.action


class CategoricalPolicy(PolicyNetwork):
    """Categorical policy for discrete action spaces.

    Implements a policy that outputs a categorical distribution
    over discrete actions.

    Args:
        state_dim: Dimension of state space.
        action_dim: Number of discrete actions.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.
        action_mask: Optional mask for invalid actions.

    Example:
        >>> policy = CategoricalPolicy(state_dim=4, action_dim=3)
        >>> output = policy(torch.randn(2, 4))
        >>> print(output.action.shape)
        torch.Size([2])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        action_mask: Optional[Tensor] = None,
    ):
        super().__init__(state_dim, action_dim, hidden_dims, activation)
        self.action_mask = action_mask

        self.policy_net = self._build_mlp(state_dim, action_dim)

    def forward(self, state: Tensor) -> PolicyOutput:
        """Forward pass returning sampled action and distribution info."""
        logits = self.policy_net(state)

        if self.action_mask is not None:
            logits = logits.masked_fill(~self.action_mask, float("-inf"))

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            distribution=dist,
        )

    def evaluate_actions(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate log probability and entropy of actions."""
        logits = self.policy_net(state)

        if self.action_mask is not None:
            logits = logits.masked_fill(~self.action_mask, float("-inf"))

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy

    def get_logits(self, state: Tensor) -> Tensor:
        """Get raw action logits."""
        return self.policy_net(state)


class GaussianPolicy(PolicyNetwork):
    """Gaussian policy for continuous action spaces.

    Implements a policy that outputs a Gaussian distribution
    with learnable mean and std.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.
        action_scale: Scale factor for actions.
        action_bias: Bias for actions.
        std_init: Initial standard deviation.
        std_min: Minimum standard deviation.
        std_max: Maximum standard deviation.
        learn_std: Whether to learn std.

    Example:
        >>> policy = GaussianPolicy(state_dim=4, action_dim=2)
        >>> output = policy(torch.randn(1, 4))
        >>> print(output.action.shape)
        torch.Size([1, 2])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        std_init: float = 1.0,
        std_min: float = 1e-6,
        std_max: float = 10.0,
        learn_std: bool = True,
    ):
        super().__init__(state_dim, action_dim, hidden_dims, activation)

        self.action_scale = action_scale
        self.action_bias = action_bias
        self.std_min = std_min
        self.std_max = std_max

        self.mean_net = self._build_mlp(state_dim, action_dim)

        if learn_std:
            self.log_std = nn.Parameter(
                torch.ones(action_dim) * torch.log(torch.tensor(std_init))
            )
        else:
            self.register_buffer(
                "log_std", torch.ones(action_dim) * torch.log(torch.tensor(std_init))
            )

    def _get_std(self) -> Tensor:
        """Get standard deviation clamped to valid range."""
        return torch.clamp(torch.exp(self.log_std), self.std_min, self.std_max)

    def forward(self, state: Tensor) -> PolicyOutput:
        """Forward pass returning sampled action and distribution info."""
        mean = self.mean_net(state)
        std = self._get_std()

        dist = Independent(Normal(mean, std), 1)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        scaled_action = self.action_scale * action + self.action_bias

        return PolicyOutput(
            action=scaled_action,
            log_prob=log_prob,
            entropy=entropy,
            distribution=dist,
            mean=self.action_scale * mean + self.action_bias,
            std=self.action_scale * std,
        )

    def evaluate_actions(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate log probability and entropy of actions."""
        unscaled_action = (action - self.action_bias) / self.action_scale

        mean = self.mean_net(state)
        std = self._get_std()

        dist = Independent(Normal(mean, std), 1)
        log_prob = dist.log_prob(unscaled_action)
        entropy = dist.entropy()

        return log_prob, entropy


class TanhGaussianPolicy(PolicyNetwork):
    """Tanh Gaussian policy with squashed actions.

    Implements a Gaussian policy with tanh squashing to bound
    actions to [-1, 1]. Uses the TanhTransform to properly
    account for the change of variables in log probability.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.
        std_init: Initial standard deviation.
        std_min: Minimum standard deviation.
        std_max: Maximum standard deviation.
        learn_std: Whether to learn std.
        epsilon: Small value for numerical stability.

    Example:
        >>> policy = TanhGaussianPolicy(state_dim=4, action_dim=2)
        >>> output = policy(torch.randn(1, 4))
        >>> print(output.action.min(), output.action.max())
        tensor(-1.) tensor(1.)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        std_init: float = 1.0,
        std_min: float = 1e-6,
        std_max: float = 10.0,
        learn_std: bool = True,
        epsilon: float = 1e-6,
    ):
        super().__init__(state_dim, action_dim, hidden_dims, activation)

        self.std_min = std_min
        self.std_max = std_max
        self.epsilon = epsilon

        self.mean_net = self._build_mlp(state_dim, action_dim)

        if learn_std:
            self.log_std = nn.Parameter(
                torch.ones(action_dim) * torch.log(torch.tensor(std_init))
            )
        else:
            self.register_buffer(
                "log_std", torch.ones(action_dim) * torch.log(torch.tensor(std_init))
            )

    def _get_std(self) -> Tensor:
        """Get standard deviation clamped to valid range."""
        return torch.clamp(torch.exp(self.log_std), self.std_min, self.std_max)

    def forward(self, state: Tensor) -> PolicyOutput:
        """Forward pass returning sampled action and distribution info."""
        mean = self.mean_net(state)
        std = self._get_std()

        base_dist = Independent(Normal(mean, std), 1)
        dist = TransformedDistribution(base_dist, [TanhTransform()])

        raw_action = base_dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = base_dist.log_prob(raw_action)
        log_prob -= torch.sum(torch.log((1 - action.pow(2)) + self.epsilon), dim=-1)

        entropy = base_dist.entropy()

        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            distribution=dist,
            mean=torch.tanh(mean),
            std=std,
        )

    def evaluate_actions(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate log probability and entropy of actions."""
        mean = self.mean_net(state)
        std = self._get_std()

        base_dist = Independent(Normal(mean, std), 1)

        raw_action = torch.atanh(
            torch.clamp(action, -1 + self.epsilon, 1 - self.epsilon)
        )

        log_prob = base_dist.log_prob(raw_action)
        log_prob -= torch.sum(torch.log((1 - action.pow(2)) + self.epsilon), dim=-1)

        entropy = base_dist.entropy()

        return log_prob, entropy


class DeterministicPolicy(PolicyNetwork):
    """Deterministic policy for continuous action spaces.

    Implements a deterministic policy that directly outputs
    actions without sampling. Useful for DDPG/TD3 style algorithms.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.
        action_scale: Scale factor for actions (output range [-scale, scale]).
        action_bias: Bias for actions.

    Example:
        >>> policy = DeterministicPolicy(state_dim=4, action_dim=2)
        >>> output = policy(torch.randn(1, 4))
        >>> print(output.action.shape)
        torch.Size([1, 2])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__(state_dim, action_dim, hidden_dims, activation)
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.policy_net = self._build_mlp(state_dim, action_dim)

    def forward(self, state: Tensor) -> PolicyOutput:
        """Forward pass returning deterministic action."""
        action = torch.tanh(self.policy_net(state))
        scaled_action = self.action_scale * action + self.action_bias

        batch_size = state.shape[0]
        return PolicyOutput(
            action=scaled_action,
            log_prob=torch.zeros(batch_size, device=state.device),
            entropy=torch.zeros(batch_size, device=state.device),
            mean=scaled_action,
            std=torch.zeros(batch_size, self.action_dim, device=state.device),
        )

    def evaluate_actions(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate deterministic action (zero entropy)."""
        batch_size = state.shape[0]
        return (
            torch.zeros(batch_size, device=state.device),
            torch.zeros(batch_size, device=state.device),
        )


class MultiCategoricalPolicy(PolicyNetwork):
    """Multi-categorical policy for multi-discrete action spaces.

    Implements a policy that outputs independent categorical
    distributions for each action dimension. Useful when the
    action space is a product of discrete spaces.

    Args:
        state_dim: Dimension of state space.
        action_dims: List of action dimensions for each discrete space.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.

    Example:
        >>> policy = MultiCategoricalPolicy(state_dim=4, action_dims=[3, 2])
        >>> output = policy(torch.randn(1, 4))
        >>> print(output.action.shape)
        torch.Size([1, 2])
    """

    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        self.action_dims = action_dims
        total_action_dim = sum(action_dims)

        super().__init__(state_dim, total_action_dim, hidden_dims, activation)

        self.policy_net = self._build_mlp(state_dim, total_action_dim)

    def forward(self, state: Tensor) -> PolicyOutput:
        """Forward pass returning sampled actions and distribution info."""
        logits = self.policy_net(state)

        split_logits = torch.split(logits, self.action_dims, dim=-1)
        actions = []
        log_probs = []
        entropies = []

        for split_logit in split_logits:
            dist = Categorical(logits=split_logit)
            action = dist.sample()
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

        actions = torch.stack(actions, dim=-1)
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropies, dim=-1).sum(dim=-1)

        return PolicyOutput(
            action=actions,
            log_prob=log_prob,
            entropy=entropy,
        )

    def evaluate_actions(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluate log probability and entropy of actions."""
        logits = self.policy_net(state)
        split_logits = torch.split(logits, self.action_dims, dim=-1)

        log_probs = []
        entropies = []

        for i, split_logit in enumerate(split_logits):
            dist = Categorical(logits=split_logit)
            log_probs.append(dist.log_prob(action[:, i]))
            entropies.append(dist.entropy())

        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropies, dim=-1).sum(dim=-1)

        return log_prob, entropy


def create_policy_network(
    policy_type: str,
    state_dim: int,
    action_dim: int,
    hidden_dims: List[int] = None,
    **kwargs,
) -> PolicyNetwork:
    """Factory function to create policy networks.

    Args:
        policy_type: Type of policy ('categorical', 'gaussian', 'tanh_gaussian',
                     'deterministic', 'multi_categorical').
        state_dim: Dimension of state space.
        action_dim: Dimension of action space (or list for multi_categorical).
        hidden_dims: List of hidden layer dimensions.
        **kwargs: Additional arguments passed to the policy constructor.

    Returns:
        PolicyNetwork instance.

    Example:
        >>> policy = create_policy_network('gaussian', state_dim=4, action_dim=2)
    """
    policy_classes = {
        "categorical": CategoricalPolicy,
        "gaussian": GaussianPolicy,
        "tanh_gaussian": TanhGaussianPolicy,
        "deterministic": DeterministicPolicy,
        "multi_categorical": MultiCategoricalPolicy,
    }

    if policy_type not in policy_classes:
        raise ValueError(
            f"Unknown policy type: {policy_type}. "
            f"Available: {list(policy_classes.keys())}"
        )

    policy_class = policy_classes[policy_type]

    if policy_type == "multi_categorical":
        return policy_class(state_dim, action_dim, hidden_dims, **kwargs)

    return policy_class(state_dim, action_dim, hidden_dims, **kwargs)

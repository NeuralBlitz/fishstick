"""
Value Networks for Reinforcement Learning.

This module provides value network architectures including
state-value (V), action-value (Q), and advantage estimators
with support for dueling architecture.

Example:
    >>> from fishstick.reinforcement import StateValueNetwork, DuelingQNetwork
    >>> v_net = StateValueNetwork(state_dim=4)
    >>> q_net = DuelingQNetwork(state_dim=4, action_dim=2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class ValueOutput:
    """Output from a value network.

    Attributes:
        value: Predicted value(s).
        advantage: Advantage estimate (optional, for dueling networks).
        q_values: Q-values for all actions (optional).
    """

    value: Tensor
    advantage: Optional[Tensor] = None
    q_values: Optional[Tensor] = None

    def detach(self) -> ValueOutput:
        """Detach all tensors from computation graph."""
        return ValueOutput(
            value=self.value.detach(),
            advantage=self.advantage.detach() if self.advantage is not None else None,
            q_values=self.q_values.detach() if self.q_values is not None else None,
        )


class ValueNetwork(Module, ABC):
    """Base class for value networks.

    All value networks should inherit from this class and implement
    the forward method.

    Args:
        state_dim: Dimension of state space.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.state_dim = state_dim
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


class StateValueNetwork(ValueNetwork):
    """State value network V(s).

    Estimates the expected return from a given state.
    Used in actor-critic methods as the critic.

    Args:
        state_dim: Dimension of state space.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.

    Example:
        >>> v_net = StateValueNetwork(state_dim=4)
        >>> output = v_net(torch.randn(2, 4))
        >>> print(output.value.shape)
        torch.Size([2, 1])
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__(state_dim, hidden_dims, activation)
        self.value_net = self._build_mlp(state_dim, 1)

    def forward(self, state: Tensor) -> ValueOutput:
        """Forward pass returning state value."""
        value = self.value_net(state)
        return ValueOutput(value=value)


class ActionValueNetwork(ValueNetwork):
    """Action value network Q(s, a).

    Estimates the expected return from a given state-action pair.
    Used in Q-learning and DQN style algorithms.

    Args:
        state_dim: Dimension of state space.
        action_dim: Number of discrete actions.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.

    Example:
        >>> q_net = ActionValueNetwork(state_dim=4, action_dim=2)
        >>> output = q_net(torch.randn(2, 4))
        >>> print(output.q_values.shape)
        torch.Size([2, 2])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__(state_dim, hidden_dims, activation)
        self.action_dim = action_dim
        self.q_net = self._build_mlp(state_dim, action_dim)

    def forward(self, state: Tensor) -> ValueOutput:
        """Forward pass returning Q-values for all actions."""
        q_values = self.q_net(state)
        return ValueOutput(
            value=q_values.max(dim=-1, keepdim=True)[0], q_values=q_values
        )

    def get_q_value(self, state: Tensor, action: Tensor) -> Tensor:
        """Get Q-value for specific state-action pairs.

        Args:
            state: State tensor.
            action: Action tensor (discrete indices).

        Returns:
            Q-values for the given state-action pairs.
        """
        q_values = self.q_net(state)
        if action.dim() == 1:
            return q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        return q_values.gather(1, action.long()).squeeze(1)


class ContinuousQNetwork(ValueNetwork):
    """Q-network for continuous action spaces.

    Takes both state and action as input and outputs a single Q-value.
    Used in DDPG, TD3, and SAC algorithms.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.

    Example:
        >>> q_net = ContinuousQNetwork(state_dim=4, action_dim=2)
        >>> output = q_net(torch.randn(2, 4), torch.randn(2, 2))
        >>> print(output.value.shape)
        torch.Size([2, 1])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__(state_dim, hidden_dims, activation)
        self.action_dim = action_dim

        self.q_net = self._build_mlp(state_dim + action_dim, 1)

    def forward(self, state: Tensor, action: Tensor) -> ValueOutput:
        """Forward pass returning Q-value for state-action pair."""
        x = torch.cat([state, action], dim=-1)
        value = self.q_net(x)
        return ValueOutput(value=value)


class DuelingQNetwork(ValueNetwork):
    """Dueling Q-network for discrete action spaces.

    Implements the dueling architecture that separately estimates
    state value V(s) and advantage A(s, a), combining them as:
    Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))

    This leads to better policy evaluation in the presence of
    many similar-valued actions.

    Args:
        state_dim: Dimension of state space.
        action_dim: Number of discrete actions.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.
        advantage_type: How to combine V and A ('mean', 'max').

    Reference:
        Dueling Network Architectures for Deep RL (Wang et al., 2016)

    Example:
        >>> q_net = DuelingQNetwork(state_dim=4, action_dim=2)
        >>> output = q_net(torch.randn(2, 4))
        >>> print(output.q_values.shape)
        torch.Size([2, 2])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        advantage_type: str = "mean",
    ):
        super().__init__(state_dim, hidden_dims, activation)
        self.action_dim = action_dim
        self.advantage_type = advantage_type

        self.feature_net = self._build_feature_net(state_dim)

        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            self.activation_fn,
            nn.Linear(self.hidden_dims[-1], 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            self.activation_fn,
            nn.Linear(self.hidden_dims[-1], action_dim),
        )

    def _build_feature_net(self, input_dim: int) -> nn.Sequential:
        """Build feature extraction network."""
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

        return nn.Sequential(*layers)

    def forward(self, state: Tensor) -> ValueOutput:
        """Forward pass returning Q-values using dueling architecture."""
        features = self.feature_net(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        if self.advantage_type == "mean":
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = value + advantage - advantage.max(dim=-1, keepdim=True)[0]

        return ValueOutput(
            value=value,
            advantage=advantage,
            q_values=q_values,
        )

    def get_q_value(self, state: Tensor, action: Tensor) -> Tensor:
        """Get Q-value for specific state-action pairs."""
        output = self.forward(state)
        if action.dim() == 1:
            return output.q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        return output.q_values.gather(1, action.long()).squeeze(1)


class DuelingContinuousQNetwork(ValueNetwork):
    """Dueling Q-network for continuous action spaces.

    Combines dueling architecture with continuous action inputs.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__(state_dim, hidden_dims, activation)
        self.action_dim = action_dim

        self.feature_net = self._build_feature_net(state_dim + action_dim)

        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            self.activation_fn,
            nn.Linear(self.hidden_dims[-1], 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            self.activation_fn,
            nn.Linear(self.hidden_dims[-1], 1),
        )

    def _build_feature_net(self, input_dim: int) -> nn.Sequential:
        """Build feature extraction network."""
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

        return nn.Sequential(*layers)

    def forward(self, state: Tensor, action: Tensor) -> ValueOutput:
        """Forward pass returning Q-value."""
        x = torch.cat([state, action], dim=-1)
        features = self.feature_net(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_value = value + advantage

        return ValueOutput(value=q_value, advantage=advantage)


class AdvantageNetwork(ValueNetwork):
    """Advantage network A(s, a).

    Estimates the advantage of taking action a in state s
    compared to the average action value.

    Args:
        state_dim: Dimension of state space.
        action_dim: Number of discrete actions.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.

    Example:
        >>> adv_net = AdvantageNetwork(state_dim=4, action_dim=2)
        >>> output = adv_net(torch.randn(2, 4))
        >>> print(output.advantage.shape)
        torch.Size([2, 2])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__(state_dim, hidden_dims, activation)
        self.action_dim = action_dim
        self.advantage_net = self._build_mlp(state_dim, action_dim)

    def forward(self, state: Tensor) -> ValueOutput:
        """Forward pass returning advantage values."""
        advantage = self.advantage_net(state)
        return ValueOutput(
            value=advantage.mean(dim=-1, keepdim=True), advantage=advantage
        )


class EnsembleQNetwork(ValueNetwork):
    """Ensemble of Q-networks for uncertainty estimation.

    Maintains multiple Q-networks and aggregates their predictions.
    Useful for reducing overestimation bias in Q-learning.

    Args:
        state_dim: Dimension of state space.
        action_dim: Number of discrete actions.
        num_ensemble: Number of Q-networks in ensemble.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.
        aggregation: How to aggregate predictions ('mean', 'min', 'max').

    Example:
        >>> q_net = EnsembleQNetwork(state_dim=4, action_dim=2, num_ensemble=3)
        >>> output = q_net(torch.randn(2, 4))
        >>> print(output.q_values.shape)
        torch.Size([2, 2])
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_ensemble: int = 2,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        aggregation: str = "min",
    ):
        super().__init__(state_dim, hidden_dims, activation)
        self.action_dim = action_dim
        self.num_ensemble = num_ensemble
        self.aggregation = aggregation

        self.q_networks = nn.ModuleList(
            [
                ActionValueNetwork(state_dim, action_dim, hidden_dims, activation)
                for _ in range(num_ensemble)
            ]
        )

    def forward(self, state: Tensor) -> ValueOutput:
        """Forward pass returning aggregated Q-values."""
        q_values_list = [net(state).q_values for net in self.q_networks]
        q_values_stack = torch.stack(q_values_list, dim=0)

        if self.aggregation == "mean":
            q_values = q_values_stack.mean(dim=0)
        elif self.aggregation == "min":
            q_values = q_values_stack.min(dim=0)[0]
        elif self.aggregation == "max":
            q_values = q_values_stack.max(dim=0)[0]
        else:
            q_values = q_values_stack.mean(dim=0)

        return ValueOutput(
            value=q_values.max(dim=-1, keepdim=True)[0],
            q_values=q_values,
        )

    def get_all_q_values(self, state: Tensor) -> Tensor:
        """Get Q-values from all ensemble members.

        Returns:
            Tensor of shape (num_ensemble, batch_size, action_dim).
        """
        q_values_list = [net(state).q_values for net in self.q_networks]
        return torch.stack(q_values_list, dim=0)


class ContinuousEnsembleQNetwork(ValueNetwork):
    """Ensemble of continuous Q-networks.

    Args:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        num_ensemble: Number of Q-networks in ensemble.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function name.
        aggregation: How to aggregate predictions ('mean', 'min').
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_ensemble: int = 2,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        aggregation: str = "min",
    ):
        super().__init__(state_dim, hidden_dims, activation)
        self.action_dim = action_dim
        self.num_ensemble = num_ensemble
        self.aggregation = aggregation

        self.q_networks = nn.ModuleList(
            [
                ContinuousQNetwork(state_dim, action_dim, hidden_dims, activation)
                for _ in range(num_ensemble)
            ]
        )

    def forward(self, state: Tensor, action: Tensor) -> ValueOutput:
        """Forward pass returning aggregated Q-value."""
        q_values_list = [net(state, action).value for net in self.q_networks]
        q_values_stack = torch.cat(q_values_list, dim=-1)

        if self.aggregation == "mean":
            q_value = q_values_stack.mean(dim=-1, keepdim=True)
        elif self.aggregation == "min":
            q_value = q_values_stack.min(dim=-1, keepdim=True)[0]
        else:
            q_value = q_values_stack.mean(dim=-1, keepdim=True)

        return ValueOutput(value=q_value)

    def get_all_q_values(self, state: Tensor, action: Tensor) -> Tensor:
        """Get Q-values from all ensemble members."""
        q_values_list = [net(state, action).value for net in self.q_networks]
        return torch.cat(q_values_list, dim=-1)


def create_value_network(
    value_type: str,
    state_dim: int,
    action_dim: int = 0,
    hidden_dims: List[int] = None,
    **kwargs,
) -> ValueNetwork:
    """Factory function to create value networks.

    Args:
        value_type: Type of value network ('state', 'action', 'dueling',
                    'continuous', 'ensemble', 'continuous_ensemble').
        state_dim: Dimension of state space.
        action_dim: Dimension of action space (required for most types).
        hidden_dims: List of hidden layer dimensions.
        **kwargs: Additional arguments passed to the constructor.

    Returns:
        ValueNetwork instance.

    Example:
        >>> v_net = create_value_network('state', state_dim=4)
        >>> q_net = create_value_network('dueling', state_dim=4, action_dim=2)
    """
    if value_type == "state":
        return StateValueNetwork(state_dim, hidden_dims, **kwargs)
    elif value_type == "action":
        return ActionValueNetwork(state_dim, action_dim, hidden_dims, **kwargs)
    elif value_type == "dueling":
        return DuelingQNetwork(state_dim, action_dim, hidden_dims, **kwargs)
    elif value_type == "continuous":
        return ContinuousQNetwork(state_dim, action_dim, hidden_dims, **kwargs)
    elif value_type == "dueling_continuous":
        return DuelingContinuousQNetwork(state_dim, action_dim, hidden_dims, **kwargs)
    elif value_type == "advantage":
        return AdvantageNetwork(state_dim, action_dim, hidden_dims, **kwargs)
    elif value_type == "ensemble":
        return EnsembleQNetwork(
            state_dim, action_dim, hidden_dims=hidden_dims, **kwargs
        )
    elif value_type == "continuous_ensemble":
        return ContinuousEnsembleQNetwork(
            state_dim, action_dim, hidden_dims=hidden_dims, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown value type: {value_type}. "
            f"Available: 'state', 'action', 'dueling', 'continuous', "
            f"'dueling_continuous', 'advantage', 'ensemble', 'continuous_ensemble'"
        )

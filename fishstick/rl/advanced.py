"""
fishstick Advanced Reinforcement Learning Toolkit

Comprehensive RL module providing:
- RL Environments Interface
- Policy Networks
- RL Algorithms (DQN, Double DQN, Dueling DQN, PPO, A2C, SAC)
- Experience Replay Buffers
- Utilities (GAE, Noisy Networks, Return computation)
- Training Loop

Usage:
    from fishstick.rl.advanced import DQNAgent, PPOAgent, RLTrainer

    # Create environment
    env = GymWrapper(gym.make('CartPole-v1'))

    # Initialize agent
    agent = DQNAgent(state_dim=4, action_dim=2)

    # Train
    trainer = RLTrainer(agent, env)
    trainer.train(num_episodes=1000)
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
    overload,
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
from torch.distributions import Categorical, Normal

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
# RL Environments Interface
# =============================================================================


class BaseEnv(ABC):
    """Abstract base class for RL environments.

    All environments in fishstick RL must inherit from this class
    and implement the required methods.

    Example:
        class CustomEnv(BaseEnv):
            def reset(self) -> np.ndarray:
                return self._get_initial_state()

            def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
                # Execute action and return transition
                return state, reward, done, info
    """

    @abstractmethod
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment to initial state.

        Returns:
            Initial observation as numpy array
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one timestep within the environment.

        Args:
            action: Action to take in the environment

        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment.

        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)

        Returns:
            Rendered frame if mode is 'rgb_array', else None
        """
        pass

    def close(self) -> None:
        """Close the environment and cleanup resources."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """Return the observation space of the environment."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """Return the action space of the environment."""
        pass

    @property
    def unwrapped(self) -> "BaseEnv":
        """Return the base environment (without wrappers)."""
        return self


class GymWrapper(BaseEnv):
    """Wrapper for OpenAI Gym/Gymnasium environments.

    Provides a unified interface for gym environments within
    the fishstick RL framework.

    Args:
        env: Gym environment instance or string name
        render_mode: Rendering mode for visualization

    Example:
        import gymnasium as gym
        env = GymWrapper(gym.make('CartPole-v1'))
        state = env.reset()
        state, reward, done, info = env.step(0)
    """

    def __init__(
        self, env: Union[str, Any], render_mode: Optional[str] = None, **kwargs
    ):
        self._env_name = env if isinstance(env, str) else None
        self._render_mode = render_mode
        self._env_kwargs = kwargs
        self._env = None

        if not isinstance(env, str):
            self._env = env

    def _get_env(self) -> Any:
        """Lazy load the gym environment."""
        if self._env is None:
            try:
                import gymnasium as gym
            except ImportError:
                try:
                    import gym
                except ImportError:
                    raise ImportError(
                        "Either gymnasium or gym must be installed. "
                        "Install with: pip install gymnasium"
                    )

            if self._env_name:
                self._env = gym.make(
                    self._env_name, render_mode=self._render_mode, **self._env_kwargs
                )
        return self._env

    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment to initial state."""
        env = self._get_env()
        result = env.reset(**kwargs)
        # Handle both gym (old) and gymnasium (new) APIs
        if isinstance(result, tuple):
            state, info = result
            return state
        return result

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one timestep within the environment."""
        env = self._get_env()
        result = env.step(action)

        # Handle both gym (old) and gymnasium (new) APIs
        if len(result) == 5:
            state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            state, reward, done, info = result

        return state, reward, done, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        env = self._get_env()
        return env.render()

    def close(self) -> None:
        """Close the environment."""
        if self._env:
            self._env.close()

    @property
    def observation_space(self) -> Any:
        """Return the observation space."""
        return self._get_env().observation_space

    @property
    def action_space(self) -> Any:
        """Return the action space."""
        return self._get_env().action_space

    @property
    def unwrapped(self) -> BaseEnv:
        """Return the base environment."""
        if hasattr(self._get_env(), "unwrapped"):
            return GymWrapper(self._get_env().unwrapped)
        return self

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped environment."""
        return getattr(self._get_env(), name)


class VectorizedEnv(BaseEnv):
    """Vectorized environment for parallel execution.

    Runs multiple environments in parallel for efficient
    data collection. Supports both synchronous and
    asynchronous execution modes.

    Args:
        env_fns: List of functions that create environments
        num_envs: Number of parallel environments
        asynchronous: If True, use async execution (when available)

    Example:
        def make_env():
            return GymWrapper(gym.make('CartPole-v1'))

        vec_env = VectorizedEnv([make_env] * 4)
        states = vec_env.reset()  # Shape: (4, state_dim)
    """

    def __init__(
        self,
        env_fns: List[Callable[[], BaseEnv]],
        num_envs: Optional[int] = None,
        asynchronous: bool = False,
    ):
        if num_envs is not None:
            env_fns = [env_fns[0]] * num_envs if len(env_fns) == 1 else env_fns

        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.asynchronous = asynchronous
        self.envs: List[BaseEnv] = []
        self._closed = False

        # Initialize environments
        for fn in env_fns:
            self.envs.append(fn())

    def reset(self, **kwargs) -> np.ndarray:
        """Reset all environments."""
        states = [env.reset(**kwargs) for env in self.envs]
        return np.array(states)

    def step(
        self, actions: Sequence[Action]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all environments with given actions."""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        states, rewards, dones, infos = zip(*results)
        return (np.array(states), np.array(rewards), np.array(dones), list(infos))

    def step_async(self, actions: Sequence[Action]) -> None:
        """Asynchronously step environments (placeholder for async support)."""
        self._async_results = [
            env.step(action) for env, action in zip(self.envs, actions)
        ]

    def step_wait(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Wait for async step to complete."""
        states, rewards, dones, infos = zip(*self._async_results)
        return np.array(states), np.array(rewards), np.array(dones), list(infos)

    def close(self) -> None:
        """Close all environments."""
        if self._closed:
            return
        for env in self.envs:
            env.close()
        self._closed = True

    @property
    def observation_space(self) -> Any:
        """Return observation space of first environment."""
        return self.envs[0].observation_space

    @property
    def action_space(self) -> Any:
        """Return action space of first environment."""
        return self.envs[0].action_space

    def __len__(self) -> int:
        return self.num_envs

    def __getitem__(self, idx: int) -> BaseEnv:
        return self.envs[idx]


class NormalizedEnv(BaseEnv):
    """Environment wrapper for observation and reward normalization.

    Normalizes observations and/or rewards using running statistics.
    Useful for stabilizing training across different scales.

    Args:
        env: Base environment to wrap
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        obs_clip: Clip value for normalized observations
        gamma: Discount factor for reward normalization
        epsilon: Small constant for numerical stability

    Example:
        env = GymWrapper(gym.make('CartPole-v1'))
        norm_env = NormalizedEnv(env, normalize_obs=True)
    """

    def __init__(
        self,
        env: BaseEnv,
        normalize_obs: bool = True,
        normalize_reward: bool = False,
        obs_clip: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        self.env = env
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.obs_clip = obs_clip
        self.gamma = gamma
        self.epsilon = epsilon

        # Running statistics for observations
        self.obs_mean = np.zeros(0)
        self.obs_var = np.ones(0)
        self.obs_count = epsilon

        # Running statistics for rewards
        self.ret = 0.0  # Running return for reward normalization
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = epsilon

    def _update_obs_stats(self, obs: np.ndarray) -> None:
        """Update running observation statistics."""
        if self.obs_mean.size == 0:
            self.obs_mean = np.zeros_like(obs, dtype=np.float64)
            self.obs_var = np.ones_like(obs, dtype=np.float64)

        batch_mean = np.mean(obs)
        batch_var = np.var(obs)
        batch_count = 1

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count

        self.obs_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.obs_count * batch_count / total_count
        self.obs_var = M2 / total_count
        self.obs_count = total_count

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        if not self.normalize_obs or self.obs_count <= self.epsilon:
            return obs

        obs_normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(obs_normalized, -self.obs_clip, self.obs_clip)

    def _update_reward_stats(self, reward: float) -> None:
        """Update running reward statistics."""
        self.ret = self.ret * self.gamma + reward

        self.ret_count += 1
        delta = self.ret - self.ret_mean
        self.ret_mean += delta / self.ret_count
        delta2 = self.ret - self.ret_mean
        self.ret_var += delta * delta2

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        if not self.normalize_reward:
            return reward
        return reward / np.sqrt(self.ret_var / self.ret_count + self.epsilon)

    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and return normalized observation."""
        self.ret = 0.0
        obs = self.env.reset(**kwargs)

        if self.normalize_obs:
            self._update_obs_stats(obs)
            obs = self._normalize_obs(obs)

        return obs

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step environment and normalize outputs."""
        obs, reward, done, info = self.env.step(action)

        if self.normalize_obs:
            self._update_obs_stats(obs)
            obs = self._normalize_obs(obs)

        if self.normalize_reward:
            self._update_reward_stats(reward)
            reward = self._normalize_reward(reward)

        # Store raw values in info
        info["raw_obs"] = obs
        info["raw_reward"] = reward

        return obs, reward, done, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render(mode)

    def close(self) -> None:
        self.env.close()

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def unwrapped(self) -> BaseEnv:
        return self.env.unwrapped


# =============================================================================
# Policy Networks
# =============================================================================


class PolicyNetwork(nn.Module):
    """Base policy network with action distribution.

    Provides a foundation for stochastic policies with methods
    for sampling actions and computing log probabilities.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        activation: Activation function class

    Example:
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        action = policy.get_action(state)
        log_prob = policy.log_prob(state, action)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), activation()])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass to get action logits."""
        features = self.feature_extractor(state)
        logits = self.action_head(features)
        return logits

    def get_action(
        self, state: Union[np.ndarray, Tensor], deterministic: bool = False
    ) -> int:
        """Sample action from policy.

        Args:
            state: Current state
            deterministic: If True, return argmax action

        Returns:
            Sampled action
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            logits = self.forward(state)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = torch.multinomial(probs, 1).squeeze(-1)

            return action.item()

    def log_prob(self, state: Tensor, action: Tensor) -> Tensor:
        """Compute log probability of action."""
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, action.unsqueeze(1)).squeeze(1)

    def entropy(self, state: Tensor) -> Tensor:
        """Compute policy entropy."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(probs * log_probs).sum(dim=-1)


class ActorCriticNetwork(nn.Module):
    """Combined actor-critic network.

    Shares feature extraction between policy (actor) and
    value function (critic) for better sample efficiency.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        activation: Activation function class
        share_features: Whether to share feature extractor

    Example:
        ac = ActorCriticNetwork(state_dim=4, action_dim=2)
        action, log_prob, value = ac.get_action_and_value(state)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
        share_features: bool = True,
        continuous: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.share_features = share_features
        self.continuous = continuous

        if share_features:
            # Shared feature extractor
            layers = []
            prev_dim = state_dim
            for hidden_dim in hidden_dims:
                layers.extend([nn.Linear(prev_dim, hidden_dim), activation()])
                prev_dim = hidden_dim
            self.shared_features = nn.Sequential(*layers)

            # Actor and critic heads
            if continuous:
                self.actor_mean = nn.Linear(prev_dim, action_dim)
                self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            else:
                self.actor_head = nn.Linear(prev_dim, action_dim)
            self.critic_head = nn.Linear(prev_dim, 1)
        else:
            # Separate networks for actor and critic
            self.actor = PolicyNetwork(state_dim, action_dim, hidden_dims, activation)
            self.critic = QNetwork(state_dim, 1, hidden_dims, activation)

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning action logits and value."""
        if self.share_features:
            features = self.shared_features(state)

            if self.continuous:
                action_mean = self.actor_mean(features)
                action_std = self.actor_log_std.exp()
                value = self.critic_head(features).squeeze(-1)
                return action_mean, action_std, value
            else:
                action_logits = self.actor_head(features)
                value = self.critic_head(features).squeeze(-1)
                return action_logits, value
        else:
            action_logits = self.actor(state)
            value = self.critic(state).squeeze(-1)
            return action_logits, value

    def get_action_and_value(
        self, state: Tensor, deterministic: bool = False
    ) -> Tuple[int, Tensor, Tensor]:
        """Get action, log probability, and value estimate.

        Returns:
            Tuple of (action, log_prob, value)
        """
        if self.share_features and self.continuous:
            action_mean, action_std, value = self.forward(state)
            dist = Normal(action_mean, action_std)

            if deterministic:
                action = action_mean
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, value
        else:
            action_logits, value = self.forward(state)
            probs = F.softmax(action_logits, dim=-1)

            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = torch.multinomial(probs, 1).squeeze(-1)

            log_prob = (
                F.log_softmax(action_logits, dim=-1)
                .gather(1, action.unsqueeze(1))
                .squeeze(1)
            )
            return action, log_prob, value

    def evaluate_actions(
        self, state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate actions for PPO update.

        Returns:
            Tuple of (log_prob, value, entropy)
        """
        if self.share_features and self.continuous:
            action_mean, action_std, value = self.forward(state)
            dist = Normal(action_mean, action_std)

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, value, entropy
        else:
            action_logits, value = self.forward(state)
            probs = F.softmax(action_logits, dim=-1)
            log_probs = F.log_softmax(action_logits, dim=-1)

            log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
            entropy = -(probs * log_probs).sum(dim=-1)

            return log_prob, value, entropy


class QNetwork(nn.Module):
    """Q-value network for DQN and variants.

    Estimates action-values Q(s,a) for state-action pairs.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space (output actions)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function class
        dueling: Whether to use dueling architecture

    Example:
        q_net = QNetwork(state_dim=4, action_dim=2)
        q_values = q_net(state)  # Returns Q-values for all actions
        best_action = q_values.argmax(dim=-1)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
        dueling: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling

        # Feature extraction layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), activation()])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        if dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1] // 2),
                activation(),
                nn.Linear(hidden_dims[-1] // 2, 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1] // 2),
                activation(),
                nn.Linear(hidden_dims[-1] // 2, action_dim),
            )
        else:
            # Standard Q-network
            self.q_head = nn.Linear(prev_dim, action_dim)

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass to get Q-values for all actions."""
        features = self.feature_extractor(state)

        if self.dueling:
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
            q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        else:
            q_values = self.q_head(features)

        return q_values

    def get_action(self, state: Tensor, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state
            epsilon: Probability of random action

        Returns:
            Selected action
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            q_values = self.forward(state)
            return q_values.argmax(dim=-1).item()


class DeterministicPolicy(nn.Module):
    """Deterministic policy for continuous control (DDPG, TD3).

    Maps states directly to continuous actions without
    stochastic sampling.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        activation: Activation function class
        action_scale: Scale factor for output actions
        action_bias: Bias for output actions

    Example:
        policy = DeterministicPolicy(state_dim=3, action_dim=1)
        action = policy.get_action(state)  # Continuous action in [-1, 1]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.action_bias = action_bias

        # Build network
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), activation()])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.action_head = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, state: Tensor) -> Tensor:
        """Forward pass to get continuous action."""
        features = self.feature_extractor(state)
        action = self.action_head(features)
        # Scale and shift to action range
        return action * self.action_scale + self.action_bias

    def get_action(
        self, state: Union[np.ndarray, Tensor], noise: float = 0.0
    ) -> np.ndarray:
        """Get deterministic action with optional exploration noise.

        Args:
            state: Current state
            noise: Standard deviation of Gaussian noise to add

        Returns:
            Continuous action
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action = self.forward(state)

            if noise > 0:
                action = action + torch.randn_like(action) * noise
                action = torch.clamp(
                    action,
                    -self.action_scale + self.action_bias,
                    self.action_scale + self.action_bias,
                )

            return action.cpu().numpy().flatten()


# =============================================================================
# Experience Replay
# =============================================================================


class ReplayBuffer:
    """Standard experience replay buffer.

    Stores and samples transitions (s, a, r, s', done) uniformly.
    Essential for off-policy RL algorithms like DQN.

    Args:
        capacity: Maximum number of transitions to store
        seed: Random seed for reproducibility

    Example:
        buffer = ReplayBuffer(capacity=100000)
        buffer.push(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = buffer.sample(32)
    """

    def __init__(self, capacity: int = 100000, seed: Optional[int] = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

        if seed is not None:
            random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample random batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions))
            if isinstance(actions[0], (float, np.floating))
            else torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Experience:
        return self.buffer[idx]

    def clear(self) -> None:
        """Clear all transitions from buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer.

    Samples transitions with probability proportional to their
    TD error, prioritizing important experiences.

    Based on "Prioritized Experience Replay" (Schaul et al., 2016)

    Args:
        capacity: Maximum number of transitions
        alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        beta: Importance sampling exponent annealing start
        beta_increment: Beta increment per sampling
        epsilon: Small constant to avoid zero priorities

    Example:
        buffer = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)
        buffer.push(state, action, reward, next_state, done)
        batch, weights, indices = buffer.sample(32)
        buffer.update_priorities(indices, td_errors)
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = Experience(
                state, action, reward, next_state, done
            )

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, np.ndarray]:
        """Sample batch with importance sampling weights.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Compute sampling probabilities
        priorities = self.priorities[: len(self.buffer)]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer), batch_size, p=probabilities, replace=False
        )

        # Compute importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions))
            if isinstance(actions[0], (float, np.floating))
            else torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            weights,
            indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon

    def reset_beta(self) -> None:
        """Reset beta to initial value."""
        self.beta = self.beta_start

    def __len__(self) -> int:
        return len(self.buffer)


class MultiStepBuffer:
    """N-step return experience replay buffer.

    Computes n-step returns for each transition, improving
    value estimation and credit assignment.

    Based on "Learning to Predict by the Methods of Temporal Differences" (Sutton, 1988)

    Args:
        capacity: Maximum number of transitions
        n_step: Number of steps for n-step return
        gamma: Discount factor

    Example:
        buffer = MultiStepBuffer(capacity=100000, n_step=3, gamma=0.99)
        buffer.push(state, action, reward)
        # When done, call end_episode()
        batch = buffer.sample(32)
    """

    def __init__(self, capacity: int = 100000, n_step: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma

        self.n_step_buffer = deque(maxlen=n_step)
        self.buffer = deque(maxlen=capacity)

    def _compute_n_step_return(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """Compute n-step return from buffer."""
        state, action = self.n_step_buffer[0][:2]

        # Compute discounted return
        reward = 0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            reward += (self.gamma**i) * r

        # Get final state and done flag
        next_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        return state, action, reward, next_state, done

    def push(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[Experience]:
        """Add transition and return n-step experience if ready."""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_step and not done:
            return None

        # Compute n-step return
        n_step_exp = self._compute_n_step_return()
        self.buffer.append(n_step_exp)

        return n_step_exp

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample batch of n-step transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def end_episode(self) -> None:
        """Flush remaining transitions at episode end."""
        while len(self.n_step_buffer) > 0:
            self.n_step_buffer.popleft()
            if len(self.n_step_buffer) > 0:
                exp = self._compute_n_step_return()
                self.buffer.append(exp)

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# RL Algorithms
# =============================================================================


class DQNAgent:
    """Deep Q-Network (DQN) Agent.

    Classic off-policy RL algorithm using experience replay
    and target network for stable learning.

    Based on "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Decay rate for epsilon
        target_update_freq: Frequency of target network updates
        buffer_capacity: Capacity of replay buffer
        batch_size: Batch size for training
        dueling: Whether to use dueling architecture

    Example:
        agent = DQNAgent(state_dim=4, action_dim=2)
        action = agent.get_action(state)
        agent.store_transition(state, action, reward, next_state, done)
        loss = agent.update()
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 1000,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        dueling: bool = False,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.update_count = 0

        # Device configuration
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_network = QNetwork(
            state_dim, action_dim, hidden_dims, dueling=dueling
        ).to(self.device)
        self.target_network = QNetwork(
            state_dim, action_dim, hidden_dims, dueling=dueling
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def get_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy."""
        eps = epsilon if epsilon is not None else self.epsilon

        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """Update Q-network using experience replay.

        Returns:
            Loss value or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]


class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent.

    Reduces overestimation bias by decoupling action selection
    from action evaluation using the online and target networks.

    Based on "Deep Reinforcement Learning with Double Q-learning" (Hasselt et al., 2016)

    Example:
        agent = DoubleDQNAgent(state_dim=4, action_dim=2)
        # Same interface as DQNAgent
    """

    def update(self) -> Optional[float]:
        """Update using Double DQN."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use online network to select actions
        with torch.no_grad():
            # Select best actions using online network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate using target network
            next_q = (
                self.target_network(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN Agent.

    Uses dueling architecture to separately estimate state value
    and action advantages, improving learning efficiency.

    Based on "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)

    Example:
        agent = DuelingDQNAgent(state_dim=4, action_dim=2)
        # Same interface as DQNAgent
    """

    def __init__(self, *args, **kwargs):
        """Initialize with dueling=True."""
        kwargs["dueling"] = True
        super().__init__(*args, **kwargs)


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent.

    On-policy algorithm using clipped surrogate objective
    to prevent destructive large policy updates.

    Based on "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_epsilon: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Gradient clipping norm
        ppo_epochs: Number of PPO update epochs
        batch_size: Mini-batch size

    Example:
        agent = PPOAgent(state_dim=4, action_dim=2)

        # Collect trajectories
        for step in range(num_steps):
            action, log_prob, value = agent.select_action(state)
            agent.store_transition(log_prob, value, reward, done)
            state = next_state

        # Update policy
        agent.compute_returns_and_advantages(next_state, done)
        agent.update()
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Device configuration
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Network
        self.policy = ActorCriticNetwork(state_dim, action_dim, hidden_dims).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Memory buffers
        self.clear_memory()

    def clear_memory(self) -> None:
        """Clear trajectory memory."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action and store relevant values.

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action_and_value(state_tensor)

            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())

            return action.item(), log_prob.item(), value.item()

    def store_transition(self, reward: float, done: bool) -> None:
        """Store transition reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self, last_state: np.ndarray, last_done: bool
    ) -> None:
        """Compute GAE advantages and returns."""
        with torch.no_grad():
            last_state_tensor = (
                torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            )
            _, last_value = self.policy(last_state_tensor)
            last_value = last_value.item()

        # Compute GAE
        advantages = []
        gae = 0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value if not last_done else 0
                next_non_terminal = 1.0 - float(last_done)
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)

        self.advantages = torch.FloatTensor(advantages).to(self.device)
        self.returns = self.advantages + torch.FloatTensor(self.values).to(self.device)

        # Convert to tensors
        self.states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        self.actions_tensor = torch.LongTensor(self.actions).to(self.device)
        self.log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)

    def update(self) -> Dict[str, float]:
        """Update policy using PPO clipped objective.

        Returns:
            Dictionary of training metrics
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            # Generate random indices
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)

            for start in range(0, len(self.states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch data
                states = self.states_tensor[batch_indices]
                actions = self.actions_tensor[batch_indices]
                old_log_probs = self.log_probs_tensor[batch_indices]
                advantages = self.advantages[batch_indices]
                returns = self.returns[batch_indices]

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(
                    states, actions
                )

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.clear_memory()

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class A2CAgent:
    """Advantage Actor-Critic (A2C) Agent.

    Synchronous version of A3C that updates the global network
    after collecting experiences from multiple workers.

    Based on "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        gamma: Discount factor
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Gradient clipping norm

    Example:
        agent = A2CAgent(state_dim=4, action_dim=2)

        # Collect rollouts
        for step in range(num_steps):
            action, log_prob, value = agent.select_action(state)
            agent.store_transition(log_prob, value, reward, done)
            state = next_state

        # Update
        agent.compute_returns(next_state, done)
        loss = agent.update()
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Device configuration
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Network
        self.policy = ActorCriticNetwork(state_dim, action_dim, hidden_dims).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Memory
        self.clear_memory()

    def clear_memory(self) -> None:
        """Clear trajectory memory."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action and store relevant values."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action_and_value(state_tensor)

            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())

            return action.item(), log_prob.item(), value.item()

    def store_transition(self, reward: float, done: bool) -> None:
        """Store transition reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns(self, last_state: np.ndarray, last_done: bool) -> None:
        """Compute discounted returns."""
        with torch.no_grad():
            last_state_tensor = (
                torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            )
            _, last_value = self.policy(last_state_tensor)
            last_value = last_value.item()

        # Compute returns
        returns = []
        R = last_value if not last_done else 0

        for t in reversed(range(len(self.rewards))):
            R = self.rewards[t] + self.gamma * R * (1 - float(self.dones[t]))
            returns.insert(0, R)

        self.returns = torch.FloatTensor(returns).to(self.device)
        self.states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        self.actions_tensor = torch.LongTensor(self.actions).to(self.device)
        self.log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        self.values_tensor = torch.FloatTensor(self.values).to(self.device)

    def update(self) -> Dict[str, float]:
        """Update policy using advantage actor-critic.

        Returns:
            Dictionary of training metrics
        """
        # Compute advantages
        advantages = self.returns - self.values_tensor

        # Evaluate actions
        log_probs, values, entropy = self.policy.evaluate_actions(
            self.states_tensor, self.actions_tensor
        )

        # Actor loss (policy gradient with advantage)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss (value function error)
        critic_loss = F.mse_loss(values, self.returns)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            actor_loss
            + self.value_coef * critic_loss
            + self.entropy_coef * entropy_loss
        )

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.clear_memory()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": loss.item(),
        }

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class SACAgent:
    """Soft Actor-Critic (SAC) Agent.

    Off-policy maximum entropy RL algorithm that optimizes
    for both expected return and policy entropy.

    Based on "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor" (Haarnoja et al., 2018)

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        gamma: Discount factor
        tau: Soft update coefficient
        alpha: Temperature parameter (or 'auto' for automatic tuning)
        buffer_capacity: Replay buffer capacity
        batch_size: Batch size for training

    Example:
        agent = SACAgent(state_dim=3, action_dim=1, continuous=True)

        # Collect experience
        action = agent.get_action(state)
        agent.store_transition(state, action, reward, next_state, done)

        # Update
        losses = agent.update()
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: Union[float, str] = "auto",
        buffer_capacity: int = 1000000,
        batch_size: int = 256,
        device: str = "auto",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Automatic entropy tuning
        self.automatic_entropy_tuning = alpha == "auto"
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.alpha = torch.tensor([alpha])

        # Device configuration
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.alpha = self.alpha.to(self.device)

        # Actor network (Gaussian policy)
        self.actor = ActorCriticNetwork(
            state_dim, action_dim, hidden_dims, continuous=True, share_features=False
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.actor.parameters(), lr=learning_rate
        )

        # Critic networks (Q-functions)
        self.critic1 = QNetwork(state_dim + action_dim, 1, hidden_dims).to(self.device)
        self.critic2 = QNetwork(state_dim + action_dim, 1, hidden_dims).to(self.device)

        self.target_critic1 = QNetwork(state_dim + action_dim, 1, hidden_dims).to(
            self.device
        )
        self.target_critic2 = QNetwork(state_dim + action_dim, 1, hidden_dims).to(
            self.device
        )

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=learning_rate,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from Gaussian policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, log_std = self.actor.actor(state_tensor)
            std = log_std.exp()

            if deterministic:
                action = torch.tanh(mean)
            else:
                normal = Normal(mean, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)

            return action.cpu().numpy().flatten()

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        """Update actor and critic networks.

        Returns:
            Dictionary of training metrics or None
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critic
        with torch.no_grad():
            # Sample actions from current policy
            next_mean, next_log_std = self.actor.actor(next_states)
            next_std = next_log_std.exp()
            next_normal = Normal(next_mean, next_std)
            next_x_t = next_normal.rsample()
            next_actions = torch.tanh(next_x_t)
            next_log_prob = next_normal.log_prob(next_x_t) - torch.log(
                1 - next_actions.pow(2) + 1e-6
            )
            next_log_prob = next_log_prob.sum(dim=-1, keepdim=True)

            # Compute target Q-values
            next_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1))
            next_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1))
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = (
                rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
            )

        # Update Q-functions
        q1 = self.critic1(
            torch.cat(
                [states, actions.unsqueeze(1) if actions.dim() == 1 else actions], dim=1
            )
        )
        q2 = self.critic2(
            torch.cat(
                [states, actions.unsqueeze(1) if actions.dim() == 1 else actions], dim=1
            )
        )

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        mean, log_std = self.actor.actor(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        actions_pred = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - actions_pred.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        q1_new = self.critic1(torch.cat([states, actions_pred], dim=1))
        q2_new = self.critic2(torch.cat([states, actions_pred], dim=1))
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (temperature)
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            alpha_value = self.alpha.item()
            alpha_loss_value = alpha_loss.item()
        else:
            alpha_value = self.alpha.item()
            alpha_loss_value = 0

        # Soft update target networks
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss_value,
            "alpha": alpha_value,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha if self.automatic_entropy_tuning else None,
                "alpha_optimizer": self.alpha_optimizer.state_dict()
                if self.automatic_entropy_tuning
                else None,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        if self.automatic_entropy_tuning and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


# =============================================================================
# Utilities
# =============================================================================


def compute_returns(
    rewards: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    bootstrap_value: float = 0.0,
) -> List[float]:
    """Compute discounted returns for a trajectory.

    Args:
        rewards: List of rewards
        dones: List of done flags
        gamma: Discount factor
        bootstrap_value: Value estimate for final state

    Returns:
        List of discounted returns

    Example:
        returns = compute_returns([1, 1, 1], [False, False, True], gamma=0.99)
        # Returns: [2.9701, 1.99, 1.0]
    """
    returns = []
    R = bootstrap_value

    for r, done in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - float(done))
        returns.insert(0, R)

    return returns


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_value: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimation (GAE).

    GAE provides a bias-variance tradeoff for advantage estimation
    using TD(lambda)-style weighted averages.

    Based on "High-Dimensional Continuous Control Using Generalized
    Advantage Estimation" (Schulman et al., 2016)

    Args:
        rewards: List of rewards
        values: List of value estimates
        dones: List of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        bootstrap_value: Value estimate for final state

    Returns:
        Tuple of (advantages, returns)

    Example:
        advantages, returns = compute_gae(
            rewards=[1, 1, 1],
            values=[0.5, 0.6, 0.7],
            dones=[False, False, True],
            gamma=0.99,
            gae_lambda=0.95
        )
    """
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = bootstrap_value
            next_non_terminal = 1.0 - float(dones[t])
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration.

    Implements factorized noisy networks that learn to adapt
    exploration noise during training, replacing epsilon-greedy.

    Based on "Noisy Networks for Exploration" (Fortunato et al., 2018)

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        sigma_init: Initial value for noise standard deviation

    Example:
        layer = NoisyLinear(64, 10)
        output = layer(input)  # Noisy forward pass
        layer.reset_noise()    # Resample noise
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Non-learnable parameters
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize parameters using factorized Gaussian noise."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        """Reset factorized Gaussian noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> Tensor:
        """Generate scaled noise for factorization."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


# =============================================================================
# Training Loop
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for RL training.

    Attributes:
        num_episodes: Total number of training episodes
        max_steps_per_episode: Maximum steps per episode
        eval_freq: Evaluation frequency (episodes)
        num_eval_episodes: Number of evaluation episodes
        save_freq: Model save frequency (episodes)
        log_freq: Logging frequency (steps)
        render: Whether to render environment
        seed: Random seed
    """

    num_episodes: int = 1000
    max_steps_per_episode: int = 1000
    eval_freq: int = 100
    num_eval_episodes: int = 10
    save_freq: int = 500
    log_freq: int = 10
    render: bool = False
    seed: Optional[int] = None


class RLTrainer:
    """High-level training coordinator for RL agents.

    Provides a complete training loop with evaluation,
    logging, and checkpointing capabilities.

    Args:
        agent: RL agent to train
        env: Training environment
        eval_env: Optional separate evaluation environment
        config: Training configuration

    Example:
        agent = DQNAgent(state_dim=4, action_dim=2)
        env = GymWrapper(gym.make('CartPole-v1'))

        config = TrainingConfig(num_episodes=1000, eval_freq=100)
        trainer = RLTrainer(agent, env, config=config)

        metrics = trainer.train()
        trainer.evaluate(num_episodes=10)
    """

    def __init__(
        self,
        agent: Any,
        env: BaseEnv,
        eval_env: Optional[BaseEnv] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env or env
        self.config = config or TrainingConfig()

        # Training state
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.eval_results: List[Dict[str, float]] = []

        # Set seed
        if self.config.seed is not None:
            self._set_seed(self.config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def train(self) -> Dict[str, List]:
        """Run complete training loop.

        Returns:
            Dictionary containing training metrics
        """
        print(f"Starting training for {self.config.num_episodes} episodes")

        for episode in range(1, self.config.num_episodes + 1):
            episode_reward, episode_length, episode_loss = self._train_episode()

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Logging
            if episode % self.config.log_freq == 0:
                avg_reward = (
                    np.mean(self.episode_rewards[-100:])
                    if len(self.episode_rewards) >= 100
                    else np.mean(self.episode_rewards)
                )
                print(
                    f"Episode {episode}/{self.config.num_episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Length: {episode_length} | "
                    f"Loss: {episode_loss:.4f if episode_loss else 'N/A'}"
                )

            # Evaluation
            if episode % self.config.eval_freq == 0:
                eval_metrics = self.evaluate(self.config.num_eval_episodes)
                self.eval_results.append(eval_metrics)
                print(
                    f"Evaluation at episode {episode}: "
                    f"Mean Reward: {eval_metrics['mean_reward']:.2f} +/- "
                    f"{eval_metrics['std_reward']:.2f} | "
                    f"Success Rate: {eval_metrics['success_rate']:.2%}"
                )

            # Save checkpoint
            if hasattr(self.agent, "save") and episode % self.config.save_freq == 0:
                self.agent.save(f"checkpoint_episode_{episode}.pt")

        print("Training completed!")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_results": self.eval_results,
        }

    def _train_episode(self) -> Tuple[float, int, Optional[float]]:
        """Train for a single episode.

        Returns:
            Tuple of (episode_reward, episode_length, loss)
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        losses = []

        for step in range(self.config.max_steps_per_episode):
            # Select action
            if hasattr(self.agent, "get_action"):
                if hasattr(self.agent, "select_action"):
                    action = self.agent.select_action(state)[0]
                else:
                    action = self.agent.get_action(state)
            else:
                raise ValueError("Agent must have get_action or select_action method")

            # Step environment
            next_state, reward, done, info = self.env.step(action)

            # Store transition
            if hasattr(self.agent, "store_transition"):
                if hasattr(self.agent, "select_action"):
                    # On-policy agent (PPO, A2C)
                    self.agent.store_transition(reward, done)
                else:
                    # Off-policy agent (DQN, SAC)
                    self.agent.store_transition(state, action, reward, next_state, done)

            # Update agent
            if hasattr(self.agent, "update"):
                loss = self.agent.update()
                if loss is not None:
                    if isinstance(loss, dict):
                        losses.append(
                            loss.get("critic_loss", loss.get("total_loss", 0))
                        )
                    else:
                        losses.append(loss)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if self.config.render:
                self.env.render()

            if done:
                break

        # Compute returns for on-policy agents
        if hasattr(self.agent, "compute_returns"):
            self.agent.compute_returns(state, done)
        elif hasattr(self.agent, "compute_returns_and_advantages"):
            self.agent.compute_returns_and_advantages(state, done)

        # Update on-policy agents after episode
        if hasattr(self.agent, "clear_memory") and len(losses) == 0:
            loss_dict = self.agent.update()
            losses.append(
                loss_dict.get("total_loss", 0)
                if isinstance(loss_dict, dict)
                else loss_dict
            )

        avg_loss = np.mean(losses) if losses else None
        return episode_reward, episode_length, avg_loss

    def evaluate(
        self, num_episodes: int = 10, render: bool = False
    ) -> Dict[str, float]:
        """Evaluate agent performance.

        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        lengths = []
        successes = []

        for episode in range(num_episodes):
            state = self.eval_env.reset()
            episode_reward = 0.0
            episode_length = 0

            for step in range(self.config.max_steps_per_episode):
                # Select action deterministically
                if hasattr(self.agent, "get_action"):
                    action = self.agent.get_action(state, epsilon=0.0)
                elif hasattr(self.agent, "select_action"):
                    action = self.agent.select_action(state)[0]
                else:
                    raise ValueError(
                        "Agent must have get_action or select_action method"
                    )

                state, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1

                if render:
                    self.eval_env.render()

                if done:
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)
            successes.append(info.get("success", episode_reward > 0))

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_length": np.mean(lengths),
            "success_rate": np.mean(successes),
        }

    def save_metrics(self, path: str) -> None:
        """Save training metrics to file."""
        import json

        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "eval_results": self.eval_results,
        }
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Environments
    "BaseEnv",
    "GymWrapper",
    "VectorizedEnv",
    "NormalizedEnv",
    # Policy Networks
    "PolicyNetwork",
    "ActorCriticNetwork",
    "QNetwork",
    "DeterministicPolicy",
    # RL Algorithms
    "DQNAgent",
    "DoubleDQNAgent",
    "DuelingDQNAgent",
    "PPOAgent",
    "A2CAgent",
    "SACAgent",
    # Experience Replay
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "MultiStepBuffer",
    # Utilities
    "compute_returns",
    "compute_gae",
    "NoisyLinear",
    # Training
    "TrainingConfig",
    "RLTrainer",
]

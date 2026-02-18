"""
fishstick Reinforcement Learning Module

Advanced reinforcement learning components for building production-ready
RL systems, including policy networks, value networks, actor-critic algorithms,
replay buffers, and exploration strategies.

Modules:
    - policy_networks: Policy networks (categorical, gaussian, deterministic)
    - value_networks: Value networks (V, Q, advantage, dueling architecture)
    - actor_critic: Actor-Critic algorithms (A2C, PPO, SAC)
    - replay_buffers: Experience replay (uniform, prioritized, episodic, HER)
    - exploration: Exploration strategies (epsilon-greedy, entropy, noise)

Example:
    >>> from fishstick.reinforcement import (
    ...     CategoricalPolicy,
    ...     GaussianPolicy,
    ...     PPO,
    ...     PrioritizedReplayBuffer,
    ...     EpsilonGreedy,
    ... )
    >>> policy = CategoricalPolicy(state_dim=4, action_dim=2, hidden_dims=[64, 64])
    >>> ppo = PPO(policy, value_network, lr=3e-4)
"""

from __future__ import annotations

from fishstick.reinforcement.policy_networks import (
    PolicyNetwork,
    CategoricalPolicy,
    GaussianPolicy,
    DeterministicPolicy,
    TanhGaussianPolicy,
    MultiCategoricalPolicy,
    PolicyOutput,
    create_policy_network,
)

from fishstick.reinforcement.value_networks import (
    ValueNetwork,
    StateValueNetwork,
    ActionValueNetwork,
    DuelingQNetwork,
    AdvantageNetwork,
    ContinuousQNetwork,
    ValueOutput,
    create_value_network,
)

from fishstick.reinforcement.actor_critic import (
    ActorCriticBase,
    A2C,
    PPO,
    SAC,
    PPOConfig,
    SACConfig,
    A2CConfig,
    RolloutBuffer,
    compute_gae,
)

from fishstick.reinforcement.replay_buffers import (
    ReplayBufferBase,
    UniformReplayBuffer,
    PrioritizedReplayBuffer,
    EpisodicReplayBuffer,
    HERReplayBuffer,
    MultiStepReplayBuffer,
    Transition,
    Episode,
    ReplayBufferConfig,
    create_replay_buffer,
)

from fishstick.reinforcement.exploration import (
    ExplorationStrategy,
    EpsilonGreedy,
    DecayingEpsilonGreedy,
    EntropyRegularization,
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
    NoisyLinear,
    NoisyNetwork,
    ExplorationConfig,
    create_exploration,
)

__all__ = [
    "PolicyNetwork",
    "CategoricalPolicy",
    "GaussianPolicy",
    "DeterministicPolicy",
    "TanhGaussianPolicy",
    "MultiCategoricalPolicy",
    "PolicyOutput",
    "create_policy_network",
    "ValueNetwork",
    "StateValueNetwork",
    "ActionValueNetwork",
    "DuelingQNetwork",
    "AdvantageNetwork",
    "ContinuousQNetwork",
    "ValueOutput",
    "create_value_network",
    "ActorCriticBase",
    "A2C",
    "PPO",
    "SAC",
    "PPOConfig",
    "SACConfig",
    "A2CConfig",
    "RolloutBuffer",
    "compute_gae",
    "ReplayBufferBase",
    "UniformReplayBuffer",
    "PrioritizedReplayBuffer",
    "EpisodicReplayBuffer",
    "HERReplayBuffer",
    "MultiStepReplayBuffer",
    "Transition",
    "Episode",
    "ReplayBufferConfig",
    "create_replay_buffer",
    "ExplorationStrategy",
    "EpsilonGreedy",
    "DecayingEpsilonGreedy",
    "EntropyRegularization",
    "GaussianNoise",
    "OrnsteinUhlenbeckNoise",
    "NoisyLinear",
    "NoisyNetwork",
    "ExplorationConfig",
    "create_exploration",
]

__version__ = "0.1.0"

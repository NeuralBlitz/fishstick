"""
RL Extensions Module for fishstick.

Advanced reinforcement learning extensions including:
- Model-based RL (MBPO, PlaNet/Dreamer)
- Offline RL (CQL, TD3+BC)
- Curiosity-driven exploration (ICM, RND)
- Hierarchical RL (Option-Critic, HAC)
- Multi-task RL utilities

Example:
    >>> from fishstick.rl_extensions import CQL, MBPO, ICM, OptionCritic
    >>> from fishstick.rl_extensions.planet import LatentDynamicsModel
"""

from __future__ import annotations

from fishstick.rl_extensions.planet import (
    LatentDynamicsModel,
    RecurrentStateSpaceModel,
    RSSM,
    DreamerAgent,
    DreamerConfig,
)

from fishstick.rl_extensions.mbpo import (
    MBPO,
    MBPOConfig,
    EnsembleDynamicsModel,
    RolloutWorker,
    ModelBasedSampler,
)

from fishstick.rl_extensions.cql import (
    CQL,
    CQLConfig,
    ConservativeQLearning,
    ImplicitQLearning,
)

from fishstick.rl_extensions.td3_bc import (
    TD3BC,
    TD3BCConfig,
    TwinDelayedDDPGBC,
)

from fishstick.rl_extensions.curiosity import (
    IntrinsicCuriosityModule,
    RandomNetworkDistillation,
    ICMConfig,
    RNDConfig,
    CuriosityRewardWrapper,
)

from fishstick.rl_extensions.hierarchical import (
    OptionCritic,
    OptionCriticConfig,
    HAC,
    HACConfig,
    HierarchicalPolicy,
    MetaController,
    SubController,
    Option,
)

from fishstick.rl_extensions.multitask import (
    MultiTaskPolicy,
    MultiTaskValueNetwork,
    TaskEmbeddingNetwork,
    TaskIdentifier,
    MultiTaskConfig,
    create_multitask_agent,
)

__all__ = [
    # Model-based RL
    "LatentDynamicsModel",
    "RecurrentStateSpaceModel",
    "RSSM",
    "DreamerAgent",
    "DreamerConfig",
    "MBPO",
    "MBPOConfig",
    "EnsembleDynamicsModel",
    "RolloutWorker",
    "ModelBasedSampler",
    # Offline RL
    "CQL",
    "CQLConfig",
    "ConservativeQLearning",
    "ImplicitQLearning",
    "TD3BC",
    "TD3BCConfig",
    "TwinDelayedDDPGBC",
    # Curiosity
    "IntrinsicCuriosityModule",
    "RandomNetworkDistillation",
    "ICMConfig",
    "RNDConfig",
    "CuriosityRewardWrapper",
    # Hierarchical RL
    "OptionCritic",
    "OptionCriticConfig",
    "HAC",
    "HACConfig",
    "HierarchicalPolicy",
    "MetaController",
    "SubController",
    "Option",
    # Multi-task
    "MultiTaskPolicy",
    "MultiTaskValueNetwork",
    "TaskEmbeddingNetwork",
    "TaskIdentifier",
    "MultiTaskConfig",
    "create_multitask_agent",
]

__version__ = "0.1.0"

"""
fishstick Reinforcement Learning Module

RL algorithms, environments, and utilities.
"""

from fishstick.rl.agents import (
    DQN,
    PolicyGradient,
    ActorCritic,
    PPO,
    SAC,
    TD3,
)
from fishstick.rl.environments import (
    GymEnvironment,
    EnvironmentWrapper,
)
from fishstick.rl.utils import (
    PrioritizedReplayBuffer,
    OrnsteinUhlenbeckProcess,
)
from fishstick.rl.agents import ReplayBuffer

# Import from advanced2 for advanced RL algorithms
from fishstick.rl.advanced2 import (
    # Model-Based RL
    EnsembleDynamicsModel,
    PETS,
    MBPO,
    RSSM,
    Dreamer,
    # Offline RL
    CQL,
    IQL,
    DecisionTransformer,
    CRR,
    AWAC,
    # Hierarchical RL
    OptionCritic,
    HAC,
    FeudalNetwork,
    HIRO,
    # Multi-Agent RL
    MADDPG,
    MAPPO,
    QMIX,
    VDN,
    COMA,
    # Imitation Learning
    BehavioralCloning,
    DAgger,
    GAIL,
    AIRL,
    SQIL,
    # Inverse RL
    MaxEntIRL,
    DeepMaxEnt,
    GCL,
    # Meta-RL
    MAML,
    RL2,
    ProMP,
    PEARL,
    # Utilities
    soft_update,
    hard_update,
    Swish,
)

__all__ = [
    # Agents
    "DQN",
    "PolicyGradient",
    "ActorCritic",
    "PPO",
    "SAC",
    "TD3",
    "ReplayBuffer",
    # Environments
    "GymEnvironment",
    "EnvironmentWrapper",
    # Utils
    "PrioritizedReplayBuffer",
    "OrnsteinUhlenbeckProcess",
    # Model-Based RL
    "EnsembleDynamicsModel",
    "PETS",
    "MBPO",
    "RSSM",
    "Dreamer",
    # Offline RL
    "CQL",
    "IQL",
    "DecisionTransformer",
    "CRR",
    "AWAC",
    # Hierarchical RL
    "OptionCritic",
    "HAC",
    "FeudalNetwork",
    "HIRO",
    # Multi-Agent RL
    "MADDPG",
    "MAPPO",
    "QMIX",
    "VDN",
    "COMA",
    # Imitation Learning
    "BehavioralCloning",
    "DAgger",
    "GAIL",
    "AIRL",
    "SQIL",
    # Inverse RL
    "MaxEntIRL",
    "DeepMaxEnt",
    "GCL",
    # Meta-RL
    "MAML",
    "RL2",
    "ProMP",
    "PEARL",
    # Utilities
    "soft_update",
    "hard_update",
    "Swish",
]

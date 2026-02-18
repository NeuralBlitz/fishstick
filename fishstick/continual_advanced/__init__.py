from .ewc import EWC, FisherInformation, compute_fisher_matrix
from .replay import (
    ReservoirReplay,
    MemoryAwareSynapses,
    GradientEpisodicMemory,
    ExperienceReplayBuffer,
)
from .progressive import (
    ProgressiveNetwork,
    ProgressiveColumn,
    add_lateral_connections,
    adapt_to_task,
)

__all__ = [
    "EWC",
    "FisherInformation",
    "compute_fisher_matrix",
    "ReservoirReplay",
    "MemoryAwareSynapses",
    "GradientEpisodicMemory",
    "ExperienceReplayBuffer",
    "ProgressiveNetwork",
    "ProgressiveColumn",
    "add_lateral_connections",
    "adapt_to_task",
]

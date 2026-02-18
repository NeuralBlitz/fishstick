"""
Memory Networks Module.

Implements differentiable memory architectures:
- Neural Turing Machine (NTM)
- Differentiable Neural Computer (DNC)
- Compute Graph Memory
"""

from fishstick.memory_networks.ntm import (
    NeuralTuringMachine,
    NTMController,
    NTMCell,
)
from fishstick.memory_networks.dnc import (
    DifferentiableNeuralComputer,
    DNCCell,
    AccessModule,
)
from fishstick.memory_networks.compute_graph import (
    GraphMemoryNetwork,
    GraphMemoryCell,
    MemoryGraph,
)

__all__ = [
    "NeuralTuringMachine",
    "NTMController",
    "NTMCell",
    "DifferentiableNeuralComputer",
    "DNCCell",
    "AccessModule",
    "GraphMemoryNetwork",
    "GraphMemoryCell",
    "MemoryGraph",
]

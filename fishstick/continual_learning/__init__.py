"""
Continual Learning Module for Fishstick AI Framework.

Comprehensive implementation of state-of-the-art continual learning methods
focusing on preventing catastrophic forgetting in neural networks.

Key Focus Areas:
- Experience Replay: Buffer-based methods for storing and replaying past experiences
- Elastic Weight Consolidation: Regularization-based approaches
- Progressive Neural Networks: Architecture expansion methods
- Task-Agnostic Learning: Methods that work without explicit task boundaries
- Memory-Aware Training: Training techniques that optimize memory usage

References:
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017)
- Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning" (2017)
- Chaudhry et al., "Efficient Lifelong Learning with A-GEM" (2019)
- Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network" (2018)
- Serra et al., "Overcoming Catastrophic Forgetting with Hard Attention" (2018)
- Aljundi et al., "Memory Aware Synapses: Learning What (not) to forget" (2018)
"""

from typing import Optional, Dict, Any, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

import numpy as np
from collections import deque
import copy
import math

# Experience Replay Modules
from .experience_replay import (
    ExperienceReplay,
    ReservoirBuffer,
    WeightedReplayBuffer,
)

from .prioritized_replay import (
    PrioritizedReplayBuffer,
    SumTree,
    ProportionalPriority,
    RankBasedPriority,
)

from .generative_replay import (
    GenerativeReplayBuffer,
    VAEReplay,
    GANReplay,
    StableReplay,
)

# Regularization Methods
from .elastic_weight_consolidation import (
    EWCRegularizer,
    OnlineEWC,
    DiagonalEWC,
    KFAC_EWC,
)

from .synaptic_intelligence import (
    SynapticIntelligence,
    SIOptimizer,
)

from .memory_aware_synapses import (
    MemoryAwareSynapses,
    MASImportance,
)

# Progressive & Dynamic Architectures
from .progressive_networks import (
    ProgressiveColumn,
    ProgressiveNeuralNetwork,
    AdapterProgressiveNetwork,
)

from .packnet import (
    PackNetPruner,
    PackNetMethod,
    MultiTaskPackNet,
)

from .hard_attention import (
    HardAttentionTask,
    HATMethod,
    TaskEmbedding,
)

# Task-Agnostic Learning
from .task_agnostic import (
    TaskAgnosticContinualLearner,
    MetaLearningAgnostic,
    OMLearnable,
)

from .online_continual import (
    OnlineContinualLearner,
    StreamingLearner,
    SlidingWindowLearner,
)

from .streaming_learning import (
    StreamingMethod,
    ReservoirSampling,
    BoundedMemoryLearner,
)

# Memory-Aware Training
from .memory_aware_training import (
    MemoryAwareTrainer,
    AdaptiveReplayScheduler,
    MemoryEfficientTrainer,
)

from .gem import (
    GradientEpisodicMemory,
    GEMOptimizer,
)

from .agem import (
    AverageGEM,
    EfficientGEM,
    ProjectedGEM,
)

# Evaluation & Utilities
from .evaluation import (
    ContinualMetrics,
    AverageAccuracy,
    ForgettingMeasure,
    BackwardTransfer,
    ForwardTransfer,
    compute_metrics,
)

from .trainer import (
    ContinualTrainer,
    TaskSequence,
    EvaluationProtocol,
)

from .buffers import (
    RingBuffer,
    HerdingBuffer,
    MeanOfFeatures,
    FeatureBuffer,
)


__all__ = [
    # Experience Replay
    "ExperienceReplay",
    "ReservoirBuffer",
    "WeightedReplayBuffer",
    "PrioritizedReplayBuffer",
    "SumTree",
    "ProportionalPriority",
    "RankBasedPriority",
    "GenerativeReplayBuffer",
    "VAEReplay",
    "GANReplay",
    "StableReplay",
    # Regularization
    "EWCRegularizer",
    "OnlineEWC",
    "DiagonalEWC",
    "KFAC_EWC",
    "SynapticIntelligence",
    "SIOptimizer",
    "MemoryAwareSynapses",
    "MASImportance",
    # Progressive Networks
    "ProgressiveColumn",
    "ProgressiveNeuralNetwork",
    "AdapterProgressiveNetwork",
    "PackNetPruner",
    "PackNetMethod",
    "MultiTaskPackNet",
    "HardAttentionTask",
    "HATMethod",
    "TaskEmbedding",
    # Task-Agnostic
    "TaskAgnosticContinualLearner",
    "MetaLearningAgnostic",
    "OMLearnable",
    "OnlineContinualLearner",
    "StreamingLearner",
    "SlidingWindowLearner",
    "StreamingMethod",
    "ReservoirSampling",
    "BoundedMemoryLearner",
    # Memory-Aware
    "MemoryAwareTrainer",
    "AdaptiveReplayScheduler",
    "MemoryEfficientTrainer",
    "GradientEpisodicMemory",
    "GEMOptimizer",
    "AverageGEM",
    "EfficientGEM",
    "ProjectedGEM",
    # Evaluation
    "ContinualMetrics",
    "AverageAccuracy",
    "ForgettingMeasure",
    "BackwardTransfer",
    "ForwardTransfer",
    "compute_metrics",
    "ContinualTrainer",
    "TaskSequence",
    "EvaluationProtocol",
    # Buffers
    "RingBuffer",
    "HerdingBuffer",
    "MeanOfFeatures",
    "FeatureBuffer",
]


__version__ = "0.1.0"

"""
Advanced Memory and Attention Architectures.

A comprehensive library of differentiable memory systems for neural networks.

Modules:
- differentiable_stack: Differentiable data structures (stack, queue, deque, priority queue)
- ntm: Neural Turing Machine and variants
- memory_attention: Memory-augmented attention mechanisms
- content_addressing: Content-based memory addressing implementations
- hierarchical_memory: Hierarchical memory systems (working, episodic, semantic)

Key Classes:
- DifferentiableStack: Differentiable stack memory
- DifferentiableQueue: Differentiable FIFO queue
- DifferentiableDeque: Differentiable double-ended queue
- PriorityQueue: Differentiable priority queue
- NeuralTuringMachine: Complete NTM implementation
- LookupFreeNTM: Memory-efficient NTM variant
- HopfieldAttention: Hopfield network-based attention
- SparseMemoryAttention: Sparse attention for large memories
- KeyValueMemoryAttention: Key-value memory attention
- MemoryAugmentedAttention: Standard attention + memory
- CosineSimilarityAddressing: Cosine similarity content addressing
- EuclideanDistanceAddressing: Distance-based addressing
- MultiHeadContentAddressing: Multi-head addressing
- MemoryBank: Complete memory bank with read/write
- AssociativeMemory: Key-value associative memory
- WorkingMemory: Short-term working memory
- EpisodicMemory: Event-based episodic memory
- SemanticMemory: Structured semantic knowledge
- HierarchicalMemorySystem: Complete hierarchical memory
- ContinualLearningMemory: Experience replay buffer
- MetaLearningMemory: Meta-learning memory

Based on:
- Neural Turing Machines (Graves et al., 2014)
- Differentiable Neural Computers (Graves et al., 2016)
- Memory Networks (Weston et al., 2014)
- Hopfield Networks (Krotov & Hopfield, 2016)
- Differentiable Data Structures (Jang et al., 2016)
"""

from typing import (
    Optional,
    Tuple,
    Dict,
    Any,
    List,
    Union,
    Callable,
)

import torch
from torch import Tensor, nn

# Differentiable Data Structures
from .differentiable_stack import (
    DifferentiableStack,
    DifferentiableQueue,
    DifferentiableDeque,
    PriorityQueue,
    DifferentiableStackEnsemble,
)

# Neural Turing Machine
from .ntm import (
    ContentAddressing,
    LocationAddressing,
    ReadHead,
    WriteHead,
    NTMController,
    NeuralTuringMachine,
    LookupFreeNTM,
    create_ntm,
)

# Memory Attention
from .memory_attention import (
    HopfieldAttention,
    SparseMemoryAttention,
    KeyValueMemoryAttention,
    AssociativeAttention,
    MemoryAugmentedAttention,
    RoutingAttention,
    SetAttention,
)

# Content-Based Addressing
from .content_addressing import (
    AddressingResult,
    CosineSimilarityAddressing,
    EuclideanDistanceAddressing,
    DotProductAddressing,
    LearnedSimilarityAddressing,
    MultiHeadContentAddressing,
    HybridAddressing,
    AttentionBasedAddressing,
    MemoryBank,
    AssociativeMemory,
)

# Hierarchical Memory
from .hierarchical_memory import (
    MemoryType,
    MemoryState,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    HierarchicalMemorySystem,
    ContinualLearningMemory,
    MetaLearningMemory,
)


__all__ = [
    # Differentiable Data Structures
    "DifferentiableStack",
    "DifferentiableQueue",
    "DifferentiableDeque",
    "PriorityQueue",
    "DifferentiableStackEnsemble",
    # Neural Turing Machine
    "ContentAddressing",
    "LocationAddressing",
    "ReadHead",
    "WriteHead",
    "NTMController",
    "NeuralTuringMachine",
    "LookupFreeNTM",
    "create_ntm",
    # Memory Attention
    "HopfieldAttention",
    "SparseMemoryAttention",
    "KeyValueMemoryAttention",
    "AssociativeAttention",
    "MemoryAugmentedAttention",
    "RoutingAttention",
    "SetAttention",
    # Content-Based Addressing
    "AddressingResult",
    "CosineSimilarityAddressing",
    "EuclideanDistanceAddressing",
    "DotProductAddressing",
    "LearnedSimilarityAddressing",
    "MultiHeadContentAddressing",
    "HybridAddressing",
    "AttentionBasedAddressing",
    "MemoryBank",
    "AssociativeMemory",
    # Hierarchical Memory
    "MemoryType",
    "MemoryState",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "HierarchicalMemorySystem",
    "ContinualLearningMemory",
    "MetaLearningMemory",
]


__version__ = "0.1.0"

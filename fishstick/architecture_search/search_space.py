"""
Neural Architecture Search Space Definitions.

Provides comprehensive search space primitives for NAS including:
- Operation primitives (convolution, pooling, attention, etc.)
- Cell-based search spaces
- DARTS-style continuous spaces
- Graph-based architecture representation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import random
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class OperationType(Enum):
    """Types of operations available in the search space."""

    CONV_1X1 = "conv_1x1"
    CONV_3X3 = "conv_3x3"
    CONV_5X5 = "conv_5x5"
    CONV_7X7 = "conv_7x7"
    SEP_CONV_3X3 = "sep_conv_3x3"
    SEP_CONV_5X5 = "sep_conv_5x5"
    SEP_CONV_7X7 = "sep_conv_7x7"
    DIL_CONV_3X3 = "dil_conv_3x3"
    DIL_CONV_5X5 = "dil_conv_5x5"
    MAX_POOL = "max_pool"
    AVG_POOL = "avg_pool"
    GLOBAL_POOL = "global_pool"
    SKIP = "skip"
    ZERO = "zero"
    ATTENTION = "attention"
    LINEAR = "linear"


@dataclass
class OperationSpec:
    """Specification for a single operation in the search space."""

    op_type: OperationType
    kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    groups: int = 1
    expand_ratio: int = 4
    attention_heads: int = 4

    def __hash__(self) -> int:
        return hash(
            (
                self.op_type.value,
                self.kernel_size,
                self.stride,
                self.dilation,
                self.groups,
                self.expand_ratio,
            )
        )

    @property
    def is_zero(self) -> bool:
        return self.op_type == OperationType.ZERO

    @property
    def is_reduction(self) -> bool:
        return self.stride > 1


@dataclass
class EdgeSpec:
    """Specification for an edge in the architecture graph."""

    source_idx: int
    target_idx: int
    operation: OperationSpec

    def __repr__(self) -> str:
        return f"Edge({self.source_idx} -> {self.target_idx}: {self.operation.op_type.value})"


@dataclass
class CellSpec:
    """Specification for a cell (DAG) in the architecture."""

    name: str
    num_nodes: int
    num_edges_per_node: int = 2
    edges: List[EdgeSpec] = field(default_factory=list)

    def add_edge(self, source_idx: int, target_idx: int, operation: OperationSpec):
        """Add an edge to the cell."""
        edge = EdgeSpec(source_idx, target_idx, operation)
        self.edges.append(edge)

    def get_edges_between(self, i: int, j: int) -> List[EdgeSpec]:
        """Get all edges from node i to node j."""
        return [e for e in self.edges if e.source_idx == i and e.target_idx == j]

    def to_adjacency_list(self) -> Dict[int, List[Tuple[int, OperationSpec]]]:
        """Convert to adjacency list representation."""
        adj: Dict[int, List[Tuple[int, OperationSpec]]] = {
            i: [] for i in range(self.num_nodes)
        }
        for edge in self.edges:
            adj[edge.source_idx].append((edge.target_idx, edge.operation))
        return adj


@dataclass
class ArchitectureSpec:
    """
    Complete neural architecture specification.

    Can represent:
    - Cell-based architectures (normal + reduction cells)
    - Sequential architectures
    - DAG-based architectures
    """

    normal_cell: Optional[CellSpec] = None
    reduction_cell: Optional[CellSpec] = None
    stem_channels: int = 32
    num_classes: int = 10
    num_cells_per_stage: int = 5
    num_stages: int = 3
    auxiliary_head: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_cell_based(self) -> bool:
        return self.normal_cell is not None

    def num_parameters(self) -> int:
        """Estimate number of parameters (simplified)."""
        # This is a rough estimation
        base_params = self.stem_channels * 3 * 3 * 3
        cell_params = 0
        if self.normal_cell:
            cell_params += len(self.normal_cell.edges) * self.stem_channels * 64
        if self.reduction_cell:
            cell_params += len(self.reduction_cell.edges) * self.stem_channels * 64
        return base_params + cell_params * self.num_cells_per_stage * self.num_stages


class SearchSpace(ABC):
    """Abstract base class for search spaces."""

    @abstractmethod
    def sample(self, rng: Optional[np.random.RandomState] = None) -> ArchitectureSpec:
        """Sample a random architecture from the search space."""
        pass

    @abstractmethod
    def sample_operation(
        self, rng: Optional[np.random.RandomState] = None
    ) -> OperationSpec:
        """Sample a random operation."""
        pass

    @property
    @abstractmethod
    def num_operations(self) -> int:
        """Number of available operations."""
        pass


class DARTSearchSpace(SearchSpace):
    """
    DARTS-style differentiable search space.

    Uses continuous relaxation for operations and supports:
    - Edge normalization
    - Operation weighting
    - Architecture gradient optimization

    Reference: Liu et al., "DARTS: Differentiable Architecture Search", ICLR 2019
    """

    AVAILABLE_OPERATIONS = [
        OperationType.CONV_1X1,
        OperationType.CONV_3X3,
        OperationType.CONV_5X5,
        OperationType.SEP_CONV_3X3,
        OperationType.SEP_CONV_5X5,
        OperationType.DIL_CONV_3X3,
        OperationType.DIL_CONV_5X5,
        OperationType.MAX_POOL,
        OperationType.AVG_POOL,
        OperationType.SKIP,
    ]

    def __init__(
        self,
        num_nodes: int = 4,
        num_edges: int = 2,
        num_operations: Optional[int] = None,
        reduction_cells: bool = True,
    ):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_operations = num_operations or len(self.AVAILABLE_OPERATIONS)
        self.reduction_cells = reduction_cells

    @property
    def operation_list(self) -> List[OperationType]:
        """Get list of available operations."""
        return self.AVAILABLE_OPERATIONS[: self.num_operations]

    @property
    def num_operations(self) -> int:
        return self._num_operations

    @num_operations.setter
    def num_operations(self, value: int):
        self._num_operations = min(value, len(self.AVAILABLE_OPERATIONS))

    def sample_operation(
        self, rng: Optional[np.random.RandomState] = None
    ) -> OperationSpec:
        """Sample a random operation."""
        if rng is None:
            rng = np.random.RandomState()
        op_type = rng.choice(self.operation_list)

        kernel_size = 3
        if op_type in [
            OperationType.CONV_3X3,
            OperationType.SEP_CONV_3X3,
            OperationType.DIL_CONV_3X3,
        ]:
            kernel_size = 3
        elif op_type in [
            OperationType.CONV_5X5,
            OperationType.SEP_CONV_5X5,
            OperationType.DIL_CONV_5X5,
        ]:
            kernel_size = 5
        elif op_type == OperationType.CONV_7X7:
            kernel_size = 7

        return OperationSpec(op_type=op_type, kernel_size=kernel_size)

    def sample_cell(
        self,
        name: str,
        is_reduction: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> CellSpec:
        """Sample a random cell (DAG of operations)."""
        if rng is None:
            rng = np.random.RandomState()

        cell = CellSpec(
            name=name, num_nodes=self.num_nodes, num_edges_per_node=self.num_edges
        )

        # Create edges: from each previous node to each current node
        for node_idx in range(1, self.num_nodes):
            # Connect from previous nodes
            num_inputs = min(node_idx, self.num_edges)
            source_indices = rng.choice(node_idx, size=num_inputs, replace=False)

            for src_idx in source_indices:
                operation = self.sample_operation(rng)
                if is_reduction and operation.stride == 1:
                    operation = OperationSpec(
                        op_type=operation.op_type,
                        kernel_size=operation.kernel_size,
                        stride=2,  # Reduction
                        dilation=operation.dilation,
                    )
                cell.add_edge(src_idx, node_idx, operation)

        return cell

    def sample(self, rng: Optional[np.random.RandomState] = None) -> ArchitectureSpec:
        """Sample a complete architecture."""
        if rng is None:
            rng = np.random.RandomState()

        normal_cell = self.sample_cell("normal", is_reduction=False, rng=rng)

        reduction_cell = None
        if self.reduction_cells:
            reduction_cell = self.sample_cell("reduction", is_reduction=True, rng=rng)

        return ArchitectureSpec(
            normal_cell=normal_cell,
            reduction_cell=reduction_cell,
        )


class NB201SearchSpace(SearchSpace):
    """
    NAS-Bench-201 search space.

    A constrained search space used in NAS benchmarks.

    Reference: Dong et al., "NAS-Bench-201: Extending the Reproducibility
    of NAS", ICLR 2020
    """

    AVAILABLE_OPERATIONS = [
        OperationType.CONV_1X1,
        OperationType.CONV_3X3,
        OperationType.AVG_POOL,
        OperationType.SKIP,
        OperationType.ZERO,
    ]

    def __init__(self):
        self._num_operations = len(self.AVAILABLE_OPERATIONS)

    @property
    def num_operations(self) -> int:
        return self._num_operations

    def sample_operation(
        self, rng: Optional[np.random.RandomState] = None
    ) -> OperationSpec:
        """Sample a random operation."""
        if rng is None:
            rng = np.random.RandomState()
        op_type = rng.choice(self.AVAILABLE_OPERATIONS)
        return OperationSpec(op_type=op_type)

    def sample(self, rng: Optional[np.random.RandomState] = None) -> ArchitectureSpec:
        """Sample a complete architecture."""
        if rng is None:
            rng = np.random.RandomState()

        # Create a simple cell-based architecture
        num_nodes = 4

        def sample_cell(name: str) -> CellSpec:
            cell = CellSpec(name=name, num_nodes=num_nodes, num_edges_per_node=1)

            # Edge from node 0 to 1
            cell.add_edge(0, 1, self.sample_operation(rng))
            # Edge from node 0 to 2
            cell.add_edge(0, 2, self.sample_operation(rng))
            # Edge from node 1 to 3
            cell.add_edge(1, 3, self.sample_operation(rng))
            # Edge from node 2 to 3
            cell.add_edge(2, 3, self.sample_operation(rng))

            return cell

        return ArchitectureSpec(
            normal_cell=sample_cell("normal"),
            reduction_cell=sample_cell("reduction"),
        )


class MobileNetSearchSpace(SearchSpace):
    """
    MobileNet-style search space for efficient architectures.

    Focuses on operations suitable for mobile/edge deployment:
    - Depthwise separable convolutions
    - MobileInvertedResidual
    - Squeeze-and-Excitation
    """

    AVAILABLE_OPERATIONS = [
        OperationType.SEP_CONV_3X3,
        OperationType.SEP_CONV_5X5,
        OperationType.SEP_CONV_7X7,
        OperationType.CONV_1X1,
        OperationType.AVG_POOL,
        OperationType.SKIP,
    ]

    def __init__(
        self,
        max_blocks: int = 20,
        min_blocks: int = 8,
        expand_ratios: List[int] = field(default_factory=lambda: [3, 4, 6]),
        channel_sizes: List[int] = field(
            default_factory=lambda: [16, 24, 32, 64, 96, 128, 160, 320, 640]
        ),
    ):
        self.max_blocks = max_blocks
        self.min_blocks = min_blocks
        self.expand_ratios = expand_ratios
        self.channel_sizes = channel_sizes
        self._num_operations = len(self.AVAILABLE_OPERATIONS)

    @property
    def num_operations(self) -> int:
        return self._num_operations

    def sample_operation(
        self, rng: Optional[np.random.RandomState] = None
    ) -> OperationSpec:
        """Sample a random operation."""
        if rng is None:
            rng = np.random.RandomState()
        op_type = rng.choice(self.AVAILABLE_OPERATIONS)

        kernel_size = 3
        if op_type == OperationType.SEP_CONV_5X5:
            kernel_size = 5
        elif op_type == OperationType.SEP_CONV_7X7:
            kernel_size = 7

        expand_ratio = rng.choice(self.expand_ratios)

        return OperationSpec(
            op_type=op_type,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio,
        )

    def sample(self, rng: Optional[np.random.RandomState] = None) -> ArchitectureSpec:
        """Sample a complete MobileNet architecture."""
        if rng is None:
            rng = np.random.RandomState()

        # MobileNet typically uses sequential blocks
        # For simplicity, create a cell representation
        num_blocks = rng.randint(self.min_blocks, self.max_blocks + 1)

        cell = CellSpec(name="mobile_blocks", num_nodes=num_blocks + 1)

        for i in range(num_blocks):
            operation = self.sample_operation(rng)
            channel = rng.choice(self.channel_sizes)
            cell.add_edge(i, i + 1, operation)

        return ArchitectureSpec(
            normal_cell=cell,
            stem_channels=rng.choice(self.channel_sizes[:4]),
        )


class ResNetSearchSpace(SearchSpace):
    """
    ResNet-style search space.

    Searches over:
    - Block types (basic, bottleneck)
    - Block configurations
    - Channel sizes
    - Number of blocks per stage
    """

    BLOCK_TYPES = ["basic", "bottleneck"]

    def __init__(
        self,
        base_channels: int = 64,
        max_blocks_per_stage: int = 4,
        channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 8]),
    ):
        self.base_channels = base_channels
        self.max_blocks_per_stage = max_blocks_per_stage
        self.channel_multipliers = channel_multipliers
        self._num_operations = len(self.BLOCK_TYPES)

    @property
    def num_operations(self) -> int:
        return self._num_operations

    def sample_operation(
        self, rng: Optional[np.random.RandomState] = None
    ) -> OperationSpec:
        """Sample a random block type."""
        if rng is None:
            rng = np.random.RandomState()
        block_type = rng.choice(self.BLOCK_TYPES)
        return OperationSpec(
            op_type=OperationType.CONV_3X3
            if block_type == "basic"
            else OperationType.CONV_1X1,
            kernel_size=3 if block_type == "basic" else 1,
        )

    def sample(self, rng: Optional[np.random.RandomState] = None) -> ArchitectureSpec:
        """Sample a complete ResNet-style architecture."""
        if rng is None:
            rng = np.random.RandomState()

        num_stages = len(self.channel_multipliers)
        num_blocks = [
            rng.randint(1, self.max_blocks_per_stage + 1) for _ in range(num_stages)
        ]

        # Create cell representation
        total_blocks = sum(num_blocks)
        cell = CellSpec(name="resnet_blocks", num_nodes=total_blocks + 1)

        block_idx = 0
        for stage_idx in range(num_stages):
            channels = self.base_channels * self.channel_multipliers[stage_idx]
            for _ in range(num_blocks[stage_idx]):
                operation = self.sample_operation(rng)
                operation.expand_ratio = channels
                cell.add_edge(block_idx, block_idx + 1, operation)
                block_idx += 1

        return ArchitectureSpec(
            normal_cell=cell,
            stem_channels=self.base_channels,
        )


def create_search_space(space_type: str = "darts", **kwargs) -> SearchSpace:
    """
    Factory function to create search spaces.

    Args:
        space_type: Type of search space ("darts", "nb201", "mobilenet", "resnet")
        **kwargs: Additional arguments for the search space

    Returns:
        SearchSpace instance
    """
    spaces = {
        "darts": DARTSearchSpace,
        "nb201": NB201SearchSpace,
        "nasbench201": NB201SearchSpace,
        "mobilenet": MobileNetSearchSpace,
        "resnet": ResNetSearchSpace,
    }

    if space_type.lower() not in spaces:
        raise ValueError(
            f"Unknown search space: {space_type}. Available: {list(spaces.keys())}"
        )

    return spaces[space_type.lower()](**kwargs)


def get_operation_info(op_type: OperationType) -> Dict[str, Any]:
    """Get information about an operation type."""
    info = {
        "name": op_type.value,
        "flops_factor": 1.0,
        "params_factor": 1.0,
    }

    if op_type == OperationType.CONV_1X1:
        info.update({"kernel": 1, "reduction": 1.0})
    elif op_type == OperationType.CONV_3X3:
        info.update({"kernel": 3, "reduction": 1.0})
    elif op_type == OperationType.CONV_5X5:
        info.update({"kernel": 5, "reduction": 1.0})
    elif op_type == OperationType.CONV_7X7:
        info.update({"kernel": 7, "reduction": 1.0})
    elif op_type == OperationType.SEP_CONV_3X3:
        info.update({"kernel": 3, "reduction": 0.5, "params_factor": 0.5})
    elif op_type == OperationType.SEP_CONV_5X5:
        info.update({"kernel": 5, "reduction": 0.5, "params_factor": 0.5})
    elif op_type == OperationType.DIL_CONV_3X3:
        info.update({"kernel": 3, "dilation": 2, "receptive_field": "expanded"})
    elif op_type == OperationType.MAX_POOL:
        info.update({"kernel": 3, "reduction": 0.5})
    elif op_type == OperationType.AVG_POOL:
        info.update({"kernel": 3, "reduction": 0.5})
    elif op_type == OperationType.GLOBAL_POOL:
        info.update({"reduction": "global"})
    elif op_type == OperationType.SKIP:
        info.update({"reduction": 1.0})
    elif op_type == OperationType.ZERO:
        info.update({"reduction": "none", "flops_factor": 0.0, "params_factor": 0.0})

    return info

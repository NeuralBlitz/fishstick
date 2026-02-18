"""
Differentiable Neural Architecture Search (DNAS) Module for fishstick.

This module provides a comprehensive suite of differentiable NAS algorithms including:
- DARTS: Differentiable Architecture Search
- Gradient-based methods: GDAS, SNAS, ENAS, ProxylessNAS
- One-shot methods: SinglePathOneShot, FairNAS, OFA, BigNAS
- Hardware-aware NAS: Latency optimization, FLOPS constraints
- Search strategies: Evolutionary, Bayesian, Random search
- Network morphism: Net2Net, progressive growing/shrinking

References:
- DARTS: Liu et al., "DARTS: Differentiable Architecture Search", ICLR 2019
- ProxylessNAS: Cai et al., "ProxylessNAS: Direct Neural Architecture Search", ICLR 2019
- ENAS: Pham et al., "Efficient Neural Architecture Search", ICML 2018
- GDAS: Dong & Yang, "Searching for A Robust Neural Architecture", CVPR 2019
- SNAS: Xie et al., "SNAS: Stochastic Neural Architecture Search", ICLR 2019
- FairNAS: Chu et al., "FairNAS: Rethinking Evaluation Fairness", ICLR 2021
- OFA: Cai et al., "Once-for-All: Train One Network", ICLR 2020
"""

from __future__ import annotations

import copy
import math
import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Architecture Components
# =============================================================================


class OperationType:
    """Enumeration of operation types for NAS."""

    ZERO = "zero"
    SKIP = "skip"
    CONV_1X1 = "conv_1x1"
    CONV_3X3 = "conv_3x3"
    CONV_5X5 = "conv_5x5"
    CONV_7X7 = "conv_7x7"
    SEP_CONV_3X3 = "sep_conv_3x3"
    SEP_CONV_5X5 = "sep_conv_5x5"
    DIL_CONV_3X3 = "dil_conv_3x3"
    DIL_CONV_5X5 = "dil_conv_5x5"
    MAX_POOL_3X3 = "max_pool_3x3"
    AVG_POOL_3X3 = "avg_pool_3x3"
    GLOBAL_AVG_POOL = "global_avg_pool"
    IDENTITY = "identity"

    @classmethod
    def all_ops(cls) -> List[str]:
        """Return all operation types."""
        return [
            cls.ZERO,
            cls.SKIP,
            cls.CONV_1X1,
            cls.CONV_3X3,
            cls.CONV_5X5,
            cls.SEP_CONV_3X3,
            cls.SEP_CONV_5X5,
            cls.DIL_CONV_3X3,
            cls.DIL_CONV_5X5,
            cls.MAX_POOL_3X3,
            cls.AVG_POOL_3X3,
            cls.IDENTITY,
        ]


@dataclass
class ArchParam:
    """Architecture parameter for differentiable search."""

    name: str
    shape: Tuple[int, ...]
    requires_grad: bool = True

    def create(self, device: str = "cpu") -> nn.Parameter:
        """Create the parameter tensor."""
        param = nn.Parameter(
            torch.randn(*self.shape) * 1e-3, requires_grad=self.requires_grad
        )
        return param.to(device)


@dataclass
class Edge:
    """Edge in the computational DAG."""

    src: int
    dst: int
    op_weights: Optional[Tensor] = None

    def __hash__(self) -> int:
        return hash((self.src, self.dst))


@dataclass
class CellSpec:
    """Cell specification for NAS."""

    num_nodes: int
    edges: List[Edge] = field(default_factory=list)
    is_reduction: bool = False

    def add_edge(self, src: int, dst: int) -> Edge:
        """Add an edge to the cell."""
        edge = Edge(src, dst)
        self.edges.append(edge)
        return edge


@dataclass
class Genotype:
    """Discrete architecture genotype."""

    normal: List[Tuple[str, int]]
    normal_concat: List[int]
    reduce: List[Tuple[str, int]]
    reduce_concat: List[int]

    def __str__(self) -> str:
        return f"Genotype(normal={self.normal}, reduce={self.reduce})"


# =============================================================================
# Operation Implementations
# =============================================================================


class ReLUConvBN(nn.Module):
    """ReLU -> Conv -> BN block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class SepConv(nn.Module):
    """Separable convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=1,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class DilConv(nn.Module):
    """Dilated convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 2,
    ):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class IdentityOp(nn.Module):
    """Identity operation."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class ZeroOp(nn.Module):
    """Zero operation."""

    def __init__(self, stride: int = 1):
        super().__init__()
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    """Reduce spatial dimensions by factorized convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
    ):
        super().__init__()
        assert stride == 2
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels // 2, 1, stride=stride, padding=0, bias=False
        )
        self.conv_2 = nn.Conv2d(
            in_channels, out_channels // 2, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)


# =============================================================================
# DARTS Architecture
# =============================================================================


class MixedOp(nn.Module):
    """
    Mixed operation that combines multiple candidate operations.

    Each candidate operation is applied and outputs are weighted by
    learnable architecture parameters.

    Reference: DARTS: Differentiable Architecture Search
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        operations: Optional[List[str]] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Default operations if not specified
        if operations is None:
            operations = [
                OperationType.ZERO,
                OperationType.IDENTITY,
                OperationType.CONV_3X3,
                OperationType.CONV_5X5,
                OperationType.SEP_CONV_3X3,
                OperationType.SEP_CONV_5X5,
                OperationType.DIL_CONV_3X3,
                OperationType.MAX_POOL_3X3,
                OperationType.AVG_POOL_3X3,
            ]

        self.operations = operations
        self._ops = nn.ModuleList()

        for op_name in operations:
            op = self._build_operation(op_name)
            self._ops.append(op)

    def _build_operation(self, op_name: str) -> nn.Module:
        """Build a single operation module."""
        if op_name == OperationType.ZERO:
            return ZeroOp(self.stride)
        elif op_name in (OperationType.IDENTITY, OperationType.SKIP):
            if self.stride == 1 and self.in_channels == self.out_channels:
                return IdentityOp()
            else:
                return FactorizedReduce(
                    self.in_channels, self.out_channels, self.stride
                )
        elif op_name == OperationType.CONV_1X1:
            return ReLUConvBN(self.in_channels, self.out_channels, 1, self.stride, 0)
        elif op_name == OperationType.CONV_3X3:
            return ReLUConvBN(self.in_channels, self.out_channels, 3, self.stride, 1)
        elif op_name == OperationType.CONV_5X5:
            return ReLUConvBN(self.in_channels, self.out_channels, 5, self.stride, 2)
        elif op_name == OperationType.SEP_CONV_3X3:
            return SepConv(self.in_channels, self.out_channels, 3, self.stride, 1)
        elif op_name == OperationType.SEP_CONV_5X5:
            return SepConv(self.in_channels, self.out_channels, 5, self.stride, 2)
        elif op_name == OperationType.DIL_CONV_3X3:
            return DilConv(self.in_channels, self.out_channels, 3, self.stride, 2, 2)
        elif op_name == OperationType.DIL_CONV_5X5:
            return DilConv(self.in_channels, self.out_channels, 5, self.stride, 4, 2)
        elif op_name == OperationType.MAX_POOL_3X3:
            return nn.Sequential(
                nn.MaxPool2d(3, stride=self.stride, padding=1),
                nn.BatchNorm2d(self.out_channels),
            )
        elif op_name == OperationType.AVG_POOL_3X3:
            return nn.Sequential(
                nn.AvgPool2d(3, stride=self.stride, padding=1),
                nn.BatchNorm2d(self.out_channels),
            )
        else:
            raise ValueError(f"Unknown operation: {op_name}")

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        """
        Forward pass with weighted operations.

        Args:
            x: Input tensor
            weights: Architecture weights [num_operations]

        Returns:
            Weighted sum of operation outputs
        """
        outputs = []
        for w, op in zip(weights, self._ops):
            if w > 1e-8:  # Skip near-zero weights
                outputs.append(w * op(x))

        return sum(outputs) if outputs else torch.zeros_like(x)


class DARTSCell(nn.Module):
    """
    Differentiable cell for DARTS.

    Implements a directed acyclic graph (DAG) where each node
    aggregates outputs from previous nodes through mixed operations.

    Args:
        steps: Number of intermediate nodes
        multiplier: Number of nodes to concatenate for output
        C_prev_prev: Channels from cell k-2
        C_prev: Channels from cell k-1
        C: Number of channels
        reduction: Whether this is a reduction cell
        reduction_prev: Whether previous cell was reduction
    """

    def __init__(
        self,
        steps: int,
        multiplier: int,
        C_prev_prev: int,
        C_prev: int,
        C: int,
        reduction: bool,
        reduction_prev: bool,
        operations: Optional[List[str]] = None,
    ):
        super().__init__()

        self.reduction = reduction
        self.num_edges = steps * (steps + 3) // 2  # Edges in DAG

        # Preprocess inputs
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self._steps = steps
        self._multiplier = multiplier

        # Build mixed operations for each edge
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):  # 2 inputs + previous nodes
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, C, stride, operations)
                self._ops.append(op)

    def forward(self, s0: Tensor, s1: Tensor, weights: Tensor) -> Tensor:
        """
        Forward pass through the cell.

        Args:
            s0: Input from cell k-2
            s1: Input from cell k-1
            weights: Architecture weights [num_edges, num_operations]

        Returns:
            Cell output tensor
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0

        for i in range(self._steps):
            # Compute node i+2 (intermediate nodes)
            s = sum(
                self._ops[offset + j](h, weights[offset + j])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        # Concatenate last multiplier nodes
        return torch.cat(states[-self._multiplier :], dim=1)


class ArchitectureParameters(nn.Module):
    """
    Learnable architecture parameters for differentiable search.

    Maintains alpha parameters for normal and reduction cells
    that control operation selection via softmax.
    """

    def __init__(
        self,
        num_edges: int,
        num_ops: int,
        init_alphas: Optional[Tensor] = None,
    ):
        super().__init__()

        self.num_edges = num_edges
        self.num_ops = num_ops

        if init_alphas is not None:
            self.alphas_normal = nn.Parameter(init_alphas.clone())
            self.alphas_reduce = nn.Parameter(init_alphas.clone())
        else:
            self.alphas_normal = nn.Parameter(torch.randn(num_edges, num_ops) * 1e-3)
            self.alphas_reduce = nn.Parameter(torch.randn(num_edges, num_ops) * 1e-3)

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Return normalized architecture weights."""
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)
        return weights_normal, weights_reduce

    def get_weights(self, cell_type: str = "normal") -> Tensor:
        """Get architecture weights for specific cell type."""
        if cell_type == "normal":
            return F.softmax(self.alphas_normal, dim=-1)
        elif cell_type == "reduce":
            return F.softmax(self.alphas_reduce, dim=-1)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

    def arch_parameters(self) -> List[nn.Parameter]:
        """Return list of architecture parameters."""
        return [self.alphas_normal, self.alphas_reduce]


class DARTSSearchSpace:
    """
    Search space definition for DARTS.

    Defines the complete space of architectures including:
    - Operations: convolutions, pooling, skip connections
    - Topology: DAG structure with N nodes
    - Constraints: max edges, valid operations
    """

    def __init__(
        self,
        num_nodes: int = 4,
        num_ops: int = 8,
        operations: Optional[List[str]] = None,
    ):
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        self.operations = operations or self._default_operations()

        # Calculate number of edges
        self.num_edges = sum(range(2, 2 + num_nodes))

    def _default_operations(self) -> List[str]:
        """Default operation set for DARTS."""
        return [
            OperationType.ZERO,
            OperationType.IDENTITY,
            OperationType.CONV_3X3,
            OperationType.CONV_5X5,
            OperationType.SEP_CONV_3X3,
            OperationType.SEP_CONV_5X5,
            OperationType.DIL_CONV_3X3,
            OperationType.AVG_POOL_3X3,
        ]

    def sample_random(self) -> Genotype:
        """Sample a random architecture from the search space."""
        normal = []
        reduce = []

        for i in range(self.num_nodes):
            for j in range(2):  # 2 inputs per node
                op = random.choice(self.operations)
                src = random.randint(0, i + 1)
                normal.append((op, src))
                reduce.append((op, src))

        return Genotype(
            normal=normal,
            normal_concat=list(range(2, 2 + self.num_nodes)),
            reduce=reduce,
            reduce_concat=list(range(2, 2 + self.num_nodes)),
        )

    def is_valid(self, genotype: Genotype) -> bool:
        """Check if genotype is valid in this search space."""
        expected_edges = self.num_nodes * 2
        if len(genotype.normal) != expected_edges:
            return False
        if len(genotype.reduce) != expected_edges:
            return False

        for op, _ in genotype.normal + genotype.reduce:
            if op not in self.operations:
                return False

        return True

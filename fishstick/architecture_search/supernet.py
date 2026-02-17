"""
Super-Net Training Utilities for Neural Architecture Search.

Provides super-net implementations and training utilities for:
- Weight sharing in NAS
- Gradient-based architecture search
- Performance estimation
- Architecture evaluation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np

from .search_space import (
    ArchitectureSpec,
    CellSpec,
    DARTSearchSpace,
    EdgeSpec,
    OperationSpec,
    OperationType,
    SearchSpace,
)
from .controller import ArchitectureController


class MixedOperation(nn.Module):
    """
    Mixed operation with multiple paths, selected via softmax weights.

    Implements the differentiable architecture search mixture where
    all operations are applied and their outputs are weighted-summed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        operation_specs: List[OperationSpec],
        stride: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_operations = len(operation_specs)

        # Build operation modules
        self.operations = nn.ModuleList()
        for spec in operation_specs:
            op = self._build_operation(spec)
            self.operations.append(op)

        # Zero operation (for skipping)
        self.zero_op = ZeroOperation(out_channels, stride)

    def _build_operation(self, spec: OperationSpec) -> nn.Module:
        """Build a single operation from specification."""
        op_type = spec.op_type

        if op_type == OperationType.CONV_1X1:
            return nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    1,
                    stride=self.stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_type == OperationType.CONV_3X3:
            return nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    3,
                    stride=self.stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_type == OperationType.CONV_5X5:
            return nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    5,
                    stride=self.stride,
                    padding=2,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_type == OperationType.SEP_CONV_3X3:
            return nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    3,
                    stride=self.stride,
                    padding=1,
                    groups=self.in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_type == OperationType.SEP_CONV_5X5:
            return nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.in_channels,
                    5,
                    stride=self.stride,
                    padding=2,
                    groups=self.in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_type == OperationType.DIL_CONV_3X3:
            return nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    3,
                    stride=self.stride,
                    padding=2,
                    dilation=2,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            )

        elif op_type == OperationType.MAX_POOL:
            return nn.MaxPool2d(3, stride=self.stride, padding=1)

        elif op_type == OperationType.AVG_POOL:
            return nn.AvgPool2d(3, stride=self.stride, padding=1)

        elif op_type == OperationType.SKIP:
            return SkipConnection(self.in_channels, self.out_channels, self.stride)

        else:
            # Default: 1x1 conv
            return nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.out_channels,
                    1,
                    stride=self.stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, x: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with optional weights for operation selection.

        Args:
            x: Input tensor
            weights: Optional weights for each operation [num_operations]

        Returns:
            Weighted sum of operation outputs
        """
        if weights is None:
            # Equal weights
            weights = (
                torch.ones(self.num_operations, device=x.device) / self.num_operations
            )

        # Apply zero operation
        zero_output = self.zero_op(x)

        # Apply each operation and compute weighted sum
        result = torch.zeros_like(x)

        for i, (op, weight) in enumerate(zip(self.operations, weights)):
            if weight > 1e-10:  # Skip near-zero weights
                result = result + weight * op(x)

        # Handle zero operation
        zero_weight = weights[-1] if len(weights) > self.num_operations else 0.0
        if zero_weight > 1e-10:
            result = result + zero_weight * zero_output

        return result


class ZeroOperation(nn.Module):
    """Zero operation that outputs zeros."""

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.channels = channels
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # Return zero tensor with same shape
        return torch.zeros_like(x)


class SkipConnection(nn.Module):
    """Skip connection operation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if self.stride > 1:
            x = F.adaptive_avg_pool2d(x, 1)

        if self.in_channels != self.out_channels:
            # Project to output channels
            padding = (self.out_channels - self.in_channels) // 2
            # Pad channels
            x = F.pad(x, [0, 0, 0, 0, 0, self.out_channels - self.in_channels])

        return x


class MixedCell(nn.Module):
    """
    Mixed cell with multiple edges and operations.

    Each edge has a mixed operation, and edges are combined
    based on learnable weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        operation_specs: List[OperationSpec],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes

        # Create mixed operations for each edge
        # Edges: node i gets input from nodes 0..i-1
        self.edge_ops = nn.ModuleDict()

        for node_idx in range(1, num_nodes):
            for src_idx in range(node_idx):
                edge_key = f"{src_idx}_{node_idx}"

                # For DARTS, each edge has all operations as mixed
                self.edge_ops[edge_key] = MixedOperation(
                    in_channels,
                    out_channels,
                    operation_specs,
                )

        # Output concatenation
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(
        self, inputs: List[Tensor], weights: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """
        Forward pass through the cell.

        Args:
            inputs: List of input tensors from previous cells
            weights: Optional weights for each edge

        Returns:
            Output tensor
        """
        # Preprocess inputs
        processed = [self.preprocess(inp) for inp in inputs]

        # Node states
        states = [processed[0]] if processed else [torch.zeros_like(processed[0])]

        # Process each node
        for node_idx in range(1, self.num_nodes):
            # Aggregate inputs from previous nodes
            node_input = torch.zeros_like(processed[0])

            for src_idx in range(node_idx):
                edge_key = f"{src_idx}_{node_idx}"

                if weights is not None and edge_key in weights:
                    edge_weights = weights[edge_key]
                else:
                    edge_weights = None

                edge_output = self.edge_ops[edge_key](processed[src_idx], edge_weights)
                node_input = node_input + edge_output

            states.append(node_input)

        # Concatenate all node outputs
        return torch.cat(states[1:], dim=1)


class SuperNet(nn.Module):
    """
    Super-net for differentiable architecture search.

    A supernet that contains all possible architectures as subgraphs,
    with shared weights and differentiable operation selection.

    Reference: Liu et al., "DARTS: Differentiable Architecture Search", ICLR 2019
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_nodes: int = 4,
        num_operations: int = 8,
        stem_channels: int = 32,
        num_classes: int = 10,
        num_stages: int = 3,
        auxiliary_head: bool = False,
    ):
        super().__init__()

        self.search_space = search_space
        self.num_nodes = num_nodes
        self.stem_channels = stem_channels
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.auxiliary_head = auxiliary_head

        # Operation specifications
        if isinstance(search_space, DARTSearchSpace):
            self.operation_specs = [
                OperationSpec(op_type=op)
                for op in search_space.AVAILABLE_OPERATIONS[:num_operations]
            ]
        else:
            self.operation_specs = [
                OperationSpec(op_type=op)
                for op in search_space.AVAILABLE_OPERATIONS[:num_operations]
            ]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
        )

        # Cells for each stage
        self.cells = nn.ModuleDict()
        channels = stem_channels

        for stage_idx in range(num_stages):
            is_reduction = stage_idx > 0

            # Reduction cell (every 3 stages)
            if is_reduction and stage_idx % 3 == 0:
                channels = channels * 2
                self.cells[f"reduction_{stage_idx}"] = MixedCell(
                    channels // 2,
                    channels,
                    num_nodes,
                    self.operation_specs,
                )

            # Normal cell
            self.cells[f"normal_{stage_idx}"] = MixedCell(
                channels,
                channels,
                num_nodes,
                self.operation_specs,
            )

        # Final layers
        self.final_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * num_nodes, num_classes),
        )

        # Auxiliary head for early exit training
        if auxiliary_head:
            self.auxiliary_head_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, num_classes),
            )

    def forward(
        self,
        x: Tensor,
        weights: Optional[Dict[str, Tensor]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        """
        Forward pass through the super-net.

        Args:
            x: Input tensor
            weights: Optional architecture weights for operation selection

        Returns:
            Output logits, or (logits, auxiliary_logits) if auxiliary_head is True
        """
        # Stem
        x = self.stem(x)

        # Cells
        for stage_idx in range(self.num_stages):
            # Reduction cell
            if stage_idx > 0 and stage_idx % 3 == 0:
                reduction_key = f"reduction_{stage_idx}"
                x = self.cells[reduction_key]([x], weights)

            # Normal cell
            normal_key = f"normal_{stage_idx}"
            x = self.cells[normal_key]([x], weights)

        # Final prediction
        logits = self.final_conv(x)

        if self.auxiliary_head and self.training:
            aux_logits = self.auxiliary_head_layer(x)
            return logits, aux_logits

        return logits

    def get_architecture_weights(self) -> Dict[str, Tensor]:
        """Get current architecture weights from mixed operations."""
        weights = {}

        for cell_name, cell in self.cells.items():
            for edge_name, edge_op in cell.edge_ops.items():
                key = f"{cell_name}_{edge_name}"
                # Get average operation weights (simplified)
                weights[key] = torch.ones(len(edge_op.operations)) / len(
                    edge_op.operations
                )

        return weights

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class SuperNetTrainer:
    """
    Trainer for super-net architecture search.

    Handles:
    - Joint optimization of architecture and model weights
    - Learning rate scheduling
    - Architecture weight updates
    - Logging and checkpointing
    """

    def __init__(
        self,
        supernet: SuperNet,
        controller: Optional[ArchitectureController] = None,
        lr_model: float = 0.01,
        lr_arch: float = 3e-4,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        unrolled_steps: int = 1,
        grad_clip: float = 5.0,
    ):
        self.supernet = supernet
        self.controller = controller
        self.grad_clip = grad_clip

        # Optimizers
        self.model_optimizer = torch.optim.SGD(
            supernet.parameters(),
            lr=lr_model,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        if controller is not None:
            self.arch_optimizer = torch.optim.Adam(
                controller.parameters(),
                lr=lr_arch,
                betas=(0.5, 0.999),
            )
        else:
            self.arch_optimizer = None

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer,
            T_max=100,
        )

    def train_step(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        step: int,
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            step: Current step number

        Returns:
            Dictionary of metrics
        """
        self.supernet.train()

        # Get training batch
        inputs, targets = next(iter(train_loader))

        # Forward pass
        outputs = self.supernet(inputs)

        if isinstance(outputs, tuple):
            logits, aux_logits = outputs
            # Auxiliary loss
            loss = criterion(logits, targets) + 0.4 * criterion(aux_logits, targets)
        else:
            loss = criterion(outputs, targets)

        # Backward
        self.model_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.supernet.parameters(), self.grad_clip)

        self.model_optimizer.step()

        # Update architecture weights (if controller available)
        if self.controller is not None and self.arch_optimizer is not None:
            self._update_arch_weights(val_loader, criterion)

        # Update scheduler
        self.scheduler.step()

        return {
            "loss": loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def _update_arch_weights(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ):
        """Update architecture weights using validation loss."""
        # Simplified DARTS-style update
        # In full implementation, this would use unrolled optimization

        self.controller.train()
        self.arch_optimizer.zero_grad()

        # Get validation batch
        val_inputs, val_targets = next(iter(val_loader))

        # Forward with current architecture
        outputs = self.supernet(val_inputs, weights=None)
        val_loss = criterion(outputs, val_targets)

        # Architecture regularization
        arch_weights = self.controller.forward_edge_weights("normal")
        reg_loss = 1e-3 * (arch_weights**2).mean()

        total_loss = val_loss + reg_loss
        total_loss.backward()

        self.arch_optimizer.step()

    def search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        criterion: nn.Module = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> ArchitectureSpec:
        """
        Run architecture search.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of search epochs
            criterion: Loss function (default: CrossEntropyLoss)
            callback: Optional callback function

        Returns:
            Best architecture found
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        best_arch = None
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            # Training step
            metrics = self.train_step(train_loader, val_loader, criterion, epoch)

            # Get current architecture
            if self.controller is not None:
                current_arch = self.controller.sample_architecture(hard=True)
            else:
                current_arch = self.supernet.search_space.sample()

            # Evaluate (simplified)
            self.supernet.eval()
            with torch.no_grad():
                val_inputs, val_targets = next(iter(val_loader))
                outputs = self.supernet(val_inputs)
                _, predicted = outputs.max(1)
                val_acc = (
                    100.0 * predicted.eq(val_targets).sum().item() / val_targets.size(0)
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_arch = current_arch

            if callback:
                callback(epoch, {"val_acc": val_acc, **metrics})

        return best_arch


def build_supernet_from_architecture(
    architecture: ArchitectureSpec,
    stem_channels: int = 32,
    num_classes: int = 10,
) -> nn.Module:
    """
    Build a concrete model from an architecture specification.

    Args:
        architecture: Architecture specification
        stem_channels: Number of stem channels
        num_classes: Number of output classes

    Returns:
        PyTorch model
    """
    # This would build a concrete model from the architecture spec
    # For now, return a simple ResNet-like model
    model = nn.Sequential(
        nn.Conv2d(3, stem_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(stem_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(stem_channels, stem_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(stem_channels),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(stem_channels, num_classes),
    )

    return model


@dataclass
class SuperNetConfig:
    """Configuration for super-net."""

    search_space: SearchSpace
    num_nodes: int = 4
    num_operations: int = 8
    stem_channels: int = 32
    num_classes: int = 10
    num_stages: int = 3
    auxiliary_head: bool = False


def create_supernet(config: SuperNetConfig) -> SuperNet:
    """Create a super-net from configuration."""
    return SuperNet(
        search_space=config.search_space,
        num_nodes=config.num_nodes,
        num_operations=config.num_operations,
        stem_channels=config.stem_channels,
        num_classes=config.num_classes,
        num_stages=config.num_stages,
        auxiliary_head=config.auxiliary_head,
    )

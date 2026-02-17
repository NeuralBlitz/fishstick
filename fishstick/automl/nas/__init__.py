"""
Neural Architecture Search (NAS) Module for fishstick

Provides automated neural architecture search capabilities including:
- Random search over architectures
- Evolutionary algorithm-based search
- Cell-based search spaces
- Architecture evaluation and complexity estimation

This module integrates with the fishstick training infrastructure for seamless
architecture discovery and optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple, Union, Set
import copy
import random
import time
import warnings
from collections import defaultdict
from pathlib import Path
import json

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


# ============================================================================
# Search Space Definitions
# ============================================================================


@dataclass
class Operation:
    """Base class for operations in the search space."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.params is None:
            self.params = {}


class ConvOperation(Operation):
    """Convolution operation with various configurations."""

    KERNEL_SIZES = [1, 3, 5, 7]

    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        separable: bool = False,
        name: Optional[str] = None,
    ):
        if kernel_size not in self.KERNEL_SIZES:
            raise ValueError(f"kernel_size must be one of {self.KERNEL_SIZES}")

        if padding is None:
            padding = kernel_size // 2

        super().__init__(
            name=name or f"conv{kernel_size}x{kernel_size}",
            params={
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
                "separable": separable,
            },
        )


class PoolingOperation(Operation):
    """Pooling operation."""

    TYPES = ["max", "avg", "global_max", "global_avg"]

    def __init__(
        self,
        pool_type: str = "max",
        kernel_size: int = 3,
        stride: Optional[int] = None,
        padding: Optional[int] = None,
        name: Optional[str] = None,
    ):
        if pool_type not in self.TYPES:
            raise ValueError(f"pool_type must be one of {self.TYPES}")

        if stride is None:
            stride = kernel_size if "global" not in pool_type else 1
        if padding is None:
            padding = kernel_size // 2

        super().__init__(
            name=name or f"{pool_type}_pool",
            params={
                "pool_type": pool_type,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
            },
        )


class ActivationOperation(Operation):
    """Activation function operation."""

    TYPES = ["relu", "leaky_relu", "elu", "selu", "gelu", "swish", "mish", "none"]

    def __init__(self, activation_type: str = "relu", name: Optional[str] = None):
        if activation_type not in self.TYPES:
            raise ValueError(f"activation_type must be one of {self.TYPES}")
        super().__init__(
            name=name or activation_type, params={"activation_type": activation_type}
        )


class SkipOperation(Operation):
    """Skip connection or identity operation."""

    def __init__(self, name: str = "skip"):
        super().__init__(name=name, params={"type": "skip"})


class NoneOperation(Operation):
    """Null operation (no connection)."""

    def __init__(self, name: str = "none"):
        super().__init__(name=name, params={"type": "none"})


@dataclass
class SearchSpace:
    """
    Define the search space for neural architecture search.

    Args:
        operations: List of available operations
        max_depth: Maximum depth of the network
        min_depth: Minimum depth of the network
        width_multipliers: Available width multipliers
        cell_based: Whether to use cell-based search (normal + reduction cells)
        num_nodes: Number of nodes per cell (for cell-based search)
        num_edges: Number of edges per node

    Example:
        >>> space = SearchSpace(
        ...     operations=[
        ...         ConvOperation(3),
        ...         ConvOperation(5),
        ...         PoolingOperation("max"),
        ...         SkipOperation(),
        ...     ],
        ...     max_depth=8,
        ...     cell_based=True,
        ... )
    """

    operations: List[Operation]
    max_depth: int = 8
    min_depth: int = 2
    width_multipliers: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5]
    )
    cell_based: bool = False
    num_nodes: int = 4
    num_edges: int = 2

    def __post_init__(self):
        if not self.operations:
            raise ValueError("Search space must have at least one operation")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth cannot be greater than max_depth")

    def sample_operation(
        self, rng: Optional[np.random.RandomState] = None
    ) -> Operation:
        """Sample a random operation from the search space."""
        if rng is None:
            rng = np.random.RandomState()
        op_idx = rng.randint(0, len(self.operations))
        return copy.deepcopy(self.operations[op_idx])

    def sample_width(self, rng: Optional[np.random.RandomState] = None) -> float:
        """Sample a random width multiplier."""
        if rng is None:
            rng = np.random.RandomState()
        return rng.choice(self.width_multipliers)


# ============================================================================
# Architecture Representation
# ============================================================================


@dataclass
class LayerSpec:
    """Specification for a single layer."""

    operation: Operation
    in_channels: int
    out_channels: int
    width_multiplier: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "operation": self.operation.name,
            "operation_params": self.operation.params,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "width_multiplier": self.width_multiplier,
        }


@dataclass
class CellSpec:
    """Specification for a cell in cell-based architectures."""

    name: str
    nodes: List[Dict[str, Any]] = field(default_factory=list)

    def add_node(self, inputs: List[int], operations: List[Operation]):
        """Add a node to the cell."""
        self.nodes.append({"inputs": inputs, "operations": operations})

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "nodes": [
                {
                    "inputs": node["inputs"],
                    "operations": [op.name for op in node["operations"]],
                }
                for node in self.nodes
            ],
        }


@dataclass
class Architecture:
    """
    Represents a neural network architecture.

    Can represent either:
    - Sequential architectures (layers list)
    - Cell-based architectures (normal_cell + reduction_cell)
    """

    layers: List[LayerSpec] = field(default_factory=list)
    normal_cell: Optional[CellSpec] = None
    reduction_cell: Optional[CellSpec] = None
    cell_based: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert architecture to dictionary."""
        result = {"cell_based": self.cell_based, "metadata": self.metadata}

        if self.cell_based:
            result["normal_cell"] = (
                self.normal_cell.to_dict() if self.normal_cell else None
            )
            result["reduction_cell"] = (
                self.reduction_cell.to_dict() if self.reduction_cell else None
            )
        else:
            result["layers"] = [layer.to_dict() for layer in self.layers]

        return result

    def save(self, filepath: str):
        """Save architecture to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Architecture":
        """Load architecture from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        # Reconstruct architecture from dict (simplified)
        return cls(metadata=data.get("metadata", {}))


# ============================================================================
# Architecture Builder
# ============================================================================


class ArchitectureBuilder:
    """Build PyTorch models from Architecture specifications."""

    @staticmethod
    def build_layer(layer_spec: LayerSpec) -> nn.Module:
        """Build a single layer from specification."""
        op = layer_spec.operation
        out_channels = int(layer_spec.out_channels * layer_spec.width_multiplier)

        if isinstance(op, ConvOperation):
            return ArchitectureBuilder._build_conv(layer_spec, out_channels)
        elif isinstance(op, PoolingOperation):
            return ArchitectureBuilder._build_pool(op)
        elif isinstance(op, ActivationOperation):
            return ArchitectureBuilder._build_activation(op)
        elif isinstance(op, SkipOperation):
            return nn.Identity()
        elif isinstance(op, NoneOperation):
            return None
        else:
            raise ValueError(f"Unknown operation type: {type(op)}")

    @staticmethod
    def _build_conv(layer_spec: LayerSpec, out_channels: int) -> nn.Module:
        """Build convolution layer."""
        op = layer_spec.operation
        in_channels = layer_spec.in_channels

        if op.params.get("separable", False):
            # Depthwise separable convolution
            layers = [
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=op.params["kernel_size"],
                    stride=op.params["stride"],
                    padding=op.params["padding"],
                    dilation=op.params["dilation"],
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        else:
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=op.params["kernel_size"],
                    stride=op.params["stride"],
                    padding=op.params["padding"],
                    dilation=op.params["dilation"],
                    groups=op.params.get("groups", 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

        return nn.Sequential(*layers)

    @staticmethod
    def _build_pool(op: PoolingOperation) -> nn.Module:
        """Build pooling layer."""
        params = op.params
        pool_type = params["pool_type"]

        if pool_type == "max":
            return nn.MaxPool2d(
                kernel_size=params["kernel_size"],
                stride=params["stride"],
                padding=params["padding"],
            )
        elif pool_type == "avg":
            return nn.AvgPool2d(
                kernel_size=params["kernel_size"],
                stride=params["stride"],
                padding=params["padding"],
            )
        elif pool_type == "global_max":
            return nn.AdaptiveMaxPool2d(1)
        elif pool_type == "global_avg":
            return nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")

    @staticmethod
    def _build_activation(op: ActivationOperation) -> nn.Module:
        """Build activation layer."""
        act_type = op.params["activation_type"]

        activations = {
            "relu": nn.ReLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
            "elu": nn.ELU(inplace=True),
            "selu": nn.SELU(inplace=True),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(inplace=True),  # Swish is SiLU
            "mish": nn.Mish(inplace=True),
            "none": nn.Identity(),
        }

        return activations.get(act_type, nn.ReLU(inplace=True))

    @classmethod
    def build_model(
        cls,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 64,
    ) -> nn.Module:
        """Build a complete model from architecture specification."""
        if architecture.cell_based:
            return cls._build_cell_based_model(
                architecture, input_channels, num_classes, base_width
            )
        else:
            return cls._build_sequential_model(
                architecture, input_channels, num_classes, base_width
            )

    @classmethod
    def _build_sequential_model(
        cls,
        architecture: Architecture,
        input_channels: int,
        num_classes: int,
        base_width: int,
    ) -> nn.Module:
        """Build sequential model from layer specifications."""
        layers = []
        current_channels = input_channels
        final_channels = base_width

        for i, layer_spec in enumerate(architecture.layers):
            if isinstance(layer_spec.operation, NoneOperation):
                continue

            # Update the layer_spec with correct in_channels
            updated_spec = copy.deepcopy(layer_spec)
            updated_spec.in_channels = current_channels

            # Determine output channels
            if isinstance(updated_spec.operation, ConvOperation):
                if i == 0:
                    updated_spec.out_channels = base_width
                elif i % 2 == 1:
                    updated_spec.out_channels = current_channels * 2
                else:
                    updated_spec.out_channels = current_channels
                final_channels = int(
                    updated_spec.out_channels * updated_spec.width_multiplier
                )

            layer = cls.build_layer(updated_spec)
            if layer is not None:
                layers.append(layer)
                # Update current_channels for next layer
                if isinstance(updated_spec.operation, ConvOperation):
                    current_channels = final_channels

        # Global average pooling and classifier
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())

        layers.append(nn.Linear(final_channels, num_classes))

        return nn.Sequential(*layers)

    @classmethod
    def _build_cell_based_model(
        cls,
        architecture: Architecture,
        input_channels: int,
        num_classes: int,
        base_width: int,
    ) -> nn.Module:
        """Build cell-based model (simplified implementation)."""
        # This is a simplified implementation
        # Full implementation would parse the cell DAG structure

        layers = [
            nn.Conv2d(input_channels, base_width, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        ]

        # Add cells (simplified - just add some conv layers)
        current_channels = base_width
        for i in range(3):  # 3 stages
            out_channels = current_channels * 2 if i > 0 else current_channels
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        out_channels,
                        3,
                        stride=2 if i > 0 else 1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = out_channels

        layers.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(current_channels, num_classes),
            ]
        )

        return nn.Sequential(*layers)


# ============================================================================
# Architecture Evaluator
# ============================================================================


class ArchitectureEvaluator:
    """
    Evaluate architectures without full training.

    Provides methods for:
    - Quick performance estimation (few epochs)
    - Parameter count estimation
    - FLOPs estimation
    - Latency estimation
    """

    def __init__(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_epochs_estimate: int = 5,
        lr: float = 0.01,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs_estimate = num_epochs_estimate
        self.lr = lr

    def estimate_performance(
        self,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 64,
    ) -> Dict[str, float]:
        """
        Estimate architecture performance by training for few epochs.

        Returns dict with keys:
        - val_accuracy: Validation accuracy
        - train_accuracy: Training accuracy
        - train_loss: Training loss
        """
        if self.train_loader is None or self.val_loader is None:
            raise ValueError(
                "train_loader and val_loader required for performance estimation"
            )

        model = ArchitectureBuilder.build_model(
            architecture, input_channels, num_classes, base_width
        ).to(self.device)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs_estimate
        )

        best_val_acc = 0.0

        for epoch in range(self.num_epochs_estimate):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

                if batch_idx >= 10:  # Limit batches for speed
                    break

            scheduler.step()

            # Validate
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

                    if batch_idx >= 5:  # Limit batches for speed
                        break

            val_acc = 100.0 * val_correct / val_total
            best_val_acc = max(best_val_acc, val_acc)

        train_acc = 100.0 * train_correct / train_total

        return {
            "val_accuracy": best_val_acc,
            "train_accuracy": train_acc,
            "train_loss": train_loss / min(len(self.train_loader), 11),
        }

    def count_parameters(
        self,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 64,
    ) -> int:
        """Count the number of parameters in the architecture."""
        model = ArchitectureBuilder.build_model(
            architecture, input_channels, num_classes, base_width
        )
        return sum(p.numel() for p in model.parameters())

    def estimate_flops(
        self,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 64,
        input_size: Tuple[int, int] = (224, 224),
    ) -> int:
        """
        Estimate FLOPs (floating point operations) for the architecture.

        Uses a simple estimation based on layer dimensions.
        """
        model = ArchitectureBuilder.build_model(
            architecture, input_channels, num_classes, base_width
        )

        total_flops = 0
        h, w = input_size

        # Simple FLOP estimation
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # FLOPs = 2 * Cin * Cout * K * K * H * W / stride
                kernel_ops = module.kernel_size[0] * module.kernel_size[1]
                output_h = h // module.stride[0]
                output_w = w // module.stride[1]
                flops = (
                    2
                    * module.in_channels
                    * module.out_channels
                    * kernel_ops
                    * output_h
                    * output_w
                )
                total_flops += flops
                h, w = output_h, output_w
            elif isinstance(module, nn.Linear):
                # FLOPs = 2 * in_features * out_features
                total_flops += 2 * module.in_features * module.out_features
            elif isinstance(module, nn.BatchNorm2d):
                # FLOPs = 2 * num_features * H * W
                total_flops += 2 * module.num_features * h * w

        return total_flops

    def estimate_latency(
        self,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 64,
        input_size: Tuple[int, int] = (224, 224),
        num_runs: int = 50,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Estimate inference latency of the architecture.

        Returns dict with:
        - mean_latency_ms: Mean latency in milliseconds
        - std_latency_ms: Standard deviation of latency
        - min_latency_ms: Minimum latency
        - max_latency_ms: Maximum latency
        """
        model = (
            ArchitectureBuilder.build_model(
                architecture, input_channels, num_classes, base_width
            )
            .to(self.device)
            .eval()
        )

        dummy_input = torch.randn(1, input_channels, *input_size).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

        # Measure
        if self.device == "cuda":
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()

                start = time.time()
                _ = model(dummy_input)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                end = time.time()
                latencies.append((end - start) * 1000)  # Convert to ms

        return {
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
        }

    def evaluate(
        self,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 64,
        input_size: Tuple[int, int] = (224, 224),
        evaluate_performance: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of an architecture.

        Returns dict with all metrics including:
        - Performance metrics (if evaluate_performance=True)
        - Parameter count
        - FLOPs estimate
        - Latency estimate
        """
        results = {
            "params": self.count_parameters(
                architecture, input_channels, num_classes, base_width
            ),
            "flops": self.estimate_flops(
                architecture, input_channels, num_classes, base_width, input_size
            ),
            "latency": self.estimate_latency(
                architecture, input_channels, num_classes, base_width, input_size
            ),
        }

        if evaluate_performance and self.train_loader is not None:
            perf = self.estimate_performance(
                architecture, input_channels, num_classes, base_width
            )
            results.update(perf)

        return results


# ============================================================================
# Base NAS Class
# ============================================================================


class NeuralArchitectureSearch(ABC):
    """
    Base class for Neural Architecture Search algorithms.

    Args:
        search_space: SearchSpace defining available operations and constraints
        evaluator: ArchitectureEvaluator for evaluating architectures
        max_iterations: Maximum number of search iterations
        population_size: Size of population (for evolutionary methods)
        seed: Random seed for reproducibility

    Example:
        >>> space = SearchSpace(operations=[...], max_depth=8)
        >>> evaluator = ArchitectureEvaluator(train_loader, val_loader)
        >>> nas = RandomNAS(space, evaluator, max_iterations=100)
        >>> best_arch, best_score = nas.search()
    """

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Optional[ArchitectureEvaluator] = None,
        max_iterations: int = 100,
        population_size: int = 50,
        seed: Optional[int] = None,
    ):
        self.search_space = search_space
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.seed = seed

        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.history = []
        self.best_architecture = None
        self.best_score = float("-inf")

    @abstractmethod
    def search(
        self, input_channels: int = 3, num_classes: int = 10, base_width: int = 64
    ) -> Tuple[Architecture, float]:
        """
        Perform architecture search.

        Returns:
            Tuple of (best_architecture, best_score)
        """
        pass

    def sample_random_architecture(
        self, input_channels: int = 3, base_width: int = 64
    ) -> Architecture:
        """Sample a random architecture from the search space."""
        if self.search_space.cell_based:
            return self._sample_cell_based_architecture(input_channels, base_width)
        else:
            return self._sample_sequential_architecture(input_channels, base_width)

    def _sample_sequential_architecture(
        self, input_channels: int = 3, base_width: int = 64
    ) -> Architecture:
        """Sample a random sequential architecture."""
        depth = self.rng.randint(
            self.search_space.min_depth, self.search_space.max_depth + 1
        )
        width_multiplier = self.search_space.sample_width(self.rng)

        layers = []
        current_channels = base_width

        for i in range(depth):
            op = self.search_space.sample_operation(self.rng)

            # Skip None operations occasionally
            if isinstance(op, NoneOperation) and self.rng.random() > 0.3:
                op = self.search_space.sample_operation(self.rng)

            layer_spec = LayerSpec(
                operation=op,
                in_channels=input_channels if i == 0 else current_channels,
                out_channels=current_channels * 2
                if i % 2 == 1 and i > 0
                else current_channels,
                width_multiplier=width_multiplier,
            )

            layers.append(layer_spec)

            # Update channels for next layer
            if isinstance(op, ConvOperation) and i > 0 and i % 2 == 1:
                current_channels *= 2

        return Architecture(layers=layers, cell_based=False)

    def _sample_cell_based_architecture(
        self, input_channels: int = 3, base_width: int = 64
    ) -> Architecture:
        """Sample a random cell-based architecture."""
        normal_cell = CellSpec(name="normal")
        reduction_cell = CellSpec(name="reduction")

        # Sample nodes for normal cell
        for node_idx in range(self.search_space.num_nodes):
            num_inputs = min(node_idx + 2, self.search_space.num_edges)
            inputs = self.rng.choice(
                node_idx + 2, size=num_inputs, replace=False
            ).tolist()
            operations = [
                self.search_space.sample_operation(self.rng) for _ in range(num_inputs)
            ]
            normal_cell.add_node(inputs, operations)

        # Sample nodes for reduction cell
        for node_idx in range(self.search_space.num_nodes):
            num_inputs = min(node_idx + 2, self.search_space.num_edges)
            inputs = self.rng.choice(
                node_idx + 2, size=num_inputs, replace=False
            ).tolist()
            operations = [
                self.search_space.sample_operation(self.rng) for _ in range(num_inputs)
            ]
            reduction_cell.add_node(inputs, operations)

        return Architecture(
            normal_cell=normal_cell, reduction_cell=reduction_cell, cell_based=True
        )

    def evaluate_architecture(
        self,
        architecture: Architecture,
        input_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 64,
    ) -> float:
        """
        Evaluate an architecture and return its fitness score.

        Uses the evaluator if available, otherwise returns a random score
        for demonstration purposes.
        """
        if self.evaluator is not None:
            try:
                results = self.evaluator.evaluate(
                    architecture,
                    input_channels,
                    num_classes,
                    base_width,
                    evaluate_performance=True,
                )
                # Use validation accuracy as fitness
                return results.get("val_accuracy", 0.0)
            except Exception as e:
                warnings.warn(f"Error evaluating architecture: {e}")
                return 0.0
        else:
            # Random score for testing
            return self.rng.random() * 100

    def save_history(self, filepath: str):
        """Save search history to file."""
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_results(self) -> List[Dict]:
        """Get search results as list of dictionaries."""
        return self.history


# ============================================================================
# Random NAS
# ============================================================================


class RandomNAS(NeuralArchitectureSearch):
    """
    Random Neural Architecture Search.

    Simple baseline that randomly samples architectures from the search space
    and returns the best one found.

    Args:
        search_space: SearchSpace defining available operations
        evaluator: ArchitectureEvaluator for evaluating architectures
        max_iterations: Number of random architectures to sample
        seed: Random seed

    Example:
        >>> space = SearchSpace(operations=[...], max_depth=8)
        >>> evaluator = ArchitectureEvaluator(train_loader, val_loader)
        >>> nas = RandomNAS(space, evaluator, max_iterations=100)
        >>> best_arch, best_score = nas.search()
    """

    def search(
        self, input_channels: int = 3, num_classes: int = 10, base_width: int = 64
    ) -> Tuple[Architecture, float]:
        """
        Perform random architecture search.

        Returns:
            Tuple of (best_architecture, best_score)
        """
        print(f"RandomNAS: Searching {self.max_iterations} random architectures")

        for iteration in range(self.max_iterations):
            # Sample random architecture
            architecture = self.sample_random_architecture(input_channels, base_width)

            # Evaluate
            score = self.evaluate_architecture(
                architecture, input_channels, num_classes, base_width
            )

            # Record
            self.history.append(
                {
                    "iteration": iteration,
                    "score": score,
                    "architecture": architecture.to_dict(),
                }
            )

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = architecture
                print(
                    f"  Iteration {iteration + 1}/{self.max_iterations}: New best score: {score:.4f}"
                )
            else:
                print(
                    f"  Iteration {iteration + 1}/{self.max_iterations}: Score: {score:.4f}"
                )

        print(f"\nRandomNAS Complete. Best score: {self.best_score:.4f}")
        return self.best_architecture, self.best_score


# ============================================================================
# Evolutionary NAS
# ============================================================================


class EvolutionaryNAS(NeuralArchitectureSearch):
    """
    Evolutionary Neural Architecture Search.

    Uses genetic algorithms to evolve architectures through:
    - Selection of fittest individuals
    - Mutation (add/remove/change layers)
    - Crossover between architectures

    Reference: Real et al., "Regularized Evolution for Image Classifier Architecture Search", 2019

    Args:
        search_space: SearchSpace defining available operations
        evaluator: ArchitectureEvaluator for evaluating architectures
        max_iterations: Number of evolution iterations
        population_size: Size of the population
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        tournament_size: Size of tournament for selection
        seed: Random seed

    Example:
        >>> space = SearchSpace(operations=[...], max_depth=8)
        >>> evaluator = ArchitectureEvaluator(train_loader, val_loader)
        >>> nas = EvolutionaryNAS(
        ...     space, evaluator,
        ...     max_iterations=100,
        ...     population_size=50,
        ...     mutation_rate=0.3
        ... )
        >>> best_arch, best_score = nas.search()
    """

    def __init__(
        self,
        search_space: SearchSpace,
        evaluator: Optional[ArchitectureEvaluator] = None,
        max_iterations: int = 100,
        population_size: int = 50,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        tournament_size: int = 5,
        seed: Optional[int] = None,
    ):
        super().__init__(search_space, evaluator, max_iterations, population_size, seed)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.population = []

    def search(
        self, input_channels: int = 3, num_classes: int = 10, base_width: int = 64
    ) -> Tuple[Architecture, float]:
        """
        Perform evolutionary architecture search.

        Returns:
            Tuple of (best_architecture, best_score)
        """
        print(f"EvolutionaryNAS: Starting evolution")
        print(f"  Population size: {self.population_size}")
        print(f"  Max iterations: {self.max_iterations}")
        print(f"  Mutation rate: {self.mutation_rate}")
        print(f"  Crossover rate: {self.crossover_rate}")

        # Initialize population
        print("\nInitializing population...")
        self._initialize_population(input_channels, num_classes, base_width)

        # Evolve
        for iteration in range(self.max_iterations):
            # Selection
            parent = self._tournament_selection()

            # Mutation
            child = self._mutate(parent, input_channels, base_width)

            # Crossover (occasionally)
            if self.rng.random() < self.crossover_rate:
                parent2 = self._tournament_selection()
                child = self._crossover(child, parent2)

            # Evaluate child
            score = self.evaluate_architecture(
                child, input_channels, num_classes, base_width
            )

            # Add to population and remove oldest (regularized evolution)
            self.population.append((child, score))
            if len(self.population) > self.population_size:
                self.population.pop(0)  # Remove oldest

            # Record
            self.history.append(
                {
                    "iteration": iteration,
                    "score": score,
                    "population_size": len(self.population),
                }
            )

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = child
                print(
                    f"  Iteration {iteration + 1}/{self.max_iterations}: New best score: {score:.4f}"
                )
            else:
                print(
                    f"  Iteration {iteration + 1}/{self.max_iterations}: Score: {score:.4f}"
                )

        print(f"\nEvolutionaryNAS Complete. Best score: {self.best_score:.4f}")
        return self.best_architecture, self.best_score

    def _initialize_population(
        self, input_channels: int, num_classes: int, base_width: int
    ):
        """Initialize the population with random architectures."""
        for i in range(self.population_size):
            architecture = self.sample_random_architecture(input_channels, base_width)
            score = self.evaluate_architecture(
                architecture, input_channels, num_classes, base_width
            )
            self.population.append((architecture, score))

            if score > self.best_score:
                self.best_score = score
                self.best_architecture = architecture

            print(f"  Individual {i + 1}/{self.population_size}: Score: {score:.4f}")

    def _tournament_selection(self) -> Architecture:
        """Select an individual using tournament selection."""
        tournament = self.rng.choice(
            len(self.population),
            size=min(self.tournament_size, len(self.population)),
            replace=False,
        )
        best_idx = tournament[0]
        best_score = self.population[best_idx][1]

        for idx in tournament[1:]:
            if self.population[idx][1] > best_score:
                best_idx = idx
                best_score = self.population[idx][1]

        return copy.deepcopy(self.population[best_idx][0])

    def _mutate(
        self, architecture: Architecture, input_channels: int, base_width: int
    ) -> Architecture:
        """Apply mutation to an architecture."""
        if not architecture.cell_based:
            return self._mutate_sequential(architecture, input_channels, base_width)
        else:
            return self._mutate_cell_based(architecture)

    def _mutate_sequential(
        self, architecture: Architecture, input_channels: int, base_width: int
    ) -> Architecture:
        """Mutate a sequential architecture."""
        mutated = copy.deepcopy(architecture)
        mutation_type = self.rng.choice(["add", "remove", "change", "width"])

        if mutation_type == "add" and len(mutated.layers) < self.search_space.max_depth:
            # Add a new layer
            insert_idx = self.rng.randint(0, len(mutated.layers) + 1)
            op = self.search_space.sample_operation(self.rng)

            in_ch = (
                input_channels
                if insert_idx == 0
                else mutated.layers[insert_idx - 1].out_channels
            )
            out_ch = (
                mutated.layers[insert_idx].in_channels
                if insert_idx < len(mutated.layers)
                else base_width
            )

            new_layer = LayerSpec(
                operation=op,
                in_channels=in_ch,
                out_channels=out_ch,
                width_multiplier=mutated.layers[0].width_multiplier
                if mutated.layers
                else 1.0,
            )
            mutated.layers.insert(insert_idx, new_layer)

        elif (
            mutation_type == "remove"
            and len(mutated.layers) > self.search_space.min_depth
        ):
            # Remove a layer
            remove_idx = self.rng.randint(0, len(mutated.layers))
            mutated.layers.pop(remove_idx)

        elif mutation_type == "change" and mutated.layers:
            # Change an operation
            change_idx = self.rng.randint(0, len(mutated.layers))
            mutated.layers[change_idx].operation = self.search_space.sample_operation(
                self.rng
            )

        elif mutation_type == "width" and mutated.layers:
            # Change width multiplier
            new_width = self.search_space.sample_width(self.rng)
            for layer in mutated.layers:
                layer.width_multiplier = new_width

        return mutated

    def _mutate_cell_based(self, architecture: Architecture) -> Architecture:
        """Mutate a cell-based architecture."""
        mutated = copy.deepcopy(architecture)

        # Mutate either normal or reduction cell
        cell_to_mutate = (
            mutated.normal_cell if self.rng.random() < 0.5 else mutated.reduction_cell
        )

        if cell_to_mutate and cell_to_mutate.nodes:
            # Mutate a random node
            node_idx = self.rng.randint(0, len(cell_to_mutate.nodes))
            node = cell_to_mutate.nodes[node_idx]

            # Change one operation
            if node["operations"]:
                op_idx = self.rng.randint(0, len(node["operations"]))
                node["operations"][op_idx] = self.search_space.sample_operation(
                    self.rng
                )

        return mutated

    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Perform crossover between two architectures."""
        if parent1.cell_based != parent2.cell_based:
            return parent1  # Can't crossover different types

        if not parent1.cell_based:
            return self._crossover_sequential(parent1, parent2)
        else:
            return self._crossover_cell_based(parent1, parent2)

    def _crossover_sequential(
        self, parent1: Architecture, parent2: Architecture
    ) -> Architecture:
        """Crossover two sequential architectures."""
        child = copy.deepcopy(parent1)

        if not parent1.layers or not parent2.layers:
            return child

        # Choose crossover point
        min_len = min(len(parent1.layers), len(parent2.layers))
        if min_len > 0:
            crossover_point = self.rng.randint(1, min_len)
            child.layers = (
                parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
            )

        return child

    def _crossover_cell_based(
        self, parent1: Architecture, parent2: Architecture
    ) -> Architecture:
        """Crossover two cell-based architectures."""
        child = copy.deepcopy(parent1)

        # Swap cells
        if self.rng.random() < 0.5 and parent2.normal_cell:
            child.normal_cell = copy.deepcopy(parent2.normal_cell)
        elif parent2.reduction_cell:
            child.reduction_cell = copy.deepcopy(parent2.reduction_cell)

        return child


# ============================================================================
# Convenience Functions
# ============================================================================


def create_default_search_space(
    cell_based: bool = False,
    include_separable: bool = True,
    include_dilated: bool = True,
) -> SearchSpace:
    """
    Create a default search space with common operations.

    Args:
        cell_based: Whether to use cell-based search
        include_separable: Include separable convolutions
        include_dilated: Include dilated convolutions

    Returns:
        SearchSpace with common operations
    """
    operations = [
        # Standard convolutions
        ConvOperation(3, name="conv3x3"),
        ConvOperation(5, name="conv5x5"),
        ConvOperation(7, name="conv7x7"),
        ConvOperation(1, name="conv1x1"),
        # Pooling
        PoolingOperation("max", 3, name="maxpool3x3"),
        PoolingOperation("avg", 3, name="avgpool3x3"),
        # Activations
        ActivationOperation("relu", name="relu"),
        ActivationOperation("leaky_relu", name="leaky_relu"),
        # Skip and none
        SkipOperation(),
        NoneOperation(),
    ]

    if include_separable:
        operations.extend(
            [
                ConvOperation(3, separable=True, name="sep_conv3x3"),
                ConvOperation(5, separable=True, name="sep_conv5x5"),
            ]
        )

    if include_dilated:
        operations.extend(
            [
                ConvOperation(3, dilation=2, name="dil_conv3x3"),
                ConvOperation(5, dilation=2, name="dil_conv5x5"),
            ]
        )

    return SearchSpace(
        operations=operations,
        max_depth=12,
        min_depth=3,
        width_multipliers=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        cell_based=cell_based,
        num_nodes=4,
        num_edges=2,
    )


def search_architecture(
    search_space: Optional[SearchSpace] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    method: str = "evolutionary",
    max_iterations: int = 100,
    population_size: int = 50,
    input_channels: int = 3,
    num_classes: int = 10,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[Architecture, float, Dict]:
    """
    Main entry point for architecture search.

    Args:
        search_space: Search space (creates default if None)
        train_loader: Training data loader
        val_loader: Validation data loader
        method: Search method ('random', 'evolutionary')
        max_iterations: Maximum search iterations
        population_size: Population size for evolutionary methods
        input_channels: Number of input channels
        num_classes: Number of output classes
        seed: Random seed
        **kwargs: Additional arguments for the search algorithm

    Returns:
        Tuple of (best_architecture, best_score, search_history)

    Example:
        >>> best_arch, score, history = search_architecture(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     method="evolutionary",
        ...     max_iterations=100
        ... )
    """
    # Create default search space if not provided
    if search_space is None:
        search_space = create_default_search_space()

    # Create evaluator
    evaluator = None
    if train_loader is not None and val_loader is not None:
        evaluator = ArchitectureEvaluator(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs_estimate=kwargs.get("num_epochs_estimate", 5),
        )

    # Create search algorithm
    if method == "random":
        nas = RandomNAS(
            search_space=search_space,
            evaluator=evaluator,
            max_iterations=max_iterations,
            seed=seed,
        )
    elif method == "evolutionary":
        nas = EvolutionaryNAS(
            search_space=search_space,
            evaluator=evaluator,
            max_iterations=max_iterations,
            population_size=population_size,
            mutation_rate=kwargs.get("mutation_rate", 0.3),
            crossover_rate=kwargs.get("crossover_rate", 0.5),
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown search method: {method}. Choose 'random' or 'evolutionary'"
        )

    # Search
    best_arch, best_score = nas.search(input_channels, num_classes)

    return best_arch, best_score, {"history": nas.get_results()}


def estimate_model_complexity(
    architecture: Architecture,
    input_channels: int = 3,
    num_classes: int = 10,
    base_width: int = 64,
    input_size: Tuple[int, int] = (224, 224),
) -> Dict[str, Any]:
    """
    Estimate model complexity without training.

    Args:
        architecture: Architecture to evaluate
        input_channels: Number of input channels
        num_classes: Number of output classes
        base_width: Base width multiplier
        input_size: Input image size

    Returns:
        Dict with complexity metrics:
        - params: Number of parameters
        - flops: FLOPs estimate
        - latency: Latency estimate (if CUDA available)

    Example:
        >>> complexity = estimate_model_complexity(architecture)
        >>> print(f"Parameters: {complexity['params']:,}")
        >>> print(f"FLOPs: {complexity['flops']:,}")
    """
    evaluator = ArchitectureEvaluator()

    return evaluator.evaluate(
        architecture=architecture,
        input_channels=input_channels,
        num_classes=num_classes,
        base_width=base_width,
        input_size=input_size,
        evaluate_performance=False,
    )


def build_model_from_architecture(
    architecture: Architecture,
    input_channels: int = 3,
    num_classes: int = 10,
    base_width: int = 64,
) -> nn.Module:
    """
    Build a PyTorch model from an architecture specification.

    Args:
        architecture: Architecture specification
        input_channels: Number of input channels
        num_classes: Number of output classes
        base_width: Base width multiplier

    Returns:
        PyTorch nn.Module

    Example:
        >>> arch = sample_random_architecture(search_space)
        >>> model = build_model_from_architecture(arch, num_classes=10)
        >>> print(model)
    """
    return ArchitectureBuilder.build_model(
        architecture, input_channels, num_classes, base_width
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Search Space
    "Operation",
    "ConvOperation",
    "PoolingOperation",
    "ActivationOperation",
    "SkipOperation",
    "NoneOperation",
    "SearchSpace",
    # Architecture
    "LayerSpec",
    "CellSpec",
    "Architecture",
    "ArchitectureBuilder",
    # Evaluator
    "ArchitectureEvaluator",
    # NAS Algorithms
    "NeuralArchitectureSearch",
    "RandomNAS",
    "EvolutionaryNAS",
    # Convenience Functions
    "create_default_search_space",
    "search_architecture",
    "estimate_model_complexity",
    "build_model_from_architecture",
]


# ============================================================================
# Legacy Components (from original nas/__init__.py)
# ============================================================================


class NASCell(nn.Module):
    """NAS cell for DARTS-style architecture search.

    Args:
        n_nodes: Number of nodes in the cell
        n_ops: Number of operations to choose from
        n_edges: Number of edges
        reduction: Whether this is a reduction cell
    """

    def __init__(
        self,
        n_nodes: int = 4,
        n_ops: int = 5,
        reduction: bool = False,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_ops = n_ops
        self.reduction = reduction

        self.edges = nn.ModuleDict()
        for i in range(2, n_nodes):
            for j in range(i):
                self.edges[f"{j}_{i}"] = nn.ModuleList(
                    [
                        nn.Identity(),
                        nn.MaxPool1d(3, stride=1, padding=1),
                        nn.AvgPool1d(3, stride=1, padding=1),
                    ]
                )

    def forward(self, x):
        from typing import List

        states = x
        for i in range(2, self.n_nodes):
            new_state = 0
            for j in range(i):
                for op in self.edges[f"{j}_{i}"]:
                    new_state = new_state + op(states[j])
            states.append(new_state / len(states))
        return states


class MixedOp(nn.Module):
    """Mixed operation for neural architecture search.

    Args:
        c_in: Input channels
        c_out: Output channels
        stride: Stride for convolution
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self._ops = nn.ModuleList(
            [
                nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False),
                nn.Conv1d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                nn.AvgPool1d(3, stride=stride, padding=1),
                nn.MaxPool1d(3, stride=stride, padding=1),
                nn.Identity() if stride == 1 else None,
            ]
        )

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops) if op is not None)


class SuperNet(nn.Module):
    """Supernet for one-shot neural architecture search.

    Args:
        num_classes: Number of output classes
        num_cells: Number of cells in the network
        num_nodes: Number of nodes per cell
        channels: Number of initial channels
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_cells: int = 8,
        num_nodes: int = 4,
        channels: int = 16,
    ):
        super().__init__()
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.channels = channels

        self.stem = nn.Conv1d(3, channels, kernel_size=3, padding=1)

        self.cells = nn.ModuleList()
        for i in range(num_cells):
            reduction = i >= num_cells // 2
            self.cells.append(NASCell(n_nodes=num_nodes, reduction=reduction))

        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell([x])[-1]
        return self.classifier(x.mean(dim=-1))


class DARTSOptimizer(nn.Module):
    """DARTS differentiable architecture search optimizer.

    Args:
        num_ops: Number of operations to choose from
        num_nodes: Number of nodes in cell
    """

    def __init__(self, num_ops: int = 5, num_nodes: int = 4):
        super().__init__()
        self.num_ops = num_ops
        self.num_nodes = num_nodes

        self.alphas = nn.ParameterList(
            [
                nn.Parameter(torch.randn(num_ops) * 1e-3)
                for _ in range(num_nodes * (num_nodes - 1) // 2)
            ]
        )

    def forward(self):
        weights = []
        for alpha in self.alphas:
            weights.append(torch.nn.functional.softmax(alpha, dim=-1))
        return weights

    def arch_parameters(self):
        return list(self.alphas)


class EfficientNASNet(nn.Module):
    """Efficient Neural Architecture Search Network.

    Args:
        num_classes: Number of output classes
        channel_list: List of channels per layer
    """

    def __init__(
        self,
        num_classes: int = 10,
        channel_list=[16, 32, 64, 128],
    ):
        super().__init__()

        self.stem = nn.Conv1d(3, channel_list[0], 3, padding=1)

        self.layers = nn.ModuleList()
        for i in range(len(channel_list) - 1):
            self.layers.append(self._make_layer(channel_list[i], channel_list[i + 1]))

        self.classifier = nn.Linear(channel_list[-1], num_classes)

    def _make_layer(self, in_ch: int, out_ch: int):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x.mean(dim=-1))


class NASBench101Cell(nn.Module):
    """NASBench-101 style cell.

    Args:
        num_vertices: Number of vertices in cell
        num_operations: Number of operations
    """

    def __init__(
        self,
        num_vertices: int = 7,
        num_operations: int = 5,
    ):
        super().__init__()
        self.num_vertices = num_vertices

        self.vertex_op = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(64, 64),
                    nn.ReLU(),
                )
                for _ in range(num_vertices)
            ]
        )

    def forward(self, inputs, adjacency):
        features = inputs[:]
        for i in range(len(features), self.num_vertices):
            new_feat = 0
            for j in range(i):
                if adjacency[j, i]:
                    new_feat = new_feat + features[j]
            if i > 0:
                features.append(self.vertex_op[i](new_feat))
            else:
                features.append(inputs[0])

        return features[-1]


# Add legacy classes to __all__
__all__.extend(
    [
        "NASCell",
        "MixedOp",
        "SuperNet",
        "DARTSOptimizer",
        "EfficientNASNet",
        "NASBench101Cell",
    ]
)

"""
Architecture Performance Predictor.

Provides tools for predicting neural architecture performance:
- Regression-based predictors
- Neural network predictors
- Transfer learning from pretrained models
- Ensemble predictors
- Cost models (FLOPs, latency, memory)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .search_space import (
    ArchitectureSpec,
    CellSpec,
    EdgeSpec,
    OperationSpec,
    OperationType,
    SearchSpace,
)


class PerformancePredictor(ABC, nn.Module):
    """Abstract base class for architecture performance predictors."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Predict performance from architecture encoding."""
        pass

    @abstractmethod
    def encode_architecture(self, arch: ArchitectureSpec) -> Tensor:
        """Encode architecture into feature vector."""
        pass

    @abstractmethod
    def train_predictor(
        self,
        train_data: List[Tuple[ArchitectureSpec, float]],
        val_data: Optional[List[Tuple[ArchitectureSpec, float]]] = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """Train the predictor."""
        pass

    @abstractmethod
    def predict(self, arch: ArchitectureSpec) -> float:
        """Predict performance for an architecture."""
        pass


class LinearPredictor(PerformancePredictor):
    """
    Simple linear regression predictor.

    Uses architecture features to predict performance via linear model.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
        )

        self.predictor = nn.Linear(32, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        features = self.encoder(x)
        return self.predictor(features)

    def encode_architecture(self, arch: ArchitectureSpec) -> Tensor:
        """Encode architecture to feature vector."""
        features = []

        # Cell-based features
        if arch.normal_cell:
            features.append(len(arch.normal_cell.edges))
            features.append(arch.normal_cell.num_nodes)
        else:
            features.append(0)
            features.append(0)

        if arch.reduction_cell:
            features.append(len(arch.reduction_cell.edges))
        else:
            features.append(0)

        # Stem channels
        features.append(arch.stem_channels)
        features.append(arch.num_classes)

        # Operation type counts
        op_counts = self._count_operations(arch)
        features.extend(op_counts)

        # Pad or truncate to fixed size
        while len(features) < 64:
            features.append(0)
        features = features[:64]

        return torch.tensor(features, dtype=torch.float32)

    def _count_operations(self, arch: ArchitectureSpec) -> List[int]:
        """Count operations in architecture."""
        op_types = [
            OperationType.CONV_1X1,
            OperationType.CONV_3X3,
            OperationType.CONV_5X5,
            OperationType.SEP_CONV_3X3,
            OperationType.SEP_CONV_5X5,
            OperationType.DIL_CONV_3X3,
            OperationType.MAX_POOL,
            OperationType.AVG_POOL,
            OperationType.SKIP,
            OperationType.ZERO,
        ]

        counts = [0] * len(op_types)

        for cell in [arch.normal_cell, arch.reduction_cell]:
            if cell is None:
                continue

            for edge in cell.edges:
                try:
                    idx = op_types.index(edge.operation.op_type)
                    counts[idx] += 1
                except ValueError:
                    pass

        return counts

    def train_predictor(
        self,
        train_data: List[Tuple[ArchitectureSpec, float]],
        val_data: Optional[List[Tuple[ArchitectureSpec, float]]] = None,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """Train the predictor."""
        # Encode architectures
        X_train = torch.stack(
            [self.encode_architecture(arch) for arch, _ in train_data]
        )
        y_train = torch.tensor(
            [y for _, y in train_data], dtype=torch.float32
        ).unsqueeze(1)

        # Create dataset
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        history = {"train_loss": []}

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.forward(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_loss)

        return history

    def predict(self, arch: ArchitectureSpec) -> float:
        """Predict performance."""
        self.eval()
        with torch.no_grad():
            x = self.encode_architecture(arch).unsqueeze(0)
            return self.forward(x).item()


class NeuralPredictor(PerformancePredictor):
    """
    Neural network-based performance predictor.

    Uses a neural network to learn complex relationships between
    architecture features and performance.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        super().__init__()

        # Encoder
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        features = self.encoder(x)
        return self.head(features)

    def encode_architecture(self, arch: ArchitectureSpec) -> Tensor:
        """Encode architecture to feature vector."""
        features = []

        # Basic architecture info
        features.append(float(arch.num_classes))
        features.append(float(arch.stem_channels))
        features.append(float(arch.num_stages))

        # Cell info
        if arch.normal_cell:
            features.append(float(len(arch.normal_cell.edges)))
            features.append(float(arch.normal_cell.num_nodes))

            # Edge statistics
            edge_counts = self._get_edge_stats(arch.normal_cell)
            features.extend(edge_counts)
        else:
            features.extend([0] * 8)

        if arch.reduction_cell:
            features.append(float(len(arch.reduction_cell.edges)))

            edge_counts = self._get_edge_stats(arch.reduction_cell)
            features.extend(edge_counts)
        else:
            features.extend([0] * 4)

        # Operation type distribution
        op_dist = self._get_operation_distribution(arch)
        features.extend(op_dist)

        # Parameter estimation
        features.append(self._estimate_params(arch))
        features.append(self._estimate_flops(arch))

        # Pad to input_dim
        while len(features) < 64:
            features.append(0)
        features = features[:64]

        return torch.tensor(features, dtype=torch.float32)

    def _get_edge_stats(self, cell: CellSpec) -> List[float]:
        """Get edge statistics."""
        num_edges = len(cell.edges)

        if num_edges == 0:
            return [0, 0, 0, 0]

        # Compute average, max, min stride
        strides = [e.operation.stride for e in cell.edges]
        kernels = [e.operation.kernel_size for e in cell.edges]

        return [
            np.mean(strides),
            np.max(strides),
            np.mean(kernels),
            np.max(kernels),
        ]

    def _get_operation_distribution(self, arch: ArchitectureSpec) -> List[float]:
        """Get operation type distribution."""
        op_types = [
            OperationType.CONV_1X1,
            OperationType.CONV_3X3,
            OperationType.CONV_5X5,
            OperationType.SEP_CONV_3X3,
            OperationType.MAX_POOL,
            OperationType.AVG_POOL,
            OperationType.SKIP,
            OperationType.ZERO,
        ]

        dist = [0.0] * len(op_types)
        total = 0

        for cell in [arch.normal_cell, arch.reduction_cell]:
            if cell is None:
                continue

            for edge in cell.edges:
                total += 1
                try:
                    idx = op_types.index(edge.operation.op_type)
                    dist[idx] += 1
                except ValueError:
                    pass

        # Normalize
        if total > 0:
            dist = [d / total for d in dist]

        return dist

    def _estimate_params(self, arch: ArchitectureSpec) -> float:
        """Estimate number of parameters."""
        # Simplified estimation
        base_params = arch.stem_channels * 3 * 3 * 3

        edge_params = 0
        for cell in [arch.normal_cell, arch.reduction_cell]:
            if cell is None:
                continue

            edge_params += len(cell.edges) * arch.stem_channels * 64

        total = base_params + edge_params * arch.num_cells_per_stage * arch.num_stages

        return np.log1p(total)

    def _estimate_flops(self, arch: ArchitectureSpec) -> float:
        """Estimate FLOPs."""
        # Simplified estimation
        base_flops = arch.stem_channels * 3 * 3 * 3 * 224 * 224

        edge_flops = 0
        for cell in [arch.normal_cell, arch.reduction_cell]:
            if cell is None:
                continue

            edge_flops += len(cell.edges) * arch.stem_channels * 64 * 7 * 7

        total = base_flops + edge_flops * arch.num_cells_per_stage * arch.num_stages

        return np.log1p(total)

    def train_predictor(
        self,
        train_data: List[Tuple[ArchitectureSpec, float]],
        val_data: Optional[List[Tuple[ArchitectureSpec, float]]] = None,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
    ) -> Dict[str, List[float]]:
        """Train the predictor."""
        # Encode architectures
        X_train = torch.stack(
            [self.encode_architecture(arch) for arch, _ in train_data]
        )
        y_train = torch.tensor(
            [y for _, y in train_data], dtype=torch.float32
        ).unsqueeze(1)

        # Normalize targets
        self.mean = y_train.mean()
        self.std = y_train.std()
        y_normalized = (y_train - self.mean) / (self.std + 1e-8)

        dataset = TensorDataset(X_train, y_normalized)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.forward(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            scheduler.step()

            history["train_loss"].append(train_loss / len(loader))

            # Validation
            if val_data is not None:
                val_loss = self._evaluate(val_data)
                history["val_loss"].append(val_loss)

        return history

    def _evaluate(self, val_data: List[Tuple[ArchitectureSpec, float]]) -> float:
        """Evaluate on validation data."""
        self.eval()

        X_val = torch.stack([self.encode_architecture(arch) for arch, _ in val_data])
        y_val = torch.tensor([y for _, y in val_data], dtype=torch.float32).unsqueeze(1)
        y_normalized = (y_val - self.mean) / (self.std + 1e-8)

        with torch.no_grad():
            pred = self.forward(X_val)
            loss = F.mse_loss(pred, y_normalized)

        return loss.item()

    def predict(self, arch: ArchitectureSpec) -> float:
        """Predict performance."""
        self.eval()

        with torch.no_grad():
            x = self.encode_architecture(arch).unsqueeze(0)
            pred = self.forward(x)

            # Denormalize
            pred = pred * (self.std + 1e-8) + self.mean

            return pred.item()


class EnsemblePredictor(PerformancePredictor):
    """
    Ensemble of multiple predictors.

    Combines multiple predictors for improved accuracy and robustness.
    """

    def __init__(
        self,
        predictors: Optional[List[PerformancePredictor]] = None,
    ):
        super().__init__()

        if predictors is None:
            # Create default ensemble
            self.predictors = nn.ModuleList(
                [
                    LinearPredictor(),
                    NeuralPredictor(),
                ]
            )
        else:
            self.predictors = nn.ModuleList(predictors)

        self._mean = None
        self._std = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass (ensemble average)."""
        predictions = []

        for predictor in self.predictors:
            pred = predictor(x)
            predictions.append(pred)

        # Stack and average
        stacked = torch.stack(predictions, dim=0)
        return stacked.mean(dim=0)

    def encode_architecture(self, arch: ArchitectureSpec) -> Tensor:
        """Encode architecture (use first predictor's encoding)."""
        return self.predictors[0].encode_architecture(arch)

    def train_predictor(
        self,
        train_data: List[Tuple[ArchitectureSpec, float]],
        val_data: Optional[List[Tuple[ArchitectureSpec, float]]] = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """Train all predictors."""
        histories = []

        for i, predictor in enumerate(self.predictors):
            print(f"Training predictor {i + 1}/{len(self.predictors)}")

            history = predictor.train_predictor(train_data, val_data, **kwargs)
            histories.append(history)

        return {"histories": histories}

    def predict(self, arch: ArchitectureSpec) -> float:
        """Predict with ensemble."""
        self.eval()

        predictions = []

        with torch.no_grad():
            for predictor in self.predictors:
                pred = predictor.predict(arch)
                predictions.append(pred)

        return np.mean(predictions)

    def predict_variance(self, arch: ArchitectureSpec) -> float:
        """Get prediction variance from ensemble."""
        self.eval()

        predictions = []

        with torch.no_grad():
            for predictor in self.predictors:
                pred = predictor.predict(arch)
                predictions.append(pred)

        return np.var(predictions)


class CostModel:
    """
    Cost model for architecture efficiency metrics.

    Estimates:
    - Number of parameters
    - FLOPs
    - Latency
    - Memory footprint
    """

    def __init__(
        self,
        latency_table: Optional[Dict[str, float]] = None,
    ):
        self.latency_table = latency_table or self._default_latency_table()

    def _default_latency_table(self) -> Dict[str, float]:
        """Default latency lookup table in ms."""
        return {
            "conv_1x1": 0.1,
            "conv_3x3": 0.3,
            "conv_5x5": 0.5,
            "sep_conv_3x3": 0.2,
            "sep_conv_5x5": 0.4,
            "max_pool": 0.1,
            "avg_pool": 0.1,
            "skip": 0.01,
            "zero": 0.0,
        }

    def estimate_params(self, arch: ArchitectureSpec) -> int:
        """Estimate number of parameters."""
        total_params = 0

        # Stem
        total_params += 3 * 3 * 3 * arch.stem_channels

        # Cells
        channels = arch.stem_channels

        for stage_idx in range(arch.num_stages):
            is_reduction = stage_idx > 0 and stage_idx % 3 == 0
            if is_reduction:
                channels *= 2

            for _ in range(arch.num_cells_per_stage):
                cell = arch.normal_cell if not is_reduction else arch.reduction_cell

                if cell:
                    for edge in cell.edges:
                        # Approximate params for this edge
                        op = edge.operation.op_type

                        if op == OperationType.CONV_1X1:
                            total_params += channels * channels
                        elif op == OperationType.CONV_3X3:
                            total_params += channels * channels * 9
                        elif op == OperationType.CONV_5X5:
                            total_params += channels * channels * 25
                        elif op == OperationType.SEP_CONV_3X3:
                            total_params += channels * 9 + channels * channels
                        elif op == OperationType.SKIP:
                            total_params += 0
                        else:
                            total_params += channels * channels

        # Final classifier
        total_params += channels * arch.num_classes

        return total_params

    def estimate_flops(self, arch: ArchitectureSpec, input_size: int = 224) -> int:
        """Estimate FLOPs for a forward pass."""
        total_flops = 0
        h, w = input_size, input_size
        channels = arch.stem_channels

        # Stem
        total_flops += 2 * 3 * channels * 9 * h * w

        for stage_idx in range(arch.num_stages):
            is_reduction = stage_idx > 0 and stage_idx % 3 == 0
            if is_reduction:
                channels *= 2
                h, w = h // 2, w // 2

            for _ in range(arch.num_cells_per_stage):
                cell = arch.normal_cell if not is_reduction else arch.reduction_cell

                if cell:
                    for edge in cell.edges:
                        op = edge.operation.op_type

                        kernel = edge.operation.kernel_size
                        stride = edge.operation.stride

                        # FLOPs for this edge
                        if op == OperationType.CONV_1X1:
                            total_flops += 2 * channels * channels * h * w
                        elif op in [OperationType.CONV_3X3, OperationType.CONV_5X5]:
                            total_flops += (
                                2 * channels * channels * kernel * kernel * h * w
                            )
                        elif (
                            op == OperationType.MAX_POOL or op == OperationType.AVG_POOL
                        ):
                            total_flops += 2 * channels * h * w
                        elif op == OperationType.SKIP:
                            total_flops += 0
                        else:
                            total_flops += 2 * channels * channels * h * w

        # Final layers
        total_flops += 2 * channels * arch.num_classes

        return total_flops

    def estimate_latency(self, arch: ArchitectureSpec) -> float:
        """Estimate latency in milliseconds."""
        total_latency = 0.0

        for cell in [arch.normal_cell, arch.reduction_cell]:
            if cell is None:
                continue

            for edge in cell.edges:
                op_name = edge.operation.op_type.value
                latency = self.latency_table.get(op_name, 0.1)
                total_latency += latency

        return total_latency

    def estimate_memory(self, arch: ArchitectureSpec, batch_size: int = 1) -> int:
        """Estimate memory usage in bytes."""
        params = self.estimate_params(arch)

        # Assume 4 bytes per parameter (float32)
        param_memory = params * 4

        # Activation memory (approximate)
        channels = arch.stem_channels
        h, w = 224, 224
        activation_memory = 0

        for stage_idx in range(arch.num_stages):
            is_reduction = stage_idx > 0 and stage_idx % 3 == 0
            if is_reduction:
                channels *= 2
                h, w = h // 2, w // 2

            for _ in range(arch.num_cells_per_stage):
                # Input activations
                activation_memory += batch_size * channels * h * w * 4

        return param_memory + activation_memory

    def full_estimate(self, arch: ArchitectureSpec) -> Dict[str, Any]:
        """Get full cost estimate."""
        return {
            "params": self.estimate_params(arch),
            "params_millions": self.estimate_params(arch) / 1e6,
            "flops": self.estimate_flops(arch),
            "flops_gflops": self.estimate_flops(arch) / 1e9,
            "latency_ms": self.estimate_latency(arch),
            "memory_mb": self.estimate_memory(arch) / 1e6,
        }


def create_predictor(
    predictor_type: str = "neural",
    **kwargs,
) -> PerformancePredictor:
    """
    Factory function to create predictors.

    Args:
        predictor_type: Type of predictor ("linear", "neural", "ensemble")
        **kwargs: Additional arguments

    Returns:
        PerformancePredictor instance
    """
    predictors = {
        "linear": LinearPredictor,
        "neural": NeuralPredictor,
        "ensemble": EnsemblePredictor,
    }

    if predictor_type.lower() not in predictors:
        raise ValueError(f"Unknown predictor type: {predictor_type}")

    return predictors[predictor_type.lower()](**kwargs)


def load_nasbench_data(
    path: str,
) -> List[Tuple[ArchitectureSpec, float]]:
    """
    Load NAS-Bench-201 style data.

    Args:
        path: Path to JSON file

    Returns:
        List of (architecture, accuracy) pairs
    """
    with open(path, "r") as f:
        data = json.load(f)

    result = []

    for arch_data in data:
        # Parse architecture
        # This is simplified - full implementation would parse actual NAS-Bench format
        arch = ArchitectureSpec(
            normal_cell=CellSpec(name="normal", num_nodes=4),
            reduction_cell=CellSpec(name="reduction", num_nodes=4),
            stem_channels=16,
        )

        accuracy = arch_data.get("accuracy", 0.0)
        result.append((arch, accuracy))

    return result

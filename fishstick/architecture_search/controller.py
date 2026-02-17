"""
Neural Architecture Search Controller.

Provides DARTS-style differentiable controllers for architecture search:
- Edge operation weights (softmax over operations)
- Architecture gradient computation
- Controller training and optimization
- Multiple controller variants (DARTS, ProxylessNAS, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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


class ArchitectureController(nn.Module):
    """
    Differentiable architecture controller using softmax relaxation.

    Maintains learnable weights for each edge-operation pair and uses
    gumbel-softmax for sampling during discrete architecture selection.

    Reference: Liu et al., "DARTS: Differentiable Architecture Search", ICLR 2019
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_nodes: int = 4,
        num_operations: Optional[int] = None,
        temperature: float = 1.0,
        gumbel_hard: bool = False,
        reduction_cell: bool = True,
    ):
        super().__init__()

        self.search_space = search_space
        self.num_nodes = num_nodes
        self.temperature = temperature
        self.gumbel_hard = gumbel_hard
        self.reduction_cell = reduction_cell

        # Get operation list
        if isinstance(search_space, DARTSearchSpace):
            self.operation_list = search_space.AVAILABLE_OPERATIONS
        else:
            self.operation_list = search_space.AVAILABLE_OPERATIONS[:num_operations]

        self.num_operations = len(self.operation_list)

        # Create edge weights (for each node pair and each operation)
        # For a DAG with num_nodes, we have:
        # node 0: no incoming edges (input)
        # node 1: edge from node 0
        # node 2: edges from nodes 0,1
        # ...
        # node num_nodes-1: edges from nodes 0..num_nodes-2
        normal_num_edges = num_nodes * (num_nodes - 1) // 2

        # Normal cell edges
        self.edge_weight_normal = nn.Parameter(
            torch.randn(normal_num_edges, self.num_operations) * 1e-3
        )

        # Reduction cell edges (if applicable)
        if reduction_cell:
            self.edge_weight_reduction = nn.Parameter(
                torch.randn(normal_num_edges, self.num_operations) * 1e-3
            )
        else:
            self.register_buffer("edge_weight_reduction", None)

        # Stem and head parameters
        self.stem_channels = 32
        self.num_classes = 10

    def _get_edge_index(self, node_idx: int) -> int:
        """Get the edge index for a node (0-indexed from node 1)."""
        # Edges come from all previous nodes
        return node_idx * (node_idx - 1) // 2

    def _get_num_edges_up_to(self, node_idx: int) -> int:
        """Get total number of edges up to (and including) a node."""
        return node_idx * (node_idx + 1) // 2

    def forward_edge_weights(
        self,
        cell_type: str = "normal",
        temperature: Optional[float] = None,
    ) -> Tensor:
        """
        Get softmax weights for each edge-operation pair.

        Args:
            cell_type: "normal" or "reduction"
            temperature: Override temperature for softmax

        Returns:
            Tensor of shape [num_edges, num_operations]
        """
        temp = temperature if temperature is not None else self.temperature
        if cell_type == "normal":
            weights = self.edge_weight_normal
        else:
            weights = self.edge_weight_reduction
        return F.softmax(weights / temp, dim=-1)

    def sample_architecture(
        self,
        temperature: Optional[float] = None,
        gumbel: bool = False,
        hard: bool = False,
    ) -> ArchitectureSpec:
        """
        Sample an architecture from the current weights.

        Args:
            temperature: Temperature for softmax
            gumbel: Use Gumbel-softmax
            hard: Use hard Gumbel-softmax (discrete)

        Returns:
            ArchitectureSpec
        """
        temp = temperature if temperature is not None else self.temperature

        # Sample normal cell
        normal_cell = self._sample_cell("normal", temp, gumbel, hard)

        # Sample reduction cell
        reduction_cell = None
        if self.reduction_cell:
            reduction_cell = self._sample_cell("reduction", temp, gumbel, hard)

        return ArchitectureSpec(
            normal_cell=normal_cell,
            reduction_cell=reduction_cell,
            stem_channels=self.stem_channels,
            num_classes=self.num_classes,
        )

    def _sample_cell(
        self,
        cell_type: str,
        temperature: float,
        gumbel: bool,
        hard: bool,
    ) -> CellSpec:
        """Sample a single cell from edge weights."""
        weights = self.forward_edge_weights(cell_type, temperature)

        if gumbel:
            # Gumbel-softmax sampling
            weights = F.gumbel_softmax(weights, tau=temperature, hard=hard)
        else:
            # Argmax for hard selection, softmax for soft
            if hard:
                weights = F.one_hot(
                    weights.argmax(dim=-1), num_classes=self.num_operations
                ).float()
            else:
                weights = weights

        cell = CellSpec(name=cell_type, num_nodes=self.num_nodes)

        # Convert weights to edges
        edge_idx = 0
        for node_idx in range(1, self.num_nodes):
            # Number of edges pointing to this node
            num_inputs = min(node_idx, self.num_nodes - 1)

            for src_idx in range(node_idx):
                # Get operation weights for this edge
                edge_weights = weights[edge_idx]

                # Get selected operation (argmax)
                op_idx = edge_weights.argmax().item()
                op_type = self.operation_list[op_idx]

                operation = OperationSpec(
                    op_type=op_type,
                    kernel_size=self._get_kernel_size(op_type),
                )

                cell.add_edge(src_idx, node_idx, operation)
                edge_idx += 1

        return cell

    def _get_kernel_size(self, op_type: OperationType) -> int:
        """Get kernel size for operation type."""
        if op_type in [
            OperationType.CONV_3X3,
            OperationType.SEP_CONV_3X3,
            OperationType.DIL_CONV_3X3,
        ]:
            return 3
        elif op_type in [
            OperationType.CONV_5X5,
            OperationType.SEP_CONV_5X5,
            OperationType.DIL_CONV_5X5,
        ]:
            return 5
        elif op_type == OperationType.CONV_7X7:
            return 7
        return 1

    def get_architecture_weights(self) -> Dict[str, Tensor]:
        """Get current architecture weights for visualization/analysis."""
        return {
            "normal": self.forward_edge_weights("normal"),
            "reduction": self.forward_edge_weights("reduction")
            if self.reduction_cell
            else None,
        }

    def arch_parameters(self) -> List[nn.Parameter]:
        """Return architecture parameters (edge weights)."""
        params = [self.edge_weight_normal]
        if self.reduction_cell and self.edge_weight_reduction is not None:
            params.append(self.edge_weight_reduction)
        return params

    def genotype(self) -> Dict[str, Any]:
        """
        Get the current genotype (discrete architecture).

        Returns a dictionary representing the architecture.
        """
        arch = self.sample_architecture(temperature=1e-5, hard=True)

        def cell_to_dict(cell: CellSpec) -> Dict:
            edge_dict = {}
            for edge in cell.edges:
                key = f"{edge.source_idx}_{edge.target_idx}"
                edge_dict[key] = edge.operation.op_type.value
            return edge_dict

        result = {
            "normal": cell_to_dict(arch.normal_cell),
        }
        if arch.reduction_cell:
            result["reduction"] = cell_to_dict(arch.reduction_cell)

        return result


class ProxylessNASController(ArchitectureController):
    """
    ProxylessNAS-style differentiable controller.

    Uses reinforcement learning-inspired architecture gradient estimation
    with path dropout for efficient search.

    Reference: Cai et al., "ProxylessNAS: Direct Neural Architecture Search",
    ICLR 2019
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_nodes: int = 4,
        dropout_rate: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__(search_space, num_nodes, temperature=temperature)

        self.dropout_rate = dropout_rate

    def forward(
        self,
        temperature: Optional[float] = None,
        use_dropout: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass returns architecture decisions.

        Args:
            temperature: Softmax temperature
            use_dropout: Whether to use path dropout

        Returns:
            Dict with "normal" and "reduction" edge weights
        """
        temp = temperature if temperature is not None else self.temperature

        weights = {
            "normal": self.forward_edge_weights("normal", temp),
        }

        if self.reduction_cell:
            weights["reduction"] = self.forward_edge_weights("reduction", temp)

        # Apply dropout during training
        if use_dropout and self.training:
            for cell_type in weights:
                mask = torch.rand_like(weights[cell_type]) > self.dropout_rate
                weights[cell_type] = weights[cell_type] * mask.float()

        return weights

    def compute_arch_gradient(
        self,
        val_loss: Tensor,
        reward: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Compute architecture gradient using REINFORCE-style estimation.

        Args:
            val_loss: Validation loss
            reward: Reward signal
            temperature: Temperature for sampling

        Returns:
            Gradient signal for architecture weights
        """
        # Sample architecture
        arch = self.sample_architecture(temperature=temperature, gumbel=True)

        # Compute gradient (simplified REINFORCE)
        # In practice, this would use the full RL formulation
        weights = self.forward(temperature)

        # This is a placeholder - full implementation would involve
        # proper RL policy gradient computation
        return val_loss * reward


class FBNetController(ArchitectureController):
    """
    FBNet-style hardware-aware controller.

    Uses latency-aware loss function for hardware-specific optimization.

    Reference: Wang et al., "FBNet: Hardware-Aware Efficient ConvNet Design",
    CVPR 2019
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_nodes: int = 4,
        latency_table: Optional[Dict[str, float]] = None,
        latency_weight: float = 1.0,
    ):
        super().__init__(search_space, num_nodes)

        self.latency_table = latency_table or self._default_latency_table()
        self.latency_weight = latency_weight

    def _default_latency_table(self) -> Dict[str, float]:
        """Default latency lookup table (in ms)."""
        return {
            "conv_1x1": 0.1,
            "conv_3x3": 0.3,
            "conv_5x5": 0.5,
            "sep_conv_3x3": 0.2,
            "sep_conv_5x5": 0.4,
            "dil_conv_3x3": 0.3,
            "dil_conv_5x5": 0.5,
            "max_pool": 0.1,
            "avg_pool": 0.1,
            "global_pool": 0.2,
            "skip": 0.01,
            "zero": 0.0,
        }

    def estimate_latency(self, architecture: ArchitectureSpec) -> float:
        """
        Estimate latency for an architecture.

        Args:
            architecture: Architecture to estimate

        Returns:
            Estimated latency in milliseconds
        """
        total_latency = 0.0

        for cell in [architecture.normal_cell, architecture.reduction_cell]:
            if cell is None:
                continue

            for edge in cell.edges:
                op_name = edge.operation.op_type.value
                latency = self.latency_table.get(op_name, 0.1)
                total_latency += latency

        return total_latency

    def hardware_aware_loss(
        self,
        val_loss: Tensor,
        architecture: ArchitectureSpec,
    ) -> Tensor:
        """
        Compute hardware-aware loss.

        Args:
            val_loss: Validation loss
            architecture: Current architecture

        Returns:
            Combined loss with latency penalty
        """
        latency = self.estimate_latency(architecture)
        return val_loss + self.latency_weight * latency * torch.tensor(
            latency, device=val_loss.device
        )


class RandomNASController(ArchitectureController):
    """
    Random architecture sampling controller.

    Provides random architecture sampling for baseline comparison
    and exploration.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_nodes: int = 4,
    ):
        super().__init__(search_space, num_nodes)

    def sample_architecture(
        self, rng: Optional[np.random.RandomState] = None
    ) -> ArchitectureSpec:
        """Sample a random architecture."""
        if rng is None:
            rng = np.random.RandomState()

        return self.search_space.sample(rng)

    def forward(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Random controller doesn't use forward weights."""
        return {}


@dataclass
class ControllerState:
    """State of the architecture controller."""

    iteration: int
    current_architecture: ArchitectureSpec
    best_architecture: Optional[ArchitectureSpec]
    best_score: float
    history: List[Dict[str, Any]] = field(default_factory=list)


class ArchitectureSearchOptimizer:
    """
    Optimizer for architecture search.

    Handles the joint optimization of:
    - Architecture weights
    - Model weights
    - Learning rate scheduling
    """

    def __init__(
        self,
        controller: ArchitectureController,
        model: nn.Module,
        unrolled_steps: int = 1,
        reg_loss_weight: float = 1e-3,
    ):
        self.controller = controller
        self.model = model
        self.unrolled_steps = unrolled_steps
        self.reg_loss_weight = reg_loss_weight

    def compute_arch_gradient(
        self,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Tuple[Tensor, Dict]:
        """
        Compute architecture gradient using unrolled optimization.

        Reference: Liu et al., "DARTS: Differentiable Architecture Search"

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for model weights
            criterion: Loss function

        Returns:
            Tuple of (architecture loss, metrics dict)
        """
        # Get validation data
        val_inputs, val_targets = next(iter(val_loader))

        # Save weights
        weights_backup = {n: p.clone() for n, p in self.model.named_parameters()}

        # Unrolled optimization step
        optimizer.zero_grad()

        # Simplified unrolled step
        # In full implementation, this would do multiple gradient steps
        outputs = self.model(val_inputs)
        val_loss = criterion(outputs, val_targets)

        # Get architecture weights for regularization
        arch_weights = self.controller.forward_edge_weights("normal")

        # Regularization: prefer diverse operations
        reg_loss = self.reg_loss_weight * (arch_weights**2).mean()

        total_loss = val_loss + reg_loss
        total_loss.backward()

        # Get gradient for architecture weights
        arch_grad = {
            n: p.grad.clone()
            for n, p in self.controller.named_parameters()
            if p.grad is not None
        }

        # Restore weights
        with torch.no_grad():
            for n, p in weights_backup.items():
                self.model.state_dict()[n].copy_(p)

        return total_loss, {"val_loss": val_loss.item(), "reg_loss": reg_loss.item()}


def create_controller(
    controller_type: str = "darts", search_space: Optional[SearchSpace] = None, **kwargs
) -> ArchitectureController:
    """
    Factory function to create architecture controllers.

    Args:
        controller_type: Type of controller ("darts", "proxyless", "fbnet", "random")
        search_space: Search space to use
        **kwargs: Additional arguments

    Returns:
        ArchitectureController instance
    """
    if search_space is None:
        search_space = DARTSearchSpace()

    controllers = {
        "darts": ArchitectureController,
        "proxyless": ProxylessNASController,
        "fbnet": FBNetController,
        "random": RandomNASController,
    }

    if controller_type.lower() not in controllers:
        raise ValueError(
            f"Unknown controller: {controller_type}. Available: {list(controllers.keys())}"
        )

    return controllers[controller_type.lower()](search_space, **kwargs)

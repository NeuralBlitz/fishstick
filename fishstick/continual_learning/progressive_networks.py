"""
Progressive Neural Networks for Continual Learning.

Dynamic architecture method that adds new columns for each task
while preserving previous columns.

Classes:
- ProgressiveColumn: Single task column in progressive network
- ProgressiveNeuralNetwork: Complete progressive network
- AdapterProgressiveNetwork: Progressive network with adapters
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np


@dataclass
class ColumnConfig:
    """Configuration for a progressive column."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    use_batch_norm: bool = True
    dropout: float = 0.0


class ProgressiveColumn(nn.Module):
    """
    Single Column in Progressive Neural Network.

    Each column is a feedforward network that processes
    input along with lateral connections from previous columns.

    Args:
        config: Column configuration
        num_prev_columns: Number of previous columns for lateral connections
    """

    def __init__(
        self,
        config: ColumnConfig,
        num_prev_columns: int = 0,
    ):
        super().__init__()

        self.config = config
        self.num_prev_columns = num_prev_columns

        self.layers = nn.ModuleList()

        input_dim = config.input_dim + num_prev_columns * config.hidden_dim

        for i in range(config.num_layers):
            if i == config.num_layers - 1:
                out_dim = config.output_dim
            else:
                out_dim = config.hidden_dim

            layers = []

            layers.append(nn.Linear(input_dim, out_dim))

            if config.use_batch_norm and i < config.num_layers - 1:
                layers.append(nn.BatchNorm1d(out_dim))

            if config.dropout > 0 and i < config.num_layers - 1:
                layers.append(nn.Dropout(config.dropout))

            if i < config.num_layers - 1:
                layers.append(nn.ReLU())

            self.layers.append(nn.Sequential(*layers))

            input_dim = out_dim

    def forward(
        self, x: Tensor, prev_features: Optional[List[Tensor]] = None
    ) -> Tensor:
        """
        Forward pass with lateral connections.

        Args:
            x: Input tensor
            prev_features: Features from previous columns

        Returns:
            Output tensor
        """
        if prev_features is not None and len(prev_features) > 0:
            lateral = torch.cat(prev_features, dim=-1)
            x = torch.cat([x, lateral], dim=-1)

        for layer in self.layers:
            x = layer(x)

        return x


class ProgressiveNeuralNetwork(nn.Module):
    """
    Progressive Neural Network for Continual Learning.

    Adds a new neural network column for each task while
    keeping previous columns frozen.

    Reference:
        Rusu et al., "Progressive Neural Networks", arXiv 2016

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_tasks: Maximum number of tasks
        num_layers: Number of layers per column
        freeze_columns: Whether to freeze previous columns
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_tasks: int = 10,
        num_layers: int = 2,
        freeze_columns: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_tasks = num_tasks
        self.freeze_columns = freeze_columns

        self.columns: nn.ModuleList[ProgressiveColumn] = nn.ModuleList()
        self.active_task: int = 0
        self.task_to_column: Dict[int, int] = {}

        config = ColumnConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )

        for task_id in range(num_tasks):
            num_prev = task_id if task_id > 0 else 0
            column = ProgressiveColumn(config, num_prev)
            self.columns.append(column)
            self.task_to_column[task_id] = task_id

    def add_task(self, task_id: int) -> None:
        """Add a new task column."""
        if task_id >= self.num_tasks:
            raise ValueError(f"Cannot add task {task_id}, max {self.num_tasks} tasks")

        self.active_task = task_id

        if task_id >= len(self.columns):
            config = ColumnConfig(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=2,
            )

            column = ProgressiveColumn(config, task_id)
            self.columns.append(column)

    def forward(
        self,
        x: Tensor,
        task_id: Optional[int] = None,
    ) -> Tensor:
        """
        Forward pass for specific task.

        Args:
            x: Input tensor
            task_id: Task ID (uses active task if None)

        Returns:
            Output logits
        """
        if task_id is None:
            task_id = self.active_task

        column_idx = self.task_to_column.get(task_id, 0)
        column = self.columns[column_idx]

        if self.freeze_columns:
            prev_features = []

            for i in range(column_idx):
                with torch.no_grad():
                    prev_input = x if i == 0 else prev_features[-1]
                    prev_features.append(self.columns[i](prev_input))

            return column(x, prev_features)
        else:
            prev_features = []

            for i in range(column_idx):
                prev_input = x if i == 0 else prev_features[-1]
                prev_features.append(self.columns[i](prev_input))

            return column(x, prev_features)

    def get_column_features(
        self,
        x: Tensor,
        task_id: int,
    ) -> List[Tensor]:
        """Get intermediate features from all columns."""
        features = []

        column_idx = self.task_to_column.get(task_id, 0)

        for i in range(column_idx + 1):
            if i == 0:
                feat = self.columns[i](x)
            else:
                feat = self.columns[i](x, features)
            features.append(feat)

        return features

    def freeze_task(self, task_id: int) -> None:
        """Freeze parameters for a specific task."""
        column_idx = self.task_to_column.get(task_id, 0)

        for param in self.columns[column_idx].parameters():
            param.requires_grad = False

    def unfreeze_task(self, task_id: int) -> None:
        """Unfreeze parameters for a specific task."""
        column_idx = self.task_to_column.get(task_id, 0)

        for param in self.columns[column_idx].parameters():
            param.requires_grad = True


class AdapterProgressiveNetwork(nn.Module):
    """
    Progressive Network with Lightweight Adapters.

    Uses adapter modules instead of full columns for
    memory efficiency.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_tasks: Maximum number of tasks
        adapter_dim: Dimension of adapter bottleneck
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_tasks: int = 10,
        adapter_dim: int = 64,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_tasks = num_tasks
        self.adapter_dim = adapter_dim

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.task_adapters = nn.ModuleDict()

        for task_id in range(num_tasks):
            self.task_adapters[str(task_id)] = nn.ModuleDict(
                {
                    "down": nn.Linear(hidden_dim, adapter_dim),
                    "up": nn.Linear(adapter_dim, hidden_dim),
                }
            )

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: Tensor,
        task_id: int = 0,
    ) -> Tensor:
        """
        Forward pass with task-specific adapter.

        Args:
            x: Input tensor
            task_id: Task ID

        Returns:
            Output logits
        """
        h = self.shared_encoder(x)

        if str(task_id) in self.task_adapters:
            adapter = self.task_adapters[str(task_id)]
            h = adapter["up"](F.relu(adapter["down"](h)))

        return self.classifier(h)

    def add_task(self, task_id: int) -> None:
        """Add new task adapter."""
        self.task_adapters[str(task_id)] = nn.ModuleDict(
            {
                "down": nn.Linear(self.hidden_dim, self.adapter_dim),
                "up": nn.Linear(self.adapter_dim, self.hidden_dim),
            }
        )

    def freeze_task(self, task_id: int) -> None:
        """Freeze adapter for task."""
        if str(task_id) in self.task_adapters:
            for param in self.task_adapters[str(task_id)].parameters():
                param.requires_grad = False


class LateralConnectionProgressive(nn.Module):
    """
    Progressive Network with Different Lateral Connection Types.

    Implements different strategies for connecting
    task columns.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_tasks: Maximum tasks
        connection_type: Type of connection ('dense', 'residual', 'attention')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_tasks: int = 10,
        connection_type: str = "dense",
    ):
        super().__init__()

        self.connection_type = connection_type

        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.columns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_tasks)
            ]
        )

        if connection_type == "attention":
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads=4, batch_first=True
            )

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: Tensor,
        task_id: int = 0,
    ) -> Tensor:
        """Forward pass with lateral connections."""
        h = F.relu(self.encoder(x))

        prev_features = []

        for i in range(task_id):
            if self.connection_type == "dense":
                prev = self.columns[i](h)
                prev_features.append(prev)
            elif self.connection_type == "residual" and i == task_id - 1:
                prev = self.columns[i](h)
                prev_features.append(prev)

        if len(prev_features) > 0:
            if self.connection_type == "attention":
                h_expanded = h.unsqueeze(1)
                prev_stacked = torch.stack(prev_features, dim=1)

                attn_out, _ = self.attention(h_expanded, prev_stacked, prev_stacked)
                h = h + attn_out.squeeze(1)
            else:
                lateral = torch.stack(prev_features, dim=0).mean(dim=0)
                h = h + lateral

        out = self.columns[task_id](h)

        return self.classifier(out)

"""
Hard Attention to the Task (HAT) Implementation.

Uses hard attention masks to protect task-specific parameters.

Classes:
- HardAttentionTask: HAT attention mechanism
- HATMethod: Complete HAT method
- TaskEmbedding: Task embedding for HAT
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np


class HardAttentionTask(nn.Module):
    """
    Hard Attention to the Task (HAT).

    Uses attention masks to selectively enable/disable
    network components for different tasks.

    Reference:
        Serra et al., "Overcoming Catastrophic Forgetting with Hard
        Attention to the Task", ICML 2018

    Args:
        num_tasks: Number of tasks
        num_layers: Number of layers to apply attention
        attention_dim: Dimension of attention embeddings
    """

    def __init__(
        self,
        num_tasks: int,
        num_layers: int,
        attention_dim: int = 128,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.attention_dim = attention_dim

        self.task_embeddings = nn.Embedding(num_tasks, attention_dim)

        self.attention_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(attention_dim, attention_dim),
                    nn.ReLU(),
                    nn.Linear(attention_dim, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_scales: Dict[int, Dict[int, Tensor]] = {}

    def forward(
        self,
        task_id: int,
        layer_idx: int,
    ) -> Tensor:
        """
        Get attention mask for specific task and layer.

        Args:
            task_id: Task identifier
            layer_idx: Layer index

        Returns:
            Attention mask
        """
        task_embedding = self.task_embeddings(task_id)

        attention_layer = self.attention_layers[layer_idx]

        mask = attention_layer(task_embedding)

        if self.training:
            scale = mask
        else:
            scale = (mask > 0.5).float()

        return scale

    def get_full_mask(self, task_id: int) -> List[Tensor]:
        """Get attention masks for all layers."""
        masks = []

        for layer_idx in range(self.num_layers):
            masks.append(self.forward(task_id, layer_idx))

        return masks

    def compute_regularization(self, task_id: int) -> Tensor:
        """
        Compute attention regularization penalty.

        Penalizes deviation from uniform attention.

        Args:
            task_id: Task identifier

        Returns:
            Regularization loss
        """
        loss = torch.tensor(0.0, device=self.task_embeddings.weight.device)

        for layer_idx in range(self.num_layers):
            mask = self.forward(task_id, layer_idx)

            uniform = torch.ones_like(mask) * 0.5

            loss += F.mse_loss(mask, uniform)

        return loss


class HATMethod(nn.Module):
    """
    Complete HAT Continual Learning Method.

    Args:
        backbone: Feature extractor network
        num_tasks: Number of tasks
        hidden_dim: Hidden dimension
        num_classes: Number of classes
        attention_layers: Number of attention layers
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_tasks: int,
        hidden_dim: int,
        num_classes: int,
        attention_layers: int = 4,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim

        self.attention = HardAttentionTask(num_tasks, attention_layers)

        self.task_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_tasks)]
        )

        self.smax = 4.0
        self.current_task = 0

    def forward(
        self,
        x: Tensor,
        task_id: int,
    ) -> Tensor:
        """
        Forward pass with task-specific attention.

        Args:
            x: Input tensor
            task_id: Task identifier

        Returns:
            Task-specific logits
        """
        features = self.backbone(x)

        attention_masks = self.attention.get_full_mask(task_id)

        if self.training:
            attention_scale = torch.sigmoid(attention_masks[0] * self.smax)
        else:
            attention_scale = (
                torch.sigmoid(attention_masks[0] * self.smax) > 0.5
            ).float()

        scaled_features = features * attention_scale

        if task_id < len(self.task_heads):
            return self.task_heads[task_id](scaled_features)
        else:
            return self.task_heads[0](scaled_features)

    def compute_loss(
        self,
        x: Tensor,
        y: Tensor,
        task_id: int,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute loss with regularization.

        Args:
            x: Input
            y: Target
            task_id: Task ID

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        logits = self.forward(x, task_id)

        task_loss = F.cross_entropy(logits, y)

        attention_reg = self.attention.compute_regularization(task_id)

        total_loss = task_loss + 0.001 * attention_reg

        loss_dict = {
            "task_loss": task_loss,
            "attention_reg": attention_reg,
            "total": total_loss,
        }

        return total_loss, loss_dict

    def set_task(self, task_id: int) -> None:
        """Set active task."""
        self.current_task = task_id

    def get_attention_stats(self, task_id: int) -> Dict[str, float]:
        """Get attention statistics for a task."""
        masks = self.attention.get_full_mask(task_id)

        stats = {}

        for i, mask in enumerate(masks):
            stats[f"layer_{i}_mean"] = mask.mean().item()
            stats[f"layer_{i}_std"] = mask.std().item()

        return stats


class TaskEmbedding(nn.Module):
    """
    Learnable Task Embedding for HAT.

    Provides task-specific embeddings that drive
    attention mask generation.

    Args:
        num_tasks: Maximum number of tasks
        embedding_dim: Dimension of task embeddings
        num_layers: Number of layers to attend
    """

    def __init__(
        self,
        num_tasks: int,
        embedding_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_tasks, embedding_dim)

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(num_layers)]
        )

        self.mask_predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, 1),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        task_id: Tensor,
    ) -> List[Tensor]:
        """
        Generate attention masks for task.

        Args:
            task_id: Task ID tensor

        Returns:
            List of attention masks per layer
        """
        emb = self.embedding(task_id)

        masks = []

        for i, (norm, predictor) in enumerate(
            zip(self.layer_norms, self.mask_predictors)
        ):
            normed = norm(emb)
            mask = torch.sigmoid(predictor(normed))
            masks.append(mask)

        return masks

    def get_task_embedding(self, task_id: int) -> Tensor:
        """Get embedding for specific task."""
        return self.embedding(
            torch.tensor(task_id, device=self.embedding.weight.device)
        )


class SupSupMethod(nn.Module):
    """
    SupSup: Super-Sets of Weights.

    Extension of HAT that allows multiple task-specific
    weight subsets.

    Reference:
        Wortsman et al., "SupSup: Learning to Switch Model",
        NeurIPS 2020

    Args:
        base_model: Base network
        num_tasks: Number of tasks
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_tasks: int = 10,
    ):
        super().__init__()

        self.base_model = base_model
        self.num_tasks = num_tasks

        self.task_groups: Dict[int, List[str]] = {}

        param_groups = {}

        for name, param in base_model.named_parameters():
            prefix = name.split(".")[0]
            if prefix not in param_groups:
                param_groups[prefix] = []
            param_groups[prefix].append(name)

        for i, params in enumerate(param_groups.values()):
            self.task_groups[i] = params

    def forward(self, x: Tensor, task_id: int) -> Tensor:
        """Forward with task-specific weights."""
        return self.base_model(x)

    def get_task_params(self, task_id: int) -> Dict[str, Tensor]:
        """Get parameters for task."""
        return {
            name: param
            for name, param in self.base_model.named_parameters()
            if name in self.task_groups.get(task_id, [])
        }

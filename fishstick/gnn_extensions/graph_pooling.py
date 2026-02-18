"""
Advanced Graph Pooling Methods.

This module provides various graph pooling techniques:
- MinCut pooling
- DiffPool (Differentiable Pooling)
- Top-k pooling
- SAGPool (Self-Attention Graph Pooling)
- Attention-based pooling
- Hierarchical pooling
"""

from typing import Optional, Tuple, List, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm, Dropout, ModuleList
import math


class MinCutPool(nn.Module):
    """
    MinCut pooling layer for graph clustering.

    Performs differentiable clustering with MinCut loss optimization.

    Args:
        in_channels: Input feature dimension
        out_channels: Number of clusters (pooling ratio)
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.assign_conv = Linear(in_channels, out_channels)
        self.feature_conv = Linear(in_channels, in_channels)

        self.dropout = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass with MinCut pooling.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Optional edge weights

        Returns:
            Tuple of (pooled_x, pooled_edge_index, pooled_edge_weight, assign_matrix)
        """
        S = self.assign_conv(x)
        S = F.softmax(S, dim=-1)
        S = self.dropout(S)

        x_transformed = self.feature_conv(x)

        num_nodes = x.size(0)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        row, col = edge_index

        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[row, col] = edge_weight
        adj = adj + adj.t()

        out = torch.matmul(S.t(), adj)
        out = torch.matmul(out, S)

        out = out / (S.sum(dim=0, keepdim=True).t() + 1e-8)

        pooled_x = torch.matmul(S.t(), x_transformed)

        assign_mask = S.argmax(dim=-1)

        new_edge_index = []
        new_edge_weight = []

        for i in range(self.out_channels):
            for j in range(self.out_channels):
                if i != j:
                    new_edge_index.append([i, j])
                    new_edge_weight.append(out[i, j].item())

        if new_edge_index:
            pooled_edge_index = torch.tensor(
                new_edge_index, dtype=torch.long, device=x.device
            ).t()
            pooled_edge_weight = torch.tensor(
                new_edge_weight, dtype=torch.float, device=x.device
            )
        else:
            pooled_edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
            pooled_edge_weight = torch.empty(0, dtype=torch.float, device=x.device)

        return pooled_x, pooled_edge_index, pooled_edge_weight, S

    def loss(self, x: Tensor, S: Tensor, edge_index: Tensor) -> Tensor:
        """
        Compute MinCut loss.

        Args:
            x: Node features
            S: Assignment matrix
            edge_index: Edge connectivity

        Returns:
            MinCut loss value
        """
        num_nodes = x.size(0)

        row, col = edge_index
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[row, col] = 1.0
        adj = adj + adj.t()

        d = adj.sum(dim=1)
        D = torch.diag(d)

        L = D - adj

        cut = torch.sum(torch.matmul(torch.matmul(S.t(), L), S))
        assoc = torch.sum(torch.matmul(S.t(), D))

        return cut / (assoc + 1e-8)


class DiffPool(nn.Module):
    """
    Differentiable Pooling layer (DiffPool).

    Learns a soft clustering assignment matrix for hierarchical pooling.

    Args:
        in_channels: Input feature dimension
        out_channels: Number of clusters
        num_layers: Number of GNN layers for assignment prediction
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.assignment_layers = ModuleList(
            [
                Linear(in_channels if i == 0 else out_channels, out_channels)
                for i in range(num_layers)
            ]
        )

        self.feature_layers = ModuleList(
            [
                Linear(in_channels if i == 0 else in_channels, in_channels)
                for i in range(num_layers)
            ]
        )

        self.dropout = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass with differentiable pooling.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Optional edge weights

        Returns:
            Tuple of (pooled_x, pooled_edge_index, pooled_edge_weight, assign_matrix)
        """
        num_nodes = x.size(0)

        for i, layer in enumerate(self.assignment_layers):
            if i > 0:
                x = F.relu(x)
            S = layer(x)
        S = F.softmax(S, dim=-1)
        S = self.dropout(S)

        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i < len(self.feature_layers) - 1:
                x = F.relu(x)

        pooled_x = torch.matmul(S.t(), x)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)

        row, col = edge_index

        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[row, col] = edge_weight
        adj = adj + adj.t()

        pooled_adj = torch.matmul(torch.matmul(S.t(), adj), S)

        pooled_edge_index = pooled_adj.nonzero().t()
        pooled_edge_weight = pooled_adj[pooled_edge_index[0], pooled_edge_index[1]]

        return pooled_x, pooled_edge_index, pooled_edge_weight, S

    def link_prediction_loss(
        self,
        S: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Link prediction loss for pooler.

        Args:
            S: Assignment matrix
            edge_index: Edge connectivity

        Returns:
            Link prediction loss
        """
        num_nodes = S.size(0)

        adj = torch.zeros(num_nodes, num_nodes, device=S.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = adj + adj.t()

        adj_pooled = torch.matmul(torch.matmul(S.t(), adj), S)

        S_sum = S.sum(dim=0, keepdim=True)
        norm = torch.matmul(S_sum.t(), S_sum)

        pred_adj = adj_pooled / (norm + 1e-8)

        loss = F.mse_loss(pred_adj, adj_pooled)

        return loss

    def entropy_loss(self, S: Tensor) -> Tensor:
        """
        Entropy loss to encourage discrete assignments.

        Args:
            S: Assignment matrix

        Returns:
            Entropy loss
        """
        return -torch.sum(S * torch.log(S + 1e-8)) / S.size(0)


class TopKPool(nn.Module):
    """
    Top-k pooling layer.

    Selects top-k nodes based on learnable attention scores.

    Args:
        in_channels: Input feature dimension
        ratio: Pooling ratio (fraction of nodes to keep)
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.score = Linear(in_channels, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass with top-k pooling.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Tuple of (pooled_x, pooled_edge_index, pooled_batch, perm, scores)
        """
        scores = self.score(x).squeeze(-1)
        scores = F.tanh(scores)

        num_nodes = x.size(0)

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        batch_size = batch.max().item() + 1

        perm_list = []
        for b in range(batch_size):
            mask = batch == b
            batch_scores = scores[mask]

            k = max(1, int(mask.sum().item() * self.ratio))

            _, indices = batch_scores.topk(k)

            global_indices = torch.where(mask)[0][indices]
            perm_list.append(global_indices)

        perm = torch.cat(perm_list, dim=0)

        x_pooled = x[perm]
        edge_index_pooled, edge_mask = self._subgraph(perm, edge_index, num_nodes)
        batch_pooled = batch[perm]

        return x_pooled, edge_index_pooled, batch_pooled, perm, scores[perm]

    def _subgraph(
        self,
        perm: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tuple[Tensor, Tensor]:
        """Extract subgraph for selected nodes."""
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        mask[perm] = True

        row, col = edge_index
        edge_mask = mask[row] & mask[col]

        return edge_index[:, edge_mask], edge_mask


class SAGPool(SelfAttentionGraphPool):
    """
    Self-Attention Graph Pooling (SAGPool).

    Uses self-attention to learn node importance for pooling.

    Args:
        in_channels: Input feature dimension
        ratio: Pooling ratio
       gnn: GNN type for score computation (gcn, gin, graphSAGE)
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        gnn: str = "gcn",
    ):
        super().__init__(in_channels, ratio)

        self.gnn_type = gnn

        if gnn == "gcn":
            self.gnn = GraphConv(in_channels, 1)
        elif gnn == "gin":
            self.gnn = GINConv(
                nn.Sequential(
                    Linear(in_channels, in_channels), nn.ReLU(), Linear(in_channels, 1)
                )
            )
        elif gnn == "graphSAGE":
            self.gnn = SAGEConv(in_channels, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass with self-attention graph pooling."""
        scores = self.gnn(x, edge_index).squeeze(-1)
        scores = F.tanh(scores)

        return super().forward(x, edge_index, batch)


class GraphConv(nn.Module):
    """Simple Graph Convolution layer for SAGPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        out = self.linear(x)
        out = out + out[col].index_add(0, row, torch.ones_like(row, dtype=torch.float))
        return out


class GINConv(nn.Module):
    """Graph Isomorphism Network convolution."""

    def __init__(self, nn_module: nn.Module):
        super().__init__()
        self.nn = nn_module

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        out = self.nn(x)
        out = out + out[col].index_add(0, row, torch.ones_like(row, dtype=torch.float))
        return out


class SAGEConv(nn.Module):
    """GraphSAGE convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index
        out = self.linear(x)
        out = out + out[col].mean(dim=0, keepdim=True).index_add(
            0, row, torch.ones_like(row, dtype=torch.float)
        )
        return out


class SelfAttentionGraphPool(nn.Module):
    """
    Self-Attention Graph Pooling base class.

    Args:
        in_channels: Input feature dimension
        ratio: Pooling ratio
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.score = nn.Sequential(
            Linear(in_channels, in_channels),
            nn.Tanh(),
            Linear(in_channels, 1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass."""
        scores = self.score(x).squeeze(-1)

        num_nodes = x.size(0)

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        batch_size = batch.max().item() + 1

        perm_list = []
        for b in range(batch_size):
            mask = batch == b
            batch_scores = scores[mask]

            k = max(1, int(mask.sum().item() * self.ratio))

            _, indices = batch_scores.topk(k)

            global_indices = torch.where(mask)[0][indices]
            perm_list.append(global_indices)

        perm = torch.cat(perm_list, dim=0)

        x_pooled = x[perm]

        mask_full = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        mask_full[perm] = True

        row, col = edge_index
        edge_mask = mask_full[row] & mask_full[col]

        edge_index_pooled = edge_index[:, edge_mask]
        batch_pooled = batch[perm]

        return x_pooled, edge_index_pooled, batch_pooled, perm, scores[perm]


class AttentionPool(nn.Module):
    """
    Attention-based graph pooling.

    Uses attention mechanism to compute weighted pooling.

    Args:
        in_channels: Input feature dimension
        out_channels: Output dimension (for graph-level output)
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.query = Linear(in_channels, out_channels)
        self.key = Linear(in_channels, out_channels)
        self.value = Linear(in_channels, in_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with attention pooling.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Tuple of (graph_embeddings, attention_weights)
        """
        num_nodes = x.size(0)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        row, col = edge_index
        attention = (q[row] * k[col]).sum(-1)
        attention = F.leaky_relu(attention, 0.2)
        attention = F.softmax(attention, dim=0)

        attention_weights = torch.zeros(num_nodes, device=x.device)
        attention_weights.index_add_(0, row, attention)

        graph_emb = (attention_weights.unsqueeze(-1) * v).sum(dim=0)

        return graph_emb, attention_weights


class HierarchicalPool(nn.Module):
    """
    Hierarchical pooling with multiple levels.

    Applies multiple pooling operations in sequence for multi-scale
    graph representations.

    Args:
        in_channels: Input feature dimension
        num_pools: Number of pooling levels
        ratios: List of pooling ratios for each level
    """

    def __init__(
        self,
        in_channels: int,
        num_pools: int = 3,
        ratios: Optional[List[float]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_pools = num_pools

        if ratios is None:
            ratios = [0.5] * num_pools
        self.ratios = ratios

        self.pools = ModuleList([TopKPool(in_channels, ratio) for ratio in ratios])

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Forward pass with hierarchical pooling.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Tuple of lists: (pooled_features, pooled_edge_indices, pools)
        """
        x_list = [x]
        edge_list = [edge_index]

        current_x = x
        current_edge = edge_index
        current_batch = batch

        for pool in self.pools:
            current_x, current_edge, current_batch, perm, scores = pool(
                current_x, current_edge, current_batch
            )

            x_list.append(current_x)
            edge_list.append(current_edge)

        return x_list, edge_list, self.pools


class Set2SetPool(nn.Module):
    """
    Set2Set pooling operation.

    Uses order-invariant set representation for graph-level pooling.

    Args:
        in_channels: Input feature dimension
        processing_steps: Number of processing steps
        num_heads: Number of LSTM iterations
    """

    def __init__(
        self,
        in_channels: int,
        processing_steps: int = 4,
        num_heads: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.processing_steps = processing_steps

        self.lstm = nn.LSTM(
            in_channels,
            in_channels,
            num_layers=1,
            batch_first=True,
        )

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with Set2Set pooling.

        Args:
            x: Node features [num_nodes, in_channels]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph embedding [in_channels]
        """
        num_nodes = x.size(0)

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        batch_size = batch.max().item() + 1

        h = torch.zeros(1, batch_size, self.in_channels, device=x.device)
        c = torch.zeros(1, batch_size, self.in_channels, device=x.device)

        q = torch.randn(batch_size, 1, self.in_channels, device=x.device)

        for _ in range(self.processing_steps):
            q, (h, c) = self.lstm(q, (h, c))

            q_expanded = q.expand(num_nodes, -1, -1)
            x_expanded = x.unsqueeze(1).expand(-1, batch_size, -1)

            scores = (q_expanded * x_expanded).sum(dim=-1)

            scores = scores.masked_fill(
                batch.unsqueeze(1)
                != torch.arange(batch_size, device=x.device).unsqueeze(0),
                float("-inf"),
            )
            attn = F.softmax(scores, dim=0)

            weighted = (attn.unsqueeze(-1) * x_expanded).sum(dim=0)

            q = weighted.unsqueeze(1)

        return q.squeeze(1)

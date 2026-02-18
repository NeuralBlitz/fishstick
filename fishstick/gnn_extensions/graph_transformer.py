"""
Graph Transformer Implementations.

This module provides advanced graph transformer architectures including:
- Graph Transformer layers with multi-head attention
- Positional encodings (Laplacian, random walk, centrality-based)
- Graph attention mechanisms
"""

from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm, Dropout, ModuleList
import math


class GraphTransformerLayer(nn.Module):
    """
    Graph Transformer Layer implementing self-attention on graph structures.

    Implements the transformer architecture adapted for graphs with:
    - Multi-head self-attention over node features
    - Feed-forward network with dropout
    - Layer normalization and residual connections

    Args:
        in_channels: Input node feature dimension
        out_channels: Output node feature dimension
        num_heads: Number of attention heads (must divide out_channels)
        dropout: Dropout probability
        edge_dim: Edge feature dimension
        use_edge_features: Whether to use edge features in attention
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: int = 0,
        use_edge_features: bool = False,
    ):
        super().__init__()
        assert out_channels % num_heads == 0, (
            "out_channels must be divisible by num_heads"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.dropout = dropout
        self.use_edge_features = use_edge_features

        self.query = Linear(in_channels, out_channels)
        self.key = Linear(in_channels, out_channels)
        self.value = Linear(in_channels, out_channels)

        if use_edge_features and edge_dim > 0:
            self.edge_proj = Linear(edge_dim, out_channels)
        else:
            self.edge_proj = None

        self.out_proj = Linear(out_channels, out_channels)
        self.ln1 = LayerNorm(out_channels)
        self.ln2 = LayerNorm(out_channels)

        self.ffn = nn.Sequential(
            Linear(out_channels, out_channels * 4),
            nn.GELU(),
            Dropout(dropout),
            Linear(out_channels * 4, out_channels),
            Dropout(dropout),
        )

        self.dropout_layer = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of Graph Transformer Layer.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            batch: Batch assignment [num_nodes] (optional)

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        batch_size = 1 if batch is None else batch.max().item() + 1

        q = self.query(x).view(-1, self.num_heads, self.head_dim)
        k = self.key(x).view(-1, self.num_heads, self.head_dim)
        v = self.value(x).view(-1, self.num_heads, self.head_dim)

        row, col = edge_index
        k_local = k[row]
        v_local = v[col]

        if self.edge_proj is not None and edge_attr is not None:
            edge_bias = self.edge_proj(edge_attr)
            edge_bias = edge_bias.view(-1, self.num_heads, self.head_dim)
            k_local = k_local + edge_bias

        attn_scores = (q[row] * k_local).sum(-1) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(
            torch.zeros_like(attn_scores).bool(), float("-inf")
        )
        attn_weights = F.softmax(attn_scores, dim=0)
        attn_weights = self.dropout_layer(attn_weights)

        messages = v_local * attn_weights.unsqueeze(-1)
        messages = messages.view(-1, self.out_channels)

        h_out = torch.zeros_like(x).index_add_(0, row, messages)

        h_out = self.out_proj(h_out)
        h_out = self.dropout_layer(h_out)

        h = self.ln1(x + h_out)
        h = h + self.ffn(self.ln2(h))

        return h

    def attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        edge_index: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute multi-head attention scores and weights.

        Args:
            q: Query tensor [num_nodes, num_heads, head_dim]
            k: Key tensor [num_nodes, num_heads, head_dim]
            v: Value tensor [num_nodes, num_heads, head_dim]
            edge_index: Edge connectivity [2, num_edges]
            mask: Optional attention mask

        Returns:
            Tuple of (attention output, attention weights)
        """
        row, col = edge_index

        scores = (q[row] * k[col]).sum(-1) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=0)

        out = v[col] * attn_weights.unsqueeze(-1)
        out = out.view(-1, self.num_heads, self.head_dim)

        return out, attn_weights


class GraphTransformer(nn.Module):
    """
    Complete Graph Transformer model with multiple layers.

    Args:
        in_channels: Input node feature dimension
        hidden_channels: Hidden dimension for each layer
        out_channels: Output node feature dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        edge_dim: Edge feature dimension
        use_edge_features: Whether to use edge features
        pool: Graph pooling method ('cls', 'mean', 'max')
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: int = 0,
        use_edge_features: bool = False,
        pool: str = "mean",
    ):
        super().__init__()

        self.node_encoder = Linear(in_channels, hidden_channels)

        if pool == "cls":
            self.cls_token = Parameter(torch.randn(1, hidden_channels))

        self.pool = pool
        self.layers = ModuleList(
            [
                GraphTransformerLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    use_edge_features=use_edge_features,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            Dropout(dropout),
            Linear(hidden_channels, out_channels),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through the Graph Transformer.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Tuple of (graph embeddings, node embeddings)
        """
        x = self.node_encoder(x)

        if self.pool == "cls" and self.cls_token is not None:
            batch_size = 1 if batch is None else batch.max().item() + 1
            cls_tokens = self.cls_token.expand(batch_size, -1)
            x = torch.cat([x, cls_tokens], dim=0)

            cls_edge_index = torch.arange(batch_size, device=x.device)
            cls_edge_index = cls_edge_index.unsqueeze(0).repeat(2, batch_size)
            edge_index = torch.cat([edge_index, cls_edge_index], dim=1)

            if batch is not None:
                batch = torch.cat(
                    [batch, torch.arange(batch_size, device=batch.device)], dim=0
                )

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)

        if self.pool == "cls":
            node_emb = x[:-batch_size] if batch is not None else x[:-1]
            graph_emb = x[-batch_size:] if batch is not None else x[-1:]
        elif self.pool == "mean":
            graph_emb = x.mean(dim=0, keepdim=True)
            node_emb = x
        elif self.pool == "max":
            graph_emb = x.max(dim=0, keepdim=True)[0]
            node_emb = x
        else:
            graph_emb = x
            node_emb = x

        graph_emb = self.classifier(graph_emb)

        return graph_emb, node_emb


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer with multi-head attention.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        heads: Number of attention heads
        dropout: Dropout probability
        concat: Whether to concatenate head outputs
        negative_slope: LeakyReLU negative slope
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        concat: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.dropout = Dropout(dropout)

        if concat:
            self.out_proj = Linear(heads * out_channels, out_channels)
        else:
            self.out_proj = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with multi-head attention.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        H, C = self.heads, self.out_channels

        x_l = self.lin(x).view(-1, H, C)
        x_r = x_l

        row, col = edge_index
        x_l_i = x_l[row]
        x_r_j = x_r[col]

        alpha = torch.cat([x_l_i, x_r_j], dim=-1)
        alpha = (alpha * self.att).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=0)
        alpha = self.dropout(alpha)

        out = x_r[col] * alpha.view(-1, H, 1)

        if self.concat:
            out = out.view(-1, H * C)
            out = self.out_proj(out)
        else:
            out = out.mean(dim=1)
            out = self.out_proj(out)

        return out

    def message(self, x_i: Tensor, x_j: Tensor, alpha: Tensor) -> Tensor:
        """Message computation for PyG interface."""
        return alpha.unsqueeze(-1) * x_j


class DirectionalGraphAttention(nn.Module):
    """
    Directional Graph Attention with edge direction awareness.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        heads: Number of attention heads
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = out_channels // heads

        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)

        self.direction_proj = nn.Linear(1, heads)

        self.out_proj = nn.Linear(out_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with directional attention.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        row, col = edge_index

        q = self.query(x).view(-1, self.heads, self.head_dim)
        k = self.key(x).view(-1, self.heads, self.head_dim)
        v = self.value(x).view(-1, self.heads, self.head_dim)

        dir_vec = (x[col] - x[row]).unsqueeze(-1)
        dir_weight = self.direction_proj(dir_vec)

        scores = (q[row] * k[col]).sum(-1) / math.sqrt(self.head_dim)

        if edge_weight is not None:
            scores = scores + edge_weight.unsqueeze(-1)

        scores = scores + dir_weight.squeeze(-1)

        attn = F.softmax(scores, dim=0)

        messages = v[col] * attn.unsqueeze(-1)
        out = torch.zeros_like(x).index_add_(0, row, messages)

        return self.out_proj(out.view(-1, self.heads * self.head_dim))

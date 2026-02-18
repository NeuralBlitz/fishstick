"""
Graph Matching Algorithms.

This module provides implementations for graph matching:
- Graph Matching Networks (GMN)
- Graph similarity learning
- Subgraph matching and isomorphism
- Graph alignment
"""

from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm, Dropout, ModuleList
import math


class GraphMatchingNetwork(nn.Module):
    """
    Graph Matching Network for learning graph similarity.

    Uses cross-graph attention to compute correspondences between graphs.

    Args:
        node_dim: Node feature dimension
        edge_dim: Edge feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
        num_iterations: Number of cross-graph iterations
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_iterations: int = 3,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_iterations = num_iterations

        self.node_encoder = nn.Sequential(
            Linear(node_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [GraphMatchingLayer(hidden_dim, edge_dim) for _ in range(num_layers)]
        )

        self.cross_graph_layers = nn.ModuleList(
            [CrossGraphAttention(hidden_dim) for _ in range(num_iterations)]
        )

        self.similarity = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(
        self,
        x1: Tensor,
        edge_index1: Tensor,
        edge_attr1: Optional[Tensor],
        x2: Tensor,
        edge_index2: Tensor,
        edge_attr2: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for graph matching.

        Args:
            x1: Graph1 node features
            edge_index1: Graph1 edge connectivity
            edge_attr1: Graph1 edge features
            x2: Graph2 node features
            edge_index2: Graph2 edge connectivity
            edge_attr2: Graph2 edge features

        Returns:
            Tuple of (similarity_score, graph1_embeddings)
        """
        x1 = self.node_encoder(x1)
        x2 = self.node_encoder(x2)

        if edge_attr1 is not None:
            edge_attr1 = self.edge_encoder(edge_attr1)
        if edge_attr2 is not None:
            edge_attr2 = self.edge_encoder(edge_attr2)

        for layer in self.layers:
            x1 = layer(x1, edge_index1, edge_attr1)
            x2 = layer(x2, edge_index2, edge_attr2)

        x1_init = x1.clone()
        x2_init = x2.clone()

        for cross_layer in self.cross_graph_layers:
            x1, x2 = cross_layer(x1, x2, x1_init, x2_init)

        num_nodes1 = x1.size(0)
        num_nodes2 = x2.size(0)

        similarity_matrix = torch.matmul(x1, x2.t())

        sim_scores = []
        for i in range(num_nodes1):
            for j in range(num_nodes2):
                sim = self.similarity(x1[i : i + 1], x2[j : j + 1])
                sim_scores.append(sim)

        if sim_scores:
            similarity_score = torch.stack(sim_scores).mean()
        else:
            similarity_score = torch.tensor(0.0, device=x1.device)

        return similarity_score, x1


class GraphMatchingLayer(nn.Module):
    """Graph matching layer with edge convolution."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.node_mlp = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        """Forward pass."""
        row, col = edge_index

        if edge_attr is not None:
            edge_features = torch.cat([x[row], x[col], edge_attr], dim=-1)
        else:
            edge_features = torch.cat([x[row], x[col]], dim=-1)

        messages = self.edge_mlp(edge_features)

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, row, messages)

        out = torch.cat([x, aggregated], dim=-1)
        out = self.node_mlp(out)

        return out


class CrossGraphAttention(nn.Module):
    """Cross-graph attention layer for matching."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query1 = Linear(hidden_dim, hidden_dim)
        self.key1 = Linear(hidden_dim, hidden_dim)
        self.value1 = Linear(hidden_dim, hidden_dim)

        self.query2 = Linear(hidden_dim, hidden_dim)
        self.key2 = Linear(hidden_dim, hidden_dim)
        self.value2 = Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        x1_init: Tensor,
        x2_init: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Cross-graph attention forward pass.

        Args:
            x1: Graph1 embeddings
            x2: Graph2 embeddings
            x1_init: Initial Graph1 embeddings
            x2_init: Initial Graph2 embeddings

        Returns:
            Tuple of updated embeddings
        """
        q1 = self.query1(x1).view(-1, self.num_heads, self.head_dim)
        k1 = self.key1(x1).view(-1, self.num_heads, self.head_dim)
        v1 = self.value1(x1).view(-1, self.num_heads, self.head_dim)

        q2 = self.query2(x2).view(-1, self.num_heads, self.head_dim)
        k2 = self.key2(x2).view(-1, self.num_heads, self.head_dim)
        v2 = self.value2(x2).view(-1, self.num_heads, self.head_dim)

        attn1 = torch.matmul(q1, k2.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn1 = F.softmax(attn1, dim=-1)
        x1_update = torch.matmul(attn1, v2)
        x1_update = x1_update.view(-1, self.hidden_dim)

        attn2 = torch.matmul(q2, k1.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn2 = F.softmax(attn2, dim=-1)
        x2_update = torch.matmul(attn2, v1)
        x2_update = x2_update.view(-1, self.hidden_dim)

        x1 = x1 + x1_update + x1_init
        x2 = x2 + x2_update + x2_init

        return x1, x2


class GraphSimilarity(nn.Module):
    """
    Graph similarity learning network.

    Learns to compute similarity between graph pairs.

    Args:
        node_dim: Node feature dimension
        hidden_dim: Hidden dimension
        pooling: Pooling method (mean, max, sum)
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        pooling: str = "mean",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        self.encoder = nn.Sequential(
            Linear(node_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.similarity_mlp = nn.Sequential(
            Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x1: Tensor,
        edge_index1: Tensor,
        x2: Tensor,
        edge_index2: Tensor,
    ) -> Tensor:
        """
        Compute graph similarity.

        Args:
            x1: Graph1 features
            edge_index1: Graph1 edges
            x2: Graph2 features
            edge_index2: Graph2 edges

        Returns:
            Similarity score
        """
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x1 = F.relu(self.conv1(x1, edge_index1))
        x1 = F.relu(self.conv2(x1, edge_index1))

        x2 = F.relu(self.conv1(x2, edge_index2))
        x2 = F.relu(self.conv2(x2, edge_index2))

        if self.pooling == "mean":
            g1 = x1.mean(dim=0)
            g2 = x2.mean(dim=0)
        elif self.pooling == "max":
            g1 = x1.max(dim=0)[0]
            g2 = x2.max(dim=0)[0]
        else:
            g1 = x1.sum(dim=0)
            g2 = x2.sum(dim=0)

        combined = torch.cat([g1, g2, torch.abs(g1 - g2), g1 * g2], dim=-1)

        return self.similarity_mlp(combined)


class GCNConv(nn.Module):
    """Simple GCN convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index

        out = self.linear(x)

        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = out * norm.unsqueeze(-1)

        aggregated = torch.zeros_like(out)
        aggregated.index_add_(0, row, out[col])

        return aggregated


class SubgraphMatching(nn.Module):
    """
    Subgraph matching network.

    Finds correspondences between query and target graphs.

    Args:
        node_dim: Node feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of GNN layers
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.Sequential(
            Linear(node_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.match_layer = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_x: Tensor,
        query_edge: Tensor,
        target_x: Tensor,
        target_edge: Tensor,
    ) -> Tensor:
        """
        Match query subgraph to target graph.

        Args:
            query_x: Query graph features
            query_edge: Query graph edges
            target_x: Target graph features
            target_edge: Target graph edges

        Returns:
            Matching scores for each query node
        """
        query_x = self.encoder(query_x)
        target_x = self.encoder(target_x)

        for layer in self.layers:
            query_x = F.relu(layer(query_x, query_edge))
            target_x = F.relu(layer(target_x, target_edge))

        num_query = query_x.size(0)
        num_target = target_x.size(0)

        scores = torch.zeros(num_query, num_target, device=query_x.device)

        for i in range(num_query):
            for j in range(num_target):
                combined = torch.cat([query_x[i], target_x[j]], dim=-1)
                scores[i, j] = self.match_layer(combined.unsqueeze(0)).squeeze()

        return scores

    def find_best_match(self, scores: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Find best matching using Hungarian algorithm approximation.

        Args:
            scores: Matching scores [num_query, num_target]

        Returns:
            Tuple of (matches, max_scores)
        """
        matches = scores.argmax(dim=1)
        max_scores = scores.max(dim=1)[0]

        return matches, max_scores


class GraphAlignment(nn.Module):
    """
    Graph alignment network for learning node correspondences.

    Args:
        node_dim: Node feature dimension
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            Linear(node_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Align two graphs.

        Args:
            x1: Graph1 features [N1, node_dim]
            x2: Graph2 features [N2, node_dim]

        Returns:
            Tuple of aligned features
        """
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x1_exp = x1.unsqueeze(1)
        x2_exp = x2.unsqueeze(1)

        aligned_x1, _ = self.cross_attention(x1_exp, x2_exp, x2_exp)
        aligned_x2, _ = self.cross_attention(x2_exp, x1_exp, x1_exp)

        return aligned_x1.squeeze(1), aligned_x2.squeeze(1)

    def compute_alignment_score(
        self,
        aligned_x1: Tensor,
        aligned_x2: Tensor,
    ) -> Tensor:
        """Compute alignment score between aligned graphs."""
        return F.cosine_similarity(aligned_x1, aligned_x2, dim=-1).mean()


class GraphEditDistance(nn.Module):
    """
    Graph edit distance learning network.

    Args:
        node_dim: Node feature dimension
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            Linear(node_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.distance_mlp = nn.Sequential(
            Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x1: Tensor,
        edge_index1: Tensor,
        x2: Tensor,
        edge_index2: Tensor,
    ) -> Tensor:
        """
        Compute learnable graph edit distance.

        Args:
            x1: Graph1 node features
            edge_index1: Graph1 edges
            x2: Graph2 node features
            edge_index2: Graph2 edges

        Returns:
            Edit distance estimate
        """
        x1 = self.node_encoder(x1)
        x2 = self.node_encoder(x2)

        if x1.size(0) == 0 or x2.size(0) == 0:
            return torch.tensor(
                float("inf"), device=x1.device if x1.size(0) > 0 else x2.device
            )

        g1 = x1.mean(dim=0)
        g2 = x2.mean(dim=0)

        node_diff = torch.cat([g1, g2, torch.abs(g1 - g2), g1 * g2], dim=-1)

        dist = self.distance_mlp(node_diff.unsqueeze(0))

        return dist


class GraphKernel(nn.Module):
    """
    Learnable graph kernel for graph similarity.

    Args:
        node_dim: Node feature dimension
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            Linear(node_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.pool = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Compute graph kernel representation."""
        x = self.encoder(x)

        row, col = edge_index

        kernel_matrix = torch.matmul(x, x.t())

        kernel_vector = kernel_matrix.sum(dim=1)

        pooled = self.pool(kernel_vector.unsqueeze(0))

        return pooled


class WeisfeilerLehman(nn.Module):
    """
    Weisfeiler-Lehman graph isomorphism test.

    Args:
        num_iterations: Number of WL iterations
    """

    def __init__(
        self,
        num_iterations: int = 3,
    ):
        super().__init__()
        self.num_iterations = num_iterations

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Perform WL algorithm.

        Args:
            x: Node features
            edge_index: Edge connectivity

        Returns:
            Tuple of (final_labels, history)
        """
        labels = x.argmax(dim=-1) if x.dim() > 1 else x.long()

        row, col = edge_index
        history = [labels.clone()]

        for _ in range(self.num_iterations):
            neighborhood = torch.zeros_like(labels)

            for i, (r, c) in enumerate(zip(row.tolist(), col.tolist())):
                neighborhood[r] = neighborhood[r] * 31 + labels[c].item()

            labels = neighborhood
            history.append(labels.clone())

        return labels, history

    def compute_isomorphism_score(
        self,
        labels1: Tensor,
        labels2: Tensor,
    ) -> float:
        """Compute isomorphism score between two label sets."""
        hist1 = torch.bincount(labels1)
        hist2 = torch.bincount(labels2)

        max_len = max(hist1.size(0), hist2.size(0))
        hist1 = F.pad(hist1, (0, max_len - hist1.size(0)))
        hist2 = F.pad(hist2, (0, max_len - hist2.size(0)))

        intersection = (hist1 * hist2).sum().float()
        union = (hist1 + hist2).sum().float()

        return (intersection / (union + 1e-8)).item()

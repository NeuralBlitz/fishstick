"""
Graph Neural Operators.

Neural operator implementations for learning on graph-structured data.
Includes message-passing operators, spectral graph convolutions, and
graph Fourier transforms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch import Tensor


class MessagePassingOperator(nn.Module):
    """Message-passing neural operator on graphs."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        message_steps: int = 3,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps

        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Forward pass with message passing.

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
        """
        num_nodes = x.size(0)

        for _ in range(self.message_steps):
            src, dst = edge_index

            messages = torch.cat([x[src], x[dst]], dim=-1)
            messages = self.message_net(messages)

            aggregated = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
            aggregated.index_add_(0, dst, messages)

            combined = torch.cat([x, aggregated], dim=-1)
            x = self.update_net(combined)
            x = F.relu(x)

        return x


class SpectralGraphConv(nn.Module):
    """Spectral graph convolution using graph Fourier transform."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_eigenvectors: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_eigenvectors = num_eigenvectors

        self.weights = nn.Parameter(
            torch.Tensor(in_channels, out_channels, num_eigenvectors)
        )
        nn.init.xavier_uniform_(self.weights)

    def forward(
        self,
        x: Tensor,
        eigenvectors: Tensor,
        eigenvalues: Tensor,
    ) -> Tensor:
        """Forward pass in spectral domain.

        Args:
            x: Node features [num_nodes, in_channels]
            eigenvectors: Graph eigenvectors [num_nodes, k]
            eigenvalues: Graph eigenvalues [k]
        """
        x_ft = torch.matmul(eigenvectors.T, x)

        x_transformed = torch.einsum("ni,ijk->njk", x_ft, self.weights)

        x_out = torch.matmul(eigenvectors, x_transformed)

        return x_out


class GraphPoolingOperator(nn.Module):
    """Graph pooling for downsampling graphs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio

        self.score_net = nn.Sequential(
            nn.Linear(in_channels, 1),
        )

    def forward(
        self,
        x: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Pool nodes based on learnable scores.

        Returns:
            pooled_x: Pooled features
            cluster_index: Cluster assignment
            batch_mask: New batch tensor
        """
        scores = self.score_net(x).squeeze(-1)

        num_nodes = x.size(0)
        num_pooled = max(1, int(num_nodes * self.ratio))

        perm = torch.argsort(scores, descending=True)[:num_pooled]

        pooled_x = x[perm]
        cluster_index = perm

        return (
            pooled_x,
            cluster_index,
            torch.zeros(num_pooled, dtype=torch.long, device=x.device),
        )


class GraphUnpoolingOperator(nn.Module):
    """Graph unpooling for upsampling graphs."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: Tensor,
        cluster_index: Tensor,
    ) -> Tensor:
        """Unpool to original graph size.

        Args:
            x: Pooled features
            cluster_index: Original indices

        Returns:
            Unpooled features
        """
        num_nodes = cluster_index.max().item() + 1
        out = torch.zeros(num_nodes, x.size(-1), device=x.device)
        out[cluster_index] = x
        return out


class GraphAttentionOperator(nn.Module):
    """Graph attention operator for weighted message passing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.query = nn.Linear(in_channels, out_channels * heads)
        self.key = nn.Linear(in_channels, out_channels * heads)
        self.value = nn.Linear(in_channels, out_channels * heads)

        self.out_proj = nn.Linear(out_channels * heads, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        src, dst = edge_index

        Q = self.query(x).view(-1, self.heads, self.out_channels)
        K = self.key(x).view(-1, self.heads, self.out_channels)
        V = self.value(x).view(-1, self.heads, self.out_channels)

        src_idx = src.repeat(self.heads)
        dst_idx = dst.repeat(self.heads)

        attn_scores = (Q[src_idx] * K[dst_idx]).sum(dim=-1).view(-1, self.heads)
        attn_weights = F.softmax(attn_scores, dim=0)
        attn_weights = self.dropout(attn_weights)

        out = torch.zeros_like(V)
        out.index_add_(
            0,
            dst.repeat(self.heads),
            V[src.repeat(self.heads)] * attn_weights.unsqueeze(-1),
        )

        out = out.view(-1, self.heads * self.out_channels)
        return self.out_proj(out)


class PointCloudOperator(nn.Module):
    """Neural operator for point cloud data."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 3,
        radius: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.radius = radius

        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        points: Tensor,
        features: Tensor,
    ) -> Tensor:
        """Process point cloud.

        Args:
            points: Point coordinates [N, 3]
            features: Point features [N, in_channels]
        """
        combined = torch.cat([points, features], dim=-1)
        return self.mlp(combined)


class MeshOperator(nn.Module):
    """Neural operator for mesh-structured data."""

    def __init__(
        self,
        vertex_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.vertex_dim = vertex_dim
        self.hidden_dim = hidden_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(vertex_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.vertex_mlp = nn.Sequential(
            nn.Linear(vertex_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vertex_dim),
        )

    def forward(
        self,
        vertices: Tensor,
        edges: Tensor,
    ) -> Tensor:
        """Process mesh.

        Args:
            vertices: Vertex features [V, vertex_dim]
            edges: Edge indices [2, E]
        """
        src, dst = edges

        edge_features = torch.cat([vertices[src], vertices[dst]], dim=-1)
        edge_features = self.edge_mlp(edge_features)

        aggregated = torch.zeros_like(vertices)
        aggregated.index_add_(0, dst, edge_features)

        combined = torch.cat([vertices, aggregated], dim=-1)
        return self.vertex_mlp(combined)


class GraphNeuralOperatorBlock(nn.Module):
    """Graph neural operator building block."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.channels = channels

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index

        x_norm = self.norm1(x)

        adj = torch.zeros(x.size(0), x.size(0), device=x.device)
        adj[src, dst] = 1.0

        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=(1 - adj).bool())
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))

        return x


__all__ = [
    "MessagePassingOperator",
    "SpectralGraphConv",
    "GraphPoolingOperator",
    "GraphUnpoolingOperator",
    "GraphAttentionOperator",
    "PointCloudOperator",
    "MeshOperator",
    "GraphNeuralOperatorBlock",
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = query.size(0)
        q = (
            self.q_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.out_proj(out)
        return out


class GraphTransformerLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        use_edge_features: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_edge_features = use_edge_features and edge_dim is not None
        self.edge_dim = edge_dim

        assert out_channels % num_heads == 0

        self.node_projection = nn.Linear(in_channels, out_channels)
        self.edge_projection = (
            nn.Linear(edge_dim, num_heads) if self.use_edge_features else None
        )

        self.attention = MultiHeadAttention(out_channels, num_heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(out_channels) if layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(out_channels) if layer_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        pe: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, num_nodes, _ = x.shape

        x = self.node_projection(x)

        if pe is not None:
            x = x + pe

        edge_index_i, edge_index_j = edge_index
        attn_mask = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device)
        for b in range(batch_size):
            attn_mask[b, edge_index_i[b], edge_index_j[b]] = 1

        if self.use_edge_features and edge_attr is not None:
            edge_attr_proj = self.edge_projection(edge_attr)
            edge_attr_proj = edge_attr_proj.unsqueeze(1).expand(-1, self.num_heads, -1)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_mask = attn_mask + edge_attr_proj.unsqueeze(-1)

        x_flat = x.view(-1, self.out_channels)
        attn_out = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.view(batch_size, num_nodes, self.out_channels)
        x = self.norm1(x + attn_out)

        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class GraphTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_channels,
                    hidden_channels,
                    num_heads,
                    dropout,
                    edge_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        pe: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(
                x, edge_index, edge_attr, pe if self.use_positional_encoding else None
            )

        x = self.output_proj(x)
        return x


class LaplacianPositionalEncoding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        edge_index_i, edge_index_j = edge_index

        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index_i, edge_index_j] = 1
        adj = adj + adj.T
        adj = adj + torch.eye(num_nodes, device=adj.device)

        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        eigenvalues, eigenvectors = torch.linalg.eigh(norm)
        eigenvectors = eigenvectors[:, : self.embedding_dim]

        pe = eigenvectors.unsqueeze(0)
        return pe

    def get_embeddings(self, batch: dict, num_nodes: int) -> Tensor:
        edge_index = batch.get("edge_index")
        return self.forward(edge_index, num_nodes)


class RandomWalkPositionalEncoding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, edge_index: Tensor, num_nodes: int, num_steps: int = 3) -> Tensor:
        edge_index_i, edge_index_j = edge_index

        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index_i, edge_index_j] = 1

        deg = adj.sum(dim=1, keepdim=True)
        transition = adj / (deg + 1e-8)
        transition = transition.T

        rw_emb = []
        current = torch.eye(num_nodes, device=edge_index.device)
        for _ in range(num_steps):
            current = current @ transition
            rw_emb.append(current)

        rw_emb = torch.stack(rw_emb, dim=-1)
        rw_emb = rw_emb[:, :, : self.embedding_dim]

        return rw_emb.unsqueeze(0)

    def get_embeddings(self, batch: dict, num_nodes: int, num_steps: int = 3) -> Tensor:
        edge_index = batch.get("edge_index")
        return self.forward(edge_index, num_nodes, num_steps)

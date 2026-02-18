"""
Heterogeneous Graph Convolutions.

This module provides implementations for heterogeneous graph neural networks:
- Relational Graph Convolutional Networks (RGCN)
- Heterogeneous Graph Attention Networks (HAN)
- Metapath-based convolutions
- Learnable relation modules
"""

from typing import Optional, Tuple, List, Dict, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm, Dropout, ModuleList
import math


class RelationGraphConv(nn.Module):
    """
    Relational Graph Convolution Layer (RGCN).

    Applies different weight matrices for different relation types in
    heterogeneous graphs.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        num_relations: Number of relation types
        activation: Activation function
        basis_func: Type of basis function decomposition (basis or block)
        num_bases: Number of basis functions (for basis decomposition)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        activation: Optional[Callable] = F.relu,
        basis_func: str = "basis",
        num_bases: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.activation = activation
        self.basis_func = basis_func

        if basis_func == "basis":
            self.num_bases = min(num_bases, num_relations)
            self.weight = Parameter(
                torch.Tensor(self.num_bases, in_channels, out_channels)
            )
            self.comp = Parameter(torch.Tensor(num_relations, self.num_bases))
        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels, out_channels)
            )

        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        if self.basis_func == "basis":
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.comp)
        else:
            nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """
        Forward pass with relation-specific transformations.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_type: Edge type indices [num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        if self.basis_func == "basis":
            weight = torch.einsum("rb,rio->rio", self.comp, self.weight)
        else:
            weight = self.weight

        out = torch.zeros(num_nodes, self.out_channels, device=x.device)

        for rel_type in range(self.num_relations):
            mask = edge_type == rel_type
            if not mask.any():
                continue

            edges = edge_index[:, mask]

            row, col = edges
            messages = x[col] @ weight[rel_type]

            out.index_add_(0, row, messages)

        out = out + self.bias

        if self.activation is not None:
            out = self.activation(out)

        return out


class HeterogeneousGraphConv(nn.Module):
    """
    Heterogeneous Graph Convolution combining multiple relation convolutions.

    Combines multiple heterogeneous convolution operations with attention-based
    aggregation.

    Args:
        in_channels: Input feature dimension (can be dict for node types)
        out_channels: Output feature dimension
        num_relations: Number of relation types
        hidden_channels: Hidden dimension for intermediate layers
        num_layers: Number of convolution layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        num_relations: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.num_relations = num_relations

        if isinstance(in_channels, dict):
            self.node_type_proj = nn.ModuleDict(
                {
                    ntype: Linear(indim, hidden_channels)
                    for ntype, indim in in_channels.items()
                }
            )
            self.in_channels = hidden_channels
        else:
            self.in_channels = in_channels
            self.node_type_proj = None

        self.convs = ModuleList(
            [
                RelationGraphConv(
                    in_channels=hidden_channels if i > 0 else self.in_channels,
                    out_channels=hidden_channels,
                    num_relations=num_relations,
                )
                for i in range(num_layers - 1)
            ]
        )

        self.convs.append(
            RelationGraphConv(
                in_channels=hidden_channels,
                out_channels=out_channels,
                num_relations=num_relations,
            )
        )

        self.dropout = Dropout(dropout)
        self.ln = LayerNorm(out_channels)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index: Tensor,
        edge_type: Tensor,
        node_type: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through heterogeneous graph convolutions.

        Args:
            x_dict: Node features per node type
            edge_index: Edge connectivity [2, num_edges]
            edge_type: Edge type indices [num_edges]
            node_type: Optional mapping of node types

        Returns:
            Updated node features per node type
        """
        if self.node_type_proj is not None and x_dict:
            x_dict = {
                ntype: proj(x)
                for ntype, x in x_dict.items()
                for proj_name, proj in self.node_type_proj.items()
                if proj_name == ntype
            }

            all_x = []
            all_indices = []
            for ntype, x in x_dict.items():
                all_x.append(x)
                all_indices.extend([ntype] * x.size(0))

            if all_x:
                x = torch.cat(all_x, dim=0)
            else:
                x = torch.zeros(1, self.in_channels, device=edge_index.device)
        elif isinstance(x_dict, dict):
            x = torch.cat(list(x_dict.values()), dim=0)
        else:
            x = x_dict

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index, edge_type)
        x = self.ln(x)

        return x


class HANLayer(nn.Module):
    """
    Heterogeneous Graph Attention Network (HAN) layer.

    Uses node-level and semantic-level attention for heterogeneous graphs.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        self.query = Linear(in_channels, out_channels)
        self.key = Linear(in_channels, out_channels)
        self.value = Linear(in_channels, out_channels)

        self.node_att = Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.semantic_att = Parameter(torch.Tensor(num_heads, self.head_dim))

        self.out_proj = Linear(out_channels, out_channels)
        self.dropout = Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.node_att)
        nn.init.xavier_uniform_(self.semantic_att)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        num_relations: int,
    ) -> Tensor:
        """
        Forward pass with node-level and semantic-level attention.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_type: Edge type indices [num_edges]
            num_relations: Number of relation types

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        H = self.num_heads
        D = self.head_dim

        q = self.query(x).view(-1, H, D)
        k = self.key(x).view(-1, H, D)
        v = self.value(x).view(-1, H, D)

        row, col = edge_index

        out_per_relation = []

        for rel_type in range(num_relations):
            mask = edge_type == rel_type
            if not mask.any():
                continue

            rel_row = row[mask]
            rel_col = col[mask]

            q_i = q[rel_row]
            k_j = k[rel_col]
            v_j = v[rel_col]

            attn_scores = (q_i * k_j).sum(-1) / math.sqrt(D)
            attn_scores = F.softmax(attn_scores, dim=0)
            attn_scores = self.dropout(attn_scores)

            messages = v_j * attn_scores.unsqueeze(-1)

            rel_out = torch.zeros(num(x), H, D, device=x.device)
            rel_out.index_add_(0, rel_row, messages)

            out_per_relation.append(rel_out)

        if not out_per_relation:
            return torch.zeros(x.size(0), self.out_channels, device=x.device)

        out = torch.stack(out_per_relation, dim=0)

        alpha = torch.softmax(self.semantic_att, dim=0)
        out = (out * alpha.view(1, -1, 1)).sum(dim=0)

        out = out.view(-1, self.out_channels)
        out = self.out_proj(out)

        return out


class MetapathConv(nn.Module):
    """
    Metapath-based convolution for heterogeneous graphs.

    Performs convolutions along specified metapaths (sequences of relations).

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        metapaths: List of metapath definitions (list of relation types)
        aggregation: Aggregation function (mean, sum, max)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        metapaths: List[List[int]],
        aggregation: str = "mean",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.metapaths = metapaths
        self.aggregation = aggregation

        self.projections = nn.ModuleList(
            [nn.Linear(in_channels, out_channels) for _ in metapaths]
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """
        Forward pass along metapaths.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_type: Edge type indices [num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.out_channels, device=x.device)

        for metapath, proj in zip(self.metapaths, self.projections):
            metapath_out = self._propagate_along_metapath(
                x, edge_index, edge_type, metapath
            )
            metapath_out = proj(metapath_out)
            out = out + metapath_out

        out = out / len(self.metapaths)
        return out

    def _propagate_along_metapath(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        metapath: List[int],
    ) -> Tensor:
        """Propagate features along a metapath."""
        current = x

        for rel_type in metapath:
            mask = edge_type == rel_type
            if not mask.any():
                return torch.zeros_like(x)

            edges = edge_index[:, mask]
            row, col = edges

            next_state = torch.zeros_like(current)
            next_state.index_add_(0, row, current[col])

            if self.aggregation == "mean":
                counts = torch.zeros(current.size(0), device=x.device)
                counts.index_add_(0, row, torch.ones(row.size(0), device=x.device))
                counts = counts.clamp(min=1)
                next_state = next_state / counts.unsqueeze(-1)

            current = next_state

        return current


class RelationLearner(nn.Module):
    """
    Learnable relation module for dynamic graphs.

    Learns to represent relations between nodes based on their features.

    Args:
        node_dim: Node feature dimension
        hidden_dim: Hidden dimension for relation learning
        num_relations: Number of learnable relation types
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        num_relations: int = 10,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations

        self.node_encoder = nn.Sequential(
            Linear(node_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
        )

        self.relation_mlp = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, num_relations),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Learn relation types for edges.

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            Tuple of (learned edge types, relation probabilities)
        """
        x_encoded = self.node_encoder(x)

        row, col = edge_index
        edge_features = torch.cat([x_encoded[row], x_encoded[col]], dim=-1)

        relation_logits = self.relation_mlp(edge_features)
        relation_probs = F.softmax(relation_logits, dim=-1)

        edge_type = relation_logits.argmax(dim=-1)

        return edge_type, relation_probs


class FastRGCNConv(nn.Module):
    """
    Optimized RGCN with batched operations for efficiency.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        num_relations: Number of relation types
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weight = nn.Parameter(
            torch.Tensor(num_relations, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """
        Forward pass with batched matrix operations.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_type: Edge type indices [num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        x_expanded = x.unsqueeze(0).expand(self.num_relations, -1, -1)

        x_transformed = torch.bmm(x_expanded, self.weight)

        out = torch.zeros(num_nodes, self.out_channels, device=x.device)

        for rel_type in range(self.num_relations):
            mask = edge_type == rel_type
            if mask.any():
                edges = edge_index[:, mask]
                row = edges[0]
                col = edges[1]

                messages = x_transformed[rel_type, col]
                out.index_add_(0, row, messages)

        out = out + self.bias
        return out

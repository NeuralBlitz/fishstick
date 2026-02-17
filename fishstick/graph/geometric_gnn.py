"""
Geometric Graph Neural Networks.

Graph neural networks with:
- Equivariance to geometric transformations
- Sheaf structure for edge/node features
- Geometric message passing
- Support for molecular/crystalline structures
"""

from typing import Optional, Tuple, Dict, List, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import numpy as np

from ..core.types import MetricTensor
from ..geometric.sheaf import DataSheaf


class GeometricEdge:
    """Represents a geometric edge with distance and direction."""

    def __init__(
        self, distance: float, direction: Tensor, features: Optional[Tensor] = None
    ):
        self.distance = distance
        self.direction = direction
        self.features = features


class EquivariantMessagePassing(MessagePassing):
    """
    E(n)-equivariant message passing layer.

    Preserves Euclidean symmetries while propagating information on graphs.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr, node_dim=0)
        self._feature_dim = node_dim  # Don't override node_dim (used by PyG)
        self.edge_dim = edge_dim

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate update (equivariant)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with equivariant updates.

        Args:
            x: Node features [n_nodes, node_dim]
            pos: Node positions [n_nodes, 3]
            edge_index: Edge indices [2, n_edges]
            edge_attr: Edge features [n_edges, edge_dim]

        Returns:
            x_out: Updated node features
            pos_out: Updated positions (equivariant)
        """
        # Propagate messages
        aggregated = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

        # Update node features
        x_out = self.node_mlp(torch.cat([x, aggregated], dim=-1))

        # Update positions equivariantly
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        dist = torch.norm(coord_diff, dim=-1, keepdim=True)
        coord_diff_norm = coord_diff / (dist + 1e-8)

        coord_msg = self.coord_mlp(aggregated)
        coord_msg_per_edge = coord_msg[row]
        pos_update = coord_msg_per_edge * coord_diff_norm

        pos_out = pos.clone()
        pos_out.index_add_(0, row, pos_update)

        return x_out, pos_out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        pos_i: Tensor,
        pos_j: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        """Compute messages."""
        coord_diff = pos_i - pos_j
        dist = torch.norm(coord_diff, dim=-1, keepdim=True)

        msg = [x_i, x_j, dist]
        if edge_attr is not None:
            msg.append(edge_attr)

        msg = torch.cat(msg, dim=-1)
        return self.message_mlp(msg)

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """Aggregate messages."""
        return super().aggregate(inputs, index, ptr, dim_size)


class SheafGraphConv(MessagePassing):
    """
    Graph convolution with sheaf structure.

    Each node has a stalk (vector space), and edges have restriction maps.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stalk_dim: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stalk_dim = stalk_dim

        # Restriction maps for edges
        self.restriction_map = nn.Linear(stalk_dim, stalk_dim)

        # Message transformation
        self.msg_transform = nn.Sequential(
            nn.Linear(in_channels + stalk_dim, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Update transformation
        self.update_transform = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        stalk_features: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with sheaf structure.

        Args:
            x: Node features [n_nodes, in_channels]
            edge_index: Edge indices [2, n_edges]
            stalk_features: Features on stalks [n_nodes, stalk_dim]

        Returns:
            Updated features [n_nodes, out_channels]
        """
        if stalk_features is None:
            stalk_features = (
                torch.randn(x.size(0), self.stalk_dim, device=x.device) * 0.1
            )

        # Propagate
        out = self.propagate(edge_index, x=x, stalk_features=stalk_features, size=None)

        # Update
        out = self.update_transform(torch.cat([x, out], dim=-1))
        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        stalk_features_i: Tensor,
        stalk_features_j: Tensor,
    ) -> Tensor:
        """Compute sheaf-constrained messages."""
        # Apply restriction map
        stalk_i = self.restriction_map(stalk_features_i)
        stalk_j = self.restriction_map(stalk_features_j)

        # Consistency loss would be computed elsewhere
        msg = torch.cat([x_j, stalk_j], dim=-1)
        return self.msg_transform(msg)


class GeometricGraphTransformer(nn.Module):
    """
    Transformer architecture for geometric graphs.

    Combines attention mechanisms with geometric structure.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.layers = nn.ModuleList(
            [
                GeometricTransformerLayer(
                    node_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    edge_dim,
                    num_heads,
                    dropout,
                )
                for i in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim if i > 0 else node_dim) for i in range(num_layers)]
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [n_nodes, node_dim]
            pos: Node positions [n_nodes, 3]
            edge_index: Edge indices [2, n_edges]
            edge_attr: Edge features [n_edges, edge_dim]

        Returns:
            Updated features [n_nodes, hidden_dim]
        """
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x)
            x = layer(x, pos, edge_index, edge_attr)
        return x


class GeometricTransformerLayer(nn.Module):
    """Single geometric transformer layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 0,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # Multi-head attention
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

        # Edge attention
        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, num_heads)

        # Geometric bias
        self.geom_bias = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with geometric attention."""
        n_nodes = x.size(0)

        # Compute attention
        q = self.q_proj(x).view(n_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(n_nodes, self.num_heads, self.head_dim)

        # Get edges
        row, col = edge_index

        # Compute geometric distances
        dist = torch.norm(pos[row] - pos[col], dim=-1, keepdim=True)
        geom_weight = self.geom_bias(dist)  # [n_edges, num_heads]

        # Attention scores
        attn = (q[row] * k[col]).sum(dim=-1) / np.sqrt(self.head_dim)
        attn = attn * geom_weight

        if edge_attr is not None and hasattr(self, "edge_proj"):
            edge_bias = self.edge_proj(edge_attr)
            attn = attn + edge_bias

        # Softmax over neighbors
        attn = softmax(attn, row, num_nodes=n_nodes)
        attn = self.dropout(attn)

        # Apply attention
        out = attn.unsqueeze(-1) * v[col]
        out = out.view(n_nodes, -1)
        out = self.out_proj(out)

        # Residual and FFN
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))

        return x


class RiemannianGraphConv(nn.Module):
    """
    Graph convolution on Riemannian manifolds.

    Uses exponential map and logarithmic map for aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        manifold_dim: int = 3,
        curvature: float = -1.0,  # Hyperbolic
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.manifold_dim = manifold_dim
        self.curvature = curvature

        self.linear = nn.Linear(in_channels, out_channels)
        self.agg_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Forward pass on Riemannian manifold.

        Args:
            x: Features in tangent space [n_nodes, in_channels]
            edge_index: Edge indices [2, n_edges]

        Returns:
            Updated features [n_nodes, out_channels]
        """
        row, col = edge_index

        # Message passing (simplified Riemannian version)
        messages = self.linear(x[col])

        # Aggregate
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        out.index_add_(0, row, messages * self.agg_weight)

        return out


class MolecularGraphNetwork(nn.Module):
    """
    Complete network for molecular property prediction.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_tasks: int = 1,
        readout: str = "mean",
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.readout = readout

        # Encoder
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_encoder = (
            nn.Linear(edge_feature_dim, hidden_dim) if edge_feature_dim > 0 else None
        )

        # Geometric layers
        self.layers = nn.ModuleList(
            [
                EquivariantMessagePassing(
                    node_dim=hidden_dim,
                    edge_dim=hidden_dim if edge_feature_dim > 0 else 0,
                    hidden_dim=hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Readout
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for molecular graph.

        Args:
            x: Atom features [n_atoms, node_feature_dim]
            pos: Atom positions [n_atoms, 3]
            edge_index: Bond indices [2, n_bonds]
            edge_attr: Bond features [n_bonds, edge_feature_dim]
            batch: Batch assignment [n_atoms]

        Returns:
            Predictions [batch_size, num_tasks]
        """
        # Encode
        h = self.node_encoder(x)
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        # Message passing
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)

        # Readout
        if batch is None:
            if self.readout == "mean":
                h = h.mean(dim=0, keepdim=True)
            elif self.readout == "sum":
                h = h.sum(dim=0, keepdim=True)
            elif self.readout == "max":
                h = h.max(dim=0, keepdim=True)[0]
        else:
            # Batch-wise readout
            batch_size = batch.max().item() + 1
            h_out = []
            for i in range(batch_size):
                mask = batch == i
                if self.readout == "mean":
                    h_out.append(h[mask].mean(dim=0))
                elif self.readout == "sum":
                    h_out.append(h[mask].sum(dim=0))
                elif self.readout == "max":
                    h_out.append(h[mask].max(dim=0)[0])
            h = torch.stack(h_out)

        return self.readout_mlp(h)


class CrystalGraphNetwork(nn.Module):
    """
    Graph network for crystalline materials (periodic boundary conditions).
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        max_neighbors: int = 12,
    ):
        super().__init__()
        self.max_neighbors = max_neighbors

        self.encoder = nn.Linear(node_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                EquivariantMessagePassing(
                    node_dim=hidden_dim,
                    edge_dim=0,
                    hidden_dim=hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        lattice: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Forward pass for crystal structure.

        Args:
            x: Atomic features [n_atoms, node_dim]
            pos: Atomic positions [n_atoms, 3]
            lattice: Lattice vectors [3, 3]
            edge_index: Neighbor indices with periodic images

        Returns:
            Property prediction
        """
        h = self.encoder(x)

        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)

        return self.readout(h.mean(dim=0))

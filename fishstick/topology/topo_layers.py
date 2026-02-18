"""
Topological Neural Network Layers.

Provides differentiable topological layers for geometric deep learning,
including persistence pooling, topological attention, and message passing.
"""

from typing import Optional, List, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class PersistentHomologyLayer(nn.Module):
    """
    Persistent Homology Layer.

    Differentiable layer that computes persistent homology
    and returns topological features as learnable representations.
    """

    def __init__(
        self,
        max_dimension: int = 2,
        min_persistence: float = 0.0,
        use_cohomology: bool = False,
    ):
        """
        Initialize persistent homology layer.

        Args:
            max_dimension: Maximum homology dimension
            min_persistence: Minimum persistence threshold
            use_cohomology: Use cohomology instead of homology
        """
        super().__init__()
        self.max_dimension = max_dimension
        self.min_persistence = min_persistence
        self.use_cohomology = use_cohomology

    def forward(
        self,
        x: Tensor,
        edge_index: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        Compute topological features.

        Args:
            x: Node features [n_nodes, feature_dim]
            edge_index: Edge indices [2, n_edges]

        Returns:
            List of persistence diagrams per dimension
        """
        if edge_index is not None:
            return self._compute_from_graph(x, edge_index)
        else:
            return self._compute_from_points(x)

    def _compute_from_points(
        self,
        points: Tensor,
    ) -> List[Tensor]:
        """Compute persistence from point cloud."""
        from .vietoris_rips import VietorisRipsComplex
        from .persistence import PersistentHomology

        vr_complex = VietorisRipsComplex(max_dimension=self.max_dimension)
        simplices, filtrations = vr_complex.build_from_points(points)

        ph = PersistentHomology(max_dimension=self.max_dimension)
        from .simplicial import BoundaryOperator

        boundary_op = BoundaryOperator(simplices)
        boundary_matrices = boundary_op.get_matrices()

        diagrams = ph.compute(filtrations, boundary_matrices)

        diagram_tensors = []
        for dim, diagram in enumerate(diagrams):
            if dim <= self.max_dimension:
                diagram_tensors.append(diagram.to_tensor())

        return diagram_tensors

    def _compute_from_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> List[Tensor]:
        """Compute persistence from graph structure."""
        edge_weight = torch.ones(edge_index.shape[1], device=x.device)

        return self._compute_from_points(x)


class PersistencePooling(nn.Module):
    """
    Persistence Pooling Layer.

    Pools node features using topological structure,
    weighting nodes by their persistence importance.
    """

    def __init__(
        self,
        pooling_type: str = "attention",
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize persistence pooling.

        Args:
            pooling_type: Type of pooling ('attention', 'max', 'mean')
            hidden_dim: Hidden dimension for attention
        """
        super().__init__()
        self.pooling_type = pooling_type
        self.hidden_dim = hidden_dim

        if pooling_type == "attention" and hidden_dim is not None:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        x: Tensor,
        diagram: Tensor,
    ) -> Tensor:
        """
        Pool features using persistence weighting.

        Args:
            x: Node features [n_nodes, feature_dim]
            diagram: Persistence diagram [n_pairs, 2] or [n_pairs, 3]

        Returns:
            Pooled features [feature_dim]
        """
        if len(diagram) == 0:
            return x.mean(dim=0)

        persistences = (diagram[:, 1] - diagram[:, 0]).clamp(min=0)

        weights = F.softmax(persistences, dim=0)

        if self.pooling_type == "attention" and self.hidden_dim is not None:
            attention_scores = self.attention(x[:, : self.hidden_dim])
            attention_weights = F.softmax(attention_scores.squeeze(-1), dim=0)
            weights = weights * attention_weights

        if len(weights) < x.shape[0]:
            weights_extended = F.pad(weights, (0, x.shape[0] - len(weights)))
        else:
            weights_extended = weights[: x.shape[0]]

        pooled = torch.sum(x * weights_extended.unsqueeze(-1), dim=0)

        return pooled


class TopologicalAttention(nn.Module):
    """
    Topological Attention Layer.

    Attention mechanism that uses topological structure
    to weight feature aggregation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize topological attention.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.topological_projection = nn.Linear(1, num_heads)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        diagram: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute topological attention.

        Args:
            x: Node features [n_nodes, hidden_dim]
            edge_index: Edge indices [2, n_edges]
            diagram: Optional persistence diagram

        Returns:
            Updated features [n_nodes, hidden_dim]
        """
        n_nodes = x.shape[0]

        q = self.query(x).view(n_nodes, self.num_heads, self.head_dim)
        k = self.key(x).view(n_nodes, self.num_heads, self.head_dim)
        v = self.value(x).view(n_nodes, self.num_heads, self.head_dim)

        attn_scores = torch.einsum("ijk,ilk->ijl", q, k) / (self.head_dim**0.5)

        if diagram is not None and len(diagram) > 0:
            topo_weights = self._compute_topo_weights(n_nodes, diagram)
            attn_scores = attn_scores + topo_weights.unsqueeze(-1)

        attn_weights = F.softmax(attn_scores, dim=2)
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum("ijl,ilk->ijk", attn_weights, v)
        out = out.contiguous().view(n_nodes, self.hidden_dim)

        return self.out_proj(out)

    def _compute_topo_weights(
        self,
        n_nodes: int,
        diagram: Tensor,
    ) -> Tensor:
        """Compute topological weights from persistence diagram."""
        if len(diagram) == 0:
            return torch.zeros(self.num_heads, n_nodes, n_nodes, device=diagram.device)

        persistences = (diagram[:, 1] - diagram[:, 0]).clamp(min=0)

        topo_weights = persistences.mean() * torch.ones(
            self.num_heads, n_nodes, n_nodes, device=diagram.device
        )

        return topo_weights


class TopologicalMessagePassing(MessagePassing):
    """
    Topological Message Passing Layer.

    Message passing that incorporates topological
    persistence information into edge messages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "add",
    ):
        """
        Initialize topological message passing.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            aggr: Aggregation method
        """
        super().__init__(aggr=aggr)

        self.lin = nn.Linear(in_channels, out_channels)
        self.message_net = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_persistence: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with topological messages.

        Args:
            x: Node features [n_nodes, in_channels]
            edge_index: Edge indices [2, n_edges]
            edge_persistence: Edge persistence weights [n_edges]

        Returns:
            Updated node features [n_nodes, out_channels]
        """
        x_transformed = self.lin(x)

        return self.propagate(
            edge_index,
            x=x_transformed,
            edge_persistence=edge_persistence,
        )

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_persistence: Optional[Tensor],
    ) -> Tensor:
        """Compute message with topological weighting."""
        msg = self.message_net(x_j)

        if edge_persistence is not None:
            weight = edge_persistence.unsqueeze(-1)
            msg = msg * weight

        return msg


class PersistentGraphConv(nn.Module):
    """
    Persistent Graph Convolution.

    Graph convolution layer that uses persistence
    to weight message passing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        """
        Initialize persistent graph convolution.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            bias: Use bias
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [n_nodes, in_channels]
            edge_index: Edge indices [2, n_edges]
            edge_weight: Edge weights

        Returns:
            Output features [n_nodes, out_channels]
        """
        x = torch.matmul(x, self.weight)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=x.device)

        out = self._propagate(edge_index, x, edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def _propagate(
        self,
        edge_index: Tensor,
        x: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        """Propagate features along edges."""
        row, col = edge_index

        out = torch.zeros_like(x)

        deg = torch.zeros(x.shape[0], device=x.device)
        deg.scatter_add_(0, row, edge_weight)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = out.scatter_add(
            0, row.unsqueeze(-1).expand_as(x), x[col] * norm.unsqueeze(-1)
        )

        return out


class TopologicalPool(nn.Module):
    """
    Topological Pooling Layer.

    Pools graph nodes based on topological importance
    using persistent homology scores.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
    ):
        """
        Initialize topological pooling.

        Args:
            in_channels: Input channels
            ratio: Pooling ratio
        """
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.score_net = nn.Sequential(
            nn.Linear(in_channels, 1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        diagram: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Pool nodes by topological scores.

        Args:
            x: Node features [n_nodes, in_channels]
            edge_index: Edge indices [2, n_edges]
            diagram: Persistence diagram

        Returns:
            Tuple of (pooled_x, pooled_edge_index, edge_mask)
        """
        scores = self.score_net(x).squeeze(-1)

        if diagram is not None and len(diagram) > 0:
            persistences = (diagram[:, 1] - diagram[:, 0]).clamp(min=0)
            topo_scores = F.softmax(persistences, dim=0)

            if len(topo_scores) >= x.shape[0]:
                scores = scores + topo_scores[: x.shape[0]]
            else:
                scores = scores + topo_scores.mean()

        perm = torch.argsort(scores, descending=True)

        n_keep = max(1, int(x.shape[0] * self.ratio))
        perm = perm[:n_keep]

        new_index_map = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        new_index_map[perm] = torch.arange(n_keep, device=x.device)

        pooled_x = x[perm]

        row, col = edge_index
        mask = (new_index_map[row] != -1) & (new_index_map[col] != -1)
        edge_index_pooled = torch.stack(
            [
                new_index_map[row[mask]],
                new_index_map[col[mask]],
            ]
        )

        return pooled_x, edge_index_pooled, perm


class FiltrationAwareEmbedding(nn.Module):
    """
    Filtration-Aware Feature Embedding.

    Embeds features while respecting the filtration
    structure from persistent homology.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_filtrations: int = 10,
    ):
        """
        Initialize filtration-aware embedding.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            n_filtrations: Number of filtration levels
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filtrations = n_filtrations

        self.filtration_levels = nn.Parameter(torch.linspace(0, 1, n_filtrations))

        self.embeddings = nn.ModuleList(
            [nn.Linear(in_channels, out_channels) for _ in range(n_filtrations)]
        )

    def forward(
        self,
        x: Tensor,
        diagram: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute filtration-aware embedding.

        Args:
            x: Node features [n_nodes, in_channels]
            diagram: Persistence diagram

        Returns:
            Embedded features [n_nodes, out_channels]
        """
        if diagram is None or len(diagram) == 0:
            return self.embeddings[0](x)

        current_filtration = self._estimate_filtration(diagram)

        embeddings = []
        for i, emb in enumerate(self.embeddings):
            filt_val = self.filtration_levels[i].item()
            weight = torch.exp(-((current_filtration - filt_val) ** 2))
            embeddings.append(emb(x) * weight)

        return torch.stack(embeddings).mean(dim=0)

    def _estimate_filtration(
        self,
        diagram: Tensor,
    ) -> float:
        """Estimate current filtration level from diagram."""
        if len(diagram) == 0:
            return 0.5

        persistences = (diagram[:, 1] - diagram[:, 0]).clamp(min=0)

        return torch.mean(persistences).item()


class TopologicalAggregation(nn.Module):
    """
    Topological Aggregation Layer.

    Aggregates features from multiple persistence diagrams
    using learnable topological pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_diagrams: int = 3,
    ):
        """
        Initialize topological aggregation.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_diagrams: Number of diagrams to aggregate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_diagrams = num_diagrams

        self.weight_net = nn.Sequential(
            nn.Linear(num_diagrams, num_diagrams),
            nn.ReLU(),
            nn.Linear(num_diagrams, num_diagrams),
            nn.Softmax(dim=1),
        )

        self.out_proj = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        features_list: List[Tensor],
        diagrams_list: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """
        Aggregate features from multiple diagrams.

        Args:
            features_list: List of feature tensors [n_nodes, in_channels]
            diagrams_list: Optional list of diagrams for weighting

        Returns:
            Aggregated features [n_nodes, out_channels]
        """
        if len(features_list) != self.num_diagrams:
            self.num_diagrams = len(features_list)

        stacked = torch.stack(features_list, dim=0)

        if diagrams_list is not None:
            weights = self._compute_topo_weights(diagrams_list)
            weights = weights.unsqueeze(-1).unsqueeze(-1)
        else:
            weights = (
                torch.ones(
                    self.num_diagrams,
                    features_list[0].shape[0],
                    features_list[0].shape[1],
                    device=stacked.device,
                )
                / self.num_diagrams
            )

        aggregated = (stacked * weights).sum(dim=0)

        return self.out_proj(aggregated)

    def _compute_topo_weights(
        self,
        diagrams_list: List[Tensor],
    ) -> Tensor:
        """Compute topological weights from diagrams."""
        weights = []

        for diagram in diagrams_list:
            if len(diagram) == 0:
                weights.append(torch.tensor(1.0))
            else:
                persistences = (diagram[:, 1] - diagram[:, 0]).clamp(min=0)
                weight = persistences.mean()
                weights.append(weight)

        weights_tensor = torch.stack(weights).unsqueeze(0)

        return F.softmax(weights_tensor, dim=1)

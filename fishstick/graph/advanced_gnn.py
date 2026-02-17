"""
Advanced Graph Neural Networks Module

Comprehensive implementation of GNN layers, pooling operations, utilities,
data loaders, and models for graph learning tasks.

This module provides:
- Message Passing Layers (GCN, GAT, GraphSAGE, GIN, Transformer, EdgeConv)
- Pooling Operations (global, TopK, SAGPool, DiffPool, Set2Set)
- Graph Utilities (batching, normalization, conversion)
- Data Loaders for graph data
- Complete GNN Models for classification, link prediction, and autoencoding
- Example applications for molecular, social, and recommendation tasks
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm, Dropout, ModuleList
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from collections import namedtuple


# =============================================================================
# Graph Data Structures
# =============================================================================


@dataclass
class GraphData:
    """
    Data structure for graph data.

    Attributes:
        x: Node features [num_nodes, num_features]
        edge_index: Graph connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, num_edge_features]
        y: Target labels (node or graph level)
        batch: Batch assignment for each node [num_nodes]
        pos: Node positions (for geometric graphs)
        num_nodes: Number of nodes in the graph
    """

    x: Optional[Tensor] = None
    edge_index: Optional[Tensor] = None
    edge_attr: Optional[Tensor] = None
    y: Optional[Tensor] = None
    batch: Optional[Tensor] = None
    pos: Optional[Tensor] = None
    num_nodes: Optional[int] = None

    def __post_init__(self):
        if self.num_nodes is None and self.x is not None:
            self.num_nodes = self.x.size(0)
        if self.batch is None and self.x is not None:
            self.batch = torch.zeros(self.x.size(0), dtype=torch.long)

    def to(self, device: Union[str, torch.device]) -> "GraphData":
        """Move data to device."""
        for key, value in self.__dict__.items():
            if isinstance(value, Tensor):
                setattr(self, key, value.to(device))
        return self

    def clone(self) -> "GraphData":
        """Clone the graph data."""
        return GraphData(
            x=self.x.clone() if self.x is not None else None,
            edge_index=self.edge_index.clone() if self.edge_index is not None else None,
            edge_attr=self.edge_attr.clone() if self.edge_attr is not None else None,
            y=self.y.clone() if self.y is not None else None,
            batch=self.batch.clone() if self.batch is not None else None,
            pos=self.pos.clone() if self.pos is not None else None,
            num_nodes=self.num_nodes,
        )


# =============================================================================
# Message Passing Layers
# =============================================================================


class MessagePassing(nn.Module):
    """
    Base class for message passing layers.

    Implements the general message passing framework:
    x_i^{(l)} = gamma^{(l)}(x_i^{(l-1)}, aggregate_{j in N(i)} phi^{(l)}(x_i^{(l-1)}, x_j^{(l-1)}, e_{j,i}))
    """

    def __init__(self, aggr: str = "add", flow: str = "source_to_target"):
        super().__init__()
        self.aggr = aggr
        self.flow = flow

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError

    def message(
        self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        """Compute messages from node j to node i."""
        raise NotImplementedError

    def aggregate(
        self,
        messages: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """Aggregate messages at each node."""
        if self.aggr == "add":
            return scatter_add(messages, index, dim=0, dim_size=dim_size)
        elif self.aggr == "mean":
            return scatter_mean(messages, index, dim=0, dim_size=dim_size)
        elif self.aggr == "max":
            return scatter_max(messages, index, dim=0, dim_size=dim_size)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggr}")

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        """Update node embeddings after aggregation."""
        return aggr_out


class GCNConv(MessagePassing):
    """
    Graph Convolutional Network (GCN) Layer.

    Implementation of "Semi-Supervised Classification with Graph Convolutional Networks"
    (Kipf & Welling, ICLR 2017).

    The layer performs:
    X' = D^{-1/2} A D^{-1/2} X W

    where A is the adjacency matrix with added self-loops,
    D is the degree matrix, X is the node features, and W is the weight matrix.

    Args:
        in_channels: Input feature dimensions
        out_channels: Output feature dimensions
        improved: Use improved GCN (adds self-loops with weight 2)
        cached: Cache normalized adjacency matrix
        add_self_loops: Automatically add self-loops
        normalize: Apply symmetric normalization
        bias: Use bias term
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._cached_edge_index = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        self.lin.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            edge_weight: Edge weights [E]

        Returns:
            Output features [N, out_channels]
        """
        x = self.lin(x)

        if self.normalize:
            if self.cached and self._cached_edge_index is not None:
                edge_index, edge_weight = self._cached_edge_index
            else:
                edge_index, edge_weight = gcn_norm(
                    edge_index,
                    edge_weight,
                    x.size(0),
                    self.improved,
                    self.add_self_loops,
                )
                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)

        # Message passing
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def propagate(
        self, edge_index: Tensor, x: Tensor, edge_weight: Optional[Tensor] = None
    ) -> Tensor:
        """Propagate messages through the graph."""
        row, col = edge_index

        # Compute messages
        out = x[col]
        if edge_weight is not None:
            out = out * edge_weight.view(-1, 1)

        # Aggregate
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))

        if self.bias is not None:
            out = out + self.bias

        return out


class GATConv(MessagePassing):
    """
    Graph Attention Network (GAT) Layer with Multi-Head Attention.

    Implementation of "Graph Attention Networks" (Veličković et al., ICLR 2018).

    The layer computes attention coefficients:
    e_{ij} = LeakyReLU(a^T [W x_i || W x_j])
    alpha_{ij} = softmax_j(e_{ij})
    x_i' = ||_{k=1}^K sigma(sum_{j in N(i)} alpha_{ij}^k W^k x_j)

    Args:
        in_channels: Input feature dimensions
        out_channels: Output feature dimensions per head
        heads: Number of attention heads
        concat: Concatenate heads (if False, average)
        negative_slope: LeakyReLU negative slope
        dropout: Dropout on attention coefficients
        add_self_loops: Add self-loops to graph
        bias: Use bias term
        fill_value: Fill value for self-loops
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        fill_value: float = 1.0,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features (optional)

        Returns:
            Output features [N, heads * out_channels] if concat else [N, out_channels]
        """
        N, H, C = x.size(0), self.heads, self.out_channels

        # Add self-loops
        if self.add_self_loops:
            edge_index, _ = add_self_loops_fn(edge_index, num_nodes=N)

        # Linear transformation
        x = self.lin(x).view(-1, H, C)

        # Compute attention coefficients
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)

        # Propagate
        out = self.propagate(edge_index, x=x, alpha=(alpha_src, alpha_dst), size=None)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def propagate(
        self, edge_index: Tensor, x: Tensor, alpha: Tuple[Tensor, Tensor], size=None
    ) -> Tensor:
        """Propagate with attention."""
        row, col = edge_index
        alpha_src, alpha_dst = alpha

        # Compute attention scores
        alpha = alpha_src[row] + alpha_dst[col]
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Softmax normalization
        alpha = softmax(alpha, row, num_nodes=x.size(0))

        # Apply dropout
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Message passing
        out = x[col] * alpha.unsqueeze(-1)
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))

        return out


class GraphSAGEConv(MessagePassing):
    """
    GraphSAGE Layer for Inductive Representation Learning.

    Implementation of "Inductive Representation Learning on Large Graphs"
    (Hamilton et al., NeurIPS 2017).

    Supports multiple aggregation methods:
    - mean: Mean aggregator
    - max: Max-pool aggregator
    - lstm: LSTM aggregator

    The layer computes:
    h_{N(i)} = aggregate({h_j for j in N(i)})
    h_i' = sigma(W * concat(h_i, h_{N(i)}))

    Args:
        in_channels: Input feature dimensions
        out_channels: Output feature dimensions
        aggr: Aggregation method ('mean', 'max', 'lstm')
        normalize: L2-normalize outputs
        root_weight: Use separate weight for root node
        bias: Use bias term
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        bias: bool = True,
    ):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if aggr == "lstm":
            self.lstm = nn.LSTM(in_channels, in_channels, batch_first=True)

        self.lin_neigh = Linear(in_channels, out_channels, bias=False)

        if root_weight:
            self.lin_root = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        self.lin_neigh.reset_parameters()
        if self.root_weight:
            self.lin_root.reset_parameters()
        if hasattr(self, "lstm"):
            self.lstm.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, size=None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            size: Bipartite graph sizes (optional)

        Returns:
            Output features [N, out_channels]
        """
        row, col = edge_index

        # Aggregate neighbor features
        if self.aggr == "lstm":
            # LSTM aggregation - process neighbors sequentially
            out = self._lstm_aggregate(x, edge_index)
        else:
            # Standard aggregation
            aggr_out = x[col]
            if self.aggr == "mean":
                aggr_out = scatter_mean(aggr_out, row, dim=0, dim_size=x.size(0))
            elif self.aggr == "max":
                aggr_out = scatter_max(aggr_out, row, dim=0, dim_size=x.size(0))[0]
            else:
                aggr_out = scatter_add(aggr_out, row, dim=0, dim_size=x.size(0))

            out = self.lin_neigh(aggr_out)

        # Add root node features
        if self.root_weight:
            out = out + self.lin_root(x)
        else:
            out = out + self.lin_neigh(x)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def _lstm_aggregate(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Aggregate using LSTM."""
        row, col = edge_index
        # Group neighbors by target node
        unique_rows, inverse = torch.unique(row, return_inverse=True)
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)

        for i, node_idx in enumerate(unique_rows):
            mask = inverse == i
            neighbors = x[col[mask]]
            if len(neighbors) > 0:
                # Add batch dimension and process
                neighbors = neighbors.unsqueeze(0)
                lstm_out, _ = self.lstm(neighbors)
                out[node_idx] = self.lin_neigh(lstm_out[0, -1])

        return out


class GINConv(MessagePassing):
    """
    Graph Isomorphism Network (GIN) Layer.

    Implementation of "How Powerful are Graph Neural Networks?"
    (Xu et al., ICLR 2019).

    The layer computes:
    x_i' = MLP((1 + eps) * x_i + sum_{j in N(i)} x_j)

    GIN is proven to be as powerful as the Weisfeiler-Lehman test for graph isomorphism.

    Args:
        nn: Neural network (typically MLP) for feature transformation
        eps: Initial epsilon value (learnable if train_eps=True)
        train_eps: Whether epsilon is learnable
    """

    def __init__(self, nn: nn.Module, eps: float = 0.0, train_eps: bool = False):
        super().__init__(aggr="add")
        self.nn = nn

        if train_eps:
            self.eps = Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x: Tensor, edge_index: Tensor, size=None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            size: Bipartite graph sizes (optional)

        Returns:
            Output features [N, out_channels]
        """
        row, col = edge_index

        # Aggregate neighbor features
        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))

        # Combine with center node: (1 + eps) * x + sum(neighbors)
        out = (1 + self.eps) * x + out

        # Apply neural network
        out = self.nn(out)

        return out


class TransformerConv(MessagePassing):
    """
    Graph Transformer Layer.

    Combines Transformer-style attention with graph structure.
    Based on "Masked Label Prediction: Unified Message Passing Model for
    Semi-Supervised Classification" (Shi et al., 2021).

    Args:
        in_channels: Input feature dimensions
        out_channels: Output feature dimensions
        heads: Number of attention heads
        concat: Concatenate heads
        beta: Apply additional self-attention layer
        dropout: Dropout probability
        edge_dim: Edge feature dimensions (if using edge attributes)
        bias: Use bias term
        root_weight: Use separate weight for root node
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.beta = beta
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.root_weight = root_weight

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        if concat:
            self.lin_skip = Linear(in_channels, heads * out_channels, bias=bias)
            if beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = None
        else:
            self.lin_skip = Linear(in_channels, out_channels, bias=bias)
            if beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = None

        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.lin_beta is not None:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]

        Returns:
            Output features
        """
        N, H, C = x.size(0), self.heads, self.out_channels

        # Compute query, key, value
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        # Propagate
        out = self.propagate(
            edge_index, query=query, key=key, value=value, edge_attr=edge_attr
        )

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        # Skip connection
        if self.root_weight:
            x_skip = self.lin_skip(x)
            if self.beta is not None:
                beta = torch.sigmoid(
                    self.lin_beta(torch.cat([out, x_skip, out - x_skip], dim=-1))
                )
                out = beta * x_skip + (1 - beta) * out
            else:
                out = out + x_skip

        return out

    def propagate(
        self,
        edge_index: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Propagate with transformer attention."""
        row, col = edge_index
        H, C = self.heads, self.out_channels

        # Compute attention scores
        query_i = query[row]
        key_j = key[col]

        if edge_attr is not None and self.lin_edge is not None:
            edge_emb = self.lin_edge(edge_attr).view(-1, H, C)
            key_j = key_j + edge_emb

        # Attention: (Q * K^T) / sqrt(d_k)
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(C)
        alpha = softmax(alpha, row, num_nodes=query.size(0))

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Aggregate
        out = value[col] * alpha.unsqueeze(-1)
        out = scatter_add(out, row, dim=0, dim_size=query.size(0))

        return out


class EdgeConv(MessagePassing):
    """
    Edge Convolution Layer for Dynamic Graph CNN.

    Implementation of "Dynamic Graph CNN for Learning on Point Clouds"
    (Wang et al., ACM TOG 2019).

    Computes edge features:
    e_{ij} = h(x_i, x_j - x_i)

    where h is typically an MLP.

    Args:
        nn: Neural network for edge feature computation
        aggr: Aggregation method
    """

    def __init__(self, nn: nn.Module, aggr: str = "max"):
        super().__init__(aggr=aggr)
        self.nn = nn

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels] or [N, num_dims] for point clouds
            edge_index: Graph connectivity [2, E] (typically k-NN graph)

        Returns:
            Output features [N, out_channels]
        """
        return self.propagate(edge_index, x=x)

    def propagate(self, edge_index: Tensor, x: Tensor) -> Tensor:
        """Propagate with edge features."""
        row, col = edge_index

        # Compute edge features: h(x_i, x_j - x_i)
        x_i = x[row]
        x_j = x[col]
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        edge_features = self.nn(edge_features)

        # Aggregate
        if self.aggr == "max":
            out = scatter_max(edge_features, row, dim=0, dim_size=x.size(0))[0]
        elif self.aggr == "mean":
            out = scatter_mean(edge_features, row, dim=0, dim_size=x.size(0))
        elif self.aggr == "add":
            out = scatter_add(edge_features, row, dim=0, dim_size=x.size(0))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggr}")

        return out


# =============================================================================
# Scatter Operations (Utility Functions)
# =============================================================================


def scatter_add(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None
) -> Tensor:
    """Scatter add operation."""
    if dim_size is None:
        dim_size = int(index.max()) + 1 if len(index) > 0 else 0

    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)

    # Create index for scatter
    dim_indices = [slice(None)] * src.ndim
    dim_indices[dim] = index

    out.index_put_(tuple(dim_indices), src, accumulate=True)
    return out


def scatter_mean(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None
) -> Tensor:
    """Scatter mean operation."""
    out = scatter_add(src, index, dim, dim_size)

    # Compute counts
    if dim_size is None:
        dim_size = int(index.max()) + 1 if len(index) > 0 else 0

    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.index_put_((index,), torch.ones_like(index, dtype=src.dtype), accumulate=True)
    count = count.clamp(min=1)

    # Expand count to match output shape
    shape = [1] * out.ndim
    shape[dim] = -1
    count = count.view(shape)

    return out / count


def scatter_max(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """Scatter max operation."""
    if dim_size is None:
        dim_size = int(index.max()) + 1 if len(index) > 0 else 0

    shape = list(src.shape)
    shape[dim] = dim_size

    # Initialize with very small values
    out = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
    argmax = torch.zeros(shape, dtype=torch.long, device=src.device)

    # Manual scatter max
    dim_indices = [slice(None)] * src.ndim
    dim_indices[dim] = index

    mask = out[tuple(dim_indices)] < src
    out[tuple(dim_indices)] = torch.where(mask, src, out[tuple(dim_indices)])

    return out, argmax


def softmax(src: Tensor, index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    """Compute softmax over neighbors."""
    if num_nodes is None:
        num_nodes = int(index.max()) + 1 if len(index) > 0 else 0

    # Numerically stable softmax
    src_max = torch.zeros(num_nodes, dtype=src.dtype, device=src.device)
    src_max.index_put_((index,), src, accumulate=False)
    src = src - src_max[index]

    out = torch.zeros(num_nodes, dtype=src.dtype, device=src.device)
    out.index_put_((index,), src.exp(), accumulate=True)

    return src.exp() / (out[index] + 1e-16)


# =============================================================================
# Graph Normalization
# =============================================================================


def gcn_norm(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Normalize adjacency matrix for GCN.

    Computes: D^{-1/2} A D^{-1/2}

    Args:
        edge_index: Graph connectivity
        edge_weight: Edge weights
        num_nodes: Number of nodes
        improved: Use improved GCN
        add_self_loops: Add self-loops
        dtype: Output dtype

    Returns:
        Normalized edge_index and edge_weight
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )

    if add_self_loops:
        edge_index, edge_weight = add_self_loops_fn(
            edge_index, edge_weight, num_nodes, improved
        )

    row, col = edge_index

    # Compute degrees
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

    # Normalize
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


def add_self_loops_fn(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    fill_value: float = 1.0,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Add self-loops to the graph.

    Args:
        edge_index: Graph connectivity
        edge_weight: Edge weights
        num_nodes: Number of nodes
        fill_value: Fill value for self-loops

    Returns:
        Edge index and weights with self-loops
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    # Create self-loops
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    # Concatenate
    edge_index = torch.cat([edge_index, loop_index], dim=1)

    if edge_weight is not None:
        loop_weight = torch.full(
            (num_nodes,), fill_value, dtype=edge_weight.dtype, device=edge_weight.device
        )
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    return edge_index, edge_weight


# =============================================================================
# Pooling Operations
# =============================================================================


def global_mean_pool(x: Tensor, batch: Tensor, size: Optional[int] = None) -> Tensor:
    """
    Global mean pooling over graphs.

    Args:
        x: Node features [N, F]
        batch: Batch assignment [N]
        size: Number of graphs in batch

    Returns:
        Pooled features [batch_size, F]
    """
    if size is None:
        size = int(batch.max()) + 1

    out = torch.zeros(size, x.size(1), dtype=x.dtype, device=x.device)
    out.index_put_((batch,), x, accumulate=True)

    # Compute counts
    count = torch.zeros(size, dtype=x.dtype, device=x.device)
    count.index_put_((batch,), torch.ones_like(batch, dtype=x.dtype), accumulate=True)
    count = count.clamp(min=1).view(-1, 1)

    return out / count


def global_max_pool(x: Tensor, batch: Tensor, size: Optional[int] = None) -> Tensor:
    """
    Global max pooling over graphs.

    Args:
        x: Node features [N, F]
        batch: Batch assignment [N]
        size: Number of graphs in batch

    Returns:
        Pooled features [batch_size, F]
    """
    if size is None:
        size = int(batch.max()) + 1

    out = torch.full((size, x.size(1)), float("-inf"), dtype=x.dtype, device=x.device)
    out.index_put_((batch,), x, accumulate=False)

    return out


def global_add_pool(x: Tensor, batch: Tensor, size: Optional[int] = None) -> Tensor:
    """
    Global add/sum pooling over graphs.

    Args:
        x: Node features [N, F]
        batch: Batch assignment [N]
        size: Number of graphs in batch

    Returns:
        Pooled features [batch_size, F]
    """
    if size is None:
        size = int(batch.max()) + 1

    out = torch.zeros(size, x.size(1), dtype=x.dtype, device=x.device)
    out.index_put_((batch,), x, accumulate=True)

    return out


class TopKPool(nn.Module):
    """
    Top-K Pooling Layer.

    Implementation of "Graph U-Nets" (Gao & Ji, ICML 2019).

    Selects top-k important nodes based on a learnable projection score.

    Args:
        in_channels: Input feature dimensions
        ratio: Pooling ratio (k = ratio * N) or absolute k if < 1
        multiplier: Score multiplier
    """

    def __init__(self, in_channels: int, ratio: float = 0.5, multiplier: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.multiplier = multiplier

        self.proj = Linear(in_channels, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.proj.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [N, F]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, F_e]
            batch: Batch assignment [N]

        Returns:
            x: Pooled node features
            edge_index: Pooled edge index
            edge_attr: Pooled edge attributes
            batch: Pooled batch assignment
            perm: Indices of selected nodes
            score: Node scores
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Compute scores
        score = self.proj(x).squeeze(-1)
        score = self.multiplier * torch.tanh(score)

        # Determine k for each graph in batch
        num_graphs = int(batch.max()) + 1
        perm_list = []
        batch_out_list = []

        for i in range(num_graphs):
            mask = batch == i
            num_nodes = mask.sum().item()

            if self.ratio < 1:
                k = max(1, int(self.ratio * num_nodes))
            else:
                k = min(int(self.ratio), num_nodes)

            score_i = score[mask]
            _, idx = torch.topk(score_i, k)
            perm_i = torch.nonzero(mask, as_tuple=False).squeeze(-1)[idx]

            perm_list.append(perm_i)
            batch_out_list.append(
                torch.full((k,), i, dtype=torch.long, device=x.device)
            )

        perm = torch.cat(perm_list, dim=0)
        batch_out = torch.cat(batch_out_list, dim=0)

        # Pool features
        x = x[perm] * score[perm].view(-1, 1)

        # Filter edges
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=x.size(0)
        )

        return x, edge_index, edge_attr, batch_out, perm, score


class SAGPool(nn.Module):
    """
    Self-Attention Graph Pooling.

    Implementation of "Self-Attention Graph Pooling" (Lee et al., ICML 2019).

    Uses GNN to compute node importance scores:
    score = GNN(X, A)

    Args:
        in_channels: Input feature dimensions
        ratio: Pooling ratio
        gnn: Type of GNN for computing scores ('gcn', 'gat', 'sage')
    """

    def __init__(self, in_channels: int, ratio: float = 0.5, gnn: str = "gcn"):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        # Score computation GNN
        if gnn == "gcn":
            self.score_gnn = GCNConv(in_channels, 1)
        elif gnn == "gat":
            self.score_gnn = GATConv(in_channels, 1, heads=1, concat=False)
        elif gnn == "sage":
            self.score_gnn = GraphSAGEConv(in_channels, 1)
        else:
            raise ValueError(f"Unknown GNN type: {gnn}")

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
        """Forward pass."""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Compute attention scores
        score = self.score_gnn(x, edge_index).squeeze(-1)

        # Top-k selection
        num_graphs = int(batch.max()) + 1
        perm_list = []
        batch_out_list = []

        for i in range(num_graphs):
            mask = batch == i
            num_nodes = mask.sum().item()

            if self.ratio < 1:
                k = max(1, int(self.ratio * num_nodes))
            else:
                k = min(int(self.ratio), num_nodes)

            score_i = score[mask]
            _, idx = torch.topk(score_i, k)
            perm_i = torch.nonzero(mask, as_tuple=False).squeeze(-1)[idx]

            perm_list.append(perm_i)
            batch_out_list.append(
                torch.full((k,), i, dtype=torch.long, device=x.device)
            )

        perm = torch.cat(perm_list, dim=0)
        batch_out = torch.cat(batch_out_list, dim=0)

        # Pool
        x = x[perm] * torch.sigmoid(score[perm]).view(-1, 1)
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=x.size(0)
        )

        return x, edge_index, edge_attr, batch_out, perm, score


class DiffPool(nn.Module):
    """
    Differentiable Pooling Layer.

    Implementation of "Hierarchical Graph Representation Learning with
    Differentiable Pooling" (Ying et al., NeurIPS 2018).

    Learns a soft assignment matrix S:
    S = softmax(GNN_{embed}(A, X))
    X' = S^T X
    A' = S^T A S

    Args:
        in_channels: Input feature dimensions
        out_channels: Output feature dimensions (number of clusters)
        hidden_channels: Hidden dimensions for GNNs
        num_layers: Number of GNN layers
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # GNN for embedding
        self.embed_gnn = nn.ModuleList()
        self.embed_gnn.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.embed_gnn.append(GCNConv(hidden_channels, hidden_channels))

        # GNN for assignment
        self.assign_gnn = nn.ModuleList()
        self.assign_gnn.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.assign_gnn.append(GCNConv(hidden_channels, hidden_channels))

        self.assign_lin = Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Returns:
            x: Pooled node features
            adj: Pooled adjacency matrix
            loss: Auxiliary link prediction loss
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Compute embeddings
        z = x
        for gnn in self.embed_gnn:
            z = F.relu(gnn(z, edge_index))

        # Compute assignment matrix
        s = x
        for gnn in self.assign_gnn:
            s = F.relu(gnn(s, edge_index))
        s = self.assign_lin(s)
        s = F.softmax(s, dim=-1)

        # Convert to dense adjacency
        num_graphs = int(batch.max()) + 1
        x_pool_list = []
        adj_pool_list = []
        loss = 0

        for i in range(num_graphs):
            mask = batch == i
            x_i = z[mask]
            s_i = s[mask]

            # Create dense adjacency
            row, col = edge_index
            mask_edges = mask[row] & mask[col]
            edge_index_i = edge_index[:, mask_edges]
            edge_index_i = edge_index_i - mask.nonzero(as_tuple=False)[0, 0]

            num_nodes = mask.sum().item()
            adj = torch.zeros(num_nodes, num_nodes, device=x.device)
            adj[edge_index_i[0], edge_index_i[1]] = 1

            # Pool
            x_pool = s_i.t() @ x_i
            adj_pool = s_i.t() @ adj @ s_i

            # Auxiliary loss (entropy regularization)
            loss += -(s_i * torch.log(s_i + 1e-15)).sum() / num_nodes

            x_pool_list.append(x_pool)
            adj_pool_list.append(adj_pool)

        x_pool = torch.stack(x_pool_list)
        adj_pool = torch.stack(adj_pool_list)

        return x_pool, adj_pool, loss, s


class Set2Set(nn.Module):
    """
    Set2Set Global Pooling.

    Implementation of "Order Matters: Sequence to Sequence for Sets"
    (Vinyals et al., ICLR 2016).

    Uses an LSTM to iteratively compute a global graph representation.

    Args:
        in_channels: Input feature dimensions
        processing_steps: Number of processing steps
        num_layers: Number of LSTM layers
    """

    def __init__(
        self, in_channels: int, processing_steps: int = 3, num_layers: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.out_channels, in_channels, num_layers)
        self.lin = Linear(2 * in_channels, in_channels)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, F]
            batch: Batch assignment [N]

        Returns:
            Global graph features [batch_size, 2*F]
        """
        num_graphs = int(batch.max()) + 1
        out_list = []

        for i in range(num_graphs):
            mask = batch == i
            x_i = x[mask]

            # Initialize LSTM hidden state
            h = (
                torch.zeros(self.num_layers, 1, self.in_channels, device=x.device),
                torch.zeros(self.num_layers, 1, self.in_channels, device=x.device),
            )

            q_star = torch.zeros(1, self.out_channels, device=x.device)

            for _ in range(self.processing_steps):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.squeeze(0)

                # Attention over nodes
                e = (x_i * q).sum(dim=-1)
                a = F.softmax(e, dim=0)
                r = (a.unsqueeze(-1) * x_i).sum(dim=0, keepdim=True)

                q_star = torch.cat([q, r], dim=-1)

            out_list.append(q_star)

        return torch.cat(out_list, dim=0)


def filter_adj(
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    perm: Tensor,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Filter adjacency matrix based on selected nodes.

    Args:
        edge_index: Graph connectivity
        edge_attr: Edge attributes
        perm: Indices of nodes to keep
        num_nodes: Number of nodes

    Returns:
        Filtered edge index and attributes
    """
    # Create mask
    mask = torch.zeros(
        num_nodes or int(edge_index.max()) + 1,
        dtype=torch.bool,
        device=edge_index.device,
    )
    mask[perm] = True

    # Filter edges
    row, col = edge_index
    mask_edges = mask[row] & mask[col]
    edge_index = edge_index[:, mask_edges]

    if edge_attr is not None:
        edge_attr = edge_attr[mask_edges]

    # Remap indices
    node_map = torch.zeros(mask.size(0), dtype=torch.long, device=edge_index.device)
    node_map[perm] = torch.arange(perm.size(0), device=edge_index.device)
    edge_index = node_map[edge_index]

    return edge_index, edge_attr


# =============================================================================
# Graph Utilities
# =============================================================================


def batch_graphs(graph_list: List[GraphData]) -> GraphData:
    """
    Batch multiple graphs into a single disconnected graph.

    Args:
        graph_list: List of GraphData objects

    Returns:
        Batched GraphData
    """
    # Compute cumulative offsets
    num_nodes_list = [g.num_nodes for g in graph_list]
    num_nodes_cumsum = [0] + torch.cumsum(torch.tensor(num_nodes_list), 0).tolist()

    # Concatenate features
    x = (
        torch.cat([g.x for g in graph_list if g.x is not None], dim=0)
        if any(g.x is not None for g in graph_list)
        else None
    )

    # Concatenate and offset edge indices
    edge_indices = []
    for i, g in enumerate(graph_list):
        if g.edge_index is not None:
            edge_indices.append(g.edge_index + num_nodes_cumsum[i])
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else None

    # Concatenate edge attributes
    edge_attr = (
        torch.cat([g.edge_attr for g in graph_list if g.edge_attr is not None], dim=0)
        if any(g.edge_attr is not None for g in graph_list)
        else None
    )

    # Create batch assignment
    batch = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_nodes_list)]
    )

    # Concatenate targets
    y = None
    if graph_list[0].y is not None:
        if graph_list[0].y.dim() == 0 or (
            graph_list[0].y.dim() == 1 and len(graph_list[0].y) == 1
        ):
            # Graph-level targets
            y = torch.stack([g.y for g in graph_list])
        else:
            # Node-level targets
            y = torch.cat([g.y for g in graph_list], dim=0)

    # Concatenate positions
    pos = (
        torch.cat([g.pos for g in graph_list if g.pos is not None], dim=0)
        if any(g.pos is not None for g in graph_list)
        else None
    )

    return GraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        batch=batch,
        pos=pos,
        num_nodes=sum(num_nodes_list),
    )


def to_dense_batch(
    x: Tensor,
    batch: Tensor,
    fill_value: float = 0.0,
    max_num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Convert sparse batch to dense batch.

    Args:
        x: Node features [N, F]
        batch: Batch assignment [N]
        fill_value: Fill value for padding
        max_num_nodes: Maximum number of nodes (for padding)

    Returns:
        Dense features [batch_size, max_num_nodes, F]
        Mask [batch_size, max_num_nodes]
    """
    num_graphs = int(batch.max()) + 1

    # Compute max nodes per graph
    if max_num_nodes is None:
        max_num_nodes = int(torch.bincount(batch).max())

    # Create output tensor
    out = torch.full(
        (num_graphs, max_num_nodes, x.size(-1)),
        fill_value,
        dtype=x.dtype,
        device=x.device,
    )
    mask = torch.zeros((num_graphs, max_num_nodes), dtype=torch.bool, device=x.device)

    # Fill in values
    for i in range(num_graphs):
        mask_i = batch == i
        num_nodes = mask_i.sum().item()
        out[i, :num_nodes] = x[mask_i]
        mask[i, :num_nodes] = True

    return out, mask


def normalize_adj(adj: Tensor, mode: str = "sym") -> Tensor:
    """
    Normalize adjacency matrix.

    Args:
        adj: Dense adjacency matrix [N, N] or [B, N, N]
        mode: Normalization mode ('sym', 'rw', 'none')

    Returns:
        Normalized adjacency matrix
    """
    if mode == "none":
        return adj

    # Compute degree
    deg = adj.sum(dim=-1)

    if mode == "sym":
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt = torch.where(
            torch.isinf(deg_inv_sqrt), torch.zeros_like(deg_inv_sqrt), deg_inv_sqrt
        )

        if adj.dim() == 2:
            norm = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        else:
            norm = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
    elif mode == "rw":
        # Random walk normalization: D^{-1} A
        deg_inv = torch.pow(deg, -1)
        deg_inv = torch.where(torch.isinf(deg_inv), torch.zeros_like(deg_inv), deg_inv)

        if adj.dim() == 2:
            norm = deg_inv.unsqueeze(-1) * adj
        else:
            norm = deg_inv.unsqueeze(-1) * adj
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return norm


def dense_to_sparse(adj: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Convert dense adjacency to sparse edge index.

    Args:
        adj: Dense adjacency matrix [N, N]

    Returns:
        Edge index [2, E]
        Edge weights [E] (if adj is not binary)
    """
    index = adj.nonzero(as_tuple=False).t()
    value = adj[index[0], index[1]]
    return index, value if not torch.all(value == 1) else None


def sparse_to_dense(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    """
    Convert sparse edge index to dense adjacency.

    Args:
        edge_index: Edge index [2, E]
        edge_attr: Edge attributes [E]
        num_nodes: Number of nodes

    Returns:
        Dense adjacency [N, N]
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    adj = torch.zeros(
        num_nodes,
        num_nodes,
        dtype=edge_attr.dtype if edge_attr is not None else torch.float,
        device=edge_index.device,
    )

    if edge_attr is not None:
        adj[edge_index[0], edge_index[1]] = edge_attr
    else:
        adj[edge_index[0], edge_index[1]] = 1

    return adj


# =============================================================================
# Graph Data Loaders
# =============================================================================


class GraphDataset(Dataset):
    """
    PyTorch Dataset for graph data.

    Args:
        graphs: List of GraphData objects
        transform: Optional transform function
    """

    def __init__(self, graphs: List[GraphData], transform: Optional[Callable] = None):
        self.graphs = graphs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> GraphData:
        graph = self.graphs[idx]
        if self.transform:
            graph = self.transform(graph)
        return graph

    def get(self, idx: int) -> GraphData:
        """Get graph by index without transforms."""
        return self.graphs[idx]


def collate_graphs(batch: List[GraphData]) -> GraphData:
    """
    Collate function for graph data loader.

    Args:
        batch: List of GraphData objects

    Returns:
        Batched GraphData
    """
    return batch_graphs(batch)


class GraphDataLoader(TorchDataLoader):
    """
    DataLoader for graph data with custom collate function.

    Example:
        >>> dataset = GraphDataset(graphs)
        >>> loader = GraphDataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, dataset: Dataset, **kwargs):
        kwargs.setdefault("collate_fn", collate_graphs)
        super().__init__(dataset, **kwargs)


class NeighborLoader(TorchDataLoader):
    """
    DataLoader that samples neighbors for large graphs.

    Useful for training on large graphs that don't fit in memory.

    Args:
        data: GraphData object
        num_neighbors: Number of neighbors to sample per hop
        batch_size: Batch size (number of seed nodes)
        shuffle: Shuffle seed nodes
        num_hops: Number of hops to sample
    """

    def __init__(
        self,
        data: GraphData,
        num_neighbors: List[int],
        batch_size: int = 1,
        shuffle: bool = False,
        num_hops: int = 2,
        **kwargs,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.num_hops = num_hops

        # Create indices for all nodes
        self.indices = torch.arange(data.num_nodes)

        super().__init__(
            range(data.num_nodes),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._sample_neighbors,
            **kwargs,
        )

    def _sample_neighbors(self, seed_nodes: List[int]) -> GraphData:
        """Sample neighbors for seed nodes."""
        seed_nodes = torch.tensor(seed_nodes, dtype=torch.long)

        # Build adjacency list
        num_nodes = self.data.num_nodes
        row, col = self.data.edge_index
        adj_list = [[] for _ in range(num_nodes)]
        for i, j in zip(row.tolist(), col.tolist()):
            adj_list[i].append(j)

        # Sample neighbors for each hop
        all_nodes = set(seed_nodes.tolist())
        current_nodes = set(seed_nodes.tolist())

        for hop in range(self.num_hops):
            next_nodes = set()
            for node in current_nodes:
                neighbors = adj_list[node]
                num_samples = min(self.num_neighbors[hop], len(neighbors))
                if num_samples > 0:
                    sampled = torch.randperm(len(neighbors))[:num_samples]
                    next_nodes.update([neighbors[i] for i in sampled])

            all_nodes.update(next_nodes)
            current_nodes = next_nodes

        # Create subgraph
        node_map = {node: i for i, node in enumerate(sorted(all_nodes))}
        node_list = torch.tensor(sorted(all_nodes), dtype=torch.long)

        # Filter features
        x = self.data.x[node_list] if self.data.x is not None else None

        # Filter edges
        row, col = self.data.edge_index
        mask = torch.tensor(
            [r.item() in all_nodes and c.item() in all_nodes for r, c in zip(row, col)]
        )
        edge_index = self.data.edge_index[:, mask]
        edge_index = torch.tensor(
            [
                [node_map[n.item()] for n in edge_index[0]],
                [node_map[n.item()] for n in edge_index[1]],
            ],
            dtype=torch.long,
        )

        edge_attr = (
            self.data.edge_attr[mask] if self.data.edge_attr is not None else None
        )

        # Create batch assignment (all nodes belong to same batch in this subgraph)
        batch = torch.zeros(len(all_nodes), dtype=torch.long)

        return GraphData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            num_nodes=len(all_nodes),
            y=self.data.y[seed_nodes] if self.data.y is not None else None,
        )


# =============================================================================
# GNN Models
# =============================================================================


class GraphClassifier(nn.Module):
    """
    Graph-level classification model.

    Architecture:
    - Message passing layers
    - Global pooling
    - MLP classifier

    Args:
        in_channels: Input feature dimensions
        hidden_channels: Hidden dimensions
        out_channels: Number of output classes
        num_layers: Number of GNN layers
        gnn_type: Type of GNN ('gcn', 'gat', 'sage', 'gin', 'transformer')
        pooling: Pooling method ('mean', 'max', 'add', 'set2set')
        dropout: Dropout probability
        use_edge_attr: Use edge attributes
        edge_dim: Edge feature dimensions
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        pooling: str = "mean",
        dropout: float = 0.5,
        use_edge_attr: bool = False,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.pooling = pooling
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr

        # Build GNN layers
        self.convs = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels

            if gnn_type == "gcn":
                conv = GCNConv(in_ch, out_ch)
            elif gnn_type == "gat":
                conv = GATConv(in_ch, out_ch, heads=4, concat=False)
            elif gnn_type == "sage":
                conv = GraphSAGEConv(in_ch, out_ch)
            elif gnn_type == "gin":
                mlp = nn.Sequential(
                    Linear(in_ch, out_ch),
                    nn.ReLU(),
                    Linear(out_ch, out_ch),
                )
                conv = GINConv(mlp)
            elif gnn_type == "transformer":
                conv = TransformerConv(
                    in_ch, out_ch, edge_dim=edge_dim if use_edge_attr else None
                )
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.convs.append(conv)
            self.norms.append(LayerNorm(out_ch))

        # Pooling
        if pooling == "set2set":
            self.pool = Set2Set(hidden_channels, processing_steps=3)
            pool_out_ch = 2 * hidden_channels
        else:
            self.pool = None
            pool_out_ch = hidden_channels

        # Classifier
        self.classifier = nn.Sequential(
            Linear(pool_out_ch, hidden_channels),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, out_channels),
        )

    def forward(self, data: GraphData) -> Tensor:
        """
        Forward pass.

        Args:
            data: GraphData object

        Returns:
            Logits [batch_size, num_classes]
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Message passing
        for conv, norm in zip(self.convs, self.norms):
            if self.use_edge_attr and edge_attr is not None:
                x_new = conv(x, edge_index, edge_attr)
            else:
                x_new = conv(x, edge_index)

            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection
            if x.size(-1) == x_new.size(-1):
                x = x_new + x
            else:
                x = x_new

        # Global pooling
        if self.pool is not None:
            x = self.pool(x, batch)
        elif self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)

        # Classify
        out = self.classifier(x)

        return out


class LinkPredictor(nn.Module):
    """
    Link prediction model for graph edges.

    Predicts the existence of edges between node pairs.

    Args:
        in_channels: Input feature dimensions
        hidden_channels: Hidden dimensions
        num_layers: Number of GNN layers
        gnn_type: Type of GNN
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        gnn_type: str = "gcn",
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Encoder
        self.encoder = ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels

            if gnn_type == "gcn":
                conv = GCNConv(in_ch, hidden_channels)
            elif gnn_type == "gat":
                conv = GATConv(in_ch, hidden_channels, heads=4, concat=False)
            elif gnn_type == "sage":
                conv = GraphSAGEConv(in_ch, hidden_channels)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.encoder.append(conv)

        # Decoder (edge predictor)
        self.decoder = nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, 1),
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Encode nodes to embeddings."""
        for i, conv in enumerate(self.encoder):
            x = conv(x, edge_index)
            if i < len(self.encoder) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """
        Decode edge probabilities from node embeddings.

        Args:
            z: Node embeddings [N, F]
            edge_index: Edge pairs to predict [2, E]

        Returns:
            Edge probabilities [E]
        """
        src, dst = edge_index
        # Concatenate embeddings of source and target
        edge_emb = torch.cat([z[src], z[dst]], dim=-1)
        return self.decoder(edge_emb).squeeze(-1)

    def forward(
        self, data: GraphData, edge_label_index: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass.

        Args:
            data: GraphData object
            edge_label_index: Edges to predict (uses data.edge_index if None)

        Returns:
            Edge logits
        """
        z = self.encode(data.x, data.edge_index)

        if edge_label_index is None:
            edge_label_index = data.edge_index

        return self.decode(z, edge_label_index)


class GraphAutoencoder(nn.Module):
    """
    Graph Autoencoder for unsupervised representation learning.

    Learns to reconstruct the graph structure from node embeddings.

    Args:
        in_channels: Input feature dimensions
        hidden_channels: Hidden dimensions
        latent_channels: Latent space dimensions
        num_layers: Number of encoder layers
        gnn_type: Type of GNN
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_channels: int,
        num_layers: int = 2,
        gnn_type: str = "gcn",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels

        # Encoder
        self.encoder = ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = latent_channels if i == num_layers - 1 else hidden_channels

            if gnn_type == "gcn":
                conv = GCNConv(in_ch, out_ch)
            elif gnn_type == "gat":
                conv = GATConv(in_ch, out_ch, heads=4, concat=False)
            elif gnn_type == "sage":
                conv = GraphSAGEConv(in_ch, out_ch)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.encoder.append(conv)

        # Decoder (inner product)
        self.decoder = InnerProductDecoder()

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Encode to latent space."""
        for i, conv in enumerate(self.encoder):
            x = conv(x, edge_index)
            if i < len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Decode edge probabilities."""
        return self.decoder(z, edge_index)

    def forward(self, data: GraphData) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Returns:
            z: Latent embeddings
            adj_recon: Reconstructed adjacency scores
        """
        z = self.encode(data.x, data.edge_index)
        adj_recon = self.decode(z, data.edge_index)
        return z, adj_recon

    def loss(
        self, data: GraphData, pos_edge_index: Tensor, neg_edge_index: Tensor
    ) -> Tensor:
        """
        Compute reconstruction loss.

        Args:
            data: GraphData
            pos_edge_index: Positive edges
            neg_edge_index: Negative edges

        Returns:
            Loss value
        """
        z = self.encode(data.x, data.edge_index)

        # Positive edges
        pos_pred = self.decoder(z, pos_edge_index)
        pos_loss = -torch.log(pos_pred + 1e-15).mean()

        # Negative edges
        neg_pred = self.decoder(z, neg_edge_index)
        neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()

        return pos_loss + neg_loss


class InnerProductDecoder(nn.Module):
    """Decoder using inner product of embeddings."""

    def forward(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """
        Compute edge probabilities via inner product.

        Args:
            z: Node embeddings [N, F]
            edge_index: Edge pairs [2, E]

        Returns:
            Probabilities [E]
        """
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))


# =============================================================================
# Example Applications
# =============================================================================


class MolecularPropertyPredictor(nn.Module):
    """
    Graph neural network for molecular property prediction.

    Designed for molecular graphs with atom/bond features.

    Args:
        num_atom_features: Number of atom feature types
        num_bond_features: Number of bond feature types
        hidden_channels: Hidden dimensions
        out_channels: Number of output tasks
        num_layers: Number of GNN layers
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 5,
    ):
        super().__init__()

        # Atom embedding
        self.atom_encoder = nn.Linear(num_atom_features, hidden_channels)

        # Bond embedding
        self.bond_encoder = nn.Linear(num_bond_features, hidden_channels)

        # GNN layers
        self.convs = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            # Use TransformerConv for better modeling of bond features
            conv = TransformerConv(
                hidden_channels,
                hidden_channels,
                heads=4,
                concat=False,
                edge_dim=hidden_channels,
            )
            self.convs.append(conv)
            self.norms.append(LayerNorm(hidden_channels))

        # Readout
        self.pool = Set2Set(hidden_channels, processing_steps=3)

        # Predictor
        self.predictor = nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            Dropout(0.3),
            Linear(hidden_channels, out_channels),
        )

    def forward(self, data: GraphData) -> Tensor:
        """
        Forward pass.

        Args:
            data: GraphData with atom features (x) and bond features (edge_attr)

        Returns:
            Property predictions [batch_size, out_channels]
        """
        x = self.atom_encoder(data.x)
        edge_attr = (
            self.bond_encoder(data.edge_attr) if data.edge_attr is not None else None
        )

        # Message passing
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, data.edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x = x + x_new  # Residual

        # Global pooling
        x = self.pool(x, data.batch)

        # Predict
        return self.predictor(x)


class SocialNetworkClassifier(nn.Module):
    """
    GNN for social network analysis tasks.

    Supports node classification (user attributes) and graph classification
    (community detection).

    Args:
        in_channels: Input feature dimensions (e.g., user attributes)
        hidden_channels: Hidden dimensions
        out_channels: Number of classes
        num_layers: Number of GNN layers
        task: 'node' or 'graph' classification
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        task: str = "node",
    ):
        super().__init__()
        self.task = task

        # GNN layers
        self.convs = ModuleList()
        self.norms = ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels

            # Use GAT for social networks (attention over neighbors)
            self.convs.append(
                GATConv(in_ch, out_ch, heads=4, concat=False, dropout=0.3)
            )
            self.norms.append(LayerNorm(out_ch))

        # Task-specific head
        if task == "node":
            # Node classification - per-node output
            self.head = Linear(hidden_channels, out_channels)
        else:
            # Graph classification - global pooling then classify
            self.head = nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                Linear(hidden_channels, out_channels),
            )

    def forward(self, data: GraphData) -> Tensor:
        """
        Forward pass.

        Args:
            data: GraphData with user features

        Returns:
            Node logits [N, C] or graph logits [batch_size, C]
        """
        x = data.x

        # Message passing
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, data.edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x = x + x_new

        if self.task == "node":
            # Node-level predictions
            return self.head(x)
        else:
            # Graph-level predictions
            x = global_mean_pool(x, data.batch)
            return self.head(x)


class RecommendationGNN(nn.Module):
    """
    Graph Neural Network for recommendation systems.

    Implements a bipartite graph approach for user-item interactions.

    Args:
        num_users: Number of users
        num_items: Number of items
        embedding_dim: Embedding dimensions
        num_layers: Number of message passing layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Embeddings
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # Message passing layers
        self.convs = ModuleList()
        for _ in range(num_layers):
            self.convs.append(GraphSAGEConv(embedding_dim, embedding_dim))

        self.dropout = dropout

        # Initialize
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(
        self, edge_index: Tensor, user_indices: Tensor, item_indices: Tensor
    ) -> Tensor:
        """
        Forward pass for rating prediction.

        Args:
            edge_index: User-item interaction edges [2, E]
            user_indices: User indices to predict for [N]
            item_indices: Item indices to predict for [N]

        Returns:
            Predicted ratings/scores [N]
        """
        # Get initial embeddings
        num_nodes = self.num_users + self.num_items
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        # Message passing
        for conv in self.convs:
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new

        # Predict scores
        user_emb = x[user_indices]
        item_emb = x[self.num_users + item_indices]

        # Dot product
        scores = (user_emb * item_emb).sum(dim=-1)

        return torch.sigmoid(scores)

    def get_embeddings(self, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get final user and item embeddings after message passing.

        Returns:
            user_embeddings: [num_users, embedding_dim]
            item_embeddings: [num_items, embedding_dim]
        """
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        for conv in self.convs:
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            x = x + x_new

        return x[: self.num_users], x[self.num_users :]


# =============================================================================
# Training Integration
# =============================================================================


class GNNTrainer:
    """
    Specialized trainer for GNN models integrating with fishstick's Trainer.

    Handles graph-specific training procedures including:
    - Negative sampling for link prediction
    - Graph augmentation
    - Task-specific evaluation

    Example:
        >>> model = GraphClassifier(64, 128, 10)
        >>> trainer = GNNTrainer(model, optimizer, task='classification')
        >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        task: str = "classification",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.task = task
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "val_metric": []}

    def compute_loss(
        self, data: GraphData, criterion: Callable
    ) -> Tuple[Tensor, Tensor]:
        """Compute loss for different tasks."""
        data = data.to(self.device)

        if self.task == "classification":
            out = self.model(data)
            if data.y.dim() == 1 or (data.y.dim() == 2 and data.y.size(1) == 1):
                loss = criterion(out, data.y)
            else:
                loss = criterion(out, data.y)
            return loss, out

        elif self.task == "link_prediction":
            # Sample negative edges
            neg_edge_index = self._sample_negative_edges(
                data.edge_index, data.num_nodes
            )

            # Forward
            out_pos = self.model(data, data.edge_index)
            data_neg = GraphData(
                x=data.x, edge_index=neg_edge_index, num_nodes=data.num_nodes
            )
            out_neg = self.model(data_neg, neg_edge_index)

            # BCE loss
            pos_loss = F.binary_cross_entropy_with_logits(
                out_pos, torch.ones_like(out_pos)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                out_neg, torch.zeros_like(out_neg)
            )
            loss = pos_loss + neg_loss

            return loss, out_pos

        elif self.task == "regression":
            out = self.model(data)
            loss = criterion(out.squeeze(), data.y)
            return loss, out

        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _sample_negative_edges(
        self, edge_index: Tensor, num_nodes: int, num_neg_samples: Optional[int] = None
    ) -> Tensor:
        """Sample negative edges for link prediction."""
        if num_neg_samples is None:
            num_neg_samples = edge_index.size(1)

        # Get existing edges as set
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        neg_edges = []
        max_attempts = num_neg_samples * 10
        attempts = 0

        while len(neg_edges) < num_neg_samples and attempts < max_attempts:
            i = torch.randint(0, num_nodes, (1,)).item()
            j = torch.randint(0, num_nodes, (1,)).item()

            if i != j and (i, j) not in existing_edges:
                neg_edges.append((i, j))

            attempts += 1

        if len(neg_edges) < num_neg_samples:
            # Fill remaining with random edges
            while len(neg_edges) < num_neg_samples:
                i = torch.randint(0, num_nodes, (1,)).item()
                j = torch.randint(0, num_nodes, (1,)).item()
                if i != j:
                    neg_edges.append((i, j))

        neg_edge_index = torch.tensor(
            neg_edges, dtype=torch.long, device=edge_index.device
        ).t()
        return neg_edge_index

    def train_epoch(self, loader: GraphDataLoader, criterion: Callable) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for data in loader:
            self.optimizer.zero_grad()
            loss, _ = self.compute_loss(data, criterion)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(
        self, loader: GraphDataLoader, criterion: Callable
    ) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in loader:
                loss, out = self.compute_loss(data, criterion)
                total_loss += loss.item()

                # Compute accuracy for classification
                if self.task == "classification":
                    pred = out.argmax(dim=-1)
                    if data.y.dim() == 1:
                        correct += (pred == data.y.to(self.device)).sum().item()
                        total += data.y.size(0)

        avg_loss = total_loss / len(loader)
        metric = correct / total if total > 0 else 0

        return avg_loss, metric

    def fit(
        self,
        train_loader: GraphDataLoader,
        val_loader: Optional[GraphDataLoader] = None,
        epochs: int = 100,
        criterion: Optional[Callable] = None,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            criterion: Loss function
            verbose: Print progress

        Returns:
            Training history
        """
        if criterion is None:
            if self.task == "classification":
                criterion = nn.CrossEntropyLoss()
            elif self.task == "regression":
                criterion = nn.MSELoss()
            else:
                criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss, val_metric = self.evaluate(val_loader, criterion)
                self.history["val_loss"].append(val_loss)
                self.history["val_metric"].append(val_metric)

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}"
                    )
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        return self.history


# =============================================================================
# Graph Augmentation
# =============================================================================


class GraphAugmentation:
    """
    Data augmentation for graphs.

    Implements common augmentations:
    - Edge dropping
    - Node feature masking
    - Edge perturbation
    - Subgraph sampling
    """

    def __init__(
        self,
        edge_drop_prob: float = 0.2,
        feat_mask_prob: float = 0.2,
        subgraph_prob: float = 0.0,
    ):
        self.edge_drop_prob = edge_drop_prob
        self.feat_mask_prob = feat_mask_prob
        self.subgraph_prob = subgraph_prob

    def __call__(self, data: GraphData) -> GraphData:
        """Apply augmentations."""
        data = data.clone()

        # Edge dropping
        if self.edge_drop_prob > 0 and data.edge_index is not None:
            mask = torch.rand(data.edge_index.size(1)) > self.edge_drop_prob
            data.edge_index = data.edge_index[:, mask]
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[mask]

        # Feature masking
        if self.feat_mask_prob > 0 and data.x is not None:
            mask = torch.rand_like(data.x) > self.feat_mask_prob
            data.x = data.x * mask

        return data


def random_walk_augmentation(
    data: GraphData,
    walk_length: int = 10,
    num_walks: int = 1,
) -> List[GraphData]:
    """
    Create augmented graphs via random walks.

    Args:
        data: Input graph
        walk_length: Length of each random walk
        num_walks: Number of walks (and augmented graphs)

    Returns:
        List of augmented GraphData objects
    """
    # Build adjacency list
    num_nodes = data.num_nodes
    row, col = data.edge_index
    adj_list = [[] for _ in range(num_nodes)]
    for i, j in zip(row.tolist(), col.tolist()):
        adj_list[i].append(j)

    augmented_graphs = []

    for _ in range(num_walks):
        # Start from random node
        current = torch.randint(0, num_nodes, (1,)).item()
        visited = {current}

        # Random walk
        for _ in range(walk_length):
            neighbors = adj_list[current]
            if len(neighbors) > 0:
                current = neighbors[torch.randint(0, len(neighbors), (1,)).item()]
                visited.add(current)

        # Create subgraph
        visited = sorted(list(visited))
        node_map = {node: i for i, node in enumerate(visited)}

        # Filter features
        x = data.x[visited] if data.x is not None else None

        # Filter edges
        row_list, col_list = [], []
        for i, j in zip(row.tolist(), col.tolist()):
            if i in visited and j in visited:
                row_list.append(node_map[i])
                col_list.append(node_map[j])

        edge_index = torch.tensor([row_list, col_list], dtype=torch.long)

        augmented_graphs.append(
            GraphData(
                x=x,
                edge_index=edge_index,
                num_nodes=len(visited),
            )
        )

    return augmented_graphs


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data structures
    "GraphData",
    # Message passing layers
    "MessagePassing",
    "GCNConv",
    "GATConv",
    "GraphSAGEConv",
    "GINConv",
    "TransformerConv",
    "EdgeConv",
    # Pooling
    "global_mean_pool",
    "global_max_pool",
    "global_add_pool",
    "TopKPool",
    "SAGPool",
    "DiffPool",
    "Set2Set",
    # Utilities
    "batch_graphs",
    "to_dense_batch",
    "add_self_loops_fn",
    "normalize_adj",
    "gcn_norm",
    "dense_to_sparse",
    "sparse_to_dense",
    # Data loaders
    "GraphDataset",
    "GraphDataLoader",
    "NeighborLoader",
    "collate_graphs",
    # Models
    "GraphClassifier",
    "LinkPredictor",
    "GraphAutoencoder",
    "InnerProductDecoder",
    # Applications
    "MolecularPropertyPredictor",
    "SocialNetworkClassifier",
    "RecommendationGNN",
    # Training
    "GNNTrainer",
    # Augmentation
    "GraphAugmentation",
    "random_walk_augmentation",
]

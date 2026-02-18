"""
Graph Embedding Methods for Geometric Deep Learning.

Implements various graph embedding techniques:
- DeepWalk and Node2Vec random walk methods
- GraphSAGE embeddings
- Attributed graph embeddings
- Autoencoder-based embeddings

Based on:
- Perozzi et al. (2014): DeepWalk: Online Learning of Social Representations
- Grover & Leskovec (2016): Node2Vec: Scalable Feature Learning
- Hamilton et al. (2017): GraphSAGE: Inductive Representation Learning
"""

from typing import Optional, Tuple, List, Callable
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math
from collections import defaultdict


class DeepWalkEmbedder(nn.Module):
    """
    DeepWalk-style graph embeddings using random walks.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        walk_length: int = 80,
        num_walks: int = 10,
        window_size: int = 5,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings."""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)

    def generate_random_walks(
        self,
        edge_index: Tensor,
        start_nodes: Optional[Tensor] = None,
    ) -> List[List[int]]:
        """
        Generate random walks on the graph.

        Args:
            edge_index: Edge connectivity [2, E]
            start_nodes: Starting nodes for walks

        Returns:
            List of random walks
        """
        row, col = edge_index

        adj = defaultdict(list)
        for i, j in zip(row.tolist(), col.tolist()):
            adj[i].append(j)

        if start_nodes is None:
            start_nodes = torch.arange(self.num_nodes)

        walks = []

        for _ in range(self.num_walks):
            for node in start_nodes.tolist():
                walk = [node]
                current = node

                for _ in range(self.walk_length):
                    neighbors = adj.get(current, [])
                    if not neighbors:
                        break
                    current = neighbors[torch.randint(len(neighbors), (1,)).item()]
                    walk.append(current)

                walks.append(walk)

        return walks

    def skipgram_loss(
        self,
        center: Tensor,
        context: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """
        Compute skipgram loss.

        Args:
            center: Center node indices [B]
            context: Context node indices [B, 2*window]
            negative: Negative samples [B, n_neg]

        Returns:
            Loss value
        """
        center_emb = self.embedding(center)
        context_emb = self.embedding(context)
        neg_emb = self.embedding(negative)

        pos_score = (center_emb * context_emb).sum(dim=-1)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score,
            torch.ones_like(pos_score),
        )

        neg_score = (center_emb.unsqueeze(1) * neg_emb).sum(dim=-1)
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score,
            torch.zeros_like(neg_score),
        )

        return pos_loss + neg_loss

    def forward(self, node_ids: Tensor) -> Tensor:
        """
        Get embeddings for nodes.

        Args:
            node_ids: Node indices [N]

        Returns:
            Node embeddings [N, embedding_dim]
        """
        return self.embedding(node_ids)


class Node2VecEmbedder(nn.Module):
    """
    Node2Vec embeddings with biased random walks.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        walk_length: int = 80,
        num_walks: int = 10,
        window_size: int = 5,
        p: float = 1.0,
        q: float = 1.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings."""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)

    def generate_node2vec_walks(
        self,
        edge_index: Tensor,
        start_nodes: Optional[Tensor] = None,
    ) -> List[List[int]]:
        """
        Generate biased random walks (Node2Vec).

        Args:
            edge_index: Edge connectivity [2, E]
            start_nodes: Starting nodes

        Returns:
            List of biased random walks
        """
        row, col = edge_index

        adj = defaultdict(list)
        adj_rev = defaultdict(list)
        for i, j in zip(row.tolist(), col.tolist()):
            adj[i].append(j)
            adj_rev[j].append(i)

        if start_nodes is None:
            start_nodes = torch.arange(self.num_nodes)

        walks = []

        for _ in range(self.num_walks):
            for node in start_nodes.tolist():
                walk = [node]
                current = node
                prev = -1

                for _ in range(self.walk_length):
                    neighbors = []

                    for n in adj.get(current, []):
                        if n != prev:
                            neighbors.append(n)

                    if not neighbors:
                        break

                    probs = []
                    for n in neighbors:
                        if n in adj.get(prev, []):
                            probs.append(1.0 / self.p)
                        else:
                            probs.append(1.0 / self.q)

                    probs = torch.tensor(probs, dtype=torch.float32)
                    probs = probs / probs.sum()

                    idx = torch.multinomial(probs, 1).item()
                    current = neighbors[idx]
                    walk.append(current)

                walks.append(walk)

        return walks

    def forward(self, node_ids: Tensor) -> Tensor:
        """Get embeddings for nodes."""
        return self.embedding(node_ids)


class GraphSAGEEmbedder(nn.Module):
    """
    GraphSAGE-style embeddings with aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        aggr: str = "mean",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                self.norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Generate GraphSAGE embeddings.

        Args:
            features: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]

        Returns:
            Node embeddings [N, out_channels]
        """
        x = features

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)

        return x


class SAGEConv(nn.Module):
    """
    SAGE convolution layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Apply SAGE convolution.

        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]

        Returns:
            Updated features [N, out_channels]
        """
        row, col = edge_index

        neighbors = x[col]

        if self.aggr == "mean":
            out = neighbors.mean(dim=0)
        elif self.aggr == "max":
            out = neighbors.max(dim=0)[0]
        elif self.aggr == "sum":
            out = neighbors.sum(dim=0)
        else:
            out = neighbors.mean(dim=0)

        out = torch.cat([x, out], dim=-1)

        return self.linear(out)


class AttributedGraphEmbedding(nn.Module):
    """
    Embeddings for attributed graphs combining structure and attributes.
    """

    def __init__(
        self,
        num_nodes: int,
        num_attrs: int,
        embedding_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_attrs = num_attrs
        self.embedding_dim = embedding_dim

        self.struct_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.attr_embedding = nn.Embedding(num_attrs + 1, embedding_dim)

        self.attr_fc = nn.Linear(embedding_dim, hidden_dim)

        self.fusion = nn.Linear(hidden_dim + embedding_dim, embedding_dim)

    def forward(
        self,
        node_ids: Tensor,
        attrs: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Get combined embeddings.

        Args:
            node_ids: Node indices [N]
            attrs: Node attributes (optional) [N]

        Returns:
            Combined embeddings [N, embedding_dim]
        """
        struct_emb = self.struct_embedding(node_ids)

        if attrs is not None:
            attr_emb = self.attr_embedding(attrs)
            attr_feat = self.attr_fc(attr_emb)
            combined = torch.cat([struct_emb, attr_feat], dim=-1)
            out = self.fusion(combined)
        else:
            out = struct_emb

        return out


class GraphAutoEncoder(nn.Module):
    """
    Graph autoencoder for unsupervised embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
    ):
        super().__init__()

        self.encoder = GraphSAGEEmbedder(
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, features: Tensor, edge_index: Tensor) -> Tensor:
        """Encode graph to latent space."""
        return self.encoder(features, edge_index)

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """
        Decode to edge probabilities.

        Args:
            z: Node embeddings [N, out_channels]
            edge_index: Edge connectivity [2, E]

        Returns:
            Edge probabilities
        """
        row, col = edge_index

        z_i = z[row]
        z_j = z[col]

        edge_feat = torch.cat([z_i, z_j], dim=-1)

        return self.decoder(edge_feat).squeeze(-1)

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Full autoencoder forward.

        Args:
            features: Node features
            edge_index: Edge connectivity

        Returns:
            Reconstructed edges and embeddings
        """
        z = self.encode(features, edge_index)
        edge_pred = self.decode(z, edge_index)

        return edge_pred, z


class SignPredictor(nn.Module):
    """
    Predict edge signs for signed graphs.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
    ):
        super().__init__()

        self.encoder = GraphSAGEEmbedder(
            in_channels,
            hidden_channels,
            hidden_channels,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Predict edge signs.

        Args:
            features: Node features
            edge_index: Edge connectivity

        Returns:
            Sign predictions [E]
        """
        embeddings = self.encoder(features, edge_index)

        row, col = edge_index

        edge_feat = torch.cat(
            [
                embeddings[row],
                embeddings[col],
                embeddings[row] * embeddings[col],
                torch.abs(embeddings[row] - embeddings[col]),
            ],
            dim=-1,
        )

        return self.classifier(edge_feat).squeeze(-1)


class LaplacianEigenmap(nn.Module):
    """
    Laplacian Eigenmap embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.register_buffer("embeddings", torch.zeros(0, embedding_dim))

    def fit(
        self,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """
        Compute Laplacian eigenmap embeddings.

        Args:
            edge_index: Edge connectivity
            num_nodes: Number of nodes
            num_components: Number of embedding dimensions

        Returns:
            Embeddings [num_nodes, embedding_dim]
        """
        row, col = edge_index

        adj = torch.zeros(num_nodes, num_nodes)
        adj[row, col] = 1
        adj = adj + adj.t()

        degree = adj.sum(dim=1)
        D_inv_sqrt = degree.pow(-0.5)
        D_inv_sqrt[D_inv_sqrt == float("inf")] = 0

        L = torch.eye(num_nodes) - D_inv_sqrt.unsqueeze(1) * adj * D_inv_sqrt.unsqueeze(
            0
        )

        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        embeddings = eigenvectors[:, 1 : self.embedding_dim + 1]

        self.register_buffer("embeddings", embeddings)

        return embeddings

    def forward(self, node_ids: Tensor) -> Tensor:
        """Get precomputed embeddings."""
        return self.embedding(node_ids)


class HigherOrderProximity(nn.Module):
    """
    Capture higher-order graph proximity via k-step neighbors.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        k_hops: int = 2,
    ):
        super().__init__()
        self.k_hops = k_hops

        self.convs = nn.ModuleList(
            [
                GraphSAGEEmbedder(in_channels, hidden_channels, hidden_channels)
                for _ in range(k_hops)
            ]
        )

        self.fusion = nn.Linear(hidden_channels * k_hops, out_channels)

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Capture multi-hop neighborhood information.

        Args:
            features: Node features
            edge_index: Edge connectivity

        Returns:
            Multi-hop embeddings
        """
        outs = []

        for conv in self.convs:
            out = conv(features, edge_index)
            outs.append(out)

        combined = torch.cat(outs, dim=-1)

        return self.fusion(combined)


__all__ = [
    "DeepWalkEmbedder",
    "Node2VecEmbedder",
    "GraphSAGEEmbedder",
    "SAGEConv",
    "AttributedGraphEmbedding",
    "GraphAutoEncoder",
    "SignPredictor",
    "LaplacianEigenmap",
    "HigherOrderProximity",
]

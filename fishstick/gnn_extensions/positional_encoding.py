"""
Positional Encodings for Graph Neural Networks.

This module provides various positional encoding schemes for graphs:
- Laplacian positional encodings (eigenvector-based)
- Random walk positional encodings
- Centrality-based encodings
- Relative positional encodings
"""

from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import scipy.sparse as sp


class LaplacianPositionalEncoding(nn.Module):
    """
    Laplacian positional encoding using graph eigenvectors.

    Encodes node positions using the eigenvectors of the graph Laplacian.
    Captures structural position in the graph topology.

    Args:
        num_embeddings: Number of eigenvector dimensions to use
        normalization: Laplacian normalization type ('sym', 'rw')
        include_first: Whether to include the first eigenvector (constant)
    """

    def __init__(
        self,
        num_embeddings: int = 16,
        normalization: str = "sym",
        include_first: bool = False,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.normalization = normalization
        self.include_first = include_first

    def forward(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Laplacian positional encodings.

        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Number of nodes
            edge_weight: Optional edge weights

        Returns:
            Positional encodings [num_nodes, num_embeddings]
        """
        edge_index_np = edge_index.cpu().numpy()

        if edge_weight is not None:
            edge_weight_np = edge_weight.cpu().numpy()
        else:
            edge_weight_np = np.ones(edge_index_np.shape[1])

        adj = sp.coo_matrix(
            (edge_weight_np, edge_index_np), shape=(num_nodes, num_nodes)
        ).tocsr()

        if self.normalization == "sym":
            d = np.array(adj.sum(axis=1)).flatten()
            d_inv_sqrt = np.power(d, -0.5)
            d_inv_sqrt[d_inv_sqrt == np.inf] = 0.0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            laplacian = sp.eye(num_nodes) - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        else:
            d = np.array(adj.sum(axis=1)).flatten()
            d_inv = np.power(d, -1)
            d_inv[d_inv == np.inf] = 0.0
            d_mat_inv = sp.diags(d_inv)
            laplacian = sp.eye(num_nodes) - d_mat_inv @ adj

        try:
            eigenvalues, eigenvectors = sp.linalg.eigsh(
                laplacian,
                k=self.num_embeddings + (0 if self.include_first else 1),
                which="SM",
            )
        except Exception:
            eigenvalues = np.zeros(
                self.num_embeddings + (0 if self.include_first else 1)
            )
            eigenvectors = np.zeros(
                (num_nodes, self.num_embeddings + (0 if self.include_first else 1))
            )

        if not self.include_first:
            eigenvectors = eigenvectors[:, 1:]

        pos_enc = torch.from_numpy(eigenvectors[:, : self.num_embeddings]).float()

        if edge_index.device.type != "cpu":
            pos_enc = pos_enc.to(edge_index.device)

        return pos_enc

    def compute_laplacian_eigenvectors(
        self,
        adj: Union[np.ndarray, sp.spmatrix],
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Laplacian eigenvectors directly.

        Args:
            adj: Adjacency matrix
            k: Number of eigenvectors to compute

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        if sp.issparse(adj):
            adj = adj.toarray()

        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == np.inf] = 0.0

        D_inv_sqrt = np.diag(degree_inv_sqrt)
        D_inv = np.diag(1.0 / degree)

        if self.normalization == "sym":
            DAD = D_inv_sqrt @ adj @ D_inv_sqrt
            laplacian = np.eye(adj.shape[0]) - DAD
        else:
            laplacian = np.eye(adj.shape[0]) - D_inv @ adj

        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        idx = np.argsort(eigenvalues)[:k]
        return eigenvalues[idx], eigenvectors[:, idx]


class RandomWalkPositionalEncoding(nn.Module):
    """
    Random walk positional encoding using return probabilities.

    Encodes node positions based on random walk return probabilities
    at various time steps.

    Args:
        embedding_dim: Dimension of the encoding
        walk_length: Length of random walks
        num_walks: Number of random walks per node
        walk_type: Type of random walk ('first', 'second', 'personalized')
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        walk_length: int = 10,
        num_walks: int = 10,
        walk_type: str = "first",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.walk_type = walk_type

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute random walk positional encodings.

        Args:
            x: Node features (used for initialization if needed)
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment

        Returns:
            Positional encodings [num_nodes, embedding_dim]
        """
        num_nodes = x.size(0)
        row, col = edge_index

        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[row, col] = 1.0
        adj[col, row] = 1.0

        degree = adj.sum(dim=1, keepdim=True)
        adj_norm = adj / (degree + 1e-8)

        rw_enc = torch.zeros(num_nodes, self.embedding_dim, device=x.device)

        current = torch.eye(num_nodes, device=x.device)

        for t in range(min(self.embedding_dim, self.walk_length)):
            current = current @ adj_norm
            if t < self.embedding_dim:
                rw_enc[:, t] = current.diag()

        return rw_enc

    def simulate_random_walk(
        self,
        edge_index: Tensor,
        start_nodes: Tensor,
    ) -> List[List[int]]:
        """
        Simulate random walks from start nodes.

        Args:
            edge_index: Edge connectivity
            start_nodes: Starting node indices
            num_steps: Number of steps per walk

        Returns:
            List of random walk paths
        """
        row, col = edge_index
        num_nodes = max(row.max().item(), col.max().item()) + 1

        neighbors = {}
        for i in range(num_nodes):
            neighbors[i] = []
        for r, c in zip(row.tolist(), col.tolist()):
            neighbors[r].append(c)
            neighbors[c].append(r)

        walks = []
        for start in start_nodes.tolist():
            walk = [start]
            current = start
            for _ in range(self.walk_length):
                if len(neighbors[current]) == 0:
                    break
                current = np.random.choice(neighbors[current])
                walk.append(current)
            walks.append(walk)

        return walks


class CentralityPositionalEncoding(nn.Module):
    """
    Centrality-based positional encoding.

    Encodes nodes using various centrality measures:
    - Degree centrality
    - Betweenness centrality
    - PageRank
    - Eigenvector centrality

    Args:
        embedding_dim: Dimension of encoding
        centrality_types: List of centrality types to use
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        centrality_types: Optional[List[str]] = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.centrality_types = centrality_types or ["degree", "pagerank"]

    def forward(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute centrality-based positional encodings.

        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Number of nodes
            edge_weight: Optional edge weights

        Returns:
            Positional encodings [num_nodes, embedding_dim]
        """
        row, col = edge_index

        degree = torch.zeros(num_nodes)
        degree.index_add_(0, row, torch.ones(row.size(0), device=edge_index.device))

        encodings = [degree.unsqueeze(1)]

        if "pagerank" in self.centrality_types:
            pagerank = self._compute_pagerank(edge_index, num_nodes, edge_weight)
            encodings.append(pagerank.unsqueeze(1))

        if len(encodings) < self.embedding_dim:
            zero_pad = torch.zeros(
                num_nodes, self.embedding_dim - len(encodings), device=edge_index.device
            )
            encodings.append(zero_pad)

        pos_enc = torch.cat(encodings, dim=1)[:, : self.embedding_dim]
        return pos_enc

    def _compute_pagerank(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Optional[Tensor] = None,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Tensor:
        """Compute PageRank scores."""
        row, col = edge_index

        if edge_weight is not None:
            weights = edge_weight
        else:
            weights = torch.ones(row.size(0), device=edge_index.device)

        out_degree = torch.zeros(num_nodes)
        out_degree.index_add_(0, row, weights)

        pr = torch.ones(num_nodes) / num_nodes

        for _ in range(max_iter):
            new_pr = torch.zeros(num_nodes, device=edge_index.device)
            new_pr.index_add_(0, col, pr[row] * weights / (out_degree[row] + 1e-8))
            new_pr = (1 - damping) / num_nodes + damping * new_pr

            if (new_pr - pr).abs().sum().item() < tol:
                break
            pr = new_pr

        return pr


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for graph structures.

    Encodes pairwise node relationships using shortest path distances
    and learnable embeddings.

    Args:
        embedding_dim: Dimension of encoding
        max_distance: Maximum distance to consider
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        max_distance: int = 10,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_distance = max_distance

        self.distance_encoder = nn.Embedding(max_distance + 1, embedding_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Compute relative positional encodings.

        Args:
            x: Node features (not used directly)
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            Pairwise encodings [num_nodes, num_nodes, embedding_dim]
        """
        num_nodes = x.size(0)

        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        row, col = edge_index
        adj[row, col] = 1.0
        adj[col, row] = 1.0

        dist = self._compute_all_pairs_distance(adj)
        dist = dist.clamp(0, self.max_distance)

        pos_enc = self.distance_encoder(dist)

        return pos_enc

    def _compute_all_pairs_distance(self, adj: Tensor) -> Tensor:
        """Compute all-pairs shortest path distances."""
        num_nodes = adj.size(0)

        dist = adj.clone()
        dist[dist == 0] = float("inf")
        dist.fill_diagonal_(0)

        for k in range(num_nodes):
            dist = torch.min(dist, dist[:, k : k + 1] + dist[k : k + 1, :])

        dist[dist == float("inf")] = 0
        return dist.long()


class SignNetPositionalEncoding(nn.Module):
    """
    Sign-invariant network (SignNet) positional encoding.

    Uses both positive and negative eigenvalues for position encoding.
    The sign invariance is achieved through absolute value operations.

    Args:
        num_embeddings: Number of eigenvector dimensions
        hidden_dim: Hidden dimension for the MLP
    """

    def __init__(
        self,
        num_embeddings: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings

        self.fc1 = nn.Linear(num_embeddings, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_embeddings)

        self.lap_encoder = LaplacianPositionalEncoding(
            num_embeddings=num_embeddings, include_first=False
        )

    def forward(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute SignNet positional encodings.

        Args:
            edge_index: Edge connectivity
            num_nodes: Number of nodes
            edge_weight: Optional edge weights

        Returns:
            Positional encodings [num_nodes, num_embeddings]
        """
        lap_enc = self.lap_encoder(edge_index, num_nodes, edge_weight)

        pos_enc = torch.abs(lap_enc)

        h = F.relu(self.fc1(pos_enc))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)

        return h + pos_enc

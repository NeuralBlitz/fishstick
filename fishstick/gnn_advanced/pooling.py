import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class TopKPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        min_score: Optional[float] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score

        self.score_layer = nn.Linear(in_channels, 1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        scores = self.score_layer(x).squeeze(-1)
        scores = torch.tanh(scores)

        num_nodes = x.size(0)
        num_clusters = max(int(num_nodes * self.ratio), 1)

        perm = torch.topk(scores, num_clusters, sorted=False)[1]

        x_pool = x[perm]
        edge_index_i, edge_index_j = edge_index

        mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
        mask[perm] = False

        new_edge_index_i = edge_index_i[mask]
        new_edge_index_j = edge_index_j[mask]

        new_mask = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        new_mask[perm] = torch.arange(perm.size(0), device=x.device)

        new_edge_index_i = new_mask[new_edge_index_i]
        new_edge_index_j = new_mask[new_edge_index_j]

        batch_pool = batch[perm]

        edge_index_pool = torch.stack([new_edge_index_i, new_edge_index_j], dim=0)

        return x_pool, edge_index_pool, perm, batch_pool, scores[perm]


class SAGPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        score_bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        self.score_layer = nn.Sequential(
            nn.Linear(in_channels, 1, bias=score_bias),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        scores = self.score_layer(x).squeeze(-1)

        edge_index_i, edge_index_j = edge_index
        neighbor_scores = scores[edge_index_j]
        aggregated_scores = torch.zeros_like(scores)
        aggregated_scores.scatter_add_(0, edge_index_i, neighbor_scores)
        scores = scores + aggregated_scores

        num_nodes = x.size(0)
        num_clusters = max(int(num_nodes * self.ratio), 1)

        perm = torch.topk(scores, num_clusters, sorted=False)[1]

        x_pool = x[perm]
        edge_index_i, edge_index_j = edge_index

        mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
        mask[perm] = False

        new_edge_index_i = edge_index_i[mask]
        new_edge_index_j = edge_index_j[mask]

        new_mask = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        new_mask[perm] = torch.arange(perm.size(0), device=x.device)

        new_edge_index_i = new_mask[new_edge_index_i]
        new_edge_index_j = new_mask[new_edge_index_j]

        batch_pool = batch[perm]

        edge_index_pool = torch.stack([new_edge_index_i, new_edge_index_j], dim=0)

        return x_pool, edge_index_pool, perm, batch_pool, scores[perm]


class DiffPooling(nn.Module):
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

        self.assignment_layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.softmax(dim=-1),
        )

        self_embedding = torch.randn(out_channels, in_channels)
        nn.init.xavier_uniform_(self_embedding)
        self.node_embedding = nn.Parameter(self_embedding)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_nodes = x.size(0)
        num_clusters = max(int(num_nodes * self.ratio), 1)

        S = self.assignment_layer(x + self.node_embedding[: x.size(1)])
        S = S[:, :num_clusters]

        x_pool = S.T @ x

        edge_index_i, edge_index_j = edge_index

        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index_i, edge_index_j] = 1

        adj_pool = S.T @ adj @ S

        edge_index_i_pool, edge_index_j_pool = adj_pool.nonzero(as_tuple=True)
        edge_index_pool = torch.stack([edge_index_i_pool, edge_index_j_pool], dim=0)

        batch_pool = torch.zeros(num_clusters, dtype=torch.long, device=x.device)
        for i in range(num_clusters):
            nodes_in_cluster = (S[:, i] > 0).nonzero(as_tuple=True)[0]
            if nodes_in_cluster.numel() > 0:
                batch_pool[i] = batch[nodes_in_cluster].max()

        perm = torch.arange(num_clusters, device=x.device)

        return x_pool, edge_index_pool, perm, batch_pool, S.sum(dim=0)


class GraclusPooling(nn.Module):
    def __init__(
        self,
        ratio: float = 0.5,
    ):
        super().__init__()
        self.ratio = ratio

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        edge_index_i, edge_index_j = edge_index
        num_nodes = x.size(0)
        num_clusters = max(int(num_nodes * self.ratio), 1)

        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index_i, edge_index_j] = 1

        x_norm = F.normalize(x, p=2, dim=-1)
        sim = x_norm @ x_norm.T

        adj_sim = adj * sim

        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj_sim * deg_inv_sqrt.unsqueeze(-2)

        weights = norm_adj.sum(dim=1)

        perm = torch.topk(weights, num_clusters, sorted=False)[1]

        x_pool = x[perm]

        mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
        mask[perm] = False

        new_edge_index_i = edge_index_i[mask]
        new_edge_index_j = edge_index_j[mask]

        new_mask = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        new_mask[perm] = torch.arange(perm.size(0), device=x.device)

        new_edge_index_i = new_mask[new_edge_index_i]
        new_edge_index_j = new_mask[new_edge_index_j]

        batch_pool = batch[perm]

        edge_index_pool = torch.stack([new_edge_index_i, new_edge_index_j], dim=0)

        return x_pool, edge_index_pool, perm, batch_pool, weights[perm]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class RGCNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        bias: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        self.weight = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))
        if bias is not None:
            self.bias = bias
        else:
            self.bias = nn.Parameter(torch.Tensor(out_dim))

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.out_dim, device=x.device, dtype=x.dtype)

        for rel_type in range(self.num_relations):
            mask = edge_type == rel_type
            if mask.sum() == 0:
                continue

            src, dst = edge_index[:, mask]
            x_src = x[src]

            aggregated = torch.matmul(
                x_src.unsqueeze(1), self.weight[rel_type]
            ).squeeze(1)
            out[dst] += F.relu(aggregated + self.bias)

        return out


class RGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations

        self.node_embeddings = nn.Embedding(num_nodes, node_dim)

        self.layers = nn.ModuleList()
        dims = [node_dim] + [hidden_dim] * (num_layers - 1)

        for i in range(num_layers):
            self.layers.append(RGCNLayer(dims[i], hidden_dim, num_relations))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        if x is None:
            x = self.node_embeddings.weight

        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
            x = self.dropout(x)

        return x


class CompGCNLayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, num_relations: int, comp_fn: str = "mult"
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.comp_fn = comp_fn

        self.w_loop = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.w_forward = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.w_backward = nn.Parameter(torch.Tensor(in_dim, out_dim))

        self.relation_embeddings = nn.Embedding(num_relations, out_dim)

        nn.init.xavier_uniform_(self.w_loop)
        nn.init.xavier_uniform_(self.w_forward)
        nn.init.xavier_uniform_(self.w_backward)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def _composition(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.comp_fn == "mult":
            return h * r
        elif self.comp_fn == "add":
            return h + r
        elif self.comp_fn == "cross":
            return torch.cross(h, r, dim=-1)
        else:
            raise ValueError(f"Unknown composition function: {self.comp_fn}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.out_dim, device=x.device, dtype=x.dtype)

        num_edges = edge_index.size(1)
        if edge_norm is None:
            edge_norm = torch.ones(num_edges, device=x.device)

        forward_mask = edge_type < self.num_relations // 2
        backward_mask = edge_type >= self.num_relations // 2

        if forward_mask.sum() > 0:
            fwd_src, fwd_dst = edge_index[:, forward_mask]
            fwd_types = edge_type[forward_mask]
            fwd_norm = edge_norm[forward_mask].unsqueeze(-1)

            fwd_msg = torch.matmul(
                self._composition(x[fwd_src], self.relation_embeddings(fwd_types)),
                self.w_forward,
            )
            out[fwd_dst] += fwd_msg * fwd_norm

        if backward_mask.sum() > 0:
            bwd_src, bwd_dst = edge_index[:, backward_mask]
            bwd_types = edge_type[backward_mask] - self.num_relations // 2
            bwd_norm = edge_norm[backward_mask].unsqueeze(-1)

            bwd_msg = torch.matmul(
                self._composition(x[bwd_src], self.relation_embeddings(bwd_types)),
                self.w_backward,
            )
            out[bwd_dst] += bwd_msg * bwd_norm

        loop_msg = torch.matmul(x, self.w_loop)
        out += loop_msg

        return out


class CompGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        comp_fn: str = "mult",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations

        self.node_embeddings = nn.Embedding(num_nodes, node_dim)

        self.layers = nn.ModuleList()
        dims = [node_dim] + [hidden_dim] * (num_layers - 1)

        for i in range(num_layers):
            self.layers.append(
                CompGCNLayer(dims[i], hidden_dim, num_relations, comp_fn)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x is None:
            x = self.node_embeddings.weight

        for layer in self.layers:
            x = layer(x, edge_index, edge_type, edge_norm)
            x = F.relu(x)
            x = self.dropout(x)

        return x


class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        num_relations: int,
        message_fn: str = "pna",
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.message_fn = message_fn

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.relation_proj = nn.Linear(edge_dim, out_dim)

        self.aggregators = ["mean", "max", "sum"]
        self.agg_nns = nn.ModuleList(
            [nn.Linear(out_dim, out_dim) for _ in self.aggregators]
        )

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([x_i, edge_attr], dim=-1)
        return self.node_mlp(combined)

    def aggregate(
        self, messages: torch.Tensor, index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        out = torch.zeros(num_nodes, self.out_dim, device=messages.device)

        for agg_fn, agg_nn in zip(self.aggregators, self.agg_nns):
            if agg_fn == "mean":
                agg = torch.zeros(num_nodes, self.out_dim, device=messages.device)
                agg.scatter_add_(0, index.unsqueeze(-1).expand_as(messages), messages)
                count = torch.zeros(num_nodes, device=messages.device).scatter_add(
                    0, index, torch.ones_like(index, dtype=torch.float)
                )
                agg = agg / count.clamp(min=1).unsqueeze(-1)
            elif agg_fn == "max":
                agg = messages.new_full((num_nodes, self.out_dim), float("-inf"))
                agg = agg.scatter_reduce(
                    0,
                    index.unsqueeze(-1).expand_as(messages),
                    messages,
                    reduce="amax",
                    include_self=False,
                )
                agg[agg == float("-inf")] = 0
            else:
                agg = torch.zeros(num_nodes, self.out_dim, device=messages.device)
                agg.scatter_add_(0, index.unsqueeze(-1).expand_as(messages), messages)

            out += agg_nn(agg)

        return out

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index

        messages = self.message(x[dst], x[src], edge_attr)

        out = self.aggregate(messages, dst, x.size(0))

        return out


class KGGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        gnn_type: str = "rgcn",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.gnn_type = gnn_type

        self.node_embeddings = nn.Embedding(num_nodes, node_dim)

        if gnn_type == "rgcn":
            self.gnn = RGCN(
                num_nodes, num_relations, node_dim, hidden_dim, num_layers, dropout
            )
        elif gnn_type == "compgcn":
            self.gnn = CompGCN(
                num_nodes,
                num_relations,
                node_dim,
                hidden_dim,
                num_layers,
                dropout,
                "mult",
            )
        elif gnn_type == "message_passing":
            edge_dim = hidden_dim
            self.gnn = nn.ModuleList(
                [
                    MessagePassingLayer(
                        node_dim if i == 0 else hidden_dim,
                        edge_dim,
                        hidden_dim,
                        num_relations,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        edge_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x is None:
            x = self.node_embeddings.weight

        if self.gnn_type == "message_passing":
            for layer in self.gnn:
                x = layer(
                    x,
                    edge_index,
                    edge_attr
                    if edge_attr is not None
                    else torch.zeros(
                        edge_index.size(1), self.num_relations, device=x.device
                    ),
                )
                x = F.relu(x)
                x = self.dropout(x)
        elif self.gnn_type == "compgcn":
            x = self.gnn(x, edge_index, edge_type, edge_norm)
        else:
            x = self.gnn(x, edge_index, edge_type)

        return x


class KnowledgeGraphGNN(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        gnn_type: str = "rgcn",
    ):
        super().__init__()

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.gnn = KGGNN(
            num_entities, num_relations, embedding_dim, hidden_dim, num_layers, gnn_type
        )

        self.score_fn = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        entity_emb = self.entity_embeddings.weight

        updated_emb = self.gnn(entity_emb, edge_index, edge_type)

        h = updated_emb[head]
        r = self.relation_embeddings(relation)
        t = updated_emb[tail]

        combined = h * r * t
        score = self.score_fn(combined, r).squeeze(-1)

        return score

    def get_entity_embeddings(self) -> torch.Tensor:
        return self.entity_embeddings.weight

    def get_relation_embeddings(self) -> torch.Tensor:
        return self.relation_embeddings.weight

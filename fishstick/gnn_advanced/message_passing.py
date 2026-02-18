import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from typing import Optional, Tuple


class GCNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = x.size(0)
        edge_index_i, edge_index_j = edge_index

        deg = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        deg.scatter_add_(0, edge_index_i, torch.ones_like(edge_index_i))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[edge_index_i] * deg_inv_sqrt[edge_index_j]

        x = torch.matmul(x, self.weight)
        out = torch.zeros_like(x)
        out.scatter_add_(
            0,
            edge_index_j.unsqueeze(-1).expand_as(x),
            norm.unsqueeze(-1) * x[edge_index_i],
        )

        if self.bias is not None:
            out = out + self.bias
        return out


class GATConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        if concat:
            self.out_channels = out_channels // heads
            self.weight = nn.Parameter(
                torch.empty(in_channels, heads * self.out_channels)
            )
        else:
            self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
            self.out_channels = out_channels

        self.att = nn.Parameter(torch.empty(2 * self.out_channels, heads))

        if bias and concat:
            self.bias = nn.Parameter(torch.empty(heads * self.out_channels))
        elif bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        edge_index_i, edge_index_j = edge_index

        x = torch.matmul(x, self.weight)
        x = x.view(-1, self.heads, self.out_channels)

        h_i = x[edge_index_i]
        h_j = x[edge_index_j]

        h_combined = torch.cat([h_i, h_j], dim=-1)
        h_combined = h_combined.view(-1, self.heads, 2 * self.out_channels)

        e = torch.sum(h_combined * self.att.unsqueeze(0), dim=-1)
        e = F.leaky_relu(e, 0.2)

        alpha = torch.zeros(edge_index_i.size(0), self.heads, device=x.device)
        alpha.scatter_add_(
            0,
            edge_index_i.unsqueeze(-1).expand_as(alpha),
            e.exp(),
        )
        alpha = alpha[edge_index_j]
        alpha = alpha / (alpha.sum(dim=0, keepdim=True) + 1e-16)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        h_j = h_j * alpha.unsqueeze(-1)
        out = torch.zeros_like(h_j)
        out.scatter_add_(
            0, edge_index_i.unsqueeze(-1).unsqueeze(-1).expand_as(out), h_j
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out


class GraphSAGEConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregator: str = "mean",
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator = aggregator
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        edge_index_i, edge_index_j = edge_index

        x = torch.matmul(x, self.weight)

        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.out_channels, device=x.device)

        if self.aggregator == "mean":
            agg = torch.zeros(num_nodes, self.out_channels, device=x.device)
            count = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

            agg.scatter_add_(
                0, edge_index_j.unsqueeze(-1).expand_as(x), x[edge_index_i]
            )
            count.scatter_add_(0, edge_index_j, torch.ones_like(edge_index_j))
            count[count == 0] = 1
            agg = agg / count.unsqueeze(-1)

        elif self.aggregator == "max":
            agg = torch.full(
                (num_nodes, self.out_channels), float("-inf"), device=x.device
            )
            agg.scatter_add_(
                0, edge_index_j.unsqueeze(-1).expand_as(x), x[edge_index_i]
            )
            agg[agg == float("-inf")] = 0

        elif self.aggregator == "add":
            agg = torch.zeros(num_nodes, self.out_channels, device=x.device)
            agg.scatter_add_(
                0, edge_index_j.unsqueeze(-1).expand_as(x), x[edge_index_i]
            )

        out = agg + x

        if self.bias is not None:
            out = out + self.bias
        return out


class MPNNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: Optional[int] = None,
        message_steps: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.message_steps = message_steps

        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        if edge_channels is not None:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_channels + in_channels * 2, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
        else:
            self.edge_mlp = None

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.node_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.edge_mlp is not None:
            for module in self.edge_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        edge_index_i, edge_index_j = edge_index
        num_nodes = x.size(0)

        h = x.clone()
        for _ in range(self.message_steps):
            h_i = h[edge_index_i]
            h_j = h[edge_index_j]

            if self.edge_mlp is not None and edge_attr is not None:
                edge_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
                m_ij = self.edge_mlp(edge_input)
            else:
                edge_input = torch.cat([h_i, h_j], dim=-1)
                m_ij = self.node_mlp(edge_input)

            m = torch.zeros(num_nodes, self.out_channels, device=x.device)
            m.scatter_add_(0, edge_index_j.unsqueeze(-1).expand_as(m), m_ij)
            h = h + m

        if self.bias is not None:
            h = h + self.bias
        return h

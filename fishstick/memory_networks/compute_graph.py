"""
Compute Graph Memory Implementation.

Memory as differentiable graph and graph networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


class MemoryGraph(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        edge_dim: int = 16,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.nodes = nn.Parameter(torch.randn(num_nodes, node_dim))
        self.edges = nn.Parameter(torch.randn(num_nodes, num_nodes, edge_dim))

        nn.init.xavier_uniform_(self.nodes)
        nn.init.xavier_uniform_(self.edges)

    def get_node_features(
        self, node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if node_indices is None:
            return self.nodes
        return self.nodes[node_indices]

    def get_edge_features(
        self, source_indices: torch.Tensor, target_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.edges[source_indices, target_indices]

    def update_nodes(self, node_indices: torch.Tensor, updates: torch.Tensor):
        self.nodes.data[node_indices] = updates

    def update_edges(
        self,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        updates: torch.Tensor,
    ):
        self.edges.data[source_indices, target_indices] = updates


class GraphMessagePassing(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        self.message_networks = nn.ModuleList()
        for _ in range(num_layers):
            self.message_networks.append(
                nn.Sequential(
                    nn.Linear(node_dim + edge_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

        self.update_network = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def message(
        self,
        source_features: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([source_features, edge_features], dim=-1)
        messages = []
        for net in self.message_networks:
            messages.append(net(combined))
        return torch.stack(messages, dim=0).sum(dim=0)

    def aggregate(
        self,
        messages: torch.Tensor,
        num_neighbors: torch.Tensor,
    ) -> torch.Tensor:
        return messages / (num_neighbors.unsqueeze(-1) + 1e-8)

    def update(
        self,
        node_features: torch.Tensor,
        aggregated_messages: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([node_features, aggregated_messages], dim=-1)
        return self.update_network(combined)


class GraphMemoryCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        node_dim: int = 64,
        edge_dim: int = 16,
        hidden_dim: int = 128,
        num_nodes: int = 16,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        self.memory_graph = MemoryGraph(num_nodes, node_dim, edge_dim)

        self.message_passing = GraphMessagePassing(
            node_dim, edge_dim, hidden_dim, num_layers
        )

        self.node_query_net = nn.Linear(input_size, node_dim)
        self.edge_query_net = nn.Linear(input_size, edge_dim)

        self.controller = nn.LSTMCell(input_size + node_dim * num_nodes, hidden_dim)

        self.node_update_net = nn.Linear(hidden_dim, node_dim * num_nodes)
        self.edge_update_net = nn.Linear(hidden_dim, edge_dim * num_nodes * num_nodes)

        self.output_net = nn.Linear(hidden_dim + node_dim * num_nodes, output_size)

        self.h_state = None
        self.c_state = None

    def init_state(self, batch_size: int, device: torch.device):
        self.h_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.c_state = torch.zeros(batch_size, self.hidden_dim, device=device)

    def read_memory(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_query = self.node_query_net(query)
        edge_query = self.edge_query_net(query)

        node_similarity = torch.matmul(node_query, self.memory_graph.nodes.t())
        node_weights = F.softmax(node_similarity, dim=-1)

        node_read = torch.matmul(node_weights, self.memory_graph.nodes)

        edge_similarity = torch.matmul(
            edge_query.unsqueeze(1),
            self.memory_graph.edges.view(self.num_nodes, -1).t(),
        ).view(self.num_nodes, self.num_nodes)
        edge_weights = F.softmax(edge_similarity.view(-1), dim=-1).view(
            self.num_nodes, self.num_nodes
        )

        edge_read = torch.matmul(
            edge_weights.unsqueeze(-1), self.memory_graph.edges
        ).sum(dim=1)

        return node_read, edge_read

    def write_memory(self, node_updates: torch.Tensor, edge_updates: torch.Tensor):
        node_updates = node_updates.view(-1, self.num_nodes, self.node_dim)
        edge_updates = edge_updates.view(
            -1, self.num_nodes, self.num_nodes, self.edge_dim
        )

        new_nodes = self.memory_graph.nodes.unsqueeze(0) + node_updates
        new_edges = self.memory_graph.edges.unsqueeze(0) + edge_updates

        self.memory_graph.nodes.data = new_nodes.mean(dim=0)
        self.memory_graph.edges.data = new_edges.mean(dim=0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.h_state is None:
            self.init_state(x.size(0), x.device)

        node_read, edge_read = self.read_memory(x)

        controller_input = torch.cat([x, node_read.view(x.size(0), -1)], dim=-1)
        self.h_state, self.c_state = self.controller(
            controller_input, (self.h_state, self.c_state)
        )

        node_updates = self.node_update_net(self.h_state)
        edge_updates = self.edge_update_net(self.h_state)

        self.write_memory(node_updates, edge_updates)

        output_input = torch.cat([self.h_state, node_read.view(x.size(0), -1)], dim=-1)
        output = self.output_net(output_input)

        return output, (self.h_state, self.c_state)


class GraphMemoryNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        node_dim: int = 64,
        edge_dim: int = 16,
        hidden_dim: int = 128,
        num_nodes: int = 16,
        num_layers: int = 2,
    ):
        super().__init__()
        self.graph_cell = GraphMemoryCell(
            input_size=input_size,
            output_size=output_size,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.graph_cell(x)
        return output

    def reset(self):
        self.graph_cell.h_state = None
        self.graph_cell.c_state = None

    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        return {
            "nodes": self.graph_cell.memory_graph.nodes.data.clone(),
            "edges": self.graph_cell.memory_graph.edges.data.clone(),
        }

    def set_memory_state(self, state: Dict[str, torch.Tensor]):
        self.graph_cell.memory_graph.nodes.data = state["nodes"]
        self.graph_cell.memory_graph.edges.data = state["edges"]

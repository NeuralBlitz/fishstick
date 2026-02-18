"""
Graph Neural Network for Weather Modeling

Graph-based weather prediction models inspired by GraphCast and Pangu-Weather.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree


class GraphWeatherEncoder(nn.Module):
    """Encoder for graph-based weather data.

    Args:
        input_channels: Number of input variables per node
        hidden_dim: Hidden dimension size
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.node_encoder(x)


class GraphWeatherProcessor(nn.Module):
    """Graph message passing processor for weather.

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of message passing layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [GraphWeatherLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GraphWeatherLayer(MessagePassing):
    """Single graph weather layer with edge updates.

    Args:
        hidden_dim: Hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__(aggr="add")

        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.message_net(torch.cat([x_i, x_j], dim=-1))

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return self.update_net(torch.cat([x, aggr_out], dim=-1))


class MultiMeshWeatherModel(nn.Module):
    """Multi-mesh weather model with hierarchical graph structure.

    This model uses multiple graph resolutions (fine, medium, coarse)
    to capture both local and global weather patterns.

    Args:
        input_channels: Number of input variables
        hidden_dim: Hidden dimension
        num_fine_layers: Number of fine mesh layers
        num_coarse_layers: Number of coarse mesh layers
        forecast_horizon: Number of steps to forecast
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 128,
        num_fine_layers: int = 4,
        num_coarse_layers: int = 2,
        forecast_horizon: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        self.fine_encoder = GraphWeatherEncoder(input_channels, hidden_dim)

        self.fine_processor = GraphWeatherProcessor(
            hidden_dim, num_fine_layers, dropout
        )

        self.coarse_processor = GraphWeatherProcessor(
            hidden_dim, num_coarse_layers, dropout
        )

        self.mesh_transform = nn.Linear(hidden_dim * 2, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon * input_channels),
        )

    def forward(
        self,
        x: Tensor,
        fine_edges: Tensor,
        coarse_edges: Tensor,
    ) -> Tensor:
        x = self.fine_encoder(x)

        fine_out = self.fine_processor(x, fine_edges)

        coarse_out = self.coarse_processor(x, coarse_edges)

        combined = torch.cat([fine_out, coarse_out], dim=-1)
        combined = self.mesh_transform(combined)

        forecast = self.decoder(combined)

        return forecast


class SpatialGraphTransformer(nn.Module):
    """Spatial graph transformer for weather modeling.

    Args:
        input_channels: Number of input variables
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        forecast_horizon: Number of steps to forecast
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        forecast_horizon: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        self.input_proj = nn.Linear(input_channels, hidden_dim)

        self.pos_encoding = nn.Parameter(torch.randn(1, hidden_dim) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_horizon * input_channels),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.input_proj(x)

        x = x + self.pos_encoding

        x = self.transformer(x)

        forecast = self.temporal_proj(x)

        return forecast


def create_icosphere_grid(
    num_refinements: int = 3,
) -> Tuple[Tensor, Tensor]:
    """Create icosphere grid for global weather modeling.

    Args:
        num_refinements: Number of mesh refinements

    Returns:
        Tuple of (node_features, edge_index)
    """
    base_icosahedron = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            [0, 5, 1],
            [11, 6, 7],
            [11, 7, 8],
            [11, 8, 9],
            [11, 9, 10],
            [11, 10, 6],
            [1, 2, 7],
            [2, 3, 8],
            [3, 4, 9],
            [4, 5, 10],
            [5, 1, 6],
            [2, 7, 8],
            [3, 8, 9],
            [4, 9, 10],
            [5, 10, 6],
            [1, 6, 7],
        ]
    )

    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ]
    )
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    num_nodes = len(vertices)
    edges = set()
    for face in base_icosahedron:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edges.add(edge)
    edge_list = list(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    return torch.tensor(vertices, dtype=torch.float32), edge_index


def create_lat_lon_graph(
    lat_res: float,
    lon_res: float,
    connect_8_neighbors: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Create graph structure for latitude-longitude grid.

    Args:
        lat_res: Latitude resolution in degrees
        lon_res: Longitude resolution in degrees
        connect_8_neighbors: Whether to connect 8 neighbors (diagonal included)

    Returns:
        Tuple of (node_features, edge_index)
    """
    lats = torch.arange(-90, 90 + lat_res, lat_res)
    lons = torch.arange(0, 360, lon_res)

    num_lat = len(lats)
    num_lon = len(lons)
    num_nodes = num_lat * num_lon

    node_features = []
    for lat in lats:
        for lon in lons:
            lat_rad = lat * np.pi / 180
            lon_rad = lon * np.pi / 180
            node_features.append(
                [
                    np.cos(lat_rad) * np.cos(lon_rad),
                    np.cos(lat_rad) * np.sin(lon_rad),
                    np.sin(lat_rad),
                ]
            )
    x = torch.tensor(node_features, dtype=torch.float32)

    edges = []
    for i in range(num_nodes):
        lat_idx = i // num_lon
        lon_idx = i % num_lon

        neighbors = [
            (lat_idx + 1, lon_idx),
            (lat_idx - 1, lon_idx),
            (lat_idx, lon_idx + 1),
            (lat_idx, lon_idx - 1),
        ]

        if connect_8_neighbors:
            neighbors.extend(
                [
                    (lat_idx + 1, lon_idx + 1),
                    (lat_idx + 1, lon_idx - 1),
                    (lat_idx - 1, lon_idx + 1),
                    (lat_idx - 1, lon_idx - 1),
                ]
            )

        for n_lat, n_lon in neighbors:
            if 0 <= n_lat < num_lat and 0 <= n_lon < num_lon:
                j = n_lat * num_lon + n_lon
                edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    return x, edge_index


class GraphCastModel(nn.Module):
    """GraphCast-style weather prediction model.

    Args:
        input_channels: Number of input variables (e.g., pressure, temperature, wind)
        hidden_dim: Hidden dimension
        output_channels: Number of output variables
        num_layers: Number of graph layers
        edge_dim: Edge feature dimension
    """

    def __init__(
        self,
        input_channels: int = 70,
        hidden_dim: int = 256,
        output_channels: int = 70,
        num_layers: int = 16,
        edge_dim: int = 4,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels

        self.input_encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList(
            [GraphCastLayer(hidden_dim, edge_dim) for _ in range(num_layers)]
        )

        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_channels),
        )

        self.residual = nn.Linear(input_channels, output_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        identity = self.residual(x)

        h = self.input_encoder(x)

        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        out = self.output_decoder(h)

        return out + identity


class GraphCastLayer(nn.Module):
    """Single GraphCast layer with edge features.

    Args:
        hidden_dim: Hidden dimension
        edge_dim: Edge feature dimension
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        edge_dim: int = 4,
    ):
        super().__init__()

        self.sender_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.receiver_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.activation = nn.SiLU()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        src, dst = edge_index

        sender_out = self.sender_mlp(torch.cat([x[src], edge_attr], dim=-1))

        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, sender_out)

        num_neighbors = degree(dst, x.size(0)).clamp(min=1).unsqueeze(-1)
        aggregated = aggregated / num_neighbors

        combined = torch.cat([x, aggregated, edge_attr], dim=-1)
        updated = self.receiver_mlp(combined)

        x = x + self.activation(updated)
        x = self.norm1(x)

        return x

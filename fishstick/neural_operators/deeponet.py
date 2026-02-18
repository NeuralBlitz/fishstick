"""
DeepONet (Deep Operator Network) Implementations.

DeepONet learns operators between infinite-dimensional function spaces.
Based on: DeepONet: Learning nonlinear operators for identifying differential
equations (Lu et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union


class BranchNet(nn.Module):
    """Branch network for DeepONet - encodes input functions."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 128,
        num_layers: int = 4,
        hidden_dim: int = 128,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        layers = []
        layers.append(nn.Conv1d(input_channels, hidden_dim, 1))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, 1))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())

        layers.append(nn.Conv1d(hidden_dim, output_channels, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TrunkNet(nn.Module):
    """Trunk network for DeepONet - encodes query locations."""

    def __init__(
        self,
        input_dim: int,
        output_channels: int = 128,
        num_layers: int = 4,
        hidden_dim: int = 128,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, output_channels))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepONet(nn.Module):
    """Deep Operator Network for learning operators between function spaces."""

    def __init__(
        self,
        branch_channels: int,
        trunk_dim: int,
        output_dim: int = 1,
        branch_hidden: int = 128,
        trunk_hidden: int = 128,
        num_branch_layers: int = 4,
        num_trunk_layers: int = 4,
        activation: str = "relu",
    ):
        super().__init__()
        self.branch_channels = branch_channels
        self.trunk_dim = trunk_dim
        self.output_dim = output_dim

        self.branch_net = BranchNet(
            input_channels=branch_channels,
            output_channels=branch_hidden,
            num_layers=num_branch_layers,
            hidden_dim=branch_hidden,
            activation=activation,
        )
        self.trunk_net = TrunkNet(
            input_dim=trunk_dim,
            output_channels=trunk_hidden,
            num_layers=num_trunk_layers,
            hidden_dim=trunk_hidden,
            activation=activation,
        )

        self.output_bias = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        branch_output = branch_output.mean(dim=-1)
        trunk_output = trunk_output.unsqueeze(-1)

        output = torch.einsum("bo,bt->bt", branch_output, trunk_output)
        output = output.squeeze(-1) + self.output_bias

        return output


class DeepONetCartesian(DeepONet):
    """DeepONet with Cartesian product for multiple outputs."""

    def __init__(
        self,
        branch_channels: int,
        trunk_dim: int,
        output_dim: int = 1,
        branch_hidden: int = 128,
        trunk_hidden: int = 128,
        num_branch_layers: int = 4,
        num_trunk_layers: int = 4,
        activation: str = "relu",
    ):
        super().__init__(
            branch_channels=branch_channels,
            trunk_dim=trunk_dim,
            output_dim=output_dim,
            branch_hidden=branch_hidden,
            trunk_hidden=trunk_hidden,
            num_branch_layers=num_branch_layers,
            num_trunk_layers=num_trunk_layers,
            activation=activation,
        )

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        branch_output = branch_output.mean(dim=-1)
        output = torch.matmul(trunk_output, branch_output)

        return output


class DeepONetDistributed(nn.Module):
    """DeepONet with distributed sensor encoding."""

    def __init__(
        self,
        branch_channels: int,
        trunk_dim: int,
        output_dim: int = 1,
        num_sensors: int = 100,
        branch_hidden: int = 128,
        trunk_hidden: int = 128,
        num_branch_layers: int = 4,
        num_trunk_layers: int = 4,
    ):
        super().__init__()
        self.branch_channels = branch_channels
        self.trunk_dim = trunk_dim
        self.output_dim = output_dim
        self.num_sensors = num_sensors

        self.sensor_encoder = nn.Linear(branch_channels, branch_hidden)

        self.branch_net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(branch_hidden, branch_hidden),
                    nn.ReLU(),
                )
                for _ in range(num_branch_layers)
            ]
        )

        self.trunk_net = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(trunk_dim, trunk_hidden),
                    nn.ReLU(),
                )
                for _ in range(num_trunk_layers)
            ]
        )

        self.output_layer = nn.Linear(branch_hidden * trunk_hidden, output_dim)

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        x = self.sensor_encoder(branch_input)
        for layer in self.branch_net:
            x = layer(x)
        branch_output = x

        y = trunk_input
        for layer in self.trunk_net:
            y = layer(y)
        trunk_output = y

        combined = torch.einsum("bi,bj->bij", branch_output, trunk_output)
        combined = combined.view(combined.size(0), -1)

        output = self.output_layer(combined)
        return output


class NodewiseDeepONet(nn.Module):
    """Node-wise DeepONet for learning node-level operators."""

    def __init__(
        self,
        node_features_dim: int,
        node_coords_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 4,
        activation: str = "relu",
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.node_coords_dim = node_coords_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.feature_encoder = nn.Linear(node_features_dim, hidden_dim)
        self.coord_encoder = nn.Linear(node_coords_dim, hidden_dim)

        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())

        self.fusion_mlp = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        node_coords: torch.Tensor,
    ) -> torch.Tensor:
        feat_emb = self.feature_encoder(node_features)
        coord_emb = self.coord_encoder(node_coords)
        combined = feat_emb * coord_emb
        fused = self.fusion_mlp(combined)
        output = self.output_head(fused)
        return output


class DeepONetEnsemble(nn.Module):
    """Ensemble of DeepONets for improved robustness."""

    def __init__(
        self,
        num_models: int = 3,
        branch_channels: int = 1,
        trunk_dim: int = 1,
        output_dim: int = 1,
        branch_hidden: int = 128,
        trunk_hidden: int = 128,
    ):
        super().__init__()
        self.num_models = num_models

        self.models = nn.ModuleList(
            [
                DeepONet(
                    branch_channels=branch_channels,
                    trunk_dim=trunk_dim,
                    output_dim=output_dim,
                    branch_hidden=branch_hidden,
                    trunk_hidden=trunk_hidden,
                )
                for _ in range(num_models)
            ]
        )

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        outputs = [model(branch_input, trunk_input) for model in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)


class ConvolutionalDeepONet(nn.Module):
    """DeepONet with convolutional branch for spatial encoding."""

    def __init__(
        self,
        branch_channels: int,
        branch_sensor_dim: int,
        trunk_input_dim: int,
        output_dim: int = 1,
        conv_channels: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        trunk_hidden: int = 128,
    ):
        super().__init__()

        conv_layers = []
        in_ch = branch_channels
        for out_ch in conv_channels:
            conv_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
            )
            conv_layers.append(nn.ReLU())
            in_ch = out_ch

        self.conv_encoder = nn.Sequential(*conv_layers)

        conv_out_dim = conv_channels[-1] * branch_sensor_dim
        self.branch_proj = nn.Linear(conv_out_dim, trunk_hidden)

        self.trunk_net = TrunkNet(
            input_dim=trunk_input_dim,
            output_channels=trunk_hidden,
            hidden_dim=trunk_hidden,
        )

        self.output_proj = nn.Linear(trunk_hidden, output_dim)

    def forward(
        self,
        sensor_data: torch.Tensor,
        query_locations: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sensor_data.size(0)

        x = sensor_data.transpose(1, 2)
        x = self.conv_encoder(x)
        x = x.reshape(batch_size, -1)
        branch_feat = self.branch_proj(x)

        trunk_feat = self.trunk_net(query_locations)
        combined = branch_feat.unsqueeze(1) * trunk_feat

        output = self.output_proj(combined)
        return output


class AttentionDeepONet(nn.Module):
    """DeepONet with attention mechanism for sensor weighting."""

    def __init__(
        self,
        branch_channels: int,
        branch_sensor_dim: int,
        trunk_input_dim: int,
        output_dim: int = 1,
        branch_hidden: int = 128,
        trunk_hidden: int = 128,
        attention_heads: int = 4,
    ):
        super().__init__()

        self.sensor_encoder = nn.Linear(branch_channels, branch_hidden)

        self.attention = nn.MultiheadAttention(
            embed_dim=branch_hidden,
            num_heads=attention_heads,
            batch_first=True,
        )

        self.branch_proj = nn.Linear(branch_hidden, branch_hidden)

        self.trunk_net = TrunkNet(
            input_dim=trunk_input_dim,
            output_channels=trunk_hidden,
            hidden_dim=trunk_hidden,
        )

        self.output_proj = nn.Linear(branch_hidden, output_dim)

    def forward(
        self,
        sensor_data: torch.Tensor,
        query_locations: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = sensor_data.size(0)

        x = self.sensor_encoder(sensor_data)

        attended, _ = self.attention(x, x, x)

        branch_feat = self.branch_proj(attended.mean(dim=1))

        trunk_feat = self.trunk_net(query_locations)

        combined = branch_feat.unsqueeze(1) * trunk_feat

        output = self.output_proj(combined)

        return output


__all__ = [
    "BranchNet",
    "TrunkNet",
    "DeepONet",
    "DeepONetCartesian",
    "DeepONetDistributed",
    "NodewiseDeepONet",
    "DeepONetEnsemble",
    "ConvolutionalDeepONet",
    "AttentionDeepONet",
]

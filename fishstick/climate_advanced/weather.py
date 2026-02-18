from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm, Dropout, MultiheadAttention
import math


class GraphCastEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = Linear(input_dim, hidden_dim)

        self.graph_layers = nn.ModuleList(
            [GraphCastLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.layer_norms = nn.ModuleList(
            [LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        x = self.input_proj(x)

        for layer, norm in zip(self.graph_layers, self.layer_norms):
            x = norm(layer(x, edge_index) + x)

        return x


class GraphCastLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.fc1 = Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = Linear(hidden_dim * 4, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        x2 = self.self_attn(x, x, x)[0]
        x = self.norm1(x + self.dropout(x2))

        x2 = self.fc2(self.activation(self.fc1(x)))
        x = self.norm2(x + self.dropout(x2))

        return x


class GraphCastDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.graph_layers = nn.ModuleList(
            [GraphCastLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.layer_norms = nn.ModuleList(
            [LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.output_proj = Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        for layer, norm in zip(self.graph_layers, self.layer_norms):
            x = norm(layer(x, edge_index) + x)

        return self.output_proj(x)


class GraphCast(nn.Module):
    def __init__(
        self,
        input_variables: int = 69,
        pressure_levels: int = 13,
        hidden_dim: int = 256,
        output_variables: int = 69,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_variables = input_variables
        self.pressure_levels = pressure_levels
        self.hidden_dim = hidden_dim
        self.output_variables = output_variables

        input_dim = input_variables * pressure_levels

        self.encoder = GraphCastEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.decoder = GraphCastDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_variables * pressure_levels,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.cast_to_3d = CastTo3D(hidden_dim)

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        batch_size = x.size(0)

        x = x.view(batch_size, -1, self.hidden_dim)

        x = self.encoder(x, edge_index)
        x = self.cast_to_3d(x)

        x = self.decoder(x, edge_index)

        return x


class CastTo3D(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        return x


class PanguArtTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PanguArtEncoderLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(hidden_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class PanguArtEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fc = nn.Sequential(
            Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x2 = self.self_attn(x, x, x, attn_mask=mask)[0]
        x = self.norm1(x + self.dropout(x2))

        x2 = self.fc(x)
        x = self.norm2(x + self.dropout(x2))

        return x


class PanguArt(nn.Module):
    def __init__(
        self,
        input_channels: int = 69,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 16,
        output_channels: int = 69,
        forecast_steps: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps

        self.input_embedding = nn.Conv2d(input_channels, hidden_dim, 1)

        self.transformer = PanguArtTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.temporal_embedding = nn.Parameter(
            torch.randn(1, forecast_steps, hidden_dim)
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, height, width = x.shape

        x = self.input_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.transformer(x)

        temporal_emb = self.temporal_embedding.expand(batch_size, -1, -1)
        x = x[:, -1:, :] + temporal_emb[:, :1, :]

        x = self.output_head(x)

        x = x.transpose(1, 2).view(batch_size, -1, height, width)

        return x


class FourCastNetUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
    ):
        super().__init__()
        self.encoder1 = self._conv_block(in_channels, base_channels)
        self.encoder2 = self._conv_block(base_channels, base_channels * 2)
        self.encoder3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self._conv_block(base_channels * 4, base_channels * 8)

        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        self.decoder4 = self._conv_block(base_channels * 16, base_channels * 8)
        self.decoder3 = self._conv_block(base_channels * 8, base_channels * 4)
        self.decoder2 = self._conv_block(base_channels * 4, base_channels * 2)
        self.decoder1 = self._conv_block(base_channels * 2, base_channels)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)

        self.final = nn.Conv2d(base_channels, out_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(self.up4(b))
        d4 = torch.cat([d4, e4], dim=1)

        d3 = self.decoder3(self.up3(d4))
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.decoder2(self.up2(d3))
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.decoder1(self.up1(d2))
        d1 = torch.cat([d1, e1], dim=1)

        return self.final(d1)


class FourCastNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 69,
        output_channels: int = 69,
        base_channels: int = 128,
        num_iterations: int = 4,
    ):
        super().__init__()
        self.num_iterations = num_iterations

        self.backbone = FourCastNetUNet(
            in_channels=input_channels,
            out_channels=output_channels,
            base_channels=base_channels,
        )

        self.fourier_features = FourierFeatureMapping(base_channels * 2, 256)

        self.learned_iterations = nn.ModuleList(
            [
                nn.Conv2d(output_channels, output_channels, 1)
                for _ in range(num_iterations - 1)
            ]
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        outputs = []

        coords = self._get_coordinates(x)
        x = torch.cat([x, coords], dim=1)

        x = self.backbone(x)
        outputs.append(x)

        for i in range(self.num_iterations - 1):
            residual = self.learned_iterations[i](x)
            x = x + residual
            outputs.append(x)

        return outputs

    def _get_coordinates(self, x: Tensor) -> Tensor:
        batch_size, _, height, width = x.shape

        y_coords = torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype)
        x_coords = torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype)

        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

        coords = (
            torch.stack([y_grid, x_grid], dim=0)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
        )

        return coords


class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim: int, mapping_size: int):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn(input_dim, mapping_size) * 0.1, requires_grad=False
        )
        self.mapping_size = mapping_size

    def forward(self, x: Tensor) -> Tensor:
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


__all__ = [
    "GraphCast",
    "PanguArt",
    "FourCastNet",
    "GraphCastEncoder",
    "GraphCastDecoder",
    "PanguArtTransformer",
    "FourCastNetUNet",
]

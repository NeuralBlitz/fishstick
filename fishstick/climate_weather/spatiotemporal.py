"""
Spatio-Temporal Climate Modeling

Convolutional and recurrent models for spatio-temporal climate data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatioTemporalEncoder(nn.Module):
    """Encoder for spatio-temporal climate data.

    Args:
        input_channels: Number of input variables
        hidden_dim: Hidden dimension
        temporal_kernel: Kernel size for temporal convolution
    """

    def __init__(
        self,
        input_channels: int = 5,
        hidden_dim: int = 128,
        temporal_kernel: int = 3,
    ):
        super().__init__()

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.temporal_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=temporal_kernel,
            padding=temporal_kernel // 2,
        )

        self.temporal_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)
        x = self.spatial_conv(x)
        x = x.reshape(B, T, -1, H, W)
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(B * H * W, T, -1)
        x = x.permute(0, 2, 1)

        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)
        x = self.temporal_norm(x)

        x = x.reshape(B, H, W, T, -1)
        x = x.permute(0, 3, 4, 1, 2)

        return x


class TemporalConvolutionalNetwork(nn.Module):
    """Temporal Convolutional Network for climate time series.

    Args:
        input_channels: Number of input channels
        hidden_channels: Number of hidden channels
        num_layers: Number of TCN layers
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            dilation = 2**i

            padding = (kernel_size - 1) * dilation // 2

            layers.append(
                nn.Conv1d(
                    in_ch,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class ConvGRUClimate(nn.Module):
    """Convolutional GRU for climate modeling.

    Args:
        input_channels: Number of input channels
        hidden_channels: Number of hidden channels
        kernel_size: Convolution kernel size
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.update_gate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.candidate_gate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        if h is None:
            h = torch.zeros_like(x[:, : self.reset_gate.out_channels])

        combined = torch.cat([x, h], dim=1)

        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))

        candidate = torch.tanh(self.candidate_gate(torch.cat([x, reset * h], dim=1)))

        h_new = (1 - update) * h + update * candidate

        return h_new


class ConvLSTMClimate(nn.Module):
    """Convolutional LSTM for climate modeling.

    Args:
        input_channels: Number of input channels
        hidden_channels: Number of hidden channels
        kernel_size: Convolution kernel size
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.hidden_channels = hidden_channels

    def forward(
        self, x: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        if state is None:
            h = torch.zeros(
                x.size(0), self.hidden_channels, x.size(2), x.size(3), device=x.device
            )
            c = torch.zeros_like(h)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class AttentionSpatioTemporal(nn.Module):
    """Attention-based spatio-temporal model.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        spatial_attention: Whether to use spatial attention
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        spatial_attention: bool = True,
    ):
        super().__init__()

        self.spatial_attention = spatial_attention

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        if spatial_attention:
            self.spatial_attention_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True,
            )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape

        x_flat = x.permute(0, 2, 3, 1, 4)
        x_flat = x_flat.reshape(B * H * W, T, C)

        attn_out, _ = self.temporal_attention(x_flat, x_flat, x_flat)
        x = x_flat + attn_out
        x = self.norm1(x)

        ff_out = self.feed_forward(x)
        x = x + ff_out
        x = self.norm2(x)

        if self.spatial_attention:
            x_reshaped = x.reshape(B, H, W, T, C)
            x_reshaped = x_reshaped.permute(0, 3, 4, 1, 2)
            x_reshaped = x_reshaped.reshape(B * T * C, H, W)

            x_spatial = x_reshaped.permute(0, 2, 1)

            spatial_attn, _ = self.spatial_attention_layer(
                x_spatial, x_spatial, x_spatial
            )
            x_reshaped = x_reshaped + spatial_attn.permute(0, 2, 1)

        return x


class UNetClimate(nn.Module):
    """U-Net for climate downscaling and nowcasting.

    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        base_channels: Number of base channels
        depth: U-Net depth
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()

        self.depth = depth

        encoders = []
        decoders = []
        skip_connections = []

        ch = base_channels
        for i in range(depth):
            encoders.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_channels if i == 0 else ch // 2,
                        ch,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(),
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(),
                )
            )
            skip_connections.append(nn.Conv2d(ch, ch, kernel_size=1))
            ch *= 2

        for i in range(depth):
            decoders.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch // 2),
                    nn.ReLU(),
                    nn.Conv2d(ch // 2, ch // 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch // 2),
                    nn.ReLU(),
                )
            )
            ch //= 2

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        self.skip_connections = nn.ModuleList(skip_connections)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.output_conv = nn.Conv2d(base_channels, output_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C, H, W = x.shape
        x = x[:, -1]

        skip_features = []

        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < self.depth - 1:
                skip_features.append(self.skip_connections[i](x))
                x = self.pool(x)

        for i, decoder in enumerate(self.decoders):
            x = self.upsample(x)
            if i < len(skip_features):
                skip = skip_features[-(i + 1)]
                if x.shape != skip.shape:
                    x = nn.functional.interpolate(
                        x, size=skip.shape[-2:], mode="bilinear", align_corners=True
                    )
                x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return self.output_conv(x).unsqueeze(1)


class LatentClimateModel(nn.Module):
    """Latent variable model for climate.

    Args:
        input_channels: Number of input channels
        latent_dim: Dimension of latent space
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        input_channels: int = 5,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels, hidden_dim // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim * 2, kernel_size=4, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim // 2, input_channels, kernel_size=4, stride=2, padding=1
            ),
        )

        self.latent_dim = latent_dim

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        params = self.encoder(x)
        mu, logvar = params.chunk(2, dim=1)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


class ClimateDynamicsPredictor(nn.Module):
    """Neural network for learning climate dynamics.

    Args:
        input_channels: Number of input state variables
        hidden_channels: Number of hidden channels
        output_channels: Number of output variables (typically same as input)
    """

    def __init__(
        self,
        input_channels: int = 5,
        hidden_channels: int = 128,
        output_channels: int = 5,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def predict_step(self, x: Tensor, dt: float = 1.0) -> Tensor:
        dxdt = self.forward(x)
        return x + dxdt * dt

"""
Fourier Neural Operator Implementations.

This module provides comprehensive Fourier Neural Operator (FNO) architectures
for learning operators between function spaces in the Fourier domain.

Based on: Fourier Neural Operator for Parametric Partial Differential Equations
(Li et al., NeurIPS 2020)
"""

from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm, Dropout
import math


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution layer for FNO."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = Parameter(torch.Tensor(in_channels, out_channels, modes))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass in Fourier domain."""
        batch_size = x.size(0)
        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(
            batch_size, self.out_channels, x_ft.size(-1), device=x.device, dtype=x.dtype
        )

        modes_selected = min(self.modes, x_ft.size(-1))
        for i in range(modes_selected):
            out_ft[:, :, i] = torch.einsum(
                "bi,io->bo", x_ft[:, :, i], self.weights[:, :, i]
            )

        x_out = torch.fft.irfft(out_ft, dim=-1, n=x.size(-1))

        if self.bias is not None:
            x_out = x_out + self.bias.view(1, -1, 1)

        return x_out


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution layer for FNO."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = Parameter(
            torch.Tensor(in_channels, out_channels, modes1, modes2)
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass in 2D Fourier domain."""
        batch_size = x.size(0)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.size(-2),
            x_ft.size(-1),
            device=x.device,
            dtype=x.dtype,
        )

        modes1_sel = min(self.modes1, x_ft.size(-2))
        modes2_sel = min(self.modes2, x_ft.size(-1))

        for i in range(modes1_sel):
            for j in range(modes2_sel):
                out_ft[:, :, i, j] = torch.einsum(
                    "bij,io->boj", x_ft[:, :, i, j], self.weights[:, :, i, j]
                )

        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))

        if self.bias is not None:
            x_out = x_out + self.bias.view(1, -1, 1, 1)

        return x_out


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution layer for FNO."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = Parameter(
            torch.Tensor(in_channels, out_channels, modes1, modes2, modes3)
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass in 3D Fourier domain."""
        batch_size = x.size(0)
        x_ft = torch.fft.rfft3(x, dim=(-3, -2, -1))

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.size(-3),
            x_ft.size(-2),
            x_ft.size(-1),
            device=x.device,
            dtype=x.dtype,
        )

        m1, m2, m3 = (
            min(self.modes1, x_ft.size(-3)),
            min(self.modes2, x_ft.size(-2)),
            min(self.modes3, x_ft.size(-1)),
        )

        for i in range(m1):
            for j in range(m2):
                for k in range(m3):
                    out_ft[:, :, i, j, k] = torch.einsum(
                        "bijk,io->bojk",
                        x_ft[:, :, i, j, k],
                        self.weights[:, :, i, j, k],
                    )

        x_out = torch.fft.irfft3(
            out_ft, s=(x.size(-3), x.size(-2), x.size(-1)), dim=(-3, -2, -1)
        )

        if self.bias is not None:
            x_out = x_out + self.bias.view(1, -1, 1, 1, 1)

        return x_out


class FNO1d(nn.Module):
    """Fourier Neural Operator for 1D domains."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.num_layers = num_layers

        self.input_encoder = nn.Conv1d(in_channels, width, 1)
        self.output_decoder = nn.Conv1d(width, out_channels, 1)

        self.spectral_convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(num_layers)]
        )
        self.mlps = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(num_layers)]
        )
        self_norms = nn.ModuleList([LayerNorm(width) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([Dropout(dropout) for _ in range(num_layers)])

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_encoder(x)
        for i in range(self.num_layers):
            x_spectral = self.spectral_convs[i](x)
            x_mlp = self.mlps[i](x)
            x = self.activation(self_norms[i](x_spectral + x_mlp))
            x = self.dropouts[i](x)
        x = self.output_decoder(x)
        return x


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D domains."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 12,
        width: int = 64,
        num_layers: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.num_layers = num_layers

        self.input_encoder = nn.Conv2d(in_channels, width, 1)
        self.output_decoder = nn.Conv2d(width, out_channels, 1)

        self.spectral_convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(num_layers)]
        )
        self.mlps = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(num_layers)]
        )
        self_norms = nn.ModuleList([LayerNorm(width) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([Dropout(dropout) for _ in range(num_layers)])

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_encoder(x)
        for i in range(self.num_layers):
            x_spectral = self.spectral_convs[i](x)
            x_mlp = self.mlps[i](x)
            x = self.activation(self_norms[i](x_spectral + x_mlp))
            x = self.dropouts[i](x)
        x = self.output_decoder(x)
        return x


class FNO3d(nn.Module):
    """Fourier Neural Operator for 3D domains."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 8,
        width: int = 32,
        num_layers: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.num_layers = num_layers

        self.input_encoder = nn.Conv3d(in_channels, width, 1)
        self.output_decoder = nn.Conv3d(width, out_channels, 1)

        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv3d(width, width, modes, modes, modes)
                for _ in range(num_layers)
            ]
        )
        self.mlps = nn.ModuleList(
            [nn.Conv3d(width, width, 1) for _ in range(num_layers)]
        )
        self_norms = nn.ModuleList([LayerNorm(width) for _ in range(num_layers)])
        self.dropouts = nn.ModuleList([Dropout(dropout) for _ in range(num_layers)])

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_encoder(x)
        for i in range(self.num_layers):
            x_spectral = self.spectral_convs[i](x)
            x_mlp = self.mlps[i](x)
            x = self.activation(self_norms[i](x_spectral + x_mlp))
            x = self.dropouts[i](x)
        x = self.output_decoder(x)
        return x


class AdaptiveFNO2d(nn.Module):
    """Adaptive Fourier Neural Operator with learnable mode selection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_modes: int = 16,
        width: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes = max_modes
        self.width = width

        self.mode_weights = nn.Parameter(torch.ones(max_modes, max_modes))

        self.input_encoder = nn.Conv2d(in_channels, width, 1)
        self.output_decoder = nn.Conv2d(width, out_channels, 1)

        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv2d(width, width, max_modes, max_modes)
                for _ in range(num_layers)
            ]
        )
        self.mlps = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(num_layers)]
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        x = self.input_encoder(x)
        b, c, h, w = x.size()

        mode_weight = torch.sigmoid(self.mode_weights)

        for i in range(len(self.spectral_convs)):
            x_ft = torch.fft.rfft2(x, dim=(-2, -1))
            x_ft = x_ft * mode_weight.view(1, 1, self.max_modes, self.max_modes)
            x = torch.fft.irfft2(x_ft, s=(h, w), dim=(-2, -1))

            x_spectral = self.spectral_convs[i](x)
            x_mlp = self.mlps[i](x)
            x = self.activation(x_spectral + x_mlp)

        x = self.output_decoder(x)
        return x


__all__ = [
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralConv3d",
    "FNO1d",
    "FNO2d",
    "FNO3d",
    "AdaptiveFNO2d",
]

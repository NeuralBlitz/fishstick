"""
Multi-scale Neural Operator Implementations.

Multi-scale methods for operator learning including wavelet transforms,
multi-grid methods, and hierarchical operator architectures.

Based on: Multiwavelet-based Operator Learning (Wu et al., 2023)
and Multi-scale DeepONet architectures.
"""

from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, LayerNorm


class WaveletTransform1D(nn.Module):
    """1D Wavelet transform for multi-scale decomposition."""

    def __init__(
        self,
        wavelet: str = "haar",
        levels: int = 3,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels

        self._build_wavelet_filters()

    def _build_wavelet_filters(self) -> None:
        wavelet_filters = {
            "haar": ([0.7071, 0.7071], [-0.7071, 0.7071]),
            "db4": (
                [0.2304, 0.7148, 0.6309, -0.0280],
                [-0.0280, 0.6309, -0.7148, 0.2304],
            ),
        }

        if self.wavelet not in wavelet_filters:
            self.wavelet = "haar"

        self.register_buffer(
            "low_pass",
            torch.tensor(wavelet_filters[self.wavelet][0]),
        )
        self.register_buffer(
            "high_pass",
            torch.tensor(wavelet_filters[self.wavelet][1]),
        )

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        """Decompose signal into approximation and detail coefficients.

        Returns:
            List of detail coefficients at each level, approximation coefficients
        """
        details = []
        current = x

        for _ in range(self.levels):
            if current.size(-1) < 2:
                break

            padded = F.pad(current, (1, 1), mode="replicate")

            approx = F.conv1d(
                padded.unsqueeze(0),
                self.low_pass.view(1, 1, -1),
                stride=2,
            ).squeeze(0)

            detail = F.conv1d(
                padded.unsqueeze(0),
                self.high_pass.view(1, 1, -1),
                stride=2,
            ).squeeze(0)

            details.append(detail)
            current = approx

        return details, current


class WaveletNeuralOperator(nn.Module):
    """Wavelet-based Neural Operator for multi-resolution operator learning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wavelet_levels: int = 3,
        wavelet: str = "haar",
        hidden_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet_levels = wavelet_levels
        self.hidden_dim = hidden_dim

        self.wavelet_transform = WaveletTransform1D(wavelet, wavelet_levels)

        self.encoder = Linear(in_channels, hidden_dim)

        self.scale_encoders = nn.ModuleList(
            [Linear(hidden_dim, hidden_dim) for _ in range(wavelet_levels + 1)]
        )

        self.processors = nn.ModuleList(
            [
                nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    Linear(hidden_dim, hidden_dim),
                )
                for _ in range(wavelet_levels + 1)
            ]
        )

        self.decoder = Linear(hidden_dim * (wavelet_levels + 1), out_channels)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        details, approx = self.wavelet_transform(x)

        encoded = self.encoder(x)

        scale_features = []
        for i, detail in enumerate(details):
            feat = self.scale_encoders[i](encoded)
            processed = self.processors[i](feat)
            scale_features.append(
                processed.mean(dim=-1, keepdim=True).expand(-1, x.size(-1), -1)
            )

        approx_feat = self.scale_encoders[-1](encoded)
        approx_processed = self.processors[-1](approx_feat)
        scale_features.append(approx_processed)

        combined = torch.cat(scale_features, dim=-1)
        output = self.decoder(combined)

        return output


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for scale interaction."""

    def __init__(
        self,
        channels: int,
        num_scales: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales
        self.num_heads = num_heads

        self.scale_proj = Linear(channels, channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

        self.output_proj = Linear(channels, channels)

    def forward(self, scale_features: List[Tensor]) -> Tensor:
        """Apply multi-scale attention across different scales.

        Args:
            scale_features: List of tensors, one per scale

        Returns:
            Attended features
        """
        projected = [self.scale_proj(s) for s in scale_features]

        max_len = max(s.size(1) for s in projected)
        padded = []
        for s in projected:
            if s.size(1) < max_len:
                pad_len = max_len - s.size(1)
                s = F.pad(s, (0, 0, 0, pad_len))
            padded.append(s)

        stacked = torch.stack(padded, dim=1)

        attended, _ = self.attention(stacked, stacked, stacked)

        output = self.output_proj(attended.mean(dim=1))
        return output


class MultigridNeuralOperator(nn.Module):
    """Multi-grid neural operator with hierarchical representation learning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_levels: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels

        self.encoder = Linear(in_channels, hidden_dim)

        self.level_operators = nn.ModuleList(
            [self._make_level_operator(hidden_dim) for _ in range(num_levels)]
        )

        self.restriction_ops = nn.ModuleList(
            [Linear(hidden_dim, hidden_dim // 2) for _ in range(num_levels - 1)]
        )

        self.prolongation_ops = nn.ModuleList(
            [Linear(hidden_dim // 2, hidden_dim) for _ in range(num_levels - 1)]
        )

        self.decoder = Linear(hidden_dim, out_channels)

    def _make_level_operator(self, hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)

        level_outputs = [self.level_operators[0](encoded)]

        for i in range(self.num_levels - 1):
            restricted = self.restriction_ops[i](level_outputs[-1])
            processed = self.level_operators[i + 1](restricted)
            level_outputs.append(processed)

        for i in range(self.num_levels - 2, -1, -1):
            prolonged = self.prolongation_ops[i](level_outputs[i + 1])
            if prolonged.size(1) < level_outputs[i].size(1):
                pad_len = level_outputs[i].size(1) - prolonged.size(1)
                prolonged = F.pad(prolonged, (0, 0, 0, pad_len))
            level_outputs[i] = level_outputs[i] + prolonged

        output = self.decoder(level_outputs[0])
        return output


class HierarchicalOperatorBlock(nn.Module):
    """Hierarchical operator block with scale separation."""

    def __init__(
        self,
        channels: int,
        num_scales: int = 3,
    ):
        super().__init__()
        self.channels = channels
        self.num_scales = num_scales

        self.downsample = nn.Conv1d(
            channels, channels, kernel_size=3, stride=2, padding=1
        )
        self.upsample = nn.ConvTranspose1d(
            channels, channels, kernel_size=3, stride=2, padding=1
        )

        self.process = nn.Sequential(
            Linear(channels, channels * 4),
            nn.GELU(),
            Linear(channels * 4, channels),
        )

        self.norm = LayerNorm(channels)

    def forward(self, x: Tensor, scale_idx: int = 0) -> Tensor:
        if scale_idx < self.num_scales - 1:
            downsampled = self.downsample(x.transpose(1, 2)).transpose(1, 2)
            processed = self.process(self.norm(downsampled))
            upsampled = self.upsample(processed.transpose(1, 2)).transpose(1, 2)

            if upsampled.size(1) < x.size(1):
                upsampled = F.pad(upsampled, (0, 0, 0, x.size(1) - upsampled.size(1)))
            elif upsampled.size(1) > x.size(1):
                upsampled = upsampled[:, : x.size(1), :]

            x = x + upsampled
        else:
            x = x + self.process(self.norm(x))

        return x


class AdaptiveScaleOperator(nn.Module):
    """Neural operator with learnable scale selection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_scales: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_scales = num_scales

        self.scale_weights = Parameter(torch.ones(num_scales))

        self.scale_encoders = nn.ModuleList(
            [Linear(in_channels, hidden_dim) for _ in range(num_scales)]
        )

        self.scale_processors = nn.ModuleList(
            [
                nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_scales)
            ]
        )

        self.decoder = Linear(hidden_dim * num_scales, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        scale_weights = F.softmax(self.scale_weights, dim=0)

        scale_features = []
        for i in range(self.num_scales):
            encoded = self.scale_encoders[i](x)
            processed = self.scale_processors[i](encoded)
            scale_features.append(processed * scale_weights[i])

        combined = torch.cat(scale_features, dim=-1)
        output = self.decoder(combined)

        return output


class ScaleSeparationModule(nn.Module):
    """Module for separating and processing different frequency components."""

    def __init__(
        self,
        channels: int,
        num_bands: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands

        self.band_filters = nn.ModuleList(
            [
                nn.Sequential(
                    Linear(channels, channels),
                    nn.GELU(),
                )
                for _ in range(num_bands)
            ]
        )

        self.band_weights = Parameter(torch.ones(num_bands))

    def forward(self, x: Tensor) -> Tensor:
        x_ft = torch.fft.rfft(x, dim=1)

        freq = torch.abs(x_ft)
        freqs = freq.mean(dim=-1, keepdim=True)

        band_size = x_ft.size(1) // self.num_bands
        band_features = []

        for i in range(self.num_bands):
            start = i * band_size
            end = min((i + 1) * band_size, x_ft.size(1))

            band = x_ft[:, start:end, :]

            filtered = self.band_filters[i](x)
            band_features.append(filtered)

        weights = F.softmax(self.band_weights, dim=0)
        output = sum(f * w for f, w in zip(band_features, weights))

        return output


class MultiResolutionDeepONet(nn.Module):
    """DeepONet with multi-resolution branch encoding."""

    def __init__(
        self,
        branch_channels: int,
        branch_sensor_dim: int,
        trunk_dim: int,
        output_dim: int = 1,
        num_resolutions: int = 3,
        trunk_hidden: int = 128,
    ):
        super().__init__()
        self.branch_channels = branch_channels
        self.trunk_dim = trunk_dim
        self.output_dim = output_dim
        self.num_resolutions = num_resolutions

        self.resolution_encoders = nn.ModuleList(
            [
                nn.Conv1d(
                    branch_channels, trunk_hidden, kernel_size=3, stride=2**i, padding=0
                )
                for i in range(num_resolutions)
            ]
        )

        self.resolution_projs = nn.ModuleList(
            [Linear(trunk_hidden, trunk_hidden) for _ in range(num_resolutions)]
        )

        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_dim, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden),
        )

        combined_dim = trunk_hidden * (num_resolutions + 1)
        self.output_proj = Linear(combined_dim, output_dim)

    def forward(
        self,
        sensor_data: Tensor,
        query_locations: Tensor,
    ) -> Tensor:
        batch_size = sensor_data.size(0)

        resolution_features = []
        x = sensor_data.transpose(1, 2)

        for i, (encoder, proj) in enumerate(
            zip(self.resolution_encoders, self.resolution_projs)
        ):
            try:
                encoded = encoder(x)
                if encoded.size(-1) < 2:
                    continue
                pooled = encoded.mean(dim=-1)
                resolved = proj(pooled)
                resolution_features.append(resolved)
            except Exception:
                continue

        trunk_feat = self.trunk_net(query_locations)
        resolution_features.append(trunk_feat)

        combined = torch.cat(resolution_features, dim=-1)
        output = self.output_proj(combined)

        return output


class FractalNeuralOperator(nn.Module):
    """Fractal neural operator with self-similar architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.hidden_dim = hidden_dim

        self.input_proj = Linear(in_channels, hidden_dim)

        self.fractal_blocks = nn.ModuleList(
            [self._make_fractal_block(hidden_dim) for _ in range(depth)]
        )

        self.output_proj = Linear(hidden_dim, out_channels)

    def _make_fractal_block(self, channels: int) -> nn.Module:
        return nn.Sequential(
            Linear(channels, channels),
            nn.GELU(),
            Linear(channels, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)

        for block in self.fractal_blocks:
            identity = x
            x = block(x)
            x = x + identity

        output = self.output_proj(x)
        return output


class OperatorInterpolation(nn.Module):
    """Interpolation module for operator output at arbitrary points."""

    def __init__(
        self,
        embedding_dim: int,
        num_basis: int = 64,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_basis = num_basis

        self.basis_functions = nn.Parameter(torch.randn(num_basis, embedding_dim))

    def forward(
        self,
        embeddings: Tensor,
        query_points: Tensor,
    ) -> Tensor:
        basis_expanded = self.basis_functions.unsqueeze(0).unsqueeze(0)

        query_expanded = query_points.unsqueeze(2)

        weights = torch.einsum("bnd,dk->bnk", query_expanded, basis_expanded.squeeze(0))
        weights = F.softmax(weights, dim=-1)

        output = torch.einsum("bnk,bkd->bnd", weights, embeddings)

        return output


__all__ = [
    "WaveletTransform1D",
    "WaveletNeuralOperator",
    "MultiScaleAttention",
    "MultigridNeuralOperator",
    "HierarchicalOperatorBlock",
    "AdaptiveScaleOperator",
    "ScaleSeparationModule",
    "MultiResolutionDeepONet",
    "FractalNeuralOperator",
    "OperatorInterpolation",
]

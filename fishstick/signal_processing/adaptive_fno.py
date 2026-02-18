"""
Adaptive Fourier Neural Operators

FNO variants with learnable frequency components and adaptive modes.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveSpectralConv(nn.Module):
    """Spectral convolution with adaptive mode selection.

    Learns which frequency modes to use for each input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dropout = dropout

        self.scale = 1.0 / (in_channels * out_channels)

        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

        self.mode_attention = nn.Parameter(torch.ones(modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive spectral convolution.

        Args:
            x: Input tensor (batch, channels, x_size)

        Returns:
            Convolved output
        """
        batch_size = x.shape[0]

        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")

        modes = min(self.modes, x_fft.shape[-1])

        x_modes = x_fft[..., :modes]

        mode_weights = torch.softmax(self.mode_attention[:modes], dim=0)

        weights = self.weights[:, :, :modes] * mode_weights.view(1, 1, -1)

        out_modes = torch.einsum("bix,iox->box", x_modes, weights)

        x_fft_out = torch.zeros_like(x_fft)
        x_fft_out[..., :modes] = out_modes

        x_out = torch.fft.irfft(x_fft_out, n=x.shape[-1], dim=-1, norm="ortho")

        if self.dropout > 0 and self.training:
            x_out = F.dropout(x_out, p=self.dropout)

        return x_out


class AdaptiveFNOBlock(nn.Module):
    """FNO block with adaptive spectral convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()

        self.spec_conv = AdaptiveSpectralConv(in_channels, out_channels, modes, dropout)

        self.conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

        self.norm = nn.GroupNorm(1, out_channels)

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adaptive FNO block."""
        identity = x

        out = self.spec_conv(x)

        if self.conv is not None:
            identity = self.conv(identity)

        out = out + identity
        out = self.norm(out)
        out = self.activation(out)

        return out


class AdaptiveFNO1D(nn.Module):
    """1D Adaptive Fourier Neural Operator."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 4,
        modes: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        self.layers = nn.ModuleList(
            [
                AdaptiveFNOBlock(
                    hidden_channels,
                    hidden_channels,
                    modes=modes,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adaptive FNO."""
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output_proj(x)

        return x


class LearnableSpectralConv(nn.Module):
    """Spectral convolution with learnable complex weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.w_real = nn.Parameter(torch.randn(in_channels, out_channels, modes))
        self.w_imag = nn.Parameter(torch.randn(in_channels, out_channels, modes))

        self.b_real = nn.Parameter(torch.zeros(out_channels))
        self.b_imag = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable spectral convolution."""
        batch_size = x.shape[0]

        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")

        modes = min(self.modes, x_fft.shape[-1])
        x_modes = x_fft[..., :modes]

        w_complex = torch.complex(self.w_real[..., :modes], self.w_imag[..., :modes])

        out_modes = torch.einsum("bix,iox->box", x_modes, w_complex)

        x_fft_out = torch.zeros_like(x_fft)
        x_fft_out[..., :modes] = out_modes

        x_out = torch.fft.irfft(x_fft_out, n=x.shape[-1], dim=-1, norm="ortho")

        return x_out + self.b_real.view(1, -1, 1) + 1j * self.b_imag.view(1, -1, 1)


class FactorizedSpectralConv(nn.Module):
    """Factorized spectral convolution for efficiency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int = 4,
        modes: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.modes = modes

        self.U = nn.Parameter(torch.randn(in_channels, rank, dtype=torch.cfloat))
        self.V = nn.Parameter(torch.randn(rank, out_channels, dtype=torch.cfloat))
        self.S = nn.Parameter(torch.ones(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply factorized spectral convolution."""
        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")

        modes = min(self.modes, x_fft.shape[-1])
        x_modes = x_fft[..., :modes]

        w = torch.einsum("ir,ro->io", self.U * self.S.unsqueeze(0), self.V)

        w = w[..., :modes].contiguous()

        out_modes = torch.einsum("bix,io->box", x_modes, w)

        x_fft_out = torch.zeros_like(x_fft)
        x_fft_out[..., :modes] = out_modes

        return torch.fft.irfft(x_fft_out, n=x.shape[-1], dim=-1, norm="ortho")


class MultiScaleFNO(nn.Module):
    """Multi-scale Fourier Neural Operator."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        scales: List[int] = [1, 2, 4, 8],
        num_layers: int = 2,
    ):
        super().__init__()
        self.scales = scales

        self.input_proj = nn.Conv1d(in_channels * len(scales), hidden_channels, 1)

        self.scale_convs = nn.ModuleList(
            [nn.Conv1d(in_channels, hidden_channels, 1) for _ in scales]
        )

        self.layers = nn.ModuleList(
            [
                AdaptiveFNOBlock(hidden_channels, hidden_channels, modes=16)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale FNO."""
        B, C, L = x.shape

        multi_scale_feats = []

        for idx, scale in enumerate(self.scales):
            if scale == 1:
                feat = x
            else:
                feat = F.avg_pool1d(x, kernel_size=scale, stride=scale)
                feat = F.interpolate(feat, size=L, mode="linear", align_corners=False)

            feat = self.scale_convs[idx](feat)
            multi_scale_feats.append(feat)

        x = torch.cat(multi_scale_feats, dim=1)
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)


class KernelizedFNO(nn.Module):
    """FNO with learnable convolution kernels."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        num_layers: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        self.spec_layers = nn.ModuleList(
            [
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size)
                for _ in range(num_layers)
            ]
        )

        self.freq_layers = nn.ModuleList(
            [
                AdaptiveSpectralConv(hidden_channels, hidden_channels, modes=16)
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.GroupNorm(1, hidden_channels) for _ in range(num_layers)]
        )

        self.output_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining kernel and spectral convolutions."""
        x = self.input_proj(x)

        for i in range(len(self.spec_layers)):
            identity = x

            x_spec = self.freq_layers[i](x)
            x_conv = self.spec_layers[i](x)

            x = x_spec + x_conv
            x = self.norms[i](x)
            x = F.gelu(x)

            x = x + identity

        return self.output_proj(x)


class TokenFNO(nn.Module):
    """FNO with frequency token learning."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dim: int = 64,
        num_tokens: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(in_channels, hidden_dim, 1)

        self.tokens = nn.Parameter(torch.randn(num_tokens, hidden_dim))

        self.layers = nn.ModuleList(
            [
                AdaptiveFNOBlock(
                    hidden_dim + num_tokens, hidden_dim + num_tokens, modes=16
                )
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Conv1d(hidden_dim + num_tokens, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with frequency tokens."""
        B = x.shape[0]

        x = self.input_proj(x)

        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")
        x_freq = torch.abs(x_fft).mean(dim=1)

        tokens = self.tokens.unsqueeze(0).expand(B, -1, -1)

        x = x.transpose(1, 2)
        x = torch.cat([x, tokens], dim=1)

        for layer in self.layers:
            x = layer(x.transpose(1, 2)).transpose(1, 2)

        x = self.output_proj(x.transpose(1, 2))

        return x


class FNOWithPositionalEncoding(nn.Module):
    """FNO with learnable positional frequency encoding."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 4,
        modes: int = 16,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        self.freq_encoding = nn.Parameter(torch.randn(1, hidden_channels, modes, 2))

        self.layers = nn.ModuleList(
            [
                AdaptiveFNOBlock(hidden_channels, hidden_channels, modes=modes)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with positional encoding."""
        x = self.input_proj(x)

        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")
        modes = min(self.freq_encoding.shape[2], x_fft.shape[-1])

        encoding = self.freq_encoding[..., :modes, :]

        x_fft[..., :modes] = x_fft[..., :modes] * (
            encoding[..., 0] + 1j * encoding[..., 1]
        )

        x = torch.fft.irfft(x_fft, n=x.shape[-1], dim=-1, norm="ortho")

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)


class FNODynamicModeDecomposition(nn.Module):
    """FNO with dynamic mode decomposition for modal analysis."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        num_modes: int = 8,
    ):
        super().__init__()
        self.num_modes = num_modes

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        self.spatial_conv = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)

        self.modes_real = nn.Parameter(torch.randn(num_modes, hidden_channels))
        self.modes_imag = nn.Parameter(torch.randn(num_modes, hidden_channels))

        self.output_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with dynamic mode decomposition."""
        x = self.input_proj(x)

        x = self.spatial_conv(x)

        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")

        modes = torch.complex(self.modes_real, self.modes_imag)

        coeffs = torch.einsum("bx,mo->bmo", x_fft.mean(dim=-1), modes)

        x_fft_recon = coeffs.sum(dim=1, keepdim=True)

        x = torch.fft.irfft(x_fft_recon, n=x.shape[-1], dim=-1, norm="ortho")

        return self.output_proj(x)


class HierarchicalFNO(nn.Module):
    """Hierarchical FNO with different resolutions."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        levels: int = 3,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        self.level_fnos = nn.ModuleList(
            [
                AdaptiveFNOBlock(hidden_channels, hidden_channels, modes=16 // (2**i))
                for i in range(levels)
            ]
        )

        self.upsamples = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
                for _ in range(levels - 1)
            ]
        )

        self.output_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through hierarchical FNO."""
        x = self.input_proj(x)

        features = []

        for i, fno in enumerate(self.level_fnos):
            x = fno(x)
            features.append(x)

            if i < len(self.upsamples):
                x = self.upsamples[i](x)

        return self.output_proj(x)

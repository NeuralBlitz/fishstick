"""
Fourier Domain Attention Mechanisms

Attention mechanisms operating in the frequency domain for efficient
global context modeling.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FrequencyDomainAttention(nn.Module):
    """Attention mechanism applied in the frequency domain.

    Applies self-attention to Fourier coefficients for capturing
    global dependencies efficiently.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency domain attention.

        Args:
            x: Input tensor of shape (batch, length, dim)

        Returns:
            Attended features
        """
        B, N, C = x.shape

        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")

        x_freq = torch.view_as_complex(x_fft) if x_fft.is_complex() else x_fft

        qkv = self.qkv(x_freq).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_freq = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = torch.fft.irfft(torch.view_as_real(x_freq), n=N, dim=1, norm="ortho")

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SpectralAttention(nn.Module):
    """Spectral attention for feature recalibration.

    Learns to weight different frequency components.
    """

    def __init__(
        self,
        freq_bins: int = 64,
        reduction: int = 4,
    ):
        super().__init__()
        self.freq_bins = freq_bins

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(freq_bins, freq_bins // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(freq_bins // reduction, freq_bins, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral attention.

        Args:
            x: Input tensor after FFT of shape (batch, freq_bins, length)

        Returns:
            Recalibrated frequency features
        """
        b, c, _ = x.size()

        avg_out = self.avg_pool(x).squeeze(-1)
        max_out = self.max_pool(x).squeeze(-1)

        avg_weight = self.fc(avg_out)
        max_weight = self.fc(max_out)

        weight = (avg_weight + max_weight).unsqueeze(-1)

        return x * weight


class GlobalFrequencyContext(nn.Module):
    """Global context via frequency domain pooling."""

    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global frequency context.

        Args:
            x: Input spectral features

        Returns:
            Contextualized features
        """
        B, C, H, W = x.shape

        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        x_mag = torch.abs(x_fft)

        x_pooled = x_mag.mean(dim=(-2, -1))

        weight = self.fc(x_pooled).view(B, C, 1, 1)

        return x * weight


class FrequencyTokenAttention(nn.Module):
    """Token attention for frequency components.

    Treats frequency bins as tokens for attention.
    """

    def __init__(
        self,
        num_freq_bins: int,
        dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.dim = dim
        self.num_heads = num_heads

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply token attention to frequency components.

        Args:
            x: Frequency features (batch, num_bins, dim)

        Returns:
            Attended features
        """
        B, N, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls_tokens, x], dim=1)

        attn_out, _ = self.attention(x, x, x)

        x = self.norm(x + attn_out)

        return x[:, 1:]


class MultiScaleFrequencyAttention(nn.Module):
    """Multi-scale frequency attention with different FFT sizes."""

    def __init__(
        self,
        channels: int,
        fft_sizes: Tuple[int, ...] = (8, 16, 32),
        reduction: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.fft_sizes = fft_sizes

        self.conv1x1 = nn.Conv1d(channels * len(fft_sizes), channels, 1)

        self.attention = SpectralAttention(freq_bins=channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale frequency attention.

        Args:
            x: Input tensor (batch, channels, length)

        Returns:
            Attended features
        """
        B, C, L = x.shape

        multi_scale_feats = []

        for fft_size in self.fft_sizes:
            if fft_size > L:
                fft_size = L

            pad_len = (fft_size - L % fft_size) % fft_size
            x_padded = F.pad(x, (0, pad_len))

            x_fft = torch.fft.rfft(x_padded, n=fft_size, dim=-1)
            x_mag = torch.abs(x_fft)

            x_down = F.adaptive_avg_pool1d(x_mag, 1).squeeze(-1)
            multi_scale_feats.append(x_down)

        x_concat = torch.cat(multi_scale_feats, dim=-1)

        x_concat = self.conv1x1(x_concat)

        return x_concat


class PhaseAwareAttention(nn.Module):
    """Attention mechanism that considers phase information."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply phase-aware attention.

        Args:
            x: Input complex-valued tensor (batch, length, dim)

        Returns:
            Attended features
        """
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")

        x_complex = (
            torch.view_as_complex(torch.cat([x_fft.real, x_fft.imag], dim=-1))
            if not x_fft.is_complex()
            else x_fft
        )

        B, N, D_complex = x_complex.shape
        D = D_complex // 2

        qkv = self.qkv(x_complex).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.num_heads)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D_complex)

        out_real = out[..., :D]
        out_imag = out[..., D:]

        out_fft = torch.complex(out_real, out_imag)

        x_out = torch.fft.irfft(out_fft, n=N, dim=1, norm="ortho")

        return self.proj(x_out)


class FrequencyResponseAttention(nn.Module):
    """Attention based on frequency response characteristics."""

    def __init__(
        self,
        in_channels: int,
        num_freqs: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_freqs = num_freqs

        self.learnable_freq = nn.Parameter(torch.linspace(0, 0.5, num_freqs))

        self.query_net = nn.Conv1d(in_channels, in_channels, 1)
        self.key_net = nn.Conv1d(in_channels, in_channels, 1)

        self.value_net = nn.Conv1d(in_channels, in_channels, 1)

        self.out_net = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency response attention.

        Args:
            x: Input tensor (batch, channels, length)

        Returns:
            Attended features
        """
        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")
        x_mag = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        freq_response = torch.relu(self.learnable_freq).unsqueeze(0).unsqueeze(0)

        q = self.query_net(x_mag * freq_response)
        k = self.key_net(x_mag)
        v = self.value_net(x)

        attn = torch.softmax(q * k / math.sqrt(self.in_channels), dim=-1)

        out = attn * v

        out_fft = torch.fft.rfft(out, dim=-1, norm="ortho")
        out_mag = torch.abs(out_fft)

        out_recon = out_mag * torch.exp(1j * x_phase)

        return self.out_net(
            torch.fft.irfft(out_recon, n=x.shape[-1], dim=-1, norm="ortho")
        )


class ChannelFrequencyAttention(nn.Module):
    """Joint channel and frequency attention."""

    def __init__(
        self,
        channels: int,
        freq_bins: int = 64,
    ):
        super().__init__()
        self.channels = channels
        self.freq_bins = freq_bins

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(freq_bins, freq_bins // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(freq_bins // 4, freq_bins, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply joint channel and frequency attention.

        Args:
            x: Input tensor (batch, channels, freq_bins)

        Returns:
            Recalibrated features
        """
        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")
        x_mag = torch.abs(x_fft)

        channel_weight = self.channel_attention(x_mag.transpose(-2, -1)).transpose(
            -2, -1
        )
        freq_weight = self.freq_attention(x_mag)

        combined_weight = channel_weight * freq_weight

        return x * combined_weight


class CrossFrequencyAttention(nn.Module):
    """Cross-attention between time and frequency domains."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.time_to_freq = nn.Linear(dim, dim)
        self.freq_to_time = nn.Linear(dim, dim)

        self.cross_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention between time and frequency.

        Args:
            x: Input tensor (batch, length, dim)

        Returns:
            Fused features
        """
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")

        x_freq = (
            torch.view_as_real(x_fft)
            if x_fft.is_complex()
            else torch.cat([x_fft.real, x_fft.imag], dim=-1)
        )

        x_freq_proj = self.time_to_freq(x_freq)

        attn_out, _ = self.cross_attention(x, x_freq_proj, x_freq_proj)

        return self.freq_to_time(attn_out)


class FrequencyLinear(nn.Module):
    """Linear layer with frequency domain transformation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.register_buffer("freq_buffer", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation in frequency domain.

        Args:
            x: Input tensor

        Returns:
            Transformed output
        """
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.permute(0, 2, 1)
            x = F.linear(x, self.weight, self.bias)
            x = x.permute(0, 2, 1)
        else:
            x = F.linear(x, self.weight, self.bias)

        return x

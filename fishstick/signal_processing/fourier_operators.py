"""
Fourier Neural Operators

Neural network layers operating in the frequency domain for efficient
global context modeling and spectral convolutions.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math


class FourierLayer(nn.Module):
    """Fourier Layer for spectral convolutions.

    Performs convolution in the frequency domain using FFT.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bias = bias

        self.scale = 1.0 / (in_channels * out_channels)

        self.weights_real = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes)
        )

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_param", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using FFT.

        Args:
            x: Input tensor of shape (batch, channels, length)

        Returns:
            Output tensor of shape (batch, out_channels, length)
        """
        batch_size, channels, length = x.shape
        device = x.device

        x_fft = torch.fft.rfft(x, dim=-1)

        x_fft_trunc = x_fft[:, :, : self.modes]

        w_real = self.weights_real.to(x.device)
        w_imag = self.weights_imag.to(x.device)

        x_real = x_fft_trunc.real
        x_imag = x_fft_trunc.imag

        out_real = torch.einsum("bcm,com->bom", x_real, w_real) - torch.einsum(
            "bcm,com->bom", x_imag, w_imag
        )
        out_imag = torch.einsum("bcm,com->bom", x_real, w_imag) + torch.einsum(
            "bcm,com->bom", x_imag, w_real
        )

        out_fft = torch.complex(out_real, out_imag)

        out_full = torch.zeros(
            batch_size,
            self.out_channels,
            length // 2 + 1,
            dtype=torch.complex64,
            device=device,
        )
        out_full[:, :, : self.modes] = out_fft

        out = torch.fft.irfft(out_full, dim=-1, n=length)

        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1)

        return out


class SpectralConv1D(nn.Module):
    """1D Spectral Convolution using Fourier transforms."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
        factor: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.factor = factor

        self.fourier_weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, 2) * 0.02
        )

        self.complex_weights = nn.Parameter(
            torch.complex(self.fourier_weights[..., 0], self.fourier_weights[..., 1])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Args:
            x: Input tensor (batch, channels, length)

        Returns:
            Output tensor (batch, out_channels, length)
        """
        batch_size, channels, length = x.shape
        device = x.device

        x_ft = torch.fft.rfft(x, dim=-1)

        modes_selected = min(self.modes, x_ft.shape[-1])
        x_ft_truncated = x_ft[:, :, :modes_selected]

        weights = self.complex_weights[:, :, :modes_selected]

        out_ft = torch.einsum("bcm,co->bom", x_ft_truncated, weights)

        out = torch.zeros(
            batch_size, self.out_channels, length, dtype=torch.complex64, device=device
        )
        out[:, :, :modes_selected] = out_ft

        return torch.fft.irfft(out, dim=-1, n=length)


class GlobalFourierOperator(nn.Module):
    """Global Fourier Operator for capturing long-range dependencies.

    Processes entire sequence in frequency domain for global context.
    """

    def __init__(
        self,
        dim: int,
        num_frequencies: int = 32,
        learnable: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_frequencies = num_frequencies
        self.learnable = learnable

        if learnable:
            self.freq_weights = nn.Parameter(
                torch.randn(num_frequencies, dim, dim) * 0.02
            )
            self.freq_bias = nn.Parameter(torch.zeros(num_frequencies, dim))
        else:
            self.register_parameter("freq_weights", None)
            self.register_parameter("freq_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global Fourier operator.

        Args:
            x: Input tensor (batch, length, dim)

        Returns:
            Output tensor (batch, length, dim)
        """
        batch_size, length, dim = x.shape
        device = x.device

        x_ft = torch.fft.rfft(x, dim=1)

        freq_len = x_ft.shape[1]

        if self.learnable:
            weights = self.freq_weights[: min(self.num_frequencies, freq_len)]
            weights = weights.to(x.device)

            x_ft_weighted = torch.einsum(
                "bfd,de->bfe", x_ft[: min(self.num_frequencies, freq_len)], weights
            )

            output_ft = torch.zeros_like(x_ft)
            output_ft[: min(self.num_frequencies, freq_len)] = x_ft_weighted
            output_ft[min(self.num_frequencies, freq_len) :] = x_ft[
                min(self.num_frequencies, freq_len) :
            ]

            if self.freq_bias is not None:
                bias = self.freq_bias[: min(self.num_frequencies, freq_len)].to(device)
                output_ft[: min(self.num_frequencies, freq_len)] += bias

            return torch.fft.irfft(output_ft, dim=1, n=length)
        else:
            return torch.fft.irfft(x_ft, dim=1, n=length)


class FourierNeuralOperatorBlock(nn.Module):
    """Fourier Neural Operator block with residual connection."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        num_frequencies: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4

        self.norm = nn.LayerNorm(dim)

        self.fc_in = nn.Linear(dim, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, dim)

        self.fourier_op = GlobalFourierOperator(
            dim=self.hidden_dim,
            num_frequencies=num_frequencies,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x

        x = self.norm(x)
        x = self.fc_in(x)
        x = F.gelu(x)
        x = self.fourier_op(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        x = self.dropout(x)

        return x + residual


class FrequencyDomainAttention(nn.Module):
    """Attention mechanism in frequency domain."""

    def __init__(
        self,
        dim: int,
        num_frequencies: int = 16,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.num_frequencies = num_frequencies
        self.head_dim = dim // heads

        assert dim % heads == 0, "dim must be divisible by heads"

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.fourier_q = FourierLayer(dim, dim, num_frequencies)
        self.fourier_k = FourierLayer(dim, dim, num_frequencies)
        self.fourier_v = FourierLayer(dim, dim, num_frequencies)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency-domain attention.

        Args:
            x: Input tensor (batch, length, dim)

        Returns:
            Output tensor (batch, length, dim)
        """
        batch_size, length, dim = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, length, self.heads, self.head_dim).transpose(1, 2)

        q_fft = self.fourier_q(q.transpose(1, 2).transpose(2, 3))
        k_fft = self.fourier_k(k.transpose(1, 2).transpose(2, 3))
        v_fft = self.fourier_v(v.transpose(1, 2).transpose(2, 3))

        attn = torch.matmul(q_fft, k_fft.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v_fft)
        out = out.transpose(2, 3).transpose(1, 2).contiguous()
        out = out.view(batch_size, length, dim)

        out = self.proj(out)

        return out


class FNO1D(nn.Module):
    """Fourier Neural Operator for 1D signals.

    Full implementation with multiple spectral layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        num_modes: int = 16,
        n_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_modes = num_modes
        self.n_layers = n_layers

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.spectral_layers = nn.ModuleList(
            [
                SpectralConv1D(hidden_channels, hidden_channels, modes=num_modes)
                for _ in range(n_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(n_layers)]
        )

        self.output_proj = nn.Linear(hidden_channels, out_channels)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO.

        Args:
            x: Input tensor (batch, length, in_channels)

        Returns:
            Output tensor (batch, length, out_channels)
        """
        x = self.input_proj(x)

        for i in range(self.n_layers):
            residual = x
            x = self.spectral_layers[i](x)
            x = self.norms(x)
            x = self.activation(x)
            x = x + residual

        x = self.output_proj(x)

        return x


class SpectralResNetBlock(nn.Module):
    """Spectral Residual Network block."""

    def __init__(
        self,
        channels: int,
        num_modes: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = SpectralConv1D(channels, channels, modes=num_modes)
        self.conv2 = SpectralConv1D(channels, channels, modes=num_modes)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with spectral convolutions."""
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.activation(x)

        return x

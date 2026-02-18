"""
Positional Encoding for NeRF

Fourier features and sinusoidal encodings for high-frequency details.
"""

from typing import Tuple
import torch
from torch import Tensor, nn
import numpy as np


class PositionalEncoder(nn.Module):
    """
    NeRF-style positional encoding.

    Encodes inputs using sin/cos at multiple frequencies.
    """

    def __init__(self, input_dim: int = 3, num_frequencies: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim * 2 * num_frequencies

        self.register_buffer(
            "freq_bands", 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [N, input_dim]

        Returns:
            Encoded tensor [N, input_dim * 2 * num_frequencies]
        """
        encoded = []

        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))

        return torch.cat(encoded, dim=-1)


class FourierFeatures(nn.Module):
    """
    Random Fourier Features for kernel approximation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale

        self.register_buffer("B", torch.randn(input_dim, output_dim // 2) * scale)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input [N, input_dim]

        Returns:
            Fourier features [N, output_dim]
        """
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class GaussianFourierFeatures(nn.Module):
    """
    Gaussian Fourier Features with learnable transformation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = nn.Parameter(
            torch.randn(input_dim, output_dim // 2) * sigma, requires_grad=False
        )
        self.b = nn.Parameter(torch.randn(output_dim // 2) * sigma, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input [N, input_dim]

        Returns:
            Features [N, output_dim]
        """
        x_proj = x @ self.W + self.b
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SinusoidalPositionEncoder(nn.Module):
    """
    Standard sinusoidal position encoding (like in Transformer).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Positions [N] or [N, 1]

        Returns:
            Encoded positions [N, d_model]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        x = x.long().clamp(0, self.pe.shape[0] - 1)
        return self.pe[x]


def get_nerf_positional_encoding(
    input_dim: int,
    num_frequencies: int = 10,
    include_input: bool = True,
) -> PositionalEncoder:
    """
    Create NeRF-style positional encoder.

    Args:
        input_dim: Input dimension
        num_frequencies: Number of frequency bands
        include_input: Whether to include original input

    Returns:
        PositionalEncoder module
    """
    return PositionalEncoder(input_dim, num_frequencies)

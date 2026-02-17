"""
Spectral Pooling Layers

Pooling operations in the frequency domain for efficient downsampling
while preserving important spectral information.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralPooling(nn.Module):
    """Spectral pooling in frequency domain.

    Keeps low-frequency components while discarding high-frequency ones.
    """

    def __init__(
        self,
        mode: str = "low",
        keep_ratio: float = 0.5,
    ):
        super().__init__()
        self.mode = mode
        self.keep_ratio = keep_ratio

    def forward(
        self,
        x: torch.Tensor,
        output_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply spectral pooling.

        Args:
            x: Input tensor (batch, channels, length)
            output_size: Target output size (if None, uses keep_ratio)

        Returns:
            Pooled tensor
        """
        batch, channels, length = x.shape
        device = x.device

        x_fft = torch.fft.rfft(x, dim=-1)

        if output_size is None:
            output_size = int(length * self.keep_ratio)

        output_size = min(output_size, length // 2 + 1)

        if self.mode == "low":
            output_fft = torch.zeros(
                batch, channels, length // 2 + 1, dtype=torch.complex64, device=device
            )
            output_fft[:, :, :output_size] = x_fft[:, :, :output_size]

        elif self.mode == "high":
            output_fft = torch.zeros(
                batch, channels, length // 2 + 1, dtype=torch.complex64, device=device
            )
            if output_size < length // 2 + 1:
                output_fft[:, :, output_size:] = x_fft[:, :, output_size:]

        elif self.mode == "middle":
            start_idx = (length // 2 + 1 - output_size) // 2
            output_fft = torch.zeros(
                batch, channels, length // 2 + 1, dtype=torch.complex64, device=device
            )
            output_fft[:, :, start_idx : start_idx + output_size] = x_fft[
                :, :, start_idx : start_idx + output_size
            ]

        output = torch.fft.irfft(output_fft, dim=-1, n=length)

        return output[:, :, : output_size * 2 - 1]

    def pool(
        self,
        x: torch.Tensor,
        kernel_size: int,
        stride: Optional[int] = None,
    ) -> torch.Tensor:
        """Pool by selecting frequency bands."""
        stride = stride or kernel_size

        length = x.shape[-1]
        n_bands = length // stride

        pooled = []

        for i in range(n_bands):
            start = i * stride
            end = min(start + kernel_size, length)

            band = x[..., start:end]

            band_fft = torch.fft.rfft(band, dim=-1)

            band_fft_trunc = band_fft[..., : int(band_fft.shape[-1] * self.keep_ratio)]

            pooled_band = torch.fft.irfft(
                band_fft_trunc, dim=-1, n=band_fft_trunc.shape[-1] * 2 - 1
            )

            pooled.append(pooled_band.mean(dim=-1, keepdim=True))

        return torch.cat(pooled, dim=-1)


class FourierPooling(nn.Module):
    """Fourier-domain pooling that keeps phase information."""

    def __init__(
        self,
        pool_size: int = 2,
        mode: str = "average",
    ):
        super().__init__()
        self.pool_size = pool_size
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier pooling.

        Args:
            x: Input tensor (batch, channels, length)

        Returns:
            Pooled tensor
        """
        batch, channels, length = x.shape
        device = x.device

        x_fft = torch.fft.rfft(x, dim=-1)

        output_length = length // self.pool_size
        output_fft = torch.zeros(
            batch, channels, length // 2 + 1, dtype=torch.complex64, device=device
        )

        for i in range(self.pool_size):
            start = i * output_length
            end = min((i + 1) * output_length, length // 2 + 1)

            if self.mode == "average":
                output_fft[:, :, start:end] = x_fft[:, :, start:end] / self.pool_size
            elif self.mode == "max":
                output_fft[:, :, start:end] = torch.max(
                    output_fft[:, :, start:end], x_fft[:, :, start:end]
                )

        output = torch.fft.irfft(output_fft, dim=-1, n=length)

        return output[:, :, :output_length]


class WaveletPooling(nn.Module):
    """Pooling using wavelet decomposition."""

    def __init__(
        self,
        level: int = 1,
        wavelet: str = "db4",
    ):
        super().__init__()
        self.level = level
        self.wavelet = wavelet

        self._init_wavelet()

    def _init_wavelet(self):
        """Initialize wavelet filters."""
        import pywt

        wavelet = pywt.Wavelet(self.wavelet)
        self.register_buffer("dec_lo", torch.tensor(wavelet.dec_lo))
        self.register_buffer("dec_hi", torch.tensor(wavelet.dec_hi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply wavelet pooling."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        batch, channels, length = x.shape
        device = x.device

        lo = self.dec_lo.to(device).unsqueeze(0).unsqueeze(0)
        hi = self.dec_hi.to(device).unsqueeze(0).unsqueeze(0)

        approx = x
        details = []

        for _ in range(self.level):
            low = F.conv1d(approx, lo, stride=2, padding=lo.shape[-1] // 2)
            high = F.conv1d(approx, hi, stride=2, padding=hi.shape[-1] // 2)
            details.append(high)
            approx = low

        return approx


class AdaptiveSpectralPooling(nn.Module):
    """Learnable adaptive spectral pooling."""

    def __init__(
        self,
        num_frequencies: int = 16,
        learn_weights: bool = True,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies

        if learn_weights:
            self.freq_weights = nn.Parameter(torch.ones(num_frequencies))
        else:
            self.register_buffer("freq_weights", torch.ones(num_frequencies))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive spectral pooling."""
        batch, channels, length = x.shape
        device = x.device

        x_fft = torch.fft.rfft(x, dim=-1)

        freq_bins = length // 2 + 1
        freq_per_bin = freq_bins // self.num_frequencies

        weights = torch.softmax(self.freq_weights, dim=0).to(device)

        output_fft = torch.zeros(
            batch, channels, freq_bins, dtype=torch.complex64, device=device
        )

        for i in range(self.num_frequencies):
            start = i * freq_per_bin
            end = min((i + 1) * freq_per_bin, freq_bins)

            output_fft[:, :, start:end] = x_fft[:, :, start:end] * weights[i]

        output = torch.fft.irfft(output_fft, dim=-1, n=length)

        return output


class LearnableFilterBankPooling(nn.Module):
    """Pooling using learnable filter bank."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int = 8,
        pool_size: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.pool_size = pool_size

        self.filters = nn.Parameter(torch.randn(num_filters, in_channels, 7) * 0.02)

        self.output_proj = nn.Linear(num_filters, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable filter bank pooling."""
        batch, channels, length = x.shape
        device = x.device

        pooled_length = length // self.pool_size

        outputs = []

        for i in range(self.pool_size):
            start = i * pooled_length
            end = min((i + 1) * pooled_length, length)

            segment = x[:, :, start:end]

            for f in range(self.num_filters):
                filter_f = self.filters[f]
                filtered = F.conv1d(
                    segment,
                    filter_f.unsqueeze(0),
                    padding=filter_f.shape[-1] // 2,
                )
                outputs.append(filtered.mean(dim=-1))

        output = torch.stack(outputs, dim=-1)

        output = self.output_proj(output)

        return output


class SpectralPooling1D(nn.Module):
    """Complete spectral pooling layer for 1D signals."""

    def __init__(
        self,
        keep_ratio: float = 0.5,
        mode: str = "low",
        learnable: bool = False,
    ):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.mode = mode

        if learnable:
            self.ratio = nn.Parameter(torch.tensor(keep_ratio))
        else:
            self.register_parameter("ratio", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with spectral pooling."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        keep_ratio = self.ratio.item() if self.ratio is not None else self.keep_ratio

        length = x.shape[-1]
        output_size = int(length * keep_ratio)

        batch, channels, length = x.shape
        device = x.device

        x_fft = torch.fft.rfft(x, dim=-1)

        output_fft = torch.zeros(
            batch, channels, length // 2 + 1, dtype=torch.complex64, device=device
        )
        output_fft[:, :, :output_size] = x_fft[:, :, :output_size]

        output = torch.fft.irfft(output_fft, dim=-1, n=length)

        return output[:, :, :output_size]


class ScaledSpectralPooling(nn.Module):
    """Spectral pooling with learnable scaling factors."""

    def __init__(
        self,
        num_bands: int = 8,
    ):
        super().__init__()
        self.num_bands = num_bands

        self.scales = nn.Parameter(torch.ones(num_bands))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled spectral pooling."""
        batch, channels, length = x.shape
        device = x.device

        x_fft = torch.fft.rfft(x, dim=-1)

        band_size = (length // 2 + 1) // self.num_bands
        scales = torch.sigmoid(self.scales).to(device)

        output_fft = torch.zeros_like(x_fft)

        for i in range(self.num_bands):
            start = i * band_size
            end = min((i + 1) * band_size, length // 2 + 1)

            output_fft[:, :, start:end] = x_fft[:, :, start:end] * scales[i]

        output = torch.fft.irfft(output_fft, dim=-1, n=length)

        return output


class StridedSpectralConv(nn.Module):
    """Spectral convolution with strided downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        modes: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.modes = modes

        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.complex64) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply strided spectral convolution."""
        batch, channels, length = x.shape
        device = x.device

        x_fft = torch.fft.rfft(x, dim=-1)

        x_trunc = x_fft[:, :, : self.modes]

        out_fft = torch.einsum("bci,bio->bco", x_trunc, self.weights)

        output_length = length // self.stride
        output_fft = torch.zeros(
            batch,
            self.out_channels,
            length // 2 + 1,
            dtype=torch.complex64,
            device=device,
        )
        output_fft[:, :, : self.modes] = out_fft

        output = torch.fft.irfft(output_fft, dim=-1, n=length)

        return output[:, :, :: self.stride]


class FrequencyAttentionPooling(nn.Module):
    """Pooling with attention over frequency bands."""

    def __init__(
        self,
        num_bands: int = 8,
        attention_dim: int = 64,
    ):
        super().__init__()
        self.num_bands = num_bands

        self.attention = nn.Sequential(
            nn.Linear(num_bands, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, num_bands),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention-weighted pooling."""
        batch, channels, length = x.shape
        device = x.device

        x_fft = torch.fft.rfft(x, dim=-1)

        band_size = (length // 2 + 1) // self.num_bands

        band_energies = []
        for i in range(self.num_bands):
            start = i * band_size
            end = min((i + 1) * band_size, length // 2 + 1)
            band_energy = torch.abs(x_fft[:, :, start:end]).mean(dim=-1)
            band_energies.append(band_energy)

        band_energies = torch.stack(band_energies, dim=-1)

        attention_weights = self.attention(band_energies)

        pooled = (band_energies * attention_weights).sum(dim=-1, keepdim=True)

        return pooled


class MultiResolutionSpectralPooling(nn.Module):
    """Multi-resolution spectral pooling combining multiple pooling strategies."""

    def __init__(
        self,
        resolutions: list = None,
    ):
        super().__init__()
        self.resolutions = resolutions or [0.25, 0.5, 0.75]

        self.pooling_layers = nn.ModuleList(
            [SpectralPooling1D(keep_ratio=r) for r in self.resolutions]
        )

        self.fusion = nn.Linear(len(self.resolutions), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-resolution pooling and fuse."""
        outputs = []

        for pool in self.pooling_layers:
            pooled = pool(x)
            outputs.append(pooled)

        max_length = max(o.shape[-1] for o in outputs)

        padded = []
        for o in outputs:
            if o.shape[-1] < max_length:
                o = F.pad(o, (0, max_length - o.shape[-1]))
            padded.append(o)

        stacked = torch.stack(padded, dim=-1)

        fused = self.fusion(stacked).squeeze(-1)

        return fused

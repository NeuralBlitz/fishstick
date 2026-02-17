"""
Filter Bank Implementations

Various filter bank implementations for signal decomposition including
Gabor, Morlet, complex wavelets, and polynomial filter banks.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaborFilterBank(nn.Module):
    """Gabor filter bank for time-frequency analysis.

    Creates a bank of Gabor filters with different center frequencies
    and bandwidths.
    """

    def __init__(
        self,
        num_filters: int = 32,
        filter_length: int = 256,
        min_freq: float = 0.01,
        max_freq: float = 0.5,
        sigma: float = 0.1,
    ):
        super().__init__()
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sigma = sigma

        self.filters = self._create_filter_bank()

    def _create_filter_bank(self) -> torch.Tensor:
        """Create Gabor filter bank."""
        filters = []

        frequencies = np.linspace(self.min_freq, self.max_freq, self.num_filters)

        for freq in frequencies:
            filter_bank = self._gabor_filter(freq, self.sigma)
            filters.append(filter_bank)

        return torch.stack(filters)

    def _gabor_filter(self, freq: float, sigma: float) -> torch.Tensor:
        """Create a single Gabor filter."""
        t = torch.arange(self.filter_length, dtype=torch.float32)
        t = t - self.filter_length / 2

        envelope = torch.exp(-0.5 * (t / (sigma * self.filter_length)) ** 2)
        carrier = torch.exp(2j * np.pi * freq * t)

        gabor = envelope * carrier
        return gabor

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Gabor filter bank to signal.

        Args:
            signal: Input signal (batch, length) or (length,)

        Returns:
            Filtered signals (batch, num_filters, length)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size, length = signal.shape
        device = signal.device

        filters = self.filters.to(device)

        if length != self.filter_length:
            signal_padded = F.pad(signal, (0, self.filter_length - length))
        else:
            signal_padded = signal

        result = torch.zeros(
            batch_size, self.num_filters, length, dtype=torch.complex64, device=device
        )

        for i in range(self.num_filters):
            filter_real = filters[i].real.unsqueeze(0)
            filter_imag = filters[i].imag.unsqueeze(0)

            result_real = F.conv1d(
                signal_padded.float().unsqueeze(1),
                filter_real.unsqueeze(0),
                padding=self.filter_length // 2,
            )
            result_imag = F.conv1d(
                signal_padded.float().unsqueeze(1),
                filter_imag.unsqueeze(0),
                padding=self.filter_length // 2,
            )

            out_len = min(result_real.shape[-1], length)
            result[:, i, :out_len] = (
                result_real.squeeze(1)[:, :out_len]
                + 1j * result_imag.squeeze(1)[:, :out_len]
            )

        return result[:, :, :length]


class MorletWaveletBank(nn.Module):
    """Bank of Morlet wavelets at different scales."""

    def __init__(
        self,
        num_scales: int = 32,
        filter_length: int = 256,
        min_scale: float = 1.0,
        max_scale: float = 32.0,
        num_oscillations: float = 6.0,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.filter_length = filter_length
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_oscillations = num_oscillations

        self.scales = torch.linspace(min_scale, max_scale, num_scales)

    def _morlet_wavelet(self, scale: float) -> torch.Tensor:
        """Create Morlet wavelet at given scale."""
        length = min(int(self.filter_length * scale), self.filter_length)
        length = length if length % 2 == 0 else length + 1

        t = torch.arange(length, dtype=torch.float32)
        t = (t - length / 2) / scale

        envelope = torch.exp(-0.5 * t**2)
        carrier = torch.exp(1j * 2 * np.pi * self.num_oscillations * t / scale)

        wavelet = envelope * carrier
        return wavelet

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply Morlet wavelet bank."""
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size, length = signal.shape
        device = signal.device

        result = torch.zeros(
            batch_size, self.num_scales, length, dtype=torch.complex64, device=device
        )

        for i, scale in enumerate(self.scales):
            wavelet = self._morlet_wavelet(scale).to(device)
            wavelet_len = wavelet.shape[0]

            signal_padded = F.pad(signal.float(), (wavelet_len // 2, wavelet_len // 2))

            conv_real = F.conv1d(
                signal_padded.unsqueeze(1),
                wavelet.real.unsqueeze(0).unsqueeze(0),
                padding=wavelet_len // 2,
            )
            conv_imag = F.conv1d(
                signal_padded.unsqueeze(1),
                wavelet.imag.unsqueeze(0).unsqueeze(0),
                padding=wavelet_len // 2,
            )

            result[:, i, :] = conv_real.squeeze(1) + 1j * conv_imag.squeeze(1)

        return result


class ComplexWaveletBank(nn.Module):
    """Complex wavelet filter bank using dual-tree complex wavelets."""

    def __init__(
        self,
        num_levels: int = 5,
        filter_length: int = 6,
        wavelet_type: str = "farras",
    ):
        super().__init__()
        self.num_levels = num_levels

        self._init_filters(filter_length, wavelet_type)

    def _init_filters(self, filter_length: int, wavelet_type: str):
        """Initialize analysis and synthesis filters."""
        if wavelet_type == "farras":
            h0a = torch.tensor([0.0352, -0.0856, -0.1350, 0.4599, 0.8069, 0.3327])
            h1a = torch.tensor([-0.3327, 0.8069, -0.4599, -0.1350, 0.0856, 0.0352])

            h0s = torch.tensor([-0.3327, 0.8069, -0.4599, -0.1350, 0.0856, 0.0352])
            h1s = torch.tensor([-0.0352, -0.0856, 0.1350, 0.4599, -0.8069, 0.3327])
        else:
            h0a = torch.randn(filter_length) * 0.1
            h1a = torch.randn(filter_length) * 0.1
            h0s = torch.randn(filter_length) * 0.1
            h1s = torch.randn(filter_length) * 0.1

        self.register_buffer("h0a", h0a)
        self.register_buffer("h1a", h1a)
        self.register_buffer("h0s", h0s)
        self.register_buffer("h1s", h1s)

    def forward(self, signal: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Apply complex wavelet transform.

        Returns:
            Tuple of (detail coefficients list, approximation)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        approx = signal.float()
        details = []

        for level in range(self.num_levels):
            lowpass, highpass = self._analysis_step(approx)
            details.append(highpass)
            approx = lowpass

        return details, approx

    def _analysis_step(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single level analysis (decomposition) step."""
        lo = self.h0a.unsqueeze(0).unsqueeze(0)
        hi = self.h1a.unsqueeze(0).unsqueeze(0)

        lowpass = F.conv1d(signal, lo, stride=2, padding=lo.shape[-1] // 2)
        highpass = F.conv1d(signal, hi, stride=2, padding=hi.shape[-1] // 2)

        return lowpass, highpass


class HalfBandFilterBank(nn.Module):
    """Half-band filter bank for perfect reconstruction.

    Used in wavelet transforms and subband coding.
    """

    def __init__(
        self,
        num_stages: int = 4,
    ):
        super().__init__()
        self.num_stages = num_stages

        self._create_filters()

    def _create_filters(self):
        """Create analysis and synthesis half-band filters."""
        n_taps = 16

        h_hp = torch.sin(
            torch.pi * (torch.arange(n_taps, dtype=torch.float32) - (n_taps - 1) / 2)
        )
        h_hp = h_hp * torch.from_numpy(np.blackman(n_taps)).float()
        h_hp = h_hp / (torch.arange(1, n_taps + 1, dtype=torch.float32) % 2 * 2 - 1)

        h_lp = torch.roll(h_hp, 1)
        h_lp[0] = 1.0

        self.register_buffer("analysis_low", h_lp)
        self.register_buffer("analysis_high", h_hp)
        self.register_buffer("synthesis_low", h_lp)
        self.register_buffer("synthesis_high", -h_hp)

    def analysis(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Analysis (decomposition) with half-band filters."""
        lo = self.analysis_low.unsqueeze(0).unsqueeze(0)
        hi = self.analysis_high.unsqueeze(0).unsqueeze(0)

        low = F.conv1d(signal, lo, stride=2, padding=lo.shape[-1] // 2)
        high = F.conv1d(signal, hi, stride=2, padding=hi.shape[-1] // 2)

        return low, high

    def synthesis(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        """Synthesis (reconstruction) from half-band subbands."""
        lo = self.synthesis_low.unsqueeze(0).unsqueeze(0)
        hi = self.synthesis_high.unsqueeze(0).unsqueeze(0)

        low_up = F.conv_transpose1d(low, lo, stride=2, padding=lo.shape[-1] // 2)
        high_up = F.conv_transpose1d(high, hi, stride=2, padding=hi.shape[-1] // 2)

        return low_up + high_up


class PolynomialFilterBank(nn.Module):
    """Polynomial (Lagrange) filter bank for signal interpolation."""

    def __init__(
        self,
        filter_order: int = 3,
        num_bands: int = 8,
    ):
        super().__init__()
        self.filter_order = filter_order
        self.num_bands = num_bands

        self.filters = self._create_polynomial_filters()

    def _create_polynomial_filters(self) -> torch.Tensor:
        """Create polynomial interpolation filters."""
        filters = []

        for band in range(self.num_bands):
            freq = (band + 1) / self.num_bands
            coeffs = self._lagrange_coeffs(freq, self.filter_order)
            filters.append(coeffs)

        return torch.stack(filters)

    def _lagrange_coeffs(self, freq: float, order: int) -> torch.Tensor:
        """Compute Lagrange interpolation coefficients."""
        n_points = order + 1
        x = torch.linspace(0, 1, n_points)
        y = torch.sin(np.pi * freq * x) / (np.pi * freq * x + 1e-9)

        coeffs = torch.zeros(n_points)

        for i in range(n_points):
            numerator = 1.0
            denominator = 1.0

            for j in range(n_points):
                if i != j:
                    numerator *= -x[j]
                    denominator *= x[i] - x[j]

            coeffs[i] = y[i] * numerator / denominator

        return coeffs

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply polynomial filter bank."""
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size, length = signal.shape
        device = signal.device

        filters = self.filters.to(device)
        n_taps = filters.shape[1]

        result = torch.zeros(batch_size, self.num_bands, length, device=device)

        for i in range(self.num_bands):
            filter_coeffs = filters[i]

            signal_padded = F.pad(signal.float(), (n_taps // 2, n_taps // 2))

            result[:, i, :] = F.conv1d(
                signal_padded.unsqueeze(1),
                filter_coeffs.view(1, 1, -1),
                padding=n_taps // 2,
            ).squeeze(1)

        return result


class LearnableFilterBank(nn.Module):
    """Learnable filter bank with trainable filter coefficients."""

    def __init__(
        self,
        num_filters: int = 32,
        filter_length: int = 64,
        stride: int = 1,
    ):
        super().__init__()
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.stride = stride

        self.filters = nn.Parameter(torch.randn(num_filters, filter_length) * 0.02)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply learnable filter bank."""
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        signal_float = signal.float()

        if signal_float.shape[-1] < self.filter_length:
            signal_float = F.pad(
                signal_float, (0, self.filter_length - signal_float.shape[-1])
            )

        output = F.conv1d(
            signal_float.unsqueeze(1),
            self.filters.unsqueeze(0),
            stride=self.stride,
            padding=self.filter_length // 2,
        )

        return output


class BiorthogonalFilterBank(nn.Module):
    """Biorthogonal filter bank for perfect reconstruction."""

    def __init__(
        self,
        num_stages: int = 4,
        wavelet: str = "bior2.2",
    ):
        super().__init__()
        self.num_stages = num_stages
        self.wavelet = wavelet

        self._init_biorthogonal_filters()

    def _init_biorthogonal_filters(self):
        """Initialize biorthogonal filter pair."""
        import pywt

        wavelet = pywt.Wavelet(self.wavelet)

        self.register_buffer("dec_lo", torch.tensor(wavelet.dec_lo))
        self.register_buffer("dec_hi", torch.tensor(wavelet.dec_hi))
        self.register_buffer("rec_lo", torch.tensor(wavelet.rec_lo))
        self.register_buffer("rec_hi", torch.tensor(wavelet.rec_hi))

    def forward(self, signal: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Apply biorthogonal wavelet transform."""
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        approx = signal.float()
        details = []

        for _ in range(self.num_stages):
            lo = self.dec_lo.to(signal.device).unsqueeze(0).unsqueeze(0)
            hi = self.dec_hi.to(signal.device).unsqueeze(0).unsqueeze(0)

            low = F.conv1d(approx, lo, stride=2, padding=lo.shape[-1] // 2)
            high = F.conv1d(approx, hi, stride=2, padding=hi.shape[-1] // 2)

            details.append(high)
            approx = low

        return details, approx


class FilterBankLayer(nn.Module):
    """Filter bank layer that can be used as a neural network module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int = 16,
        filter_length: int = 32,
        filter_type: str = "gabor",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.filter_length = filter_length
        self.filter_type = filter_type

        if filter_type == "gabor":
            self.filter_bank = GaborFilterBank(
                num_filters=num_filters,
                filter_length=filter_length,
            )
        elif filter_type == "morlet":
            self.filter_bank = MorletWaveletBank(
                num_scales=num_filters,
                filter_length=filter_length,
            )
        else:
            self.filter_bank = LearnableFilterBank(
                num_filters=num_filters,
                filter_length=filter_length,
            )

        self.projection = nn.Linear(in_channels * num_filters, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply filter bank and project to output space."""
        batch_size, length, channels = x.shape

        filtered = []
        for c in range(channels):
            signal = x[:, :, c]
            f_out = self.filter_bank(signal)
            filtered.append(f_out)

        filtered = torch.cat(filtered, dim=1)

        filtered = filtered.permute(0, 2, 1)

        output = self.projection(filtered)

        return output

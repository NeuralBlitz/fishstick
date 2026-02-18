"""
Hilbert Transform and Envelope/Phase Analysis

Hilbert transform for computing analytic signals, envelopes,
instantaneous phase, and frequency.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HilbertTransform(nn.Module):
    """Hilbert transform for analytic signal computation.

    Computes the analytic signal using the Hilbert transform
    to extract envelope and instantaneous phase.
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute analytic signal, envelope, and instantaneous phase.

        Args:
            x: Input signal

        Returns:
            Tuple of (analytic_signal, envelope, instantaneous_phase)
        """
        x_fft = torch.fft.fft(x, dim=self.dim, norm="ortho")

        n = x.shape[self.dim]

        positive_freq_indices = self._get_positive_freq_indices(x.shape, x.device)
        negative_freq_indices = self._get_negative_freq_indices(x.shape, x.device)

        analytic_fft = torch.zeros_like(x_fft)

        analytic_fft = self._set_positive_frequencies(
            analytic_fft, x_fft, positive_freq_indices
        )

        if self.dim == -1 or self.dim == x.dim() - 1:
            analytic_fft[..., 0] = x_fft[..., 0]

        analytic_signal = torch.fft.ifft(analytic_fft, dim=self.dim, norm="ortho")

        envelope = torch.abs(analytic_signal)

        instantaneous_phase = torch.angle(analytic_signal)

        return analytic_signal, envelope, instantaneous_phase

    def _get_positive_freq_indices(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """Get indices for positive frequencies."""
        dim_size = shape[self.dim] if self.dim >= 0 else shape[-1]
        half = dim_size // 2 + 1
        return torch.arange(0, half, device=device)

    def _get_negative_freq_indices(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """Get indices for negative frequencies."""
        dim_size = shape[self.dim] if self.dim >= 0 else shape[-1]
        half = dim_size // 2 + 1
        return torch.arange(half, dim_size, device=device)

    def _set_positive_frequencies(
        self,
        analytic_fft: torch.Tensor,
        x_fft: torch.Tensor,
        pos_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Set positive frequency components."""
        slices = [slice(None)] * analytic_fft.dim()
        slices[self.dim] = pos_indices

        analytic_fft[slices] = 2.0 * x_fft[slices]

        return analytic_fft


class InstantaneousFrequency(nn.Module):
    """Instantaneous frequency computation from Hilbert transform."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.hilbert = HilbertTransform(dim=dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute instantaneous frequency and envelope.

        Args:
            x: Input signal

        Returns:
            Tuple of (instantaneous_frequency, envelope)
        """
        _, envelope, phase = self.hilbert(x)

        instantaneous_freq = torch.gradient(phase, dim=self.dim)[0]

        return instantaneous_freq, envelope


class AnalyticSignalConv(nn.Module):
    """Convolution with analytic signal representation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_real = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_imag = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

        self.hilbert = HilbertTransform(dim=-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution in analytic signal domain.

        Args:
            x: Input signal

        Returns:
            Convolved analytic signal
        """
        x_analytic, _, _ = self.hilbert(x)

        out_real = self.conv_real(x_analytic.real)
        out_imag = self.conv_imag(x_analytic.imag)

        return torch.complex(out_real, out_imag)


class EnvelopeExtraction(nn.Module):
    """Learnable envelope extraction layer."""

    def __init__(
        self,
        kernel_size: int = 51,
        stride: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.envelope_conv = nn.Conv1d(
            1, 1, kernel_size, stride=stride, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract signal envelope using learned filters.

        Args:
            x: Input signal (batch, length) or (batch, channels, length)

        Returns:
            Envelope
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x_abs = torch.abs(x)

        envelope = self.envelope_conv(x_abs)

        return envelope


class PhaseExtraction(nn.Module):
    """Phase extraction and encoding."""

    def __init__(self, num_bins: int = 32):
        super().__init__()
        self.num_bins = num_bins

        self.phase_embedding = nn.Sequential(
            nn.Linear(1, num_bins), nn.ReLU(), nn.Linear(num_bins, num_bins)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and encode phase information.

        Args:
            x: Input signal

        Returns:
            Encoded phase
        """
        hilbert = HilbertTransform(dim=-1)
        _, _, phase = hilbert(x)

        phase_normalized = (phase + np.pi) / (2 * np.pi)

        phase_encoded = self.phase_embedding(phase_normalized.unsqueeze(-1))

        return phase_encoded


class InstantaneousAttributes(nn.Module):
    """Extract all instantaneous attributes from signal."""

    def __init__(self):
        super().__init__()
        self.hilbert = HilbertTransform(dim=-1)

    def forward(self, x: torch.Tensor) -> dict:
        """Compute all instantaneous attributes.

        Args:
            x: Input signal

        Returns:
            Dictionary with envelope, phase, frequency
        """
        analytic, envelope, phase = self.hilbert(x)

        instantaneous_freq = torch.gradient(phase, dim=-1)[0]

        instantaneous_amp = torch.gradient(envelope, dim=-1)[0]

        return {
            "analytic_signal": analytic,
            "envelope": envelope,
            "instantaneous_phase": phase,
            "instantaneous_frequency": instantaneous_freq,
            "instantaneous_amplitude": instantaneous_amp,
        }


class HilbertEnvelopeLoss(nn.Module):
    """Loss function using Hilbert envelope features."""

    def __init__(self, envelope_weight: float = 0.5):
        super().__init__()
        self.envelope_weight = envelope_weight
        self.hilbert = HilbertTransform(dim=-1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss with envelope matching.

        Args:
            pred: Predicted signal
            target: Target signal

        Returns:
            Combined loss
        """
        _, pred_env, _ = self.hilbert(pred)
        _, target_env, _ = self.hilbert(target)

        signal_loss = F.mse_loss(pred, target)
        envelope_loss = F.mse_loss(pred_env, target_env)

        return signal_loss + self.envelope_weight * envelope_loss


class ComplexConv1D(nn.Module):
    """1D convolution for complex-valued signals."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv_real = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.conv_imag = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex convolution.

        Args:
            x: Complex input signal

        Returns:
            Complex output
        """
        out_real = self.conv_real(x.real) - self.conv_imag(x.imag)
        out_imag = self.conv_real(x.imag) + self.conv_imag(x.real)

        return torch.complex(out_real, out_imag)


class AnalyticFilterBank(nn.Module):
    """Bank of analytic filters for multi-band decomposition."""

    def __init__(
        self,
        num_bands: int = 8,
        kernel_size: int = 65,
    ):
        super().__init__()
        self.num_bands = num_bands

        self.filters = nn.ModuleList(
            [
                nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2)
                for _ in range(num_bands)
            ]
        )

        self.band_frequencies = nn.Parameter(torch.linspace(0.1, 0.5, num_bands))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply analytic filter bank.

        Args:
            x: Input signal

        Returns:
            Filtered components
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        outputs = []

        for i, filter_layer in enumerate(self.filters):
            filtered = filter_layer(x)

            hilbert = HilbertTransform(dim=-1)
            analytic, envelope, _ = hilbert(filtered.squeeze(1))

            outputs.append(envelope.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class FrequencyDomainPhase(nn.Module):
    """Phase extraction from frequency domain."""

    def __init__(self, num_freqs: int = 64):
        super().__init__()
        self.num_freqs = num_freqs

        self.phase_encoder = nn.Sequential(
            nn.Linear(num_freqs * 2, num_freqs),
            nn.GELU(),
            nn.Linear(num_freqs, num_freqs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract phase from frequency domain.

        Args:
            x: Input signal

        Returns:
            Encoded phase features
        """
        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")

        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        phase_features = torch.cat([magnitude, phase], dim=-1)

        encoded = self.phase_encoder(phase_features)

        return encoded


class EnvelopeConsistencyLoss(nn.Module):
    """Loss ensuring envelope consistency across transformations."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        original: torch.Tensor,
        transformed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute envelope consistency loss.

        Args:
            original: Original signal
            transformed: Transformed signal

        Returns:
            Envelope consistency loss
        """
        hilbert = HilbertTransform(dim=-1)

        _, orig_env, _ = hilbert(original)
        _, trans_env, _ = hilbert(transformed)

        return F.mse_loss(orig_env, trans_env)


class CyclicSpectrumAnalysis(nn.Module):
    """Cyclic spectrum analysis for cyclostationary signals."""

    def __init__(self, n_fft: int = 256):
        super().__init__()
        self.n_fft = n_fft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute cyclic spectrum.

        Args:
            x: Input signal

        Returns:
            Cyclic spectrum
        """
        n = x.shape[-1]

        if n < self.n_fft:
            x = F.pad(x, (0, self.n_fft - n))

        x_fft = torch.fft.fft(x, dim=-1)

        cyclic_spectrum = torch.outer(x_fft, x_fft.conj())

        cyclic_spectrum = torch.fft.fftshift(cyclic_spectrum, dim=-1)

        return torch.abs(cyclic_spectrum)


class TimeDomainHilbertLayer(nn.Module):
    """Hilbert transform as a differentiable layer."""

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

        self.weight_real = nn.Parameter(torch.randn(num_features, num_features))
        self.weight_imag = nn.Parameter(torch.randn(num_features, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable Hilbert transform.

        Args:
            x: Input signal

        Returns:
            Hilbert transformed signal
        """
        out_real = x @ self.weight_real
        out_imag = x @ self.weight_imag

        return torch.complex(out_real, out_imag)

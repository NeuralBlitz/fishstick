"""
Wavelet Transform Implementations

Advanced wavelet transforms for signal processing including continuous,
discrete, packet, and scattering transforms with PyTorch support.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MorletWavelet(nn.Module):
    """Morlet wavelet for time-frequency analysis."""

    def __init__(
        self,
        center_freq: float = 1.0,
        bandwidth: float = 1.0,
        num_oscillations: float = 6.0,
    ):
        super().__init__()
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.num_oscillations = num_oscillations

    def forward(self, length: int, device: torch.device = None) -> torch.Tensor:
        """Generate Morlet wavelet of specified length."""
        if device is None:
            device = torch.device("cpu")

        t = torch.arange(length, dtype=torch.float32, device=device)
        t = t / self.bandwidth

        envelope = torch.exp(-0.5 * t**2)
        oscillation = torch.exp(
            1j * 2 * np.pi * self.center_freq * t / self.num_oscillations
        )

        wavelet = envelope * oscillation
        return wavelet


class RickerWavelet(nn.Module):
    """Ricker (Mexican Hat) wavelet."""

    def __init__(self, center_freq: float = 1.0, bandwidth: float = 1.0):
        super().__init__()
        self.center_freq = center_freq
        self.bandwidth = bandwidth

    def forward(self, length: int, device: torch.device = None) -> torch.Tensor:
        """Generate Ricker wavelet of specified length."""
        if device is None:
            device = torch.device("cpu")

        t = torch.arange(length, dtype=torch.float32, device=device)
        t = (t - length / 2) / (self.bandwidth * length / 2)

        wavelet = (1 - t**2) * torch.exp(-0.5 * t**2)
        return wavelet


class ContinuousWaveletTransform(nn.Module):
    """Continuous Wavelet Transform for time-frequency analysis.

    Computes the CWT of a 1D signal using a bank of wavelets at different scales.
    """

    def __init__(
        self,
        wavelet: str = "morlet",
        n_scales: int = 32,
        min_scale: float = 1.0,
        max_scale: float = 64.0,
        center_freq: float = 1.0,
    ):
        super().__init__()
        self.n_scales = n_scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.center_freq = center_freq

        if wavelet == "morlet":
            self.wavelet_fn = MorletWavelet(center_freq=center_freq)
        elif wavelet == "ricker":
            self.wavelet_fn = RickerWavelet(center_freq=center_freq)
        else:
            raise ValueError(f"Unknown wavelet: {wavelet}")

        self.scales = torch.linspace(min_scale, max_scale, n_scales)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute CWT of input signal.

        Args:
            signal: Input signal of shape (batch, length) or (length,)

        Returns:
            CWT coefficients of shape (batch, n_scales, length)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size, signal_length = signal.shape
        device = signal.device

        scales = self.scales.to(device)

        cwt_result = torch.zeros(
            batch_size,
            self.n_scales,
            signal_length,
            dtype=torch.complex64,
            device=device,
        )

        for i, scale in enumerate(scales):
            wavelet_length = min(int(signal_length * 0.5), int(64 * scale))
            wavelet_length = max(wavelet_length, 8)
            wavelet_length = (
                wavelet_length if wavelet_length % 2 == 0 else wavelet_length + 1
            )

            wavelet = self.wavelet_fn(wavelet_length, device)
            wavelet = wavelet / torch.max(torch.abs(wavelet))

            signal_padded = F.pad(signal, (wavelet_length // 2, wavelet_length // 2))

            for b in range(batch_size):
                conv_result = F.conv1d(
                    signal_padded[b : b + 1],
                    wavelet.real.unsqueeze(0).unsqueeze(0),
                    padding=wavelet_length // 2,
                )
                conv_imag = F.conv1d(
                    signal_padded[b : b + 1],
                    wavelet.imag.unsqueeze(0).unsqueeze(0),
                    padding=wavelet_length // 2,
                )
                out_len = min(conv_result.shape[-1], signal_length)
                cwt_result[b, i, :out_len] = (
                    conv_result.squeeze(0).squeeze(0)[:out_len]
                    + 1j * conv_imag.squeeze(0).squeeze(0)[:out_len]
                )

        return cwt_result

    def get_scaleogram(self, signal: torch.Tensor) -> torch.Tensor:
        """Get magnitude scaleogram (time-frequency representation)."""
        cwt = self.forward(signal)
        return torch.abs(cwt)


class DiscreteWaveletTransform(nn.Module):
    """Discrete Wavelet Transform using perfect reconstruction filter banks.

    Implements multi-level DWT decomposition and reconstruction.
    """

    SUPPORTED_WAVELETS = ["db1", "db2", "db4", "db8", "haar", "sym2", "coif1"]

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 4,
        mode: str = "symmetric",
    ):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

        self._init_filters()

    def _init_filters(self):
        """Initialize wavelet decomposition filters."""
        import pywt

        wavelet = pywt.Wavelet(self.wavelet)
        self.register_buffer("dec_lo", torch.tensor(wavelet.dec_lo))
        self.register_buffer("dec_hi", torch.tensor(wavelet.dec_hi))
        self.register_buffer("rec_lo", torch.tensor(wavelet.rec_lo))
        self.register_buffer("rec_hi", torch.tensor(wavelet.rec_hi))

    def forward(self, signal: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute DWT decomposition.

        Args:
            signal: Input signal of shape (batch, length) or (length,)

        Returns:
            Tuple of (approximation coefficients, detail coefficients list)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size, length = signal.shape
        device = signal.device

        approx = signal.float()
        details = []

        for _ in range(self.level):
            if approx.shape[-1] < len(self.dec_lo):
                break

            approx, detail = self._dwt_step(approx)
            details.append(detail)

        return approx, details

    def _dwt_step(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single level DWT decomposition."""
        lo = self.dec_lo.to(signal.device)
        hi = self.dec_hi.to(signal.device)

        lo = lo.unsqueeze(0).unsqueeze(0)
        hi = hi.unsqueeze(0).unsqueeze(0)

        approx = F.conv1d(signal, lo, padding=lo.shape[-1] // 2, stride=2)
        detail = F.conv1d(signal, hi, padding=hi.shape[-1] // 2, stride=2)

        return approx, detail

    def inverse(
        self,
        approx: torch.Tensor,
        details: List[torch.Tensor],
    ) -> torch.Tensor:
        """Inverse DWT reconstruction.

        Args:
            approx: Approximation coefficients
            details: List of detail coefficients

        Returns:
            Reconstructed signal
        """
        signal = approx

        for detail in reversed(details):
            signal = self._idwt_step(signal, detail)

        return signal

    def _idwt_step(self, approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        """Single level IDWT reconstruction."""
        lo = self.rec_lo.to(approx.device)
        hi = self.rec_hi.to(approx.device)

        lo = lo.unsqueeze(0).unsqueeze(0)
        hi = hi.unsqueeze(0).unsqueeze(0)

        approx_up = F.conv_transpose1d(approx, lo, stride=2, padding=lo.shape[-1] // 2)
        detail_up = F.conv_transpose1d(detail, hi, stride=2, padding=hi.shape[-1] // 2)

        return approx_up + detail_up


class WaveletPacketTransform(nn.Module):
    """Wavelet Packet Transform - full decomposition tree."""

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 3,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

        self.dwt = DiscreteWaveletTransform(wavelet=wavelet, level=level)

    def forward(self, signal: torch.Tensor) -> Tuple[List[torch.Tensor], List[int]]:
        """Compute full wavelet packet decomposition.

        Returns:
            Tuple of (coefficients list, frequency band indices)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        approx, details = self.dwt(signal)

        packets = [approx]
        for detail in details:
            packets.append(detail)

        return packets, list(range(len(packets)))


class WaveletScatteringTransform(nn.Module):
    """Wavelet Scattering Transform for invariant feature extraction.

    Computes scattering coefficients that are invariant to translations
    and stable to deformations.
    """

    def __init__(
        self,
        signal_length: int = 8192,
        sample_rate: int = 16000,
        J: int = 12,
        Q: int = 8,
        max_order: int = 2,
        average: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.signal_length = signal_length
        self.sample_rate = sample_rate
        self.J = J
        self.Q = Q
        self.max_order = max_order
        self.average = average
        self.pad_mode = pad_mode

        self._build_wavelet_filters()
        self._compute_meta()

    def _build_wavelet_filters(self):
        """Build the Morlet wavelet filter bank."""
        J, Q = self.J, self.Q

        frequencies = []
        for j in range(J):
            for q in range(Q):
                freq = 2 ** (-j - q / Q)
                frequencies.append(freq)

        frequencies = sorted(set(frequencies), reverse=True)

        filters = []
        for freq in frequencies:
            phi, psi = self._morlet_wavelet(freq, Q)
            filters.append((phi, psi, freq))

        self.filters = filters

    def _morlet_wavelet(self, freq: float, Q: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate Morlet wavelet at given frequency."""
        sigma = freq / Q
        xi = freq

        length = int(2 ** (np.ceil(np.log2(1 / sigma))) * 8)
        length = length if length % 2 == 0 else length + 1

        t = torch.linspace(-length // 2, length // 2, length)
        t = t / sigma

        envelope = torch.exp(-0.5 * t**2)
        oscillation = torch.exp(1j * 2 * np.pi * xi * t / sigma)

        psi = envelope * oscillation

        lowpass_length = min(int(2**J), length)
        phi = torch.exp(-0.5 * (t[:lowpass_length] / (2**self.J)) ** 2)

        return phi, psi

    def _compute_meta(self):
        """Precompute metadata for scattering."""
        self.meta = []

        if self.max_order >= 1:
            for phi, psi, freq in self.filters:
                self.meta.append({"order": 1, "psi": psi, "phi": phi, "freq": freq})

        if self.max_order >= 2:
            for i, (phi1, psi1, freq1) in enumerate(self.filters):
                for j, (phi2, psi2, freq2) in enumerate(self.filters):
                    if freq2 < freq1:
                        self.meta.append(
                            {
                                "order": 2,
                                "psi1": psi1,
                                "psi2": psi2,
                                "phi": phi1,
                                "freq1": freq1,
                                "freq2": freq2,
                            }
                        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute scattering transform.

        Args:
            signal: Input signal of shape (batch, length)

        Returns:
            Scattering coefficients
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size = signal.shape[0]
        device = signal.device
        outputs = []

        for meta in self.meta:
            if meta["order"] == 1:
                S = self._first_order(signal, meta, device)
            else:
                S = self._second_order(signal, meta, device)

            outputs.append(S)

        return torch.cat(outputs, dim=1)

    def _first_order(
        self,
        signal: torch.Tensor,
        meta: dict,
        device: torch.device,
    ) -> torch.Tensor:
        """First-order scattering."""
        psi = meta["psi"].to(device)
        phi = meta["phi"].to(device)

        signal_padded = F.pad(signal, (psi.shape[-1] // 2, psi.shape[-1] // 2))

        U1 = F.conv1d(
            signal_padded, psi.unsqueeze(0).unsqueeze(0), padding=psi.shape[-1] // 2
        )
        U1 = torch.abs(U1)

        if self.average:
            phi_padded = F.pad(
                phi.unsqueeze(0).unsqueeze(0), (phi.shape[-1] // 2, phi.shape[-1] // 2)
            )
            S = F.conv1d(U1, phi_padded, padding=phi.shape[-1] // 2)
        else:
            S = U1.mean(dim=-1, keepdim=True)

        return S

    def _second_order(
        self,
        signal: torch.Tensor,
        meta: dict,
        device: torch.device,
    ) -> torch.Tensor:
        """Second-order scattering."""
        psi1 = meta["psi1"].to(device)
        psi2 = meta["psi2"].to(device)
        phi = meta["phi"].to(device)

        signal_padded = F.pad(signal, (psi1.shape[-1] // 2, psi1.shape[-1] // 2))
        U1 = F.conv1d(
            signal_padded, psi1.unsqueeze(0).unsqueeze(0), padding=psi1.shape[-1] // 2
        )
        U1 = torch.abs(U1)

        U1_padded = F.pad(U1, (psi2.shape[-1] // 2, psi2.shape[-1] // 2))
        U2 = F.conv1d(
            U1_padded, psi2.unsqueeze(0).unsqueeze(0), padding=psi2.shape[-1] // 2
        )
        U2 = torch.abs(U2)

        if self.average:
            phi_padded = F.pad(
                phi.unsqueeze(0).unsqueeze(0), (phi.shape[-1] // 2, phi.shape[-1] // 2)
            )
            S = F.conv1d(U2, phi_padded, padding=phi.shape[-1] // 2)
        else:
            S = U2.mean(dim=-1, keepdim=True)

        return S


class InverseWaveletTransform(nn.Module):
    """Inverse wavelet transform for signal reconstruction."""

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 4,
    ):
        super().__init__()
        self.dwt = DiscreteWaveletTransform(wavelet=wavelet, level=level)

    def forward(
        self,
        approx: torch.Tensor,
        details: List[torch.Tensor],
    ) -> torch.Tensor:
        """Reconstruct signal from wavelet coefficients."""
        return self.dwt.inverse(approx, details)

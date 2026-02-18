"""
Time-Frequency Analysis Tools

Tools for analyzing signals in both time and frequency domains simultaneously,
including STFT, CQT, and synchrosqueezing transforms.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ShortTimeFourierTransform(nn.Module):
    """Short-Time Fourier Transform for time-frequency analysis.

    Computes the STFT of a signal to produce a time-frequency representation.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.center = center
        self.normalized = normalized
        self.onesided = onesided

        if window == "hann":
            self.window = torch.hann_window(self.win_length)
        elif window == "hamming":
            self.window = torch.hamming_window(self.win_length)
        elif window == "blackman":
            self.window = torch.blackman_window(self.win_length)
        elif window == "kaiser":
            self.window = torch.kaiser_window(self.win_length)
        else:
            raise ValueError(f"Unknown window type: {window}")

    def forward(
        self,
        signal: torch.Tensor,
        return_complex: bool = True,
    ) -> torch.Tensor:
        """Compute STFT.

        Args:
            signal: Input signal (batch, length) or (length,)
            return_complex: Whether to return complex output

        Returns:
            STFT tensor (batch, n_fft//2+1, frames) or (n_fft//2+1, frames)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        device = signal.device
        window = self.window.to(device)

        stft_result = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )

        if not return_complex:
            return torch.abs(stft_result)

        return stft_result

    def get_magnitude(self, signal: torch.Tensor) -> torch.Tensor:
        """Get STFT magnitude."""
        stft = self.forward(signal, return_complex=True)
        return torch.abs(stft)

    def get_phase(self, signal: torch.Tensor) -> torch.Tensor:
        """Get STFT phase."""
        stft = self.forward(signal, return_complex=True)
        return torch.angle(stft)

    def get_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        """Get power spectrogram in dB."""
        mag = self.get_magnitude(signal)
        spec_db = 20 * torch.log10(mag + 1e-9)
        return spec_db


class InverseSTFT(nn.Module):
    """Inverse Short-Time Fourier Transform."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.center = center
        self.normalized = normalized

        if window == "hann":
            self.window = torch.hann_window(self.win_length)
        elif window == "hamming":
            self.window = torch.hamming_window(self.win_length)
        else:
            self.window = torch.hann_window(self.win_length)

    def forward(self, stft: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """Compute inverse STFT.

        Args:
            stft: STFT tensor (batch, freq, time) or (freq, time)
            length: Expected output length

        Returns:
            Reconstructed signal (batch, length) or (length,)
        """
        if stft.dim() == 2:
            stft = stft.unsqueeze(0)

        device = stft.device
        window = self.window.to(device)

        signal = torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            length=length,
        )

        return signal


class ConstantQTransform(nn.Module):
    """Constant-Q Transform for logarithmic frequency resolution.

    Provides constant-Q frequency bins which is more aligned with
    human auditory perception.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        fmin: float = 20.0,
        fmax: float = 8000.0,
        bins_per_octave: int = 12,
        norm: bool = True,
        filter_scale: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.bins_per_octave = bins_per_octave
        self.norm = norm
        self.filter_scale = filter_scale

        self.n_bins = self._compute_n_bins()
        self.q = self._compute_q()

        self.filters = self._create_filter_bank()

    def _compute_n_bins(self) -> int:
        """Compute number of frequency bins."""
        return int(np.ceil(self.bins_per_octave * np.log2(self.fmax / self.fmin)))

    def _compute_q(self) -> float:
        """Compute Q factor."""
        return (2 ** (1 / self.bins_per_octave) - 1) / (
            2 ** (1 / self.bins_per_octave) + 1
        )

    def _create_filter_bank(self) -> torch.Tensor:
        """Create CQT filter bank."""
        n_fft = 2048

        filters = torch.zeros(self.n_bins, n_fft // 2 + 1)

        for k in range(self.n_bins):
            fk = self.fmin * 2 ** (k / self.bins_per_octave)
            n = torch.arange(n_fft // 2 + 1)
            freq = n * self.sample_rate / n_fft

            filter_response = torch.exp(
                -2 * np.pi * 1j * freq / fk * self.q
            ) * torch.exp(-0.5 * ((freq - fk) / (fk * self.filter_scale / self.q)) ** 2)

            filters[k] = torch.abs(filter_response)

        if self.norm:
            filters = filters / torch.sqrt(torch.sum(filters**2, dim=1, keepdim=True))

        return filters

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute CQT.

        Args:
            signal: Input signal (batch, length) or (length,)

        Returns:
            CQT tensor (batch, n_bins, time)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        device = signal.device
        filters = self.filters.to(device)

        n_fft = 2048

        stft = torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=torch.hann_window(n_fft, device=device),
            center=True,
            return_complex=True,
        )

        cqt = torch.matmul(filters.unsqueeze(0).to(stft.dtype), stft)

        return torch.abs(cqt)


class SynchrosqueezingTransform(nn.Module):
    """Synchrosqueezing Transform for enhanced time-frequency representation.

    Refines STFT by reassigning energy in the time-frequency plane
    based on instantaneous frequency estimates.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft

        self.stft = ShortTimeFourierTransform(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    def compute_instantaneous_frequency(
        self,
        stft: torch.Tensor,
    ) -> torch.Tensor:
        """Compute instantaneous frequency from STFT.

        Args:
            stft: STFT tensor (batch, freq, time)

        Returns:
            Instantaneous frequency (batch, freq, time)
        """
        phase = torch.angle(stft)
        inst_freq = torch.angle(stft[:, :, 1:] * torch.conj(stft[:, :, :-1]))
        inst_freq = F.pad(inst_freq, (1, 0), mode="replicate")

        freq_bins = torch.fft.rfftfreq(self.n_fft).to(stft.device)
        inst_freq = freq_bins.view(1, -1, 1) + inst_freq / (2 * np.pi * self.hop_length)

        return inst_freq

    def forward(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute synchrosqueezing transform.

        Args:
            signal: Input signal (batch, length) or (length,)

        Returns:
            Tuple of (synchrosqueezed spectrogram, magnitude)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        stft = self.stft(signal, return_complex=True)
        magnitude = torch.abs(stft)
        inst_freq = self.compute_instantaneous_frequency(stft)

        n_freq = magnitude.shape[1]
        n_time = magnitude.shape[2]
        device = signal.device

        freq_bins = torch.fft.rfftfreq(self.n_fft).to(device)

        sst = torch.zeros_like(magnitude)

        for t in range(n_time):
            for f in range(n_freq):
                if f > 0 and f < n_freq - 1:
                    if_inst = inst_freq[0, f, t]
                    f_idx = torch.argmin(torch.abs(freq_bins - if_inst))
                    if f_idx < n_freq:
                        sst[f_idx, t] += magnitude[f, t]

        return sst, magnitude


class ReassignmentMethod(nn.Module):
    """Reassignment method for time-frequency analysis.

    Improves localization in time-frequency plane by reassigning
    energy based on group delay.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.stft = ShortTimeFourierTransform(
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def compute_group_delay(
        self,
        stft: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute group delay for reassignment.

        Returns:
            Tuple of (time reassignment, frequency reassignment)
        """
        freq_bins = torch.fft.rfftfreq(self.n_fft).unsqueeze(-1).to(stft.device)

        stft_conj = torch.conj(stft)

        d_stft_domega = (stft[:, 1:, :] - stft[:, :-1, :]) * freq_bins[1:, :]

        t_reassignment = -torch.real(d_stft_domega * stft_conj[:, :-1, :]) / (
            torch.abs(stft[:, :-1, :]) ** 2 + 1e-9
        )
        f_reassignment = torch.imag(d_stft_domega * stft_conj[:, :-1, :]) / (
            2 * np.pi * self.hop_length * torch.abs(stft[:, :-1, :]) ** 2 + 1e-9
        )

        t_reassignment = F.pad(t_reassignment, (1, 0), mode="replicate")
        f_reassignment = F.pad(f_reassignment, (1, 0), mode="replicate")

        return t_reassignment, f_reassignment

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply reassignment method."""
        stft = self.stft(signal, return_complex=True)
        magnitude = torch.abs(stft)

        t_reassign, f_reassign = self.compute_group_delay(stft)

        return magnitude


class GaborTransform(nn.Module):
    """Gabor transform for time-frequency analysis using Gaussian windows."""

    def __init__(
        self,
        num_frequencies: int = 64,
        num_times: int = 64,
        sigma: float = 0.1,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.num_times = num_times
        self.sigma = sigma

        self.frequencies = torch.linspace(0, 0.5, num_frequencies)
        self.times = torch.linspace(0, 1, num_times)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute Gabor transform.

        Args:
            signal: Input signal (batch, length) or (length,)

        Returns:
            Gabor representation (batch, freq, time)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size, length = signal.shape
        device = signal.device

        result = torch.zeros(
            batch_size, self.num_frequencies, self.num_times, device=device
        )

        time_indices = torch.linspace(0, length - 1, self.num_times).long()
        freq_indices = torch.linspace(0, length // 2, self.num_frequencies).long()

        for t_idx, t in enumerate(time_indices):
            for f_idx, f in enumerate(freq_indices):
                window = self._gaussian_window(length, t, self.sigma * length)
                modulation = torch.exp(
                    -2j * np.pi * f * torch.arange(length, device=device) / length
                )
                gabor_atom = window * modulation

                result[:, f_idx, t_idx] = torch.sum(signal * gabor_atom, dim=-1)

        return result

    def _gaussian_window(self, length: int, center: int, sigma: float) -> torch.Tensor:
        """Create Gaussian window."""
        t = torch.arange(length, dtype=torch.float32)
        window = torch.exp(-0.5 * ((t - center) / sigma) ** 2)
        return window


class AdaptiveTimeFrequency(nn.Module):
    """Adaptive time-frequency representation with learned parameters."""

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 32,
        learn_window: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.learn_window = learn_window

        if learn_window:
            self.window_params = nn.Parameter(
                torch.randn(num_filters, input_dim) * 0.02
            )
        else:
            self.register_buffer("window_params", None)

        self.freq_params = nn.Parameter(
            torch.linspace(0.1, 0.5, num_filters).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute adaptive time-frequency representation."""
        batch_size, length = x.shape
        device = x.device

        if self.learn_window:
            windows = F.softmax(self.window_params, dim=-1)
        else:
            windows = torch.eye(self.num_filters, self.input_dim, device=device)

        frequencies = torch.sigmoid(self.freq_params) * 0.5
        frequencies = frequencies.to(device)

        result = torch.zeros(batch_size, self.num_filters, length, device=device)

        for i in range(self.num_filters):
            window = windows[i]
            freq = frequencies[0, i]

            modulation = torch.exp(
                -2j * np.pi * freq * torch.arange(length, device=device)
            )

            atom = window.unsqueeze(0) * modulation

            result[:, i, :] = torch.real(
                torch.fft.ifft(
                    torch.fft.fft(x, dim=-1)
                    * torch.conj(torch.fft.fft(atom.unsqueeze(0), n=length, dim=-1)),
                    dim=-1,
                )
            )

        return result

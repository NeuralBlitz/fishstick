"""
Signal Processing Utilities

Utility functions for signal preprocessing, window functions,
and common signal processing operations.
"""

from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WindowFunctions:
    """Collection of window functions for signal processing."""

    @staticmethod
    def hann(length: int, device: torch.device = None) -> torch.Tensor:
        """Hann (Hanning) window."""
        if device is None:
            device = torch.device("cpu")
        return torch.hann_window(length, device=device)

    @staticmethod
    def hamming(length: int, device: torch.device = None) -> torch.Tensor:
        """Hamming window."""
        if device is None:
            device = torch.device("cpu")
        return torch.hamming_window(length, device=device)

    @staticmethod
    def blackman(length: int, device: torch.device = None) -> torch.Tensor:
        """Blackman window."""
        if device is None:
            device = torch.device("cpu")
        return torch.blackman_window(length, device=device)

    @staticmethod
    def kaiser(
        length: int, beta: float = 12.0, device: torch.device = None
    ) -> torch.Tensor:
        """Kaiser window."""
        if device is None:
            device = torch.device("cpu")
        return torch.kaiser_window(length, beta=beta, device=device)

    @staticmethod
    def nuttall(length: int, device: torch.device = None) -> torch.Tensor:
        """Nuttall window (4-term Blackman-Harris)."""
        if device is None:
            device = torch.device("cpu")
        n = torch.arange(length, dtype=torch.float32, device=device)
        a0, a1, a2, a3 = 0.3635819, 0.4891775, 0.1365995, 0.0106411
        return (
            a0
            - a1 * torch.cos(2 * np.pi * n / (length - 1))
            + a2 * torch.cos(4 * np.pi * n / (length - 1))
            - a3 * torch.cos(6 * np.pi * n / (length - 1))
        )

    @staticmethod
    def gaussian(
        length: int, sigma: float = 0.5, device: torch.device = None
    ) -> torch.Tensor:
        """Gaussian window."""
        if device is None:
            device = torch.device("cpu")
        n = torch.arange(length, dtype=torch.float32, device=device)
        n = (n - (length - 1) / 2) / (sigma * (length - 1) / 2)
        return torch.exp(-0.5 * n**2)

    @staticmethod
    def tukey(
        length: int, alpha: float = 0.5, device: torch.device = None
    ) -> torch.Tensor:
        """Tukey (tapered cosine) window."""
        if device is None:
            device = torch.device("cpu")
        if alpha <= 0:
            return torch.ones(length, device=device)
        elif alpha >= 1:
            return torch.hann_window(length, device=device)

        n = torch.arange(length, dtype=torch.float32, device=device)
        left = int(alpha * (length - 1) / 2)

        window = torch.ones(length, device=device)

        for i in range(left + 1):
            window[i] = 0.5 * (
                1 + np.cos(np.pi * (-1 + 2 * i / (alpha * (length - 1))))
            )

        for i in range(length - left - 1, length):
            window[i] = 0.5 * (
                1 + np.cos(np.pi * (-1 + 2 * (length - 1 - i) / (alpha * (length - 1))))
            )

        return window


class SignalNormalizer(nn.Module):
    """Signal normalization utilities."""

    def __init__(
        self,
        method: str = "standard",
        dim: int = -1,
    ):
        super().__init__()
        self.method = method
        self.dim = dim

        self.register_buffer("mean", None)
        self.register_buffer("std", None)

    def fit(self, signal: torch.Tensor):
        """Compute normalization parameters from data."""
        if self.method == "standard":
            self.mean = signal.mean(dim=self.dim, keepdim=True)
            self.std = signal.std(dim=self.dim, keepdim=True) + 1e-8
        elif self.method == "minmax":
            self.mean = signal.min(dim=self.dim, keepdim=True)[0]
            self.std = signal.max(dim=self.dim, keepdim=True)[0] - self.mean + 1e-8
        elif self.method == "l2":
            self.mean = torch.zeros_like(signal)
            self.std = torch.norm(signal, dim=self.dim, keepdim=True) + 1e-8

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Normalize signal."""
        if self.method == "standard":
            return (signal - self.mean) / self.std
        elif self.method == "minmax":
            return (signal - self.mean) / self.std
        elif self.method == "l2":
            return signal / self.std
        return signal

    def inverse(self, signal: torch.Tensor) -> torch.Tensor:
        """Denormalize signal."""
        if self.method == "standard":
            return signal * self.std + self.mean
        elif self.method == "minmax":
            return signal * self.std + self.mean
        elif self.method == "l2":
            return signal * self.std
        return signal


class SignalPreprocessor(nn.Module):
    """Complete signal preprocessing pipeline."""

    def __init__(
        self,
        sample_rate: int = 16000,
        normalize: bool = True,
        remove_dc: bool = True,
        preemphasis: float = 0.97,
        target_length: Optional[int] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.remove_dc = remove_dc
        self.preemphasis = preemphasis
        self.target_length = target_length

        if normalize:
            self.normalizer = SignalNormalizer(method="standard")
        else:
            self.normalizer = None

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Preprocess signal."""
        if self.normalize and self.normalizer is not None:
            signal = self.normalizer(signal)

        if self.remove_dc:
            signal = signal - signal.mean(dim=-1, keepdim=True)

        if self.preemphasis > 0:
            signal = self._preemphasis(signal)

        if self.target_length is not None:
            signal = self._pad_or_truncate(signal, self.target_length)

        return signal

    def _preemphasis(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply pre-emphasis filter."""
        return signal - self.preemphasis * F.pad(signal[..., :-1], (1, 0))

    def _pad_or_truncate(
        self, signal: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        """Pad or truncate signal to target length."""
        current_length = signal.shape[-1]

        if current_length < target_length:
            padding = target_length - current_length
            return F.pad(signal, (0, padding))
        elif current_length > target_length:
            return signal[..., :target_length]

        return signal


class SignalAugmentation(nn.Module):
    """Signal augmentation for training."""

    def __init__(
        self,
        sample_rate: int = 16000,
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_range: Tuple[float, float] = (-2, 2),
        noise_level: float = 0.005,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        if self.training:
            if torch.rand(1).item() < 0.5:
                signal = self.add_noise(signal)

            if torch.rand(1).item() < 0.3:
                signal = self.time_stretch(signal)

            if torch.rand(1).item() < 0.3:
                signal = self.pitch_shift(signal)

        return signal

    def add_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """Add random Gaussian noise."""
        noise = torch.randn_like(signal) * self.noise_level
        return signal + noise

    def time_stretch(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply random time stretching."""
        rate = torch.empty(1).uniform_(*self.time_stretch_range).item()

        length = signal.shape[-1]
        new_length = int(length * rate)

        indices = torch.linspace(0, length - 1, new_length, device=signal.device)
        indices = indices.long().clamp(0, length - 1)

        return signal[..., indices]

    def pitch_shift(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply random pitch shifting using phase vocoder approach."""
        semitones = torch.empty(1).uniform_(*self.pitch_shift_range).item()
        rate = 2 ** (semitones / 12)

        length = signal.shape[-1]
        new_length = int(length / rate)

        indices = torch.linspace(0, length - 1, new_length, device=signal.device)
        indices = indices.long().clamp(0, length - 1)

        return signal[..., indices]


class SignalGenerator(nn.Module):
    """Generate various synthetic signals."""

    @staticmethod
    def sine(
        length: int,
        frequency: float = 1.0,
        phase: float = 0.0,
        amplitude: float = 1.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Generate sine wave."""
        if device is None:
            device = torch.device("cpu")
        t = torch.arange(length, dtype=torch.float32, device=device)
        return amplitude * torch.sin(2 * np.pi * frequency * t + phase)

    @staticmethod
    def square(
        length: int,
        frequency: float = 1.0,
        duty_cycle: float = 0.5,
        amplitude: float = 1.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Generate square wave."""
        if device is None:
            device = torch.device("cpu")
        t = torch.arange(length, dtype=torch.float32, device=device)
        period = 1 / frequency
        phase = (t % period) / period
        return amplitude * (phase < duty_cycle).float() * 2 - amplitude

    @staticmethod
    def sawtooth(
        length: int,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Generate sawtooth wave."""
        if device is None:
            device = torch.device("cpu")
        t = torch.arange(length, dtype=torch.float32, device=device)
        period = 1 / frequency
        phase = (t % period) / period
        return amplitude * (2 * phase - 1)

    @staticmethod
    def noise(
        length: int,
        noise_type: str = "white",
        device: torch.device = None,
    ) -> torch.Tensor:
        """Generate noise signal."""
        if device is None:
            device = torch.device("cpu")

        if noise_type == "white":
            return torch.randn(length, device=device)
        elif noise_type == "brown":
            white = torch.randn(length, device=device)
            return torch.cumsum(white, dim=-1) / length
        elif noise_type == "pink":
            white = torch.randn(length, device=device)
            pink = torch.zeros(length, device=device)
            b0, b1, b2, b3, b4, b5, b6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for i in range(length):
                white_i = white[i].item()
                b0 = 0.99886 * b0 + white_i * 0.0555179
                b1 = 0.99332 * b1 + white_i * 0.0750759
                b2 = 0.96900 * b2 + white_i * 0.1538520
                b3 = 0.86650 * b3 + white_i * 0.3104856
                b4 = 0.55000 * b4 + white_i * 0.5329522
                b5 = -0.7616 * b5 - white_i * 0.0168980
                pink[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white_i * 0.5362) * 0.11
                b6 = white_i * 0.115926
            return pink
        return torch.zeros(length, device=device)

    @staticmethod
    def chirp(
        length: int,
        f0: float = 0.01,
        f1: float = 0.5,
        method: str = "linear",
        device: torch.device = None,
    ) -> torch.Tensor:
        """Generate chirp signal."""
        if device is None:
            device = torch.device("cpu")
        t = torch.arange(length, dtype=torch.float32, device=device)

        if method == "linear":
            freq = f0 + (f1 - f0) * t / length
        elif method == "exponential":
            freq = f0 * (f1 / f0) ** (t / length)
        else:
            freq = f0 + (f1 - f0) * t / length

        phase = torch.cumsum(freq, dim=0)
        return torch.sin(2 * np.pi * phase)


class SignalQualityMetrics:
    """Signal quality assessment metrics."""

    @staticmethod
    def snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
        """Calculate Signal-to-Noise Ratio in dB."""
        signal_power = torch.mean(signal**2)
        noise_power = torch.mean(noise**2)
        return 10 * torch.log10(signal_power / (noise_power + 1e-9)).item()

    @staticmethod
    def sdg(signal: torch.Tensor) -> float:
        """Spectral distortion gradient."""
        signal_fft = torch.fft.rfft(signal)
        magnitude = torch.abs(signal_fft)
        gradient = torch.abs(magnitude[1:] - magnitude[:-1])
        return torch.mean(gradient).item()

    @staticmethod
    def spectral_flatness(signal: torch.Tensor) -> float:
        """Spectral flatness (Wiener entropy)."""
        spectrum = torch.abs(torch.fft.rfft(signal)) ** 2
        geometric_mean = torch.exp(torch.mean(torch.log(spectrum + 1e-9)))
        arithmetic_mean = torch.mean(spectrum)
        return (geometric_mean / (arithmetic_mean + 1e-9)).item()

    @staticmethod
    def zero_crossing_rate(signal: torch.Tensor) -> float:
        """Zero crossing rate."""
        zero_crossings = torch.sum(torch.abs(torch.diff(torch.sign(signal)))) / 2
        return (zero_crossings / signal.shape[-1]).item()


class Resample1D(nn.Module):
    """Signal resampling layer."""

    def __init__(
        self,
        orig_freq: int = 16000,
        target_freq: int = 8000,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.target_freq = target_freq

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Resample signal to target frequency."""
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        if self.orig_freq == self.target_freq:
            return signal

        ratio = self.target_freq / self.orig_freq
        new_length = int(signal.shape[-1] * ratio)

        indices = torch.linspace(
            0, signal.shape[-1] - 1, new_length, device=signal.device
        )
        indices_int = indices.long()
        indices_frac = indices - indices_int.float()

        signal_int = signal[..., indices_int]

        signal_next = signal[
            ..., torch.min(indices_int + 1, torch.tensor(signal.shape[-1] - 1))
        ]

        output = signal_int * (1 - indices_frac) + signal_next * indices_frac

        return output.squeeze(0) if output.shape[0] == 1 else output


class SignalProcessor(nn.Module):
    """Complete signal processing module combining common operations."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        n_mels: int = 80,
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.preprocessor = SignalPreprocessor(
            sample_rate=sample_rate,
            normalize=normalize,
        )

        self.mel_basis = self._create_mel_filter()

    def _create_mel_filter(self) -> torch.Tensor:
        """Create mel filterbank."""
        n_freqs = self.n_fft // 2 + 1
        fmin_mel = 2595 * np.log10(1 + 0 / 700)
        fmax_mel = 2595 * np.log10(1 + self.sample_rate / 2 / 700)

        mel_points = torch.linspace(fmin_mel, fmax_mel, self.n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = torch.floor((self.n_fft + 1) * hz_points / self.sample_rate).long()

        filterbank = torch.zeros(self.n_mels, n_freqs)
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Process signal through full pipeline."""
        signal = self.preprocessor(signal)

        spec = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft, device=signal.device),
            return_complex=True,
        )

        spec = torch.abs(spec) ** 2

        mel_spec = torch.matmul(self.mel_basis.to(signal.device), spec)
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec

"""
Audio Feature Extraction

Audio feature extraction using torch and scipy.
"""

from typing import Optional
import torch
import numpy as np
from scipy import signal


class MelSpectrogram:
    """Extract mel spectrogram features."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        fmin: float = 0,
        fmax: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate / 2

        self.mel_basis = self._create_mel_filter()

    def _create_mel_filter(self) -> torch.Tensor:
        n_freqs = self.n_fft // 2 + 1
        fmin_mel = self._hz_to_mel(self.fmin)
        fmax_mel = self._hz_to_mel(self.fmax)

        mel_points = torch.linspace(fmin_mel, fmax_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
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

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        )
        spec = torch.abs(spec) ** 2
        mel_spec = torch.matmul(self.mel_basis.to(spec.device), spec)
        mel_spec = torch.log(mel_spec + 1e-9)
        return mel_spec


class MFCC:
    """Extract MFCC features."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mfcc: int = 13,
        n_mels: int = 80,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

        self.mel_spec = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spec(audio)
        dct = self._dct_matrix(mel.shape[1])[: self.n_mfcc]
        mfcc = torch.matmul(dct.to(mel.device), mel)
        return mfcc

    def _dct_matrix(self, n_coeff: int) -> torch.Tensor:
        dct = torch.zeros(n_coeff, n_coeff)
        for i in range(n_coeff):
            for j in range(n_coeff):
                dct[i, j] = torch.cos(np.pi * i * (j + 0.5) / n_coeff)
        return dct


class Spectrogram:
    """Extract spectrogram features."""

    def __init__(
        self,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: Optional[int] = None,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length),
            return_complex=True,
        )
        return torch.abs(spec)


class ChromaFeatures:
    """Extract chroma features."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 4096,
        hop_length: int = 512,
        n_chroma: int = 12,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        )
        spec = torch.abs(spec)

        chroma = torch.zeros(self.n_chroma, spec.shape[1])
        for i in range(self.n_chroma):
            freq = 440 * 2 ** ((i - 9) / 12)
            bin_idx = int(freq * self.n_fft / self.sample_rate)
            if bin_idx < spec.shape[0]:
                chroma[i] = spec[bin_idx]

        return chroma


class SpectralContrast:
    """Extract spectral contrast features."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_bands: int = 6,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bands = n_bands

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        )
        spec = torch.abs(spec)
        spec = spec[: self.n_fft // 2 + 1, :]

        freq_bands = torch.linspace(0, spec.shape[0], self.n_bands + 2).long()
        contrast = []
        for i in range(self.n_bands):
            low = freq_bands[i]
            high = freq_bands[i + 2]
            band = spec[low:high, :]

            peak, valley = band.max(dim=0).values, band.min(dim=0).values
            contrast.append(peak - valley)

        return torch.stack(contrast)

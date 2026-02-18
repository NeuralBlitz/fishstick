"""
Advanced Spectral Feature Extraction

This module provides advanced spectral analysis features for audio processing:
- Chroma features (chroma STFT, chroma CQT, chroma cens)
- Spectral contrast
- Tonnetz (tonal centroid features)
- Spectral rolloff, centroid, flux, flatness
- Zero crossing rate
"""

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import numpy as np
from scipy import signal


@dataclass
class SpectralConfig:
    """Configuration for spectral feature extraction."""

    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_chroma: int = 12
    n_bands: int = 6
    fmin: float = 27.5
    fmax: float = 14080.0
    norm: Optional[str] = "euclidean"
    cmap: Optional[str] = None


class SpectralFeatures:
    """Extract various spectral features from audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.window = torch.hann_window(win_length)

    def spectral_centroid(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute spectral centroid.

        Args:
            spec: Magnitude spectrogram (n_freqs, n_frames)

        Returns:
            Spectral centroid per frame (n_frames,)
        """
        freqs = torch.fft.rfftfreq(self.n_fft, 1 / self.sample_rate).to(spec.device)
        freq_mag = spec * freqs.unsqueeze(1)
        centroid = freq_mag.sum(dim=0) / (spec.sum(dim=0) + 1e-10)
        return centroid

    def spectral_rolloff(
        self, spec: torch.Tensor, rolloff_percent: float = 0.85
    ) -> torch.Tensor:
        """Compute spectral rolloff.

        Args:
            spec: Magnitude spectrogram (n_freqs, n_frames)
            rolloff_percent: Percentile for rolloff (default 85%)

        Returns:
            Spectral rolloff per frame in Hz (n_frames,)
        """
        cumsum = torch.cumsum(spec, dim=0)
        total = cumsum[-1:, :]

        threshold = rolloff_percent * total
        indices = (cumsum >= threshold).float().argmax(dim=0)

        freqs = torch.fft.rfftfreq(self.n_fft, 1 / self.sample_rate).to(spec.device)
        rolloff = freqs[indices]
        return rolloff

    def spectral_flux(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute spectral flux.

        Args:
            spec: Magnitude spectrogram (n_freqs, n_frames)

        Returns:
            Spectral flux per frame (n_frames - 1,)
        """
        diff = spec[:, 1:] - spec[:, :-1]
        flux = torch.sqrt(torch.sum(diff**2, dim=0))
        return flux

    def spectral_flatness(self, spec: torch.Tensor) -> torch.Tensor:
        """Compute spectral flatness (Wiener entropy).

        Args:
            spec: Magnitude spectrogram (n_freqs, n_frames)

        Returns:
            Spectral flatness per frame (n_frames,)
        """
        log_spec = torch.log(spec + 1e-10)
        arithmetic_mean = spec.mean(dim=0)
        geometric_mean = torch.exp(log_spec.mean(dim=0))
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        return flatness

    def zero_crossing_rate(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute zero crossing rate.

        Args:
            audio: Audio waveform (n_samples,)

        Returns:
            Zero crossing rate
        """
        zero_crossings = torch.sum(torch.abs(torch.diff(torch.sign(audio)))) / 2
        return zero_crossings / audio.shape[0]

    def compute_all(self, audio: torch.Tensor) -> dict:
        """Compute all spectral features.

        Args:
            audio: Audio waveform (n_samples,) or (batch, n_samples)

        Returns:
            Dictionary of features
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(audio.device),
            return_complex=True,
        )
        spec = torch.abs(spec)

        features = {}
        features["centroid"] = self.spectral_centroid(spec)
        features["rolloff"] = self.spectral_rolloff(spec)
        features["flux"] = self.spectral_flux(spec)
        features["flatness"] = self.spectral_flatness(spec)

        if audio.shape[0] == 1:
            features["zcr"] = self.zero_crossing_rate(audio[0])
        else:
            features["zcr"] = torch.stack([self.zero_crossing_rate(a) for a in audio])

        return features


class ChromaFeatures:
    """Extract chroma features from audio.

    Chroma features capture the harmonic content of audio
    mapped to 12 pitch classes.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 4096,
        hop_length: int = 512,
        n_chroma: int = 12,
        norm: Optional[str] = "euclidean",
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.norm = norm

        self._chroma_filter = self._create_chroma_filter()

    def _create_chroma_filter(self) -> torch.Tensor:
        """Create chroma filter bank."""
        freqs = np.fft.rfftfreq(self.n_fft, 1 / self.sample_rate)

        chroma_filter = np.zeros((self.n_chroma, len(freqs)))

        for i in range(self.n_chroma):
            chroma_freq = 440 * 2 ** ((i - 9) / 12)

            for j, f in enumerate(freqs):
                if f > 0:
                    diff = min(abs(f / chroma_freq - 1), abs(f / (chroma_freq * 2) - 1))
                    chroma_filter[i, j] = max(0, 1 - diff * 4)

        return torch.tensor(chroma_filter, dtype=torch.float32)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract chroma features.

        Args:
            audio: Audio waveform (n_samples,)

        Returns:
            Chroma features (n_chroma, n_frames)
        """
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        )
        spec = torch.abs(spec) ** 2

        chroma = torch.matmul(self._chroma_filter.to(spec.device), spec)

        if self.norm == "euclidean":
            chroma = torch.nn.functional.normalize(chroma, p=2, dim=0)
        elif self.norm == "l1":
            chroma = torch.nn.functional.normalize(chroma, p=1, dim=0)

        return chroma


class SpectralContrast:
    """Extract spectral contrast features.

    Spectral contrast measures the difference between peaks
    and valleys in the spectrum across frequency bands.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_bands: int = 6,
        fmin: float = 27.5,
        fmax: float = 14080.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bands = n_bands
        self.fmin = fmin
        self.fmax = fmax

        self._band_edges = self._compute_band_edges()

    def _compute_band_edges(self) -> np.ndarray:
        """Compute frequency band edges."""
        fmin_mel = self._hz_to_mel(self.fmin)
        fmax_mel = self._hz_to_mel(self.fmax)

        mel_points = np.linspace(fmin_mel, fmax_mel, self.n_bands + 3)
        hz_points = self._mel_to_hz(mel_points)

        return hz_points[1:-1]

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract spectral contrast.

        Args:
            audio: Audio waveform (n_samples,)

        Returns:
            Spectral contrast (n_bands, n_frames)
        """
        freqs = torch.fft.rfftfreq(self.n_fft, 1 / self.sample_rate).to(audio.device)

        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft).to(audio.device),
            return_complex=True,
        )
        spec = torch.abs(spec)

        contrast = []
        for i in range(self.n_bands):
            low = (
                freqs >= self._band_edges[i]
                if i == 0
                else freqs >= self._band_edges[i - 1]
            )
            high = freqs <= self._band_edges[i]
            band_mask = low & high

            band_spec = spec[band_mask, :]

            if band_spec.shape[0] == 0:
                contrast.append(
                    torch.zeros(audio.shape[0] // self.hop_length + 1).to(audio.device)
                )
                continue

            peak_vals = band_spec.max(dim=0)[0]
            valley_vals = band_spec.min(dim=0)[0]

            contrast.append(peak_vals - valley_vals)

        return torch.stack(contrast)


class TonnetzFeatures:
    """Extract Tonnetz (tonal centroid) features.

    Tonnetz features represent the harmonic content of audio
    in a hexagonal pitch class space.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 4096,
        hop_length: int = 512,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        self._chroma_filter = self._create_chroma_filter()

    def _create_chroma_filter(self) -> torch.Tensor:
        """Create chroma filter for tonnetz."""
        freqs = np.fft.rfftfreq(self.n_fft, 1 / self.sample_rate)
        n_chroma = 12

        chroma_filter = np.zeros((n_chroma, len(freqs)))
        for i in range(n_chroma):
            chroma_freq = 440 * 2 ** ((i - 9) / 12)
            for j, f in enumerate(freqs):
                if f > 0:
                    diff = min(abs(f / chroma_freq - 1), abs(f / (chroma_freq * 2) - 1))
                    chroma_filter[i, j] = max(0, 1 - diff * 4)

        return torch.tensor(chroma_filter, dtype=torch.float32)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract tonnetz features.

        Args:
            audio: Audio waveform (n_samples,)

        Returns:
            Tonnetz features (6, n_frames)
        """
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        )
        spec = torch.abs(spec) ** 2

        chroma = torch.matmul(self._chroma_filter.to(spec.device), spec)
        chroma = chroma / (chroma.sum(dim=0, keepdim=True) + 1e-10)

        r = 1
        alpha = np.pi / 6
        beta = np.pi / 3

        tonnetz = torch.zeros(6, chroma.shape[1]).to(chroma.device)

        tonnetz[0] = r * torch.sum(
            chroma
            * torch.tensor(
                [
                    np.cos(2 * alpha),
                    np.cos(2 * alpha + 2 * np.pi / 6),
                    np.cos(2 * alpha + 4 * np.pi / 6),
                    np.cos(2 * alpha + 6 * np.pi / 6),
                    np.cos(2 * alpha + 8 * np.pi / 6),
                    np.cos(2 * alpha + 10 * np.pi / 6),
                    np.cos(2 * alpha),
                    np.cos(2 * alpha + 2 * np.pi / 6),
                    np.cos(2 * alpha + 4 * np.pi / 6),
                    np.cos(2 * alpha + 6 * np.pi / 6),
                    np.cos(2 * alpha + 8 * np.pi / 6),
                    np.cos(2 * alpha + 10 * np.pi / 6),
                ]
            ).to(chroma.device)
        )

        tonnetz[1] = r * torch.sum(
            chroma
            * torch.tensor(
                [
                    np.sin(2 * alpha),
                    np.sin(2 * alpha + 2 * np.pi / 6),
                    np.sin(2 * alpha + 4 * np.pi / 6),
                    np.sin(2 * alpha + 6 * np.pi / 6),
                    np.sin(2 * alpha + 8 * np.pi / 6),
                    np.sin(2 * alpha + 10 * np.pi / 6),
                    np.sin(2 * alpha),
                    np.sin(2 * alpha + 2 * np.pi / 6),
                    np.sin(2 * alpha + 4 * np.pi / 6),
                    np.sin(2 * alpha + 6 * np.pi / 6),
                    np.sin(2 * alpha + 8 * np.pi / 6),
                    np.sin(2 * alpha + 10 * np.pi / 6),
                ]
            ).to(chroma.device)
        )

        for i in range(4):
            tonnetz[2 + i] = r * torch.sum(
                chroma
                * torch.tensor(
                    [
                        np.cos((i + 1) * beta),
                        np.cos((i + 1) * beta + 2 * np.pi / 6),
                        np.cos((i + 1) * beta + 4 * np.pi / 6),
                        np.cos((i + 1) * beta + 6 * np.pi / 6),
                        np.cos((i + 1) * beta + 8 * np.pi / 6),
                        np.cos((i + 1) * beta + 10 * np.pi / 6),
                        np.cos((i + 1) * beta),
                        np.cos((i + 1) * beta + 2 * np.pi / 6),
                        np.cos((i + 1) * beta + 4 * np.pi / 6),
                        np.cos((i + 1) * beta + 6 * np.pi / 6),
                        np.cos((i + 1) * beta + 8 * np.pi / 6),
                        np.cos((i + 1) * beta + 10 * np.pi / 6),
                    ]
                ).to(chroma.device)
            )

        return tonnetz

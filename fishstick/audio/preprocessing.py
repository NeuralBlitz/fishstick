"""
Audio Preprocessing

Audio preprocessing and augmentation utilities.
"""

from typing import Optional
import torch
import numpy as np
from pathlib import Path


class AudioLoader:
    """Load and decode audio files."""

    def __init__(self, sample_rate: int = 16000, mono: bool = True):
        self.sample_rate = sample_rate
        self.mono = mono

    def __call__(self, path: str) -> torch.Tensor:
        try:
            import torchaudio

            waveform, sr = torchaudio.load(path)
            if self.mono and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.sample_rate
                )
            return waveform.squeeze(0)
        except ImportError:
            import scipy.io.wavfile as wav

            sr, data = wav.read(path)
            if self.mono and len(data.shape) > 1:
                data = data.mean(axis=1)
            waveform = torch.from_numpy(data.astype(np.float32))
            if sr != self.sample_rate:
                from scipy import signal

                num_samples = int(len(waveform) * self.sample_rate / sr)
                waveform = torch.from_numpy(
                    signal.resample(waveform.numpy(), num_samples)
                )
            return waveform


class AudioNormalizer:
    """Normalize audio waveform."""

    def __init__(self, target_db: float = -20.0):
        self.target_db = target_db

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(audio**2))
        target_rms = 10 ** (self.target_db / 20)
        return audio * (target_rms / (rms + 1e-8))


class TimeStretch:
    """Time stretch audio."""

    def __init__(self, min_rate: float = 0.8, max_rate: float = 1.2):
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(
        self, audio: torch.Tensor, rate: Optional[float] = None
    ) -> torch.Tensor:
        if rate is None:
            rate = np.random.uniform(self.min_rate, self.max_rate)

        indices = torch.arange(0, len(audio), rate)
        indices = indices[indices < len(audio)].long()
        return audio[indices]


class PitchShift:
    """Pitch shift audio."""

    def __init__(self, min_steps: int = -2, max_steps: int = 2):
        self.min_steps = min_steps
        self.max_steps = max_steps

    def __call__(
        self, audio: torch.Tensor, sample_rate: int = 16000, steps: Optional[int] = None
    ) -> torch.Tensor:
        if steps is None:
            steps = np.random.randint(self.min_steps, self.max_steps + 1)

        rate = 2 ** (steps / 12)
        indices = torch.arange(0, len(audio), rate)
        indices = indices[indices < len(audio)].long()
        return audio[indices]


class AddNoise:
    """Add noise to audio."""

    def __init__(self, min_snr: float = 10, max_snr: float = 30):
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __call__(
        self, audio: torch.Tensor, snr: Optional[float] = None
    ) -> torch.Tensor:
        if snr is None:
            snr = np.random.uniform(self.min_snr, self.max_snr)

        signal_power = torch.mean(audio**2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)

        return audio + noise


class RandomCrop:
    """Randomly crop audio to target length."""

    def __init__(self, target_length: int):
        self.target_length = target_length

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if len(audio) <= self.target_length:
            padding = self.target_length - len(audio)
            return torch.cat([audio, torch.zeros(padding)])
        start = np.random.randint(0, len(audio) - self.target_length)
        return audio[start : start + self.target_length]


class PadToLength:
    """Pad audio to target length."""

    def __init__(self, target_length: int, mode: str = "constant"):
        self.target_length = target_length
        self.mode = mode

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if len(audio) >= self.target_length:
            return audio[: self.target_length]

        padding = self.target_length - len(audio)
        if self.mode == "constant":
            return torch.cat([audio, torch.zeros(padding)])
        elif self.mode == "reflect":
            return torch.cat([audio, audio.flip(0)[:padding]])
        return audio


class AudioTransform:
    """Composable audio transformations."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            audio = transform(audio)
        return audio


def create_augmentation_pipeline(
    sample_rate: int = 16000,
    add_noise: bool = True,
    pitch_shift: bool = True,
    time_stretch: bool = True,
) -> AudioTransform:
    """Create audio augmentation pipeline."""
    transforms = []

    if add_noise:
        transforms.append(AddNoise())

    if pitch_shift:
        transforms.append(PitchShift())

    if time_stretch:
        transforms.append(TimeStretch())

    transforms.append(AudioNormalizer())

    return AudioTransform(transforms)

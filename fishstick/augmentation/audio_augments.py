"""Audio augmentation techniques."""

import torch
import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class AudioAugment(ABC):
    """Base class for audio augmentation."""

    @abstractmethod
    def __call__(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        pass


class AddNoise(AudioAugment):
    """Add various types of noise to audio."""

    def __init__(
        self,
        noise_type: str = "white",
        noise_level: float = 0.005,
        probability: float = 0.5,
    ):
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.probability = probability

    def __call__(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        if torch.rand(1).item() > self.probability:
            return audio

        if self.noise_type == "white":
            noise = torch.randn_like(audio) * self.noise_level
        elif self.noise_type == "pink":
            noise = self._pink_noise(audio.shape) * self.noise_level
        elif self.noise_type == "brown":
            noise = self._brown_noise(audio.shape) * self.noise_level
        else:
            noise = torch.randn_like(audio) * self.noise_level

        return audio + noise

    def _pink_noise(self, shape: torch.Size) -> torch.Tensor:
        """Generate pink noise."""
        audio_len = shape[0] if len(shape) == 1 else shape[1]
        noise = torch.randn(audio_len)
        white = torch.arange(0, audio_len, dtype=torch.float32)
        pink = noise / white.clamp(min=1)
        pink[:10] = 0
        return pink.to(audio_len.device) if len(shape) == 1 else pink.to(shape.device)

    def _brown_noise(self, shape: torch.Size) -> torch.Tensor:
        """Generate brown noise."""
        audio_len = shape[0] if len(shape) == 1 else shape[1]
        noise = torch.randn(audio_len)
        brown = torch.cumsum(noise, dim=0)
        brown = brown - brown.mean()
        return brown.to(audio_len.device) if len(shape) == 1 else brown.to(shape.device)


class TimeStretch(AudioAugment):
    """Time stretching without pitch change."""

    def __init__(
        self, min_rate: float = 0.8, max_rate: float = 1.2, probability: float = 0.5
    ):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.probability = probability

    def __call__(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        if torch.rand(1).item() > self.probability:
            return audio

        rate = torch.empty(1).uniform_(self.min_rate, self.max_rate).item()
        indices = torch.arange(0, audio.size(-1), rate)
        indices = indices.clamp(max=audio.size(-1) - 1).long()

        return audio[..., indices]


class PitchShift(AudioAugment):
    """Pitch shifting by semitones."""

    def __init__(
        self,
        min_semitones: float = -2,
        max_semitones: float = 2,
        probability: float = 0.5,
    ):
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.probability = probability

    def __call__(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        if torch.rand(1).item() > self.probability:
            return audio

        semitones = (
            torch.empty(1).uniform_(self.min_semitones, self.max_semitones).item()
        )

        rate = 2 ** (semitones / 12)
        indices = torch.arange(0, audio.size(-1), rate)
        indices = indices.clamp(max=audio.size(-1) - 1).long()

        return audio[..., indices]


class VolumeChange(AudioAugment):
    """Change audio volume."""

    def __init__(
        self, min_gain: float = 0.5, max_gain: float = 1.5, probability: float = 0.5
    ):
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.probability = probability

    def __call__(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        if torch.rand(1).item() > self.probability:
            return audio

        gain = torch.empty(1).uniform_(self.min_gain, self.max_gain).item()
        return audio * gain


class SpecAugment(AudioAugment):
    """SpecAugment: Simple spectral augmentation."""

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        probability: float = 0.5,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.probability = probability

    def __call__(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        if torch.rand(1).item() > self.probability:
            return audio

        spec = self._audio_to_spectrogram(audio, sample_rate)

        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            f0 = torch.randint(0, max(1, spec.size(1) - f), (1,)).item()
            spec[:, f0 : f0 + f, :] = 0

        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param, (1,)).item()
            t0 = torch.randint(0, max(1, spec.size(2) - t), (1,)).item()
            spec[:, :, t0 : t0 + t] = 0

        return self._spectrogram_to_audio(spec, sample_rate)

    def _audio_to_spectrogram(
        self, audio: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        n_fft = 512
        hop_length = 160

        window = torch.hann_window(n_fft)
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        return torch.abs(spec)

    def _spectrogram_to_audio(
        self, spec: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        n_fft = 512
        hop_length = 160
        window = torch.hann_window(n_fft)

        audio = torch.istft(spec, n_fft=n_fft, hop_length=hop_length, window=window)
        return audio


class TimeShift(AudioAugment):
    """Random time shift."""

    def __init__(self, max_shift: float = 0.1, probability: float = 0.5):
        self.max_shift = max_shift
        self.probability = probability

    def __call__(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        if torch.rand(1).item() > self.probability:
            return audio

        max_shift_samples = int(audio.size(-1) * self.max_shift)
        shift = torch.randint(-max_shift_samples, max_shift_samples + 1, (1,)).item()

        if shift > 0:
            return torch.cat(
                [torch.zeros(shift, device=audio.device), audio[..., :-shift]], dim=-1
            )
        else:
            return torch.cat(
                [audio[..., -shift:], torch.zeros(-shift, device=audio.device)], dim=-1
            )


class AudioCompose:
    """Compose multiple audio augmentations."""

    def __init__(self, transforms: list, probabilities: Optional[list] = None):
        self.transforms = transforms
        self.probabilities = probabilities or [1.0] * len(transforms)

    def __call__(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        for transform, prob in zip(self.transforms, self.probabilities):
            if torch.rand(1).item() < prob:
                audio = transform(audio, sample_rate)
        return audio


def create_audio_augment(augment_type: str, **kwargs) -> AudioAugment:
    """Factory function to create audio augmentations."""
    augments = {
        "noise": AddNoise,
        "time_stretch": TimeStretch,
        "pitch_shift": PitchShift,
        "volume": VolumeChange,
        "spec_augment": SpecAugment,
        "time_shift": TimeShift,
    }

    if augment_type not in augments:
        raise ValueError(f"Unknown augment: {augment_type}")

    return augments[augment_type](**kwargs)

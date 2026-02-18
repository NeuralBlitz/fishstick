"""
Audio Augmentation for Source Separation

Provides augmentation techniques specifically designed for audio source
separation training, including SpecAugment, room impulse responses,
time stretching, and pitch shifting.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class AudioAugmentation(nn.Module):
    """Base class for audio augmentations."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return self.apply(audio)
        return audio

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SpecAugment(AudioAugmentation):
    """SpecAugment: A Simple Data Augmentation Method for ASR.

    Applies time warping, frequency masking, and time masking to
    spectrograms for robust training.

    Reference:
        "SpecAugment: A Simple Data Augmentation Method for Automatic
        Speech Recognition" (Park et al., 2019)
    """

    def __init__(
        self,
        p: float = 0.5,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        warp_window: int = 5,
        warp_mode: str = "constant",
    ):
        super().__init__(p)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.warp_window = warp_window
        self.warp_mode = warp_mode

    def apply(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram.

        Args:
            spec: Spectrogram tensor (batch, freq, time)

        Returns:
            Augmented spectrogram
        """
        spec = self._time_warp(spec)
        spec = self._freq_mask(spec)
        spec = self._time_mask(spec)
        return spec

    def _time_warp(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply time warping."""
        if random.random() > 0.5:
            return spec

        batch, freq, time = spec.shape
        if time < self.warp_window * 2:
            return spec

        center = random.randint(self.warp_window, time - self.warp_window)
        warp = random.randint(-self.warp_window, self.warp_window)

        time_index = torch.arange(time, device=spec.device)
        warped_time = time_index.float() + warp * torch.sinc(
            (time_index.float() - center) / self.warp_window
        )
        warped_time = torch.clamp(warped_time, 0, time - 1)

        spec = F.interpolate(
            spec.unsqueeze(1),
            size=(freq, time),
            mode="bilinear",
            align_corners=False,
        )
        spec = spec.squeeze(1)

        return spec

    def _freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking."""
        _, freq, _ = spec.shape

        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, freq - f))
            spec[:, f0 : f0 + f, :] = 0

        return spec

    def _time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply time masking."""
        _, _, time = spec.shape

        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, time - t))
            spec[:, :, t0 : t0 + t] = 0

        return spec


class RoomImpulseResponse(AudioAugmentation):
    """Room Impulse Response (RIR) augmentation.

    Applies convolved room acoustics to simulate different recording
    environments.

    Reference:
        "A study on data augmentation of reverberant speech for
        deep learning-based speech enhancement" (Ko et al., 2017)
    """

    def __init__(
        self,
        p: float = 0.5,
        sample_rate: int = 16000,
        modes: Optional[List[str]] = None,
    ):
        super().__init__(p)
        self.sample_rate = sample_rate
        self.modes = modes or ["small", "medium", "large", "clean"]

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply RIR to audio.

        Args:
            audio: Audio tensor (batch, channels, time)

        Returns:
            Audio with applied RIR
        """
        mode = random.choice(self.modes)
        rir = self._generate_rir(mode, audio.shape[-1], audio.device)

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        augmented = []
        for i in range(audio.shape[1]):
            convolved = self._convolve(audio[:, i], rir)
            augmented.append(convolved)

        result = torch.stack(augmented, dim=1)

        if squeeze:
            result = result.squeeze(1)

        return result

    def _generate_rir(
        self,
        mode: str,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate synthetic RIR based on room size."""
        if mode == "clean":
            rir = torch.zeros(length, device=device)
            rir[0] = 1.0
            return rir

        if mode == "small":
            rt60 = 0.3
        elif mode == "medium":
            rt60 = 0.5
        else:
            rt60 = 0.8

        decay = torch.exp(
            -6.9 * torch.arange(length, device=device) / (rt60 * self.sample_rate)
        )

        reverb = torch.randn(length, device=device) * decay

        reverb[0] = 1.0

        return reverb

    def _convolve(
        self,
        audio: torch.Tensor,
        rir: torch.Tensor,
    ) -> torch.Tensor:
        """Perform convolution with RIR."""
        output_length = audio.shape[-1] + rir.shape[-1] - 1
        result = F.conv1d(
            audio.unsqueeze(0),
            rir.unsqueeze(0).unsqueeze(0),
            padding=output_length - audio.shape[-1],
        )
        return result.squeeze(0)


class TimeStretch(AudioAugmentation):
    """Time stretching augmentation without pitch change.

    Stretches or compresses audio in time domain while maintaining pitch.
    """

    def __init__(
        self,
        p: float = 0.5,
        min_rate: float = 0.8,
        max_rate: float = 1.2,
    ):
        super().__init__(p)
        self.min_rate = min_rate
        self.max_rate = max_rate

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply time stretching.

        Args:
            audio: Audio tensor

        Returns:
            Time-stretched audio
        """
        rate = random.uniform(self.min_rate, self.max_rate)

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        stretched = []
        for i in range(audio.shape[1]):
            original_length = audio.shape[-1]
            new_length = int(original_length * rate)

            stretched_sample = F.interpolate(
                audio[:, i].unsqueeze(0),
                size=new_length,
                mode="linear",
                align_corners=False,
            )

            if new_length > original_length:
                start = (new_length - original_length) // 2
                stretched_sample = stretched_sample[
                    ..., start : start + original_length
                ]
            elif new_length < original_length:
                padded = F.pad(
                    stretched_sample,
                    (0, original_length - new_length),
                    mode="constant",
                )
                stretched_sample = padded

            stretched.append(stretched_sample.squeeze(0))

        result = torch.stack(stretched, dim=1)

        if squeeze:
            result = result.squeeze(1)

        return result


class PitchShift(AudioAugmentation):
    """Pitch shifting augmentation without duration change.

    Shifts pitch up or down while maintaining duration.
    """

    def __init__(
        self,
        p: float = 0.5,
        sample_rate: int = 16000,
        min_semitones: float = -2.0,
        max_semitones: float = 2.0,
    ):
        super().__init__(p)
        self.sample_rate = sample_rate
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply pitch shifting.

        Args:
            audio: Audio tensor

        Returns:
            Pitch-shifted audio
        """
        semitones = random.uniform(self.min_semitones, self.max_semitones)
        rate = 2 ** (semitones / 12.0)

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        shifted = []
        for i in range(audio.shape[1]):
            original_length = audio.shape[-1]
            resampled_length = int(original_length * rate)

            resampled = F.interpolate(
                audio[:, i].unsqueeze(0),
                size=resampled_length,
                mode="linear",
                align_corners=False,
            )

            if resampled_length > original_length:
                start = (resampled_length - original_length) // 2
                resampled = resampled[..., start : start + original_length]
            elif resampled_length < original_length:
                resampled = F.pad(
                    resampled,
                    (0, original_length - resampled_length),
                    mode="constant",
                )

            shifted.append(resampled.squeeze(0))

        result = torch.stack(shifted, dim=1)

        if squeeze:
            result = result.squeeze(1)

        return result


class AddNoise(AudioAugmentation):
    """Add background noise to audio.

    Simulates noisy recording conditions.
    """

    def __init__(
        self,
        p: float = 0.5,
        min_snr: float = 5.0,
        max_snr: float = 30.0,
        noise_types: Optional[List[str]] = None,
    ):
        super().__init__(p)
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.noise_types = noise_types or ["white", "pink", "brown"]

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """Add noise to audio.

        Args:
            audio: Audio tensor

        Returns:
            Noisy audio
        """
        snr = random.uniform(self.min_snr, self.max_snr)
        noise_type = random.choice(self.noise_types)

        signal_power = torch.mean(audio**2)
        noise_power = signal_power / (10 ** (snr / 10))

        if noise_type == "white":
            noise = torch.randn_like(audio)
        elif noise_type == "pink":
            noise = self._generate_pink_noise(audio.shape, audio.device)
        else:
            noise = self._generate_brown_noise(audio.shape, audio.device)

        noise = noise * torch.sqrt(noise_power)

        return audio + noise

    def _generate_pink_noise(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Generate pink noise."""
        length = shape[-1]
        white = torch.randn(length, device=device)

        b = torch.tensor(
            [0.99886, 0.0555179, 0.0750759, 0.1538520, 0.3102, 0.5329, -0.0528],
            device=device,
        )
        a = torch.tensor([1.0, -0.9763, 0.0, -0.4751, 0.0, 0.0], device=device)

        pink = F.lfilter(white, a, b)
        pink = pink / (torch.max(torch.abs(pink)) + 1e-8)

        if len(shape) > 1:
            pink = pink.unsqueeze(0).expand(*shape)

        return pink

    def _generate_brown_noise(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """Generate brown noise."""
        length = shape[-1]
        white = torch.randn(length, device=device)

        brown = F.lfilter(
            white.unsqueeze(0),
            torch.tensor([1.0, -0.99], device=device),
            torch.tensor([1.0, -0.99], device=device),
        )
        brown = brown.squeeze(0)
        brown = brown / (torch.max(torch.abs(brown)) + 1e-8)

        if len(shape) > 1:
            brown = brown.unsqueeze(0).expand(*shape)

        return brown


class Reverb(AudioAugmentation):
    """Reverb effect augmentation.

    Adds artificial reverb to simulate different room acoustics.
    """

    def __init__(
        self,
        p: float = 0.5,
        sample_rate: int = 16000,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_gain: float = 0.3,
    ):
        super().__init__(p)
        self.sample_rate = sample_rate
        self.room_size = room_size
        self.damping = damping
        self.wet_gain = wet_gain

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply reverb effect.

        Args:
            audio: Audio tensor

        Returns:
            Audio with reverb
        """
        delay_samples = int(self.room_size * self.sample_rate * 0.05)
        feedback = 0.5 - self.damping * 0.4

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        reverberated = []
        for i in range(audio.shape[1]):
            delayed = F.pad(
                audio[:, i],
                (delay_samples, 0),
                mode="constant",
            )
            delayed = delayed[:, :-delay_samples]

            output = (
                audio[:, i] * (1 - self.wet_gain) + delayed * self.wet_gain * feedback
            )
            reverberated.append(output)

        result = torch.stack(reverberated, dim=1)

        if squeeze:
            result = result.squeeze(1)

        return result


class ChannelSwap(AudioAugmentation):
    """Swap stereo channels for multi-channel audio.

    Useful for building robustness to channel ordering.
    """

    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """Swap channels.

        Args:
            audio: Audio tensor (batch, channels, time)

        Returns:
            Audio with potentially swapped channels
        """
        if audio.shape[1] < 2:
            return audio

        if random.random() < 0.5:
            return torch.flip(audio, dims=[1])

        return audio


class Compose(nn.Module):
    """Compose multiple augmentations.

    Applies a sequence of augmentations to audio.
    """

    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations in sequence.

        Args:
            audio: Input audio

        Returns:
            Augmented audio
        """
        for transform in self.transforms:
            audio = transform(audio)
        return audio


class RandomChoice(nn.Module):
    """Randomly choose one augmentation from a list.

    Applies exactly one randomly selected augmentation.
    """

    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = transforms

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation.

        Args:
            audio: Input audio

        Returns:
            Augmented audio
        """
        transform = random.choice(self.transforms)
        return transform(audio)

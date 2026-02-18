"""
Audio Augmentation Module

Augmentation techniques for audio data including:
- Time stretching and pitch shifting
- Background noise addition
- Time shifting and volume perturbation
- SpecAugment
- Audio mixing
"""

from typing import Optional, Tuple, List, Union, Dict, Any
import torch
import numpy as np
from numpy.typing import NDArray
import random

from fishstick.augmentation_ext.base import AugmentationBase


class TimeStretch(AugmentationBase):
    """Time stretching for audio."""

    def __init__(
        self,
        rate: Union[float, Tuple[float, float]] = (0.8, 1.2),
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        if isinstance(rate, (int, float)):
            self.rate_range = (rate, rate)
        else:
            self.rate_range = rate
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, audio: Union[Tensor, NDArray[np.floating]]
    ) -> Union[Tensor, NDArray[np.floating]]:
        """
        Apply time stretching to audio.

        Args:
            audio: Audio tensor or array (N, T) or (T,)

        Returns:
            Time-stretched audio
        """
        if not self._should_apply():
            return audio

        rate = self.rng.uniform(*self.rate_range)

        if isinstance(audio, torch.Tensor):
            return self._stretch_tensor(audio, rate)
        else:
            return self._stretch_numpy(audio, rate)

    def _stretch_tensor(self, audio: Tensor, rate: float) -> Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        from scipy.signal import resample

        original_length = audio.size(1)
        new_length = int(original_length * rate)

        result = []
        for i in range(audio.size(0)):
            resampled = resample(audio[i].numpy(), new_length)
            result.append(torch.from_numpy(resampled))

        return torch.stack(result)

    def _stretch_numpy(
        self, audio: NDArray[np.floating], rate: float
    ) -> NDArray[np.floating]:
        from scipy.signal import resample

        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        original_length = audio.shape[1]
        new_length = int(original_length * rate)

        result = []
        for i in range(audio.shape[0]):
            resampled = resample(audio[i], new_length)
            result.append(resampled)

        return np.stack(result)


class PitchShift(AugmentationBase):
    """Pitch shifting for audio."""

    def __init__(
        self,
        n_steps: Union[int, Tuple[int, int]] = (-2, 2),
        sample_rate: int = 16000,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        if isinstance(n_steps, (int, float)):
            self.n_steps_range = (int(n_steps), int(n_steps))
        else:
            self.n_steps_range = (int(n_steps[0]), int(n_steps[1]))
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, audio: Union[Tensor, NDArray[np.floating]]
    ) -> Union[Tensor, NDArray[np.floating]]:
        """
        Apply pitch shifting to audio.

        Args:
            audio: Audio tensor or array (N, T) or (T,)

        Returns:
            Pitch-shifted audio
        """
        if not self._should_apply():
            return audio

        n_steps = self.rng.randint(*self.n_steps_range)

        if isinstance(audio, torch.Tensor):
            return self._shift_tensor(audio, n_steps)
        else:
            return self._shift_numpy(audio, n_steps)

    def _shift_tensor(self, audio: Tensor, n_steps: int) -> Tensor:
        try:
            import librosa

            if audio.dim() == 2:
                result = []
                for a in audio:
                    shifted = librosa.effects.pitch_shift(
                        a.numpy(), sr=self.sample_rate, n_steps=n_steps
                    )
                    result.append(torch.from_numpy(shifted))
                return torch.stack(result)
            else:
                shifted = librosa.effects.pitch_shift(
                    audio.numpy(), sr=self.sample_rate, n_steps=n_steps
                )
                return torch.from_numpy(shifted)
        except ImportError:
            return audio

    def _shift_numpy(
        self, audio: NDArray[np.floating], n_steps: int
    ) -> NDArray[np.floating]:
        try:
            import librosa

            if audio.ndim == 2:
                result = []
                for a in audio:
                    shifted = librosa.effects.pitch_shift(
                        a, sr=self.sample_rate, n_steps=n_steps
                    )
                    result.append(shifted)
                return np.stack(result)
            else:
                return librosa.effects.pitch_shift(
                    audio, sr=self.sample_rate, n_steps=n_steps
                )
        except ImportError:
            return audio


class AddBackgroundNoise(AugmentationBase):
    """Add background noise to audio."""

    def __init__(
        self,
        noise_level: float = 0.1,
        noise_type: str = "gaussian",
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, audio: Union[Tensor, NDArray[np.floating]]
    ) -> Union[Tensor, NDArray[np.floating]]:
        """
        Add background noise to audio.

        Args:
            audio: Audio tensor or array (N, T) or (T,)

        Returns:
            Noisy audio
        """
        if not self._should_apply():
            return audio

        if isinstance(audio, torch.Tensor):
            return self._add_noise_tensor(audio)
        else:
            return self._add_noise_numpy(audio)

    def _add_noise_tensor(self, audio: Tensor) -> Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        noise = torch.randn_like(audio) * self.noise_level
        return audio + noise

    def _add_noise_numpy(self, audio: NDArray[np.floating]) -> NDArray[np.floating]:
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        noise = np.random.randn(*audio.shape) * self.noise_level
        return audio + noise


class TimeShift(AugmentationBase):
    """Random time shift for audio."""

    def __init__(
        self,
        shift_limit: float = 0.2,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.shift_limit = shift_limit
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, audio: Union[Tensor, NDArray[np.floating]]
    ) -> Union[Tensor, NDArray[np.floating]]:
        """
        Apply time shift to audio.

        Args:
            audio: Audio tensor or array (N, T) or (T,)

        Returns:
            Time-shifted audio
        """
        if not self._should_apply():
            return audio

        shift = int(self.rng.uniform(-self.shift_limit, self.shift_limit))

        if isinstance(audio, torch.Tensor):
            return self._shift_tensor(audio, shift)
        else:
            return self._shift_numpy(audio, shift)

    def _shift_tensor(self, audio: Tensor, shift: int) -> Tensor:
        if audio.dim() == 1:
            if shift > 0:
                return torch.cat([torch.zeros(shift), audio[:-shift]])
            else:
                return torch.cat([audio[-shift:], torch.zeros(-shift)])
        else:
            result = []
            for a in audio:
                if shift > 0:
                    result.append(
                        torch.cat([torch.zeros(shift, device=a.device), a[:-shift]])
                    )
                else:
                    result.append(
                        torch.cat([a[-shift:], torch.zeros(-shift, device=a.device)])
                    )
            return torch.stack(result)

    def _shift_numpy(
        self, audio: NDArray[np.floating], shift: int
    ) -> NDArray[np.floating]:
        if audio.ndim == 1:
            if shift > 0:
                return np.concatenate([np.zeros(shift), audio[:-shift]])
            else:
                return np.concatenate([audio[-shift:], np.zeros(-shift)])
        else:
            result = []
            for a in audio:
                if shift > 0:
                    result.append(np.concatenate([np.zeros(shift), a[:-shift]]))
                else:
                    result.append(np.concatenate([a[-shift:], np.zeros(-shift)]))
            return np.stack(result)


class VolumePerturbation(AugmentationBase):
    """Volume perturbation for audio."""

    def __init__(
        self,
        db_range: Union[float, Tuple[float, float]] = (-6, 6),
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        if isinstance(db_range, (int, float)):
            self.db_range = (-db_range, db_range)
        else:
            self.db_range = db_range
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, audio: Union[Tensor, NDArray[np.floating]]
    ) -> Union[Tensor, NDArray[np.floating]]:
        """
        Apply volume perturbation to audio.

        Args:
            audio: Audio tensor or array (N, T) or (T,)

        Returns:
            Volume-perturbed audio
        """
        if not self._should_apply():
            return audio

        db_change = self.rng.uniform(*self.db_range)
        factor = 10 ** (db_change / 20)

        if isinstance(audio, torch.Tensor):
            return audio * factor
        else:
            return audio * factor


class SpecAugment(AugmentationBase):
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.

    Reference: Park et al., "SpecAugment", 2019
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, spec: Union[Tensor, NDArray[np.floating]]
    ) -> Union[Tensor, NDArray[np.floating]]:
        """
        Apply SpecAugment to spectrogram.

        Args:
            spec: Spectrogram tensor or array (F, T) or (N, F, T)

        Returns:
            Augmented spectrogram
        """
        if not self._should_apply():
            return spec

        if isinstance(spec, torch.Tensor):
            return self._augment_tensor(spec)
        else:
            return self._augment_numpy(spec)

    def _augment_tensor(self, spec: Tensor) -> Tensor:
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        freq_dim = spec.size(1)
        time_dim = spec.size(2)

        for _ in range(self.num_freq_masks):
            f = self.rng.randint(0, self.freq_mask_param)
            f0 = self.rng.randint(0, max(1, freq_dim - f))
            spec[:, f0 : f0 + f, :] = 0

        for _ in range(self.num_time_masks):
            t = self.rng.randint(0, self.time_mask_param)
            t0 = self.rng.randint(0, max(1, time_dim - t))
            spec[:, :, t0 : t0 + t] = 0

        if squeeze:
            spec = spec.squeeze(0)

        return spec

    def _augment_numpy(self, spec: NDArray[np.floating]) -> NDArray[np.floating]:
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        freq_dim = spec.shape[1]
        time_dim = spec.shape[2]

        for _ in range(self.num_freq_masks):
            f = self.rng.randint(0, self.freq_mask_param)
            f0 = self.rng.randint(0, max(1, freq_dim - f))
            spec[:, f0 : f0 + f, :] = 0

        for _ in range(self.num_time_masks):
            t = self.rng.randint(0, self.time_mask_param)
            t0 = self.rng.randint(0, max(1, time_dim - t))
            spec[:, :, t0 : t0 + t] = 0

        if squeeze:
            spec = spec.squeeze(0)

        return spec


class AudioMixUp(AugmentationBase):
    """MixUp for audio."""

    def __init__(
        self,
        alpha: float = 0.2,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, audio: Tensor, labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, ...]:
        """
        Apply MixUp to audio batches.

        Args:
            audio: Audio tensor (N, T)
            labels: Optional labels (N,)

        Returns:
            Mixed audio and optionally (labels_a, labels_b, lambda)
        """
        if not self._should_apply() or audio.size(0) < 2:
            return (audio,) if labels is None else (audio, labels, labels, 1.0)

        if self.alpha > 0:
            lam = self.rng.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = audio.size(0)
        index = self.rng.permutation(batch_size)

        mixed_audio = lam * audio + (1 - lam) * audio[index]

        if labels is not None:
            return mixed_audio, labels, labels[index], lam

        return (mixed_audio,)


class TimeCrop(AugmentationBase):
    """Randomly crop audio in time dimension."""

    def __init__(
        self,
        crop_ratio: float = 0.8,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.crop_ratio = crop_ratio
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, audio: Union[Tensor, NDArray[np.floating]]
    ) -> Union[Tensor, NDArray[np.floating]]:
        """
        Apply time cropping to audio.

        Args:
            audio: Audio tensor or array (N, T) or (T,)

        Returns:
            Cropped audio
        """
        if not self._should_apply():
            return audio

        if isinstance(audio, torch.Tensor):
            return self._crop_tensor(audio)
        else:
            return self._crop_numpy(audio)

    def _crop_tensor(self, audio: Tensor) -> Tensor:
        if audio.dim() == 1:
            length = audio.size(0)
            crop_length = int(length * self.crop_ratio)
            start = self.rng.randint(0, length - crop_length + 1)
            return audio[start : start + crop_length]
        else:
            result = []
            for a in audio:
                length = a.size(0)
                crop_length = int(length * self.crop_ratio)
                start = self.rng.randint(0, length - crop_length + 1)
                result.append(a[start : start + crop_length])
            return torch.stack(result)

    def _crop_numpy(self, audio: NDArray[np.floating]) -> NDArray[np.floating]:
        if audio.ndim == 1:
            length = audio.shape[0]
            crop_length = int(length * self.crop_ratio)
            start = self.rng.randint(0, length - crop_length + 1)
            return audio[start : start + crop_length]
        else:
            result = []
            for a in audio:
                length = a.shape[0]
                crop_length = int(length * self.crop_ratio)
                start = self.rng.randint(0, length - crop_length + 1)
                result.append(a[start : start + crop_length])
            return np.stack(result)


class AudioSpeed(AugmentationBase):
    """Change audio speed."""

    def __init__(
        self,
        speed_range: Tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.speed_range = speed_range
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, audio: Union[Tensor, NDArray[np.floating]]
    ) -> Union[Tensor, NDArray[np.floating]]:
        """
        Apply speed change to audio.

        Args:
            audio: Audio tensor or array (N, T) or (T,)

        Returns:
            Speed-changed audio
        """
        if not self._should_apply():
            return audio

        speed = self.rng.uniform(*self.speed_range)

        if isinstance(audio, torch.Tensor):
            return self._speed_tensor(audio, speed)
        else:
            return self._speed_numpy(audio, speed)

    def _speed_tensor(self, audio: Tensor, speed: float) -> Tensor:
        from scipy.signal import resample

        if audio.dim() == 1:
            new_length = int(audio.size(0) * speed)
            return torch.from_numpy(resample(audio.numpy(), new_length))
        else:
            result = []
            for a in audio:
                new_length = int(a.size(0) * speed)
                result.append(torch.from_numpy(resample(a.numpy(), new_length)))
            return torch.stack(result)

    def _speed_numpy(
        self, audio: NDArray[np.floating], speed: float
    ) -> NDArray[np.floating]:
        from scipy.signal import resample

        if audio.ndim == 1:
            new_length = int(audio.shape[0] * speed)
            return resample(audio, new_length)
        else:
            result = []
            for a in audio:
                new_length = int(a.shape[0] * speed)
                result.append(resample(a, new_length))
            return np.stack(result)


class ReverbEffect(AugmentationBase):
    """Add reverb effect to audio."""

    def __init__(
        self,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.room_size = room_size
        self.damping = damping
        self.wet_level = wet_level

    def __call__(
        self, audio: Union[Tensor, NDArray[np.floating]]
    ) -> Union[Tensor, NDArray[np.floating]]:
        """
        Apply reverb effect to audio.

        Args:
            audio: Audio tensor or array (N, T) or (T,)

        Returns:
            Audio with reverb
        """
        if not self._should_apply():
            return audio

        if isinstance(audio, torch.Tensor):
            return self._reverb_tensor(audio)
        else:
            return self._reverb_numpy(audio)

    def _reverb_tensor(self, audio: Tensor) -> Tensor:
        impulse_length = int(16000 * self.room_size)
        impulse = torch.exp(-self.damping * torch.arange(impulse_length))
        impulse = impulse / impulse.sum()

        if audio.dim() == 1:
            return (
                torch.nn.functional.conv1d(
                    audio.unsqueeze(0).unsqueeze(0),
                    impulse.unsqueeze(0).unsqueeze(0),
                    padding=impulse_length,
                )
                .squeeze(0)
                .squeeze(0)
            )
        else:
            result = []
            for a in audio:
                convolved = (
                    torch.nn.functional.conv1d(
                        a.unsqueeze(0).unsqueeze(0),
                        impulse.unsqueeze(0).unsqueeze(0),
                        padding=impulse_length,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                result.append(self.wet_level * convolved + (1 - self.wet_level) * a)
            return torch.stack(result)

    def _reverb_numpy(self, audio: NDArray[np.floating]) -> NDArray[np.floating]:
        impulse_length = int(16000 * self.room_size)
        impulse = np.exp(-self.damping * np.arange(impulse_length))
        impulse = impulse / impulse.sum()

        from scipy.signal import convolve

        if audio.ndim == 1:
            convolved = convolve(audio, impulse, mode="full")
            return self.wet_level * convolved + (1 - self.wet_level) * audio
        else:
            result = []
            for a in audio:
                convolved = convolve(a, impulse, mode="full")
                result.append(self.wet_level * convolved + (1 - self.wet_level) * a)
            return np.stack(result)


def get_audio_augmentation_pipeline(
    task: str = "speech_recognition",
    intensity: float = 1.0,
) -> List[Any]:
    """
    Get a pre-configured audio augmentation pipeline.

    Args:
        task: Task type (speech_recognition, speaker_identification, etc.)
        intensity: Overall augmentation intensity

    Returns:
        List of augmentation operations
    """
    return [
        AddBackgroundNoise(noise_level=0.1 * intensity),
        TimeShift(shift_limit=0.2 * intensity),
        VolumePerturbation(db_range=6 * intensity),
        TimeCrop(crop_ratio=0.9),
    ]

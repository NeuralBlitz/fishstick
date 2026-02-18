"""
Audio Preprocessing for Source Separation

Provides preprocessing utilities including loading, normalization,
voice activity detection, and audio filtering for source separation.
"""

from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json


class AudioLoader:
    """Load and decode audio files.

    Supports various audio formats through torchaudio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        resample: bool = True,
    ):
        self.sample_rate = sample_rate
        self.mono = mono
        self.resample = resample

    def load(
        self,
        file_path: Union[str, Path],
    ) -> Tuple[torch.Tensor, int]:
        """Load audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio tensor, original sample rate)
        """
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(file_path))

            if self.mono and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if self.resample and sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.sample_rate
                )
                sr = self.sample_rate

            return waveform, sr

        except ImportError:
            raise ImportError("torchaudio is required for audio loading")

    def load_batch(
        self,
        file_paths: List[Union[str, Path]],
    ) -> Tuple[torch.Tensor, int]:
        """Load batch of audio files.

        Args:
            file_paths: List of audio file paths

        Returns:
            Tuple of (stacked audio tensor, sample rate)
        """
        waveforms = []
        sr = None

        for path in file_paths:
            waveform, sample_rate = self.load(path)
            waveforms.append(waveform)
            sr = sample_rate

        return torch.stack(waveforms), sr


class Normalizer:
    """Audio normalization utilities.

    Provides various normalization strategies for audio signals.
    """

    @staticmethod
    def peak_normalize(
        audio: torch.Tensor,
        target_db: float = -3.0,
    ) -> torch.Tensor:
        """Peak normalize audio to target dB.

        Args:
            audio: Input audio
            target_db: Target peak level in dB

        Returns:
            Normalized audio
        """
        peak = torch.max(torch.abs(audio))
        if peak > 0:
            target_linear = 10 ** (target_db / 20)
            audio = audio * (target_linear / peak)
        return audio

    @staticmethod
    def rms_normalize(
        audio: torch.Tensor,
        target_db: float = -20.0,
    ) -> torch.Tensor:
        """RMS normalize audio to target level.

        Args:
            audio: Input audio
            target_db: Target RMS level in dB

        Returns:
            Normalized audio
        """
        rms = torch.sqrt(torch.mean(audio**2))
        if rms > 0:
            target_linear = 10 ** (target_db / 20)
            audio = audio * (target_linear / rms)
        return audio

    @staticmethod
    def standardize(
        audio: torch.Tensor,
        per_channel: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standardize audio (zero mean, unit variance).

        Args:
            audio: Input audio
            per_channel: Whether to normalize per channel

        Returns:
            Tuple of (normalized audio, mean, std)
        """
        if per_channel and audio.dim() > 1:
            mean = audio.mean(dim=-1, keepdim=True)
            std = audio.std(dim=-1, keepdim=True)
        else:
            mean = audio.mean()
            std = audio.std()

        if std > 0:
            audio = (audio - mean) / std

        return audio, mean, std

    @staticmethod
    def compute_statistics(
        audio: torch.Tensor,
    ) -> dict:
        """Compute audio statistics.

        Args:
            audio: Input audio

        Returns:
            Dictionary with statistics
        """
        return {
            "peak": float(torch.max(torch.abs(audio))),
            "rms": float(torch.sqrt(torch.mean(audio**2))),
            "mean": float(audio.mean()),
            "std": float(audio.std()),
            "min": float(audio.min()),
            "max": float(audio.max()),
            "dynamic_range": float(audio.max() - audio.min()),
        }


class Preprocessor:
    """Audio preprocessing pipeline.

    Combines multiple preprocessing steps for source separation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        normalize: bool = True,
        remove_dc_offset: bool = True,
        preemph: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.remove_dc_offset = remove_dc_offset
        self.preemph = preemph

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing pipeline.

        Args:
            audio: Input audio

        Returns:
            Preprocessed audio
        """
        if self.remove_dc_offset:
            audio = self._remove_dc_offset(audio)

        if self.preemph is not None:
            audio = self._apply_preemphasis(audio)

        if self.normalize:
            audio = Normalizer.rms_normalize(audio, target_db=-20.0)

        return audio

    def _remove_dc_offset(self, audio: torch.Tensor) -> torch.Tensor:
        """Remove DC offset from audio."""
        if audio.dim() > 1:
            mean = audio.mean(dim=-1, keepdim=True)
        else:
            mean = audio.mean()
        return audio - mean

    def _apply_preemphasis(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply pre-emphasis filter."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        emphasized = torch.zeros_like(audio)
        emphasized[..., 0] = audio[..., 0]
        emphasized[..., 1:] = audio[..., 1:] - self.preemph * audio[..., :-1]

        return emphasized.squeeze(0) if audio.shape[0] == 1 else emphasized


class VoiceActivityDetector:
    """Voice Activity Detection (VAD).

    Detects speech segments in audio for source separation preprocessing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: float = 0.03,
        energy_threshold: float = 0.01,
        padding_duration: float = 0.1,
    ):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.energy_threshold = energy_threshold
        self.padding_duration = padding_duration

        self.frame_length = int(sample_rate * frame_duration)
        self.hop_length = self.frame_length // 2
        self.padding_samples = int(sample_rate * padding_duration)

    def detect(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Detect voice activity.

        Args:
            audio: Input audio

        Returns:
            Binary mask (1 for speech, 0 for non-speech)
        """
        energy = self._compute_energy(audio)

        threshold = self.energy_threshold
        vad = (energy > threshold).float()

        vad = self._apply_smoothing(vad)
        vad = self._add_padding(vad)

        return vad

    def _compute_energy(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute frame energy."""
        frames = self._framing(audio)
        energy = torch.mean(frames**2, dim=-1)
        return energy

    def _framing(self, audio: torch.Tensor) -> torch.Tensor:
        """Frame audio into overlapping segments."""
        audio_length = audio.shape[-1]
        num_frames = (audio_length - self.frame_length) // self.hop_length + 1

        if num_frames <= 0:
            return audio.unsqueeze(-1)

        frames = torch.zeros(num_frames, self.frame_length, device=audio.device)

        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            if end <= audio_length:
                frames[i] = audio[start:end]

        return frames

    def _apply_smoothing(self, vad: torch.Tensor) -> torch.Tensor:
        """Apply smoothing to VAD decision."""
        min_speech_frames = 3

        smoothed = torch.zeros_like(vad)
        for i in range(len(vad)):
            start = max(0, i - min_speech_frames)
            end = min(len(vad), i + min_speech_frames + 1)
            smoothed[i] = torch.max(vad[start:end])

        return smoothed

    def _add_padding(self, vad: torch.Tensor) -> torch.Tensor:
        """Add padding around speech segments."""
        padding_frames = self.padding_samples // self.hop_length

        result = vad.clone()
        for i in range(len(vad)):
            if vad[i] > 0:
                start = max(0, i - padding_frames)
                end = min(len(vad), i + padding_frames + 1)
                result[start:end] = 1

        return result


class VoiceFilter:
    """Voice filtering for speech enhancement.

    Filters audio to enhance speech components.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        low_freq: float = 80.0,
        high_freq: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.low_freq = low_freq
        self.high_freq = high_freq

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply voice filtering.

        Args:
            audio: Input audio

        Returns:
            Filtered audio
        """
        return self._bandpass_filter(audio)

    def _bandpass_filter(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filter."""
        nyquist = self.sample_rate / 2

        low_norm = self.low_freq / nyquist
        high_norm = self.high_freq / nyquist

        num_taps = 101
        t = torch.arange(num_taps, device=audio.device).float()

        filter_t = (
            torch.sin(2 * np.pi * high_norm * (t - num_taps // 2))
            - torch.sin(2 * np.pi * low_norm * (t - num_taps // 2))
        ) / (np.pi * (t - num_taps // 2 + 1e-8))

        filter_t[num_taps // 2] = 2 * (high_norm - low_norm)
        window = torch.hann_window(num_taps, device=audio.device)
        filter_t = filter_t * window

        filtered = torch.nn.functional.conv1d(
            audio.unsqueeze(0).unsqueeze(0),
            filter_t.view(1, 1, -1),
            padding=num_taps // 2,
        )

        return filtered.squeeze(0).squeeze(0)


class AudioSlicer:
    """Split audio into fixed-size segments.

    Useful for creating consistent input sizes for models.
    """

    def __init__(
        self,
        segment_length: int,
        hop_length: Optional[int] = None,
        pad: bool = True,
    ):
        self.segment_length = segment_length
        self.hop_length = hop_length or segment_length
        self.pad = pad

    def slice(
        self,
        audio: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Slice audio into segments.

        Args:
            audio: Input audio

        Returns:
            List of audio segments
        """
        audio_length = audio.shape[-1]

        segments = []
        start = 0

        while start < audio_length:
            end = start + self.segment_length

            if end <= audio_length:
                segment = audio[..., start:end]
            elif self.pad:
                padding = end - audio_length
                segment = torch.nn.functional.pad(audio[..., start:], (0, padding))
            else:
                break

            segments.append(segment)
            start += self.hop_length

        return segments

    def merge(
        self,
        segments: List[torch.Tensor],
        original_length: int,
    ) -> torch.Tensor:
        """Merge segments back together.

        Args:
            segments: List of audio segments
            original_length: Original audio length

        Returns:
            Merged audio
        """
        if not segments:
            return torch.zeros(original_length)

        device = segments[0].device
        result = torch.zeros(original_length, device=device)
        weights = torch.zeros(original_length, device=device)

        start = 0
        for segment in segments:
            seg_len = segment.shape[-1]
            result[start : start + seg_len] += segment
            weights[start : start + seg_len] += 1
            start += self.hop_length

        weights = torch.where(weights > 0, weights, 1)
        result = result / weights

        return result


class SilenceRemover:
    """Remove silence from audio.

    Useful for preprocessing to remove non-speech segments.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold_db: float = -40.0,
        min_silence_duration: float = 0.3,
    ):
        self.sample_rate = sample_rate
        self.threshold_db = threshold_db
        self.min_silence_duration = min_silence_duration

        self.min_silence_samples = int(sample_rate * min_silence_duration)

    def remove(
        self,
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Remove silence from audio.

        Args:
            audio: Input audio

        Returns:
            Tuple of (audio without silence, list of non-silent segments)
        """
        audio_length = audio.shape[-1]

        energy = audio**2

        window_size = self.min_silence_samples
        energy_smoothed = (
            torch.nn.functional.avg_pool1d(
                energy.unsqueeze(0).unsqueeze(0),
                kernel_size=window_size,
                stride=window_size // 2,
            )
            .squeeze(0)
            .squeeze(0)
        )

        threshold = 10 ** (self.threshold_db / 10)
        is_speech = energy_smoothed > threshold

        non_silent_segments = []
        in_speech = False
        start = 0

        for i, speech in enumerate(is_speech.tolist()):
            if speech and not in_speech:
                start = i * (window_size // 2)
                in_speech = True
            elif not speech and in_speech:
                end = i * (window_size // 2)
                non_silent_segments.append((start, end))
                in_speech = False

        if in_speech:
            non_silent_segments.append((start, audio_length))

        if non_silent_segments:
            result = torch.cat(
                [audio[..., s:e] for s, e in non_silent_segments], dim=-1
            )
        else:
            result = audio

        return result, non_silent_segments

"""
Prosody Conversion Module for Voice Conversion

This module provides prosody conversion components:
- PitchConverter: F0 (pitch) conversion
- DurationConverter: Duration conversion
- EnergyConverter: Energy/volume conversion
- ProsodyExtractor: Extracts prosodic features
- ProsodyConverter: Combines all prosody conversions
- HierarchicalProsodyConverter: Hierarchical approach
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


@dataclass
class ProsodyConfig:
    """Configuration for prosody conversion."""

    n_mels: int = 80
    sample_rate: int = 22050
    hop_length: int = 256
    f0_min: float = 50.0
    f0_max: float = 500.0
    f0_method: str = "pyin"
    hidden_dim: int = 128
    pitch_embed_dim: int = 256
    duration_embed_dim: int = 256
    energy_embed_dim: int = 256
    num_pitch_bins: int = 256
    num_energy_bins: int = 256


class PitchExtractor(nn.Module):
    """Extracts pitch (F0) contours from audio.

    Supports multiple pitch extraction methods.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 256,
        f0_min: float = 50.0,
        f0_max: float = 500.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max

    def forward(self, audio: Tensor) -> Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        window_length = 2048
        hop_size = self.hop_length

        stft = torch.stft(
            audio,
            n_fft=2048,
            hop_length=hop_size,
            win_length=window_length,
            window=torch.hann_window(window_length).to(audio.device),
            return_complex=True,
        )

        magnitude = stft.abs()
        phase = stft.angle()

        autocorr = self._autocorrelation(magnitude)

        f0 = self._autocorr_to_f0(autocorr)

        f0 = self._median_smoothing(f0, kernel_size=5)

        return f0

    def _autocorrelation(self, spec: Tensor) -> Tensor:
        n_fft = spec.size(0)
        frames = spec.size(1)

        autocorr = torch.zeros_like(spec)

        for i in range(frames):
            frame = spec[:, i]
            for lag in range(1, n_fft // 2):
                autocorr[lag, i] = (frame[:-lag] * frame[lag:]).sum()

        return autocorr

    def _autocorr_to_f0(self, autocorr: Tensor) -> Tensor:
        freqs = torch.fft.rfftfreq(autocorr.size(0), 1.0 / self.sample_rate).to(
            autocorr.device
        )

        f0 = torch.zeros(autocorr.size(1), device=autocorr.device)

        for i in range(autocorr.size(1)):
            ac = autocorr[:, i]
            peaks, _ = self._find_peaks(ac)
            if len(peaks) > 0:
                f0[i] = self.f0_min
            else:
                f0[i] = 0.0

        return f0

    def _find_peaks(self, signal: Tensor) -> Tuple[Tensor, Tensor]:
        signal = signal[1:]
        peaks = (
            torch.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0]
            + 1
        )

        if len(peaks) == 0:
            return peaks, torch.zeros_like(peaks)

        values = signal[peaks]
        sorted_idx = torch.argsort(values, descending=True)
        return peaks[sorted_idx], values[sorted_idx]

    def _median_smoothing(self, f0: Tensor, kernel_size: int = 5) -> Tensor:
        pad = kernel_size // 2
        padded = F.pad(f0.unsqueeze(0), (pad, pad), mode="replicate")
        smoothed = padded.unfold(1, kernel_size, 1).median(dim=2)[0]
        return smoothed.squeeze(0)


class PitchConverter(nn.Module):
    """Converts pitch (F0) contours between speakers.

    Supports linear conversion and neural conversion approaches.
    """

    def __init__(
        self,
        config: ProsodyConfig,
        conversion_method: str = "linear",
    ):
        super().__init__()
        self.config = config
        self.conversion_method = conversion_method

        if conversion_method == "neural":
            self.pitch_net = nn.Sequential(
                nn.Linear(1, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
            )

            self.log_scale = nn.Parameter(torch.zeros(1))
            self.log_shift = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        source_pitch: Tensor,
        target_statistics: Optional[Dict[str, float]] = None,
    ) -> Tensor:
        if self.conversion_method == "linear":
            if target_statistics is not None:
                source_mean = source_pitch.mean()
                source_std = source_pitch.std()

                target_mean = target_statistics.get("mean", source_mean)
                target_std = target_statistics.get("std", source_std)

                converted = (source_pitch - source_mean) / (source_std + 1e-8)
                converted = converted * target_std + target_mean

                return converted
            return source_pitch

        elif self.conversion_method == "neural":
            pitch_features = source_pitch.unsqueeze(-1)
            converted = self.pitch_net(pitch_features)
            return converted.squeeze(-1)

        return source_pitch

    def extract_statistics(self, pitch: Tensor) -> Dict[str, float]:
        return {
            "mean": pitch.mean().item(),
            "std": pitch.std().item(),
            "min": pitch.min().item(),
            "max": pitch.max().item(),
            "median": pitch.median().item(),
        }


class DurationConverter(nn.Module):
    """Converts duration between speakers.

    Handles duration alignment and expansion/contraction.
    """

    def __init__(
        self,
        config: ProsodyConfig,
    ):
        super().__init__()
        self.config = config

        self.duration_predictor = nn.Sequential(
            nn.Conv1d(config.n_mels, config.hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.hidden_dim, 1, 1),
        )

        self.duration_ratio_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def predict_duration(self, mel: Tensor) -> Tensor:
        duration = self.duration_predictor(mel.transpose(1, 2)).squeeze(1)
        duration = F.softplus(duration)
        return duration

    def forward(
        self,
        source_duration: Tensor,
        target_statistics: Optional[Dict[str, float]] = None,
    ) -> Tensor:
        if target_statistics is not None:
            source_mean = source_duration.mean()
            target_mean = target_statistics.get("mean", source_mean)

            ratio = target_mean / (source_mean + 1e-8)
            converted = source_duration * ratio

            return converted

        return source_duration


class EnergyConverter(nn.Module):
    """Converts energy/volume between speakers."""

    def __init__(
        self,
        config: ProsodyConfig,
    ):
        super().__init__()
        self.config = config

        self.energy_predictor = nn.Sequential(
            nn.Conv1d(config.n_mels, config.hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.hidden_dim, 1, 1),
        )

        self.log_scale = nn.Parameter(torch.ones(1))
        self.log_shift = nn.Parameter(torch.zeros(1))

    def extract_energy(self, mel: Tensor) -> Tensor:
        energy = (mel**2).sum(dim=1).sqrt()
        return energy

    def forward(
        self,
        source_energy: Tensor,
        target_statistics: Optional[Dict[str, float]] = None,
    ) -> Tensor:
        log_energy = torch.log(source_energy + 1e-8)

        if target_statistics is not None:
            source_mean = log_energy.mean()
            source_std = log_energy.std()

            target_mean = target_statistics.get("log_mean", source_mean)
            target_std = target_statistics.get("log_std", source_std)

            converted = (log_energy - source_mean) / (source_std + 1e-8)
            converted = (
                converted * target_std * self.log_scale.exp()
                + target_mean
                + self.log_shift
            )

            return torch.exp(converted)

        return source_energy


class ProsodyExtractor(nn.Module):
    """Extracts prosodic features from audio."""

    def __init__(self, config: ProsodyConfig):
        super().__init__()
        self.config = config

        self.pitch_extractor = PitchExtractor(
            config.sample_rate,
            config.hop_length,
            config.f0_min,
            config.f0_max,
        )

    def extract_pitch(self, audio: Tensor) -> Tensor:
        pitch = self.pitch_extractor(audio)
        return pitch

    def extract_energy(self, mel: Tensor) -> Tensor:
        energy = (mel**2).sum(dim=1).sqrt()
        return energy

    def extract_all(self, audio: Tensor, mel: Tensor) -> Dict[str, Tensor]:
        pitch = self.extract_pitch(audio)
        energy = self.extract_energy(mel)

        return {
            "pitch": pitch,
            "energy": energy,
        }


class ProsodyConverter(nn.Module):
    """Complete prosody conversion combining pitch, duration, and energy.

    Args:
        config: ProsodyConfig with model parameters
    """

    def __init__(self, config: ProsodyConfig):
        super().__init__()
        self.config = config

        self.pitch_converter = PitchConverter(config)
        self.duration_converter = DurationConverter(config)
        self.energy_converter = EnergyConverter(config)

    def convert_pitch(
        self,
        source_pitch: Tensor,
        target_stats: Optional[Dict[str, float]] = None,
    ) -> Tensor:
        return self.pitch_converter(source_pitch, target_stats)

    def convert_duration(
        self,
        source_duration: Tensor,
        target_stats: Optional[Dict[str, float]] = None,
    ) -> Tensor:
        return self.duration_converter(source_duration, target_stats)

    def convert_energy(
        self,
        source_energy: Tensor,
        target_stats: Optional[Dict[str, float]] = None,
    ) -> Tensor:
        return self.energy_converter(source_energy, target_stats)

    def forward(
        self,
        source_pitch: Tensor,
        source_duration: Tensor,
        source_energy: Tensor,
        target_pitch_stats: Optional[Dict[str, float]] = None,
        target_duration_stats: Optional[Dict[str, float]] = None,
        target_energy_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tensor]:
        converted_pitch = self.convert_pitch(source_pitch, target_pitch_stats)
        converted_duration = self.convert_duration(
            source_duration, target_duration_stats
        )
        converted_energy = self.convert_energy(source_energy, target_energy_stats)

        return {
            "pitch": converted_pitch,
            "duration": converted_duration,
            "energy": converted_energy,
        }


class HierarchicalProsodyConverter(nn.Module):
    """Hierarchical prosody conversion with separate levels.

    Converts prosody at different levels: utterance, sentence, and phoneme.
    """

    def __init__(self, config: ProsodyConfig):
        super().__init__()
        self.config = config

        self.utterance_converter = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

        self.segment_converter = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

        self.frame_converter = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        pitch: Tensor,
        segment_boundaries: Optional[Tensor] = None,
    ) -> Tensor:
        if pitch.dim() == 1:
            pitch = pitch.unsqueeze(0)

        if segment_boundaries is None:
            segment_size = pitch.size(1) // 4
            segment_boundaries = torch.arange(
                0, pitch.size(1), segment_size, device=pitch.device
            )

        pitch_utterance = pitch.mean(dim=1, keepdim=True)
        pitch_utterance = self.utterance_converter(pitch_utterance)

        converted_segments = []
        for i in range(len(segment_boundaries) - 1):
            start = segment_boundaries[i]
            end = segment_boundaries[i + 1]
            segment = pitch[:, start:end]

            segment_mean = segment.mean(dim=1, keepdim=True)
            segment_converted = self.segment_converter(segment_mean)

            converted_segments.append(segment_converted)

        pitch = self.frame_converter(pitch.unsqueeze(-1)).squeeze(-1)

        return pitch


class ContourPredictor(nn.Module):
    """Predicts prosody contours for natural-sounding conversion."""

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 128,
        output_dim: int = 1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.projection = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, mel: Tensor) -> Tensor:
        output, _ = self.lstm(mel)
        contour = self.projection(output)
        return contour.squeeze(-1)


def create_prosody_converter(
    config: Optional[ProsodyConfig] = None,
    converter_type: str = "full",
) -> nn.Module:
    """Factory function to create prosody converters.

    Args:
        config: ProsodyConfig with parameters
        converter_type: Type of converter ('full', 'pitch', 'hierarchical')

    Returns:
        Initialized prosody converter
    """
    if config is None:
        config = ProsodyConfig()

    if converter_type == "full":
        return ProsodyConverter(config)
    elif converter_type == "pitch":
        return PitchConverter(config)
    elif converter_type == "hierarchical":
        return HierarchicalProsodyConverter(config)
    else:
        raise ValueError(f"Unknown converter type: {converter_type}")

"""
Audio Postprocessing for Source Separation

Provides postprocessing utilities for separated audio including
merging, filtering, smoothing, and output formatting.
"""

from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Postprocessor:
    """Audio postprocessing pipeline.

    Applies various postprocessing steps to separated sources.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        normalize: bool = True,
        clip: bool = True,
    ):
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.clip = clip

    def __call__(
        self,
        sources: torch.Tensor,
    ) -> torch.Tensor:
        """Apply postprocessing to sources.

        Args:
            sources: Separated source signals

        Returns:
            Postprocessed sources
        """
        if self.normalize:
            sources = self._normalize_sources(sources)

        if self.clip:
            sources = self._clip_audio(sources)

        return sources

    def _normalize_sources(self, sources: torch.Tensor) -> torch.Tensor:
        """Normalize each source."""
        normalized = []
        for i in range(sources.shape[0]):
            source = sources[i]
            peak = torch.max(torch.abs(source))
            if peak > 0:
                source = source / peak * 0.95
            normalized.append(source)
        return torch.stack(normalized)

    def _clip_audio(self, sources: torch.Tensor) -> torch.Tensor:
        """Clip audio to prevent clipping distortion."""
        return torch.clamp(sources, -1.0, 1.0)


class AudioMerger:
    """Merge multiple audio sources for output.

    Handles mixing and combining source estimates.
    """

    @staticmethod
    def mix_to_mono(audio: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel audio to mono.

        Args:
            audio: Multi-channel audio

        Returns:
            Mono audio
        """
        if audio.dim() == 1:
            return audio
        return torch.mean(audio, dim=0)

    @staticmethod
    def weighted_sum(
        sources: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mix sources with weights.

        Args:
            sources: Source signals
            weights: Mixing weights

        Returns:
            Mixed audio
        """
        if weights is None:
            weights = torch.ones(sources.shape[0], device=sources.device)

        weights = weights / weights.sum()

        mixed = sum(w * s for w, s in zip(weights, sources))

        return mixed

    @staticmethod
    def concatenate(
        sources: torch.Tensor,
        overlap: float = 0.0,
    ) -> torch.Tensor:
        """Concatenate sources sequentially.

        Args:
            sources: Source signals
            overlap: Overlap ratio between segments

        Returns:
            Concatenated audio
        """
        if sources.shape[0] == 1:
            return sources[0]

        result = sources[0]
        hop = int(sources.shape[-1] * (1 - overlap))

        for i in range(1, sources.shape[0]):
            result = torch.cat([result, sources[i, hop:]], dim=-1)

        return result


class SourceFilter:
    """Filter separated sources.

    Applies frequency-based filtering to enhance source quality.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        low_cutoff: Optional[float] = None,
        high_cutoff: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply filtering.

        Args:
            audio: Input audio

        Returns:
            Filtered audio
        """
        return self._apply_filter(audio)

    def _apply_filter(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply bandpass filtering."""
        nyquist = self.sample_rate / 2

        num_taps = 65

        t = torch.arange(num_taps, device=audio.device).float() - num_taps // 2

        filter_t = torch.ones(num_taps, device=audio.device)

        if self.low_cutoff is not None:
            low_norm = self.low_cutoff / nyquist
            low_pass = 2 * low_norm * torch.sinc(2 * low_norm * t)
            filter_t = filter_t * low_pass

        if self.high_cutoff is not None:
            high_norm = self.high_cutoff / nyquist
            high_pass = 2 * high_norm * torch.sinc(2 * high_norm * t)
            filter_t = filter_t * high_pass

        window = torch.hann_window(num_taps, device=audio.device)
        filter_t = filter_t * window
        filter_t = filter_t / torch.sum(filter_t)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        filtered = F.conv1d(
            audio.unsqueeze(0),
            filter_t.view(1, 1, -1),
            padding=num_taps // 2,
        )

        return filtered.squeeze(0)


class SpectralSmoother:
    """Smooth spectral content of separated sources.

    Applies temporal and frequency smoothing to reduce artifacts.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        temporal_window: int = 5,
        freq_window: int = 3,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.temporal_window = temporal_window
        self.freq_window = freq_window

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply spectral smoothing.

        Args:
            audio: Input audio

        Returns:
            Smoothed audio
        """
        return self._smooth_spectral(audio)

    def _smooth_spectral(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply spectral smoothing in STFT domain."""
        window = torch.hann_window(self.n_fft, device=audio.device)

        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )

        mag = torch.abs(stft)
        phase = torch.angle(stft)

        smoothed_mag = F.avg_pool2d(
            mag.unsqueeze(0),
            kernel_size=(self.freq_window, self.temporal_window),
            padding=(self.freq_window // 2, self.temporal_window // 2),
            stride=1,
        ).squeeze(0)

        smoothed_stft = torch.complex(
            smoothed_mag * torch.cos(phase), smoothed_mag * torch.sin(phase)
        )

        result = torch.istft(
            smoothed_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
        )

        return result


class ArtifactReducer:
    """Reduce artifacts in separated sources.

    Applies various techniques to reduce musical noise and artifacts.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.01,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Reduce artifacts.

        Args:
            audio: Input audio

        Returns:
            Artifact-reduced audio
        """
        return self._reduce_artifacts(audio)

    def _reduce_artifacts(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply artifact reduction."""
        mask = torch.abs(audio) > self.threshold
        result = audio * mask.float()

        missing = ~mask
        if missing.any():
            result = result + audio * missing.float() * 0.1

        return result


class PhaseReconstructor:
    """Reconstruct phase for separated sources.

    Provides phase reconstruction methods for better quality.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def griffin_lim(
        self,
        magnitude: torch.Tensor,
        n_iter: int = 32,
    ) -> torch.Tensor:
        """Griffin-Lim phase reconstruction.

        Args:
            magnitude: Target magnitude spectrogram
            n_iter: Number of iterations

        Returns:
            Reconstructed audio
        """
        phase = torch.rand_like(magnitude) * 2 * np.pi - np.pi

        window = torch.hann_window(self.n_fft, device=magnitude.device)

        for _ in range(n_iter):
            complex_spec = torch.complex(
                magnitude * torch.cos(phase), magnitude * torch.sin(phase)
            )

            audio = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
            )

            stft = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
            )

            phase = torch.angle(stft)

        return audio


class SourceSelector:
    """Select best sources based on energy or quality.

    Useful for selecting most prominent sources.
    """

    @staticmethod
    def select_by_energy(
        sources: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select sources by energy.

        Args:
            sources: Source signals
            top_k: Number of top sources to select

        Returns:
            Tuple of (selected sources, indices)
        """
        energies = torch.sum(sources**2, dim=-1)

        if top_k is None:
            top_k = sources.shape[0]

        _, indices = torch.topk(energies, min(top_k, sources.shape[0]))

        return sources[indices], indices

    @staticmethod
    def filter_low_energy(
        sources: torch.Tensor,
        threshold: float = 0.01,
    ) -> torch.Tensor:
        """Filter out low energy sources.

        Args:
            sources: Source signals
            threshold: Energy threshold ratio

        Returns:
            Filtered sources
        """
        energies = torch.sum(sources**2, dim=-1)
        max_energy = torch.max(energies)

        mask = energies > threshold * max_energy

        return sources * mask.float()


class OutputFormatter:
    """Format output for different use cases.

    Handles output formatting for various purposes.
    """

    def __init__(
        self,
        output_format: str = "tensor",
        sample_rate: int = 16000,
    ):
        self.output_format = output_format
        self.sample_rate = sample_rate

    def format(
        self,
        sources: torch.Tensor,
    ) -> Any:
        """Format sources for output.

        Args:
            sources: Source signals

        Returns:
            Formatted output
        """
        if self.output_format == "tensor":
            return sources
        elif self.output_format == "numpy":
            return sources.cpu().numpy()
        elif self.output_format == "list":
            return [s.cpu().numpy() for s in sources]
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")

    def to_dict(
        self,
        sources: torch.Tensor,
        names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convert sources to dictionary.

        Args:
            sources: Source signals
            names: Source names

        Returns:
            Dictionary of sources
        """
        if names is None:
            names = [f"source_{i}" for i in range(sources.shape[0])]

        return {name: sources[i] for i, name in enumerate(names)}


class OverlapAdder:
    """Handle overlap-add for frame-based separation.

    Reconstructs continuous audio from frame-based separation.
    """

    def __init__(
        self,
        frame_length: int,
        hop_length: int,
        window: Optional[torch.Tensor] = None,
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length

        if window is None:
            self.window = torch.hann_window(frame_length)
        else:
            self.window = window

    def overlap_add(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Apply overlap-add to frames.

        Args:
            frames: Frame-wise audio

        Returns:
            Reconstructed audio
        """
        if frames.dim() == 2:
            frames = frames.unsqueeze(0)

        n_frames, frame_size = frames.shape

        output_length = (n_frames - 1) * self.hop_length + frame_size
        output = torch.zeros(output_length, device=frames.device)
        weights = torch.zeros(output_length, device=frames.device)

        for i, frame in enumerate(frames):
            start = i * self.hop_length
            output[start : start + frame_size] += frame * self.window
            weights[start : start + frame_size] += self.window

        weights = torch.where(weights > 0, weights, 1)
        output = output / weights

        return output


class EnergyNormalizer:
    """Normalize energy of separated sources.

    Ensures consistent energy levels across sources.
    """

    def __init__(
        self,
        target_db: float = -20.0,
        per_source: bool = True,
    ):
        self.target_db = target_db
        self.per_source = per_source

    def __call__(self, sources: torch.Tensor) -> torch.Tensor:
        """Normalize source energy.

        Args:
            sources: Source signals

        Returns:
            Normalized sources
        """
        if self.per_source:
            return self._normalize_per_source(sources)
        else:
            return self._normalize_global(sources)

    def _normalize_per_source(self, sources: torch.Tensor) -> torch.Tensor:
        """Normalize each source independently."""
        normalized = []
        for i in range(sources.shape[0]):
            source = sources[i]
            rms = torch.sqrt(torch.mean(source**2))
            if rms > 0:
                target_rms = 10 ** (self.target_db / 20)
                source = source * (target_rms / rms)
            normalized.append(source)
        return torch.stack(normalized)

    def _normalize_global(self, sources: torch.Tensor) -> torch.Tensor:
        """Normalize all sources together."""
        all_rms = torch.sqrt(torch.mean(sources**2, dim=-1, keepdim=True))
        target_rms = 10 ** (self.target_db / 20)
        return sources * (target_rms / (all_rms + 1e-8))

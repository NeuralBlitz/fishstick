"""
Audio Source Separation Base Classes

Base classes for audio source separation models and utilities.
"""

from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


@dataclass
class SeparationResult:
    """Container for separation results."""

    sources: torch.Tensor
    source_names: Optional[List[str]] = None
    embeddings: Optional[torch.Tensor] = None
    masks: Optional[torch.Tensor] = None

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < self.sources.shape[0]:
            return self.sources[idx]
        raise IndexError(f"Source index {idx} out of range")

    @property
    def num_sources(self) -> int:
        return self.sources.shape[0]


class SeparationModel(nn.Module, ABC):
    """Abstract base class for all audio source separation models."""

    def __init__(
        self,
        n_sources: int = 2,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
    ):
        super().__init__()
        self.n_sources = n_sources
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    @abstractmethod
    def forward(self, mixture: torch.Tensor, **kwargs) -> SeparationResult:
        """Separate mixed audio into individual sources.

        Args:
            mixture: Mixed audio signal of shape (batch, channels, time)

        Returns:
            SeparationResult containing separated sources
        """
        pass

    @abstractmethod
    def estimate_sources(self, mixture: torch.Tensor, **kwargs) -> torch.Tensor:
        """Estimate sources from mixture.

        Args:
            mixture: Mixed audio signal

        Returns:
            Estimated source signals of shape (n_sources, batch, channels, time)
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "n_sources": self.n_sources,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
        }


class Separator:
    """High-level interface for audio source separation."""

    def __init__(
        self,
        model: SeparationModel,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, mixture: torch.Tensor, **kwargs) -> SeparationResult:
        """Separate audio sources."""
        with torch.no_grad():
            if mixture.device != self.device:
                mixture = mixture.to(self.device)
            return self.model(mixture, **kwargs)

    def separate(
        self,
        mixture: torch.Tensor,
        return_tensors: str = "numpy",
    ) -> Dict[str, Any]:
        """Separate audio and return results in specified format."""
        result = self(mixture)

        output = {
            "sources": result.sources,
            "num_sources": result.num_sources,
        }

        if result.source_names:
            output["source_names"] = result.source_names

        if return_tensors == "numpy":
            output["sources"] = output["sources"].cpu().numpy()

        return output

    def separate_file(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
    ) -> SeparationResult:
        """Separate audio from file."""
        try:
            import torchaudio

            waveform, sr = torchaudio.load(file_path)
            if sr != self.model.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.model.sample_rate
                )
            waveform = waveform.unsqueeze(0).to(self.device)
            return self.model(waveform)
        except ImportError:
            raise ImportError("torchaudio required for file input")


class STFT(nn.Module):
    """Short-Time Fourier Transform module."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft

        if window is None:
            self.window = torch.hann_window(self.win_length)
        else:
            self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Complex STFT of shape (batch, channels, freq, time)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        stft = torch.stft(
            x.reshape(-1, x.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            return_complex=True,
        )
        return stft.reshape(x.shape[0], x.shape[1], stft.shape[0], stft.shape[1])

    def inverse(self, stft: torch.Tensor) -> torch.Tensor:
        """Compute inverse STFT.

        Args:
            stft: Complex STFT tensor

        Returns:
            Reconstructed audio of shape (batch, channels, time)
        """
        istft = torch.istft(
            stft.reshape(
                -1, stft.shape[0] * stft.shape[1], stft.shape[2], stft.shape[3]
            ),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(stft.device),
        )
        batch_ch = stft.shape[0] * stft.shape[1]
        return istft.reshape(batch_ch, -1)


class SeparationMetrics:
    """Metrics for evaluating source separation quality."""

    @staticmethod
    def si_sdr(
        estimate: torch.Tensor, reference: torch.Tensor, epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Compute Scale-Invariant Source-to-Distortion Ratio.

        Args:
            estimate: Estimated source signal
            reference: Reference source signal

        Returns:
            SI-SDR value in dB
        """
        estimate = estimate.reshape(-1)
        reference = reference.reshape(-1)

        dot_product = torch.dot(estimate, reference)
        ref_energy = torch.dot(reference, reference) + epsilon

        scale = dot_product / ref_energy
        scaled_ref = scale * reference
        noise = estimate - scaled_ref

        signal_power = torch.dot(scaled_ref, scaled_ref) + epsilon
        noise_power = torch.dot(noise, noise) + epsilon

        return 10 * torch.log10(signal_power / noise_power + epsilon)

    @staticmethod
    def sdr(
        estimate: torch.Tensor, reference: torch.Tensor, epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Compute Source-to-Distortion Ratio.

        Args:
            estimate: Estimated source signal
            reference: Reference source signal

        Returns:
            SDR value in dB
        """
        estimate = estimate.reshape(-1)
        reference = reference.reshape(-1)

        signal_power = torch.dot(reference, reference)
        noise = estimate - reference
        noise_power = torch.dot(noise, noise) + epsilon

        return 10 * torch.log10(signal_power / noise_power + epsilon)

    @staticmethod
    def snr(
        estimate: torch.Tensor, reference: torch.Tensor, epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Compute Signal-to-Noise Ratio.

        Args:
            estimate: Estimated signal
            reference: Reference signal

        Returns:
            SNR value in dB
        """
        return SeparationMetrics.sdr(estimate, reference, epsilon)

    @staticmethod
    def pesq(
        estimate: torch.Tensor,
        reference: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Approximate PESQ score (simplified version).

        Note: Full PESQ requires soundfile/pyworld. This is a simplified
        frequency-domain based approximation.

        Args:
            estimate: Estimated signal
            reference: Reference signal
            sample_rate: Sample rate

        Returns:
            Approximate PESQ score
        """
        est_fft = torch.fft.rfft(estimate)
        ref_fft = torch.fft.rfft(reference)

        spectral_diff = torch.abs(torch.abs(est_fft) - torch.abs(ref_fft))
        spectral_distortion = torch.mean(spectral_diff)

        pesq_approx = 4.5 - 0.5 * spectral_distortion
        return torch.clamp(torch.tensor(pesq_approx), -0.5, 4.5)


class AudioMixer:
    """Utility for mixing audio sources during training."""

    @staticmethod
    def mix_sources(
        sources: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        random_mix: bool = True,
    ) -> torch.Tensor:
        """Mix multiple sources into a single mixture.

        Args:
            sources: Source signals of shape (n_sources, batch, channels, time)
            weights: Mixing weights of shape (n_sources,), optional
            random_mix: If True, randomize source order before mixing

        Returns:
            Mixed audio signal
        """
        if random_mix:
            indices = torch.randperm(sources.shape[0])
            sources = sources[indices]

        if weights is None:
            weights = torch.ones(sources.shape[0], device=sources.device)

        weights = weights / weights.sum()
        mixture = sum(w * s for w, s in zip(weights, sources))

        return mixture

    @staticmethod
    def create_random_mixture(
        sources: torch.Tensor,
        min_sources: int = 2,
        max_sources: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a random mixture from multiple sources.

        Args:
            sources: Source signals of shape (n_sources, batch, channels, time)
            min_sources: Minimum number of sources to mix
            max_sources: Maximum number of sources to mix

        Returns:
            Tuple of (mixture, selected_indices)
        """
        n_sources = sources.shape[0]
        max_sources = max_sources or n_sources

        n_to_mix = torch.randint(
            min_sources, min(max_sources + 1, n_sources + 1), (1,)
        ).item()
        indices = torch.randperm(n_sources)[:n_to_mix]

        selected = sources[indices]
        mixture = selected.sum(dim=0)

        return mixture, indices

    @staticmethod
    def add_noise(
        signal: torch.Tensor,
        snr_db: float = 20.0,
    ) -> torch.Tensor:
        """Add noise to signal at specified SNR.

        Args:
            signal: Input signal
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Signal with added noise
        """
        signal_power = torch.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))

        noise = torch.randn_like(signal) * torch.sqrt(noise_power)

        return signal + noise

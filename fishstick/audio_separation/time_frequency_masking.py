"""
Time-Frequency Masking for Audio Source Separation

Implementation of various time-frequency masking strategies for audio source
separation in the STFT domain.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from fishstick.audio_separation.base import SeparationModel, SeparationResult, STFT


class TimeFrequencyMask(nn.Module):
    """Base class for time-frequency masks."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stft = STFT(n_fft, hop_length)

    def forward(self, mixture: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute mask for mixture."""
        raise NotImplementedError


class IdealBinaryMask(TimeFrequencyMask):
    """Ideal Binary Mask (IBM) for source separation.

    The IBM assigns 1 to time-frequency bins where a particular source
    dominates and 0 otherwise. It provides an upper bound on separation
    performance for ideal conditions.

    Reference:
        Deterministic and Data-Driven Approaches to Audio Source Separation
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        threshold_db: float = 0.0,
    ):
        super().__init__(n_fft, hop_length)
        self.threshold_db = threshold_db

    def compute_ibm(
        self,
        sources: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ideal binary mask for each source.

        Args:
            sources: Target sources of shape (n_sources, batch, channels, freq, time)
            mixture: Mixture STFT of shape (batch, channels, freq, time)

        Returns:
            Binary masks of shape (n_sources, batch, channels, freq, time)
        """
        source_mags = torch.abs(sources)
        mix_mag = torch.abs(mixture)

        threshold = 10 ** (self.threshold_db / 10)

        masks = []
        for i in range(sources.shape[0]):
            source_mask = (source_mags[i] > threshold * mix_mag).float()
            masks.append(source_mask)

        return torch.stack(masks)

    def forward(
        self, mixture: torch.Tensor, sources: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Apply or compute binary mask.

        If sources are provided, computes IBM. Otherwise, applies
        a learned or default mask to the mixture.
        """
        if sources is not None:
            mix_stft = self.stft(mixture)
            masks = self.compute_ibm(sources, mix_stft)
            return masks
        else:
            return torch.ones_like(mixture.unsqueeze(0))


class IdealRatioMask(TimeFrequencyMask):
    """Ideal Ratio Mask (IRM) for source separation.

    The IRM is a soft mask representing the power ratio of each source
    to the total mixture power. It provides better performance than
    IBM in noisy conditions.

    Reference:
        Musical Source Separation, 2001
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        alpha: float = 1.0,
    ):
        super().__init__(n_fft, hop_length)
        self.alpha = alpha

    def compute_irm(
        self,
        sources: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ideal ratio mask for each source.

        Args:
            sources: Target sources of shape (n_sources, batch, channels, freq, time)

        Returns:
            Ratio masks of shape (n_sources, batch, channels, freq, time)
        """
        source_power = torch.abs(sources) ** 2

        total_power = source_power.sum(dim=0, keepdim=True)

        masks = (source_power / (total_power + 1e-8)) ** (self.alpha / 2)

        return masks

    def forward(
        self,
        mixture: torch.Tensor,
        sources: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply or compute ratio mask."""
        if sources is not None:
            return self.compute_irm(sources)

        mix_stft = self.stft(mixture)
        mag = torch.abs(mix_stft)
        mask = mag / (mag.sum(dim=0, keepdim=True) + 1e-8)
        return mask.unsqueeze(0)


class PhaseSensitiveMask(TimeFrequencyMask):
    """Phase-Sensitive Mask (PSM) for source separation.

    The PSM accounts for both magnitude and phase differences
    between sources and mixture, providing better reconstruction
    than magnitude-only masks.

    Reference:
        Phase-Sensitive Masking for Monaural Speech Separation
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        epsilon: float = 1e-8,
    ):
        super().__init__(n_fft, hop_length)
        self.epsilon = epsilon

    def compute_psm(
        self,
        sources: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        """Compute phase-sensitive mask.

        Args:
            sources: Target sources of shape (n_sources, batch, channels, freq, time)
            mixture: Mixture STFT

        Returns:
            Complex masks of shape (n_sources, batch, channels, freq, time)
        """
        source_mag = torch.abs(sources)
        mix_mag = torch.abs(mixture)

        mix_phase = torch.angle(mixture)
        source_phase = torch.angle(sources)

        phase_diff = source_phase - mix_phase

        psm = (source_mag / (mix_mag + self.epsilon)) * torch.cos(phase_diff)
        psm = F.relu(psm)

        return psm

    def forward(
        self,
        mixture: torch.Tensor,
        sources: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply or compute phase-sensitive mask."""
        if sources is not None:
            return self.compute_psm(sources, mixture)
        return torch.ones_like(mixture.unsqueeze(0))


class ComplexMask(TimeFrequencyMask):
    """Complex-valued ratio mask for STFT domain separation.

    This mask preserves both magnitude and phase information,
    providing better phase reconstruction.

    Reference:
        Complex Ratio Masking for Monaural Speech Enhancement
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
    ):
        super().__init__(n_fft, hop_length)

    def compute_complex_mask(
        self,
        source: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        """Compute complex ratio mask.

        Args:
            source: Target source STFT
            mixture: Mixture STFT

        Returns:
            Complex mask
        """
        source_conj = torch.conj(source)
        mixture_mag_sq = torch.abs(mixture) ** 2 + 1e-8

        mask_real = torch.real(source_conj * mixture) / mixture_mag_sq
        mask_imag = torch.imag(source_conj * mixture) / mixture_mag_sq

        return torch.complex(mask_real, mask_imag)

    def apply_mask(
        self,
        mixture: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply complex mask to mixture."""
        return mixture * mask

    def forward(
        self,
        mixture: torch.Tensor,
        sources: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply or compute complex mask."""
        if sources is not None:
            masks = []
            for i in range(sources.shape[0]):
                mask = self.compute_complex_mask(sources[i], mixture)
                masks.append(mask)
            return torch.stack(masks)
        return torch.ones_like(mixture.unsqueeze(0))


class WienerFilter(TimeFrequencyMask):
    """Wiener filtering for optimal MMSE source separation.

    Implements Wiener filter in the time-frequency domain for
    optimal minimum mean square error estimation.

    Reference:
        Speech Enhancement: A Signal Subspace Perspective
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        epsilon: float = 1e-8,
    ):
        super().__init__(n_fft, hop_length)
        self.epsilon = epsilon

    def compute_wiener_gain(
        self,
        source_psd: torch.Tensor,
        noise_psd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Wiener filter gain.

        Args:
            source_psd: Source power spectral density
            noise_psd: Noise power spectral density (if None, assumes speech-dominated)

        Returns:
            Wiener filter gain
        """
        if noise_psd is None:
            total_psd = source_psd + self.epsilon
            gain = source_psd / total_psd
        else:
            total_psd = source_psd + noise_psd
            gain = source_psd / (total_psd + self.epsilon)

        return gain

    def forward(
        self,
        mixture: torch.Tensor,
        source_estimates: Optional[torch.Tensor] = None,
        noise_psd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply Wiener filtering.

        Args:
            mixture: Mixture audio
            source_estimates: Initial source estimates (if None, uses mixture)
            noise_psd: Noise PSD estimate
        """
        mix_stft = self.stft(mixture)
        mag = torch.abs(mix_stft)
        power = mag**2

        if source_estimates is None:
            source_estimates = mix_stft

        if isinstance(source_estimates, torch.Tensor):
            source_psd = torch.abs(source_estimates) ** 2
        else:
            source_psd = torch.stack([torch.abs(s) ** 2 for s in source_estimates])

        if source_psd.dim() == 4:
            source_psd = source_psd.unsqueeze(0)

        gain = self.compute_wiener_gain(source_psd, noise_psd)

        estimated_stft = mix_stft * gain

        return self.stft.inverse(estimated_stft)


class TFMaskingNetwork(nn.Module):
    """Learnable time-frequency masking network.

    A neural network that learns to predict masks for source separation.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        n_sources: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_sources = n_sources

        n_freqs = n_fft // 2 + 1

        self.encoder = nn.Sequential(
            nn.Linear(n_freqs * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_freqs * n_sources),
        )

        self.stft = STFT(n_fft, hop_length)

    def forward(
        self,
        mixture: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict masks and apply them.

        Args:
            mixture: Mixed audio of shape (batch, channels, time)

        Returns:
            Tuple of (separated sources, masks)
        """
        mix_stft = self.stft(mixture)

        batch, channels, freq, time = mix_stft.shape

        mag = torch.abs(mix_stft)
        phase = torch.angle(mix_stft)

        x = torch.cat([mag, phase], dim=2)
        x = x.permute(0, 2, 1, 3).reshape(batch, freq * 2, time)

        x = self.encoder(x)
        x = self.decoder(x)

        masks = x.reshape(batch, self.n_sources, freq, time)
        masks = F.softmax(masks, dim=1)

        separated = []
        for i in range(self.n_sources):
            source = mix_stft * masks[:, i : i + 1]
            source_wav = self.stft.inverse(source)
            separated.append(source_wav)

        separated = torch.stack(separated)

        return separated, masks


class IBMEstimator(nn.Module):
    """Estimator for IBM from mixtures without ground truth.

    Uses neural network to estimate ideal binary mask from mixture.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        n_sources: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_sources = n_sources

        self.stft = STFT(n_fft, hop_length)

        freq_bins = n_fft // 2 + 1

        self.feature_net = nn.Sequential(
            nn.Linear(freq_bins * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.mask_net = nn.Linear(hidden_dim, freq_bins * n_sources)

    def forward(
        self,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate IBM from mixture.

        Args:
            mixture: Mixed audio

        Returns:
            Binary masks of shape (n_sources, batch, freq, time)
        """
        mix_stft = self.stft(mixture)

        mag = torch.abs(mix_stft)
        phase = torch.angle(mix_stft)

        x = torch.cat([mag, phase], dim=2)
        x = x.permute(0, 2, 3, 1)

        batch, freq, time, channels = x.shape
        x = x.reshape(batch, freq * time, channels)

        features = self.feature_net(x)

        masks = self.mask_net(features)
        masks = masks.reshape(batch, freq, time, self.n_sources)
        masks = masks.permute(3, 0, 1, 2)

        masks = (masks > 0.5).float()

        return masks

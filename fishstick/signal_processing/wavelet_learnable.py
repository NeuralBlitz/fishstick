"""
Learnable and Adaptive Wavelet Transforms

Wavelet transforms with learnable parameters for adaptive signal decomposition.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnableWavelet(nn.Module):
    """Learnable wavelet with adaptive frequency and bandwidth.

    Learnable parameters for center frequency and bandwidth allow
    the wavelet to adapt to the signal statistics during training.
    """

    def __init__(
        self,
        num_scales: int = 8,
        init_center_freq: float = 1.0,
        init_bandwidth: float = 1.0,
        num_oscillations: float = 6.0,
        learnable_freq: bool = True,
        learnable_bandwidth: bool = True,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_oscillations = num_oscillations

        self.center_freq_log = nn.Parameter(
            torch.tensor(np.log(init_center_freq)), requires_grad=learnable_freq
        )
        self.bandwidth_log = nn.Parameter(
            torch.tensor(np.log(init_bandwidth)), requires_grad=learnable_bandwidth
        )

    @property
    def center_freq(self) -> torch.Tensor:
        return torch.exp(self.center_freq_log)

    @property
    def bandwidth(self) -> torch.Tensor:
        return torch.exp(self.bandwidth_log)

    def forward(self, length: int, scale: float = 1.0) -> torch.Tensor:
        """Generate learnable wavelet at given scale.

        Args:
            length: Wavelet length
            scale: Scale factor

        Returns:
            Complex wavelet of shape (length,)
        """
        t = torch.arange(length, dtype=torch.float32, device=self.center_freq.device)
        t = t / (self.bandwidth * scale.clamp(min=0.1))

        envelope = torch.exp(-0.5 * t**2)
        oscillation = torch.exp(
            1j * 2 * np.pi * self.center_freq * t / self.num_oscillations
        )

        wavelet = envelope * oscillation
        return wavelet


class AdaptiveWaveletBank(nn.Module):
    """Bank of learnable wavelets with adaptive scales.

    Learnable scales and wavelet parameters for adaptive decomposition.
    """

    def __init__(
        self,
        num_scales: int = 16,
        num_orientations: int = 1,
        min_scale: float = 1.0,
        max_scale: float = 32.0,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations

        self.scales_log = nn.Parameter(
            torch.linspace(np.log(min_scale), np.log(max_scale), num_scales)
        )

        self.wavelet = LearnableWavelet(
            num_scales=num_scales,
            init_center_freq=1.0,
            init_bandwidth=1.0,
        )

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.scales_log)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply adaptive wavelet bank to signal.

        Args:
            signal: Input signal of shape (batch, length) or (length,)

        Returns:
            Wavelet coefficients of shape (batch, num_scales, length)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size, signal_length = signal.shape
        device = signal.device

        scales = self.scales.to(device)
        coefficients = torch.zeros(
            batch_size,
            self.num_scales,
            signal_length,
            dtype=torch.complex64,
            device=device,
        )

        for i, scale in enumerate(scales):
            wavelet_length = min(int(signal_length * 0.5), int(64 * scale.item()))
            wavelet_length = max(wavelet_length, 8)
            wavelet_length = (
                wavelet_length if wavelet_length % 2 == 0 else wavelet_length + 1
            )

            wavelet = self.wavelet(wavelet_length, scale.item())
            wavelet = wavelet / (torch.max(torch.abs(wavelet)) + 1e-8)

            signal_padded = F.pad(signal, (wavelet_length // 2, wavelet_length // 2))

            for b in range(batch_size):
                conv_result = F.conv1d(
                    signal_padded[b : b + 1].real.unsqueeze(0),
                    wavelet.real.unsqueeze(0).unsqueeze(0),
                    padding=wavelet_length // 2,
                )
                conv_imag = F.conv1d(
                    signal_padded[b : b + 1].real.unsqueeze(0),
                    wavelet.imag.unsqueeze(0).unsqueeze(0),
                    padding=wavelet_length // 2,
                )
                out_len = min(conv_result.shape[-1], signal_length)
                coefficients[b, i, :out_len] = (
                    conv_result.squeeze(0).squeeze(0)[:out_len]
                    + 1j * conv_imag.squeeze(0).squeeze(0)[:out_len]
                )

        return coefficients


class LearnableWaveletLayer(nn.Module):
    """End-to-end learnable wavelet layer for neural networks.

    Combines wavelet decomposition with learnable transformation.
    """

    def __init__(
        self,
        num_scales: int = 8,
        hidden_dim: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()
        self.wavelet_bank = AdaptiveWaveletBank(num_scales=num_scales)

        self.attention = nn.Sequential(
            nn.Linear(num_scales, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
            nn.Sigmoid(),
        )

        self.projection = nn.Conv1d(num_scales, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through learnable wavelet layer.

        Args:
            x: Input signal of shape (batch, length) or (batch, length, channels)

        Returns:
            Transformed signal of shape (batch, length, out_channels)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, length, channels = x.shape

        coeff = self.wavelet_bank(x.squeeze(-1))

        coeff_mag = torch.abs(coeff)
        attention_weights = self.attention(coeff_mag.mean(dim=-1))
        attention_weights = attention_weights.unsqueeze(-1)

        weighted_coeff = coeff * attention_weights

        x_transposed = weighted_coeff.transpose(1, 2)
        output = self.projection(x_transposed)

        return output.transpose(1, 2)


class WaveletScattering1D(nn.Module):
    """1D Wavelet Scattering Transform with learnable paths.

    Implements scattering transform with optional learnable components.
    """

    def __init__(
        self,
        num_scales: int = 8,
        num_order: int = 2,
        J: int = None,
        Q: int = 1,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_order = num_order
        self.J = J or num_scales
        self.Q = Q

        self.wavelet_bank = AdaptiveWaveletBank(num_scales=num_scales)

        self.order1_pool = nn.AdaptiveAvgPool1d(1)
        self.order2_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scattering transform.

        Args:
            x: Input signal of shape (batch, length)

        Returns:
            Tuple of (order-1 scattering, order-2 scattering)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        S1 = self.wavelet_bank(x)
        S1_mag = torch.abs(S1)

        S1_pooled = self.order1_pool(S1_mag).squeeze(-1)

        if self.num_order >= 2:
            S2 = torch.zeros_like(S1)
            for i in range(self.num_scales):
                S2[:, i, :] = (
                    S1_mag[:, i, :] * S1_mag[:, (i + 1) % self.num_scales, :].detach()
                )

            S2_pooled = self.order2_pool(torch.abs(S2)).squeeze(-1)
        else:
            S2_pooled = torch.zeros(batch_size, 0, device=x.device)

        return S1_pooled, S2_pooled

    def get_invariant_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get translation-invariant scattering features."""
        S1, S2 = self.forward(x)
        return torch.cat([S1, S2], dim=-1)


class StationaryWaveletTransform(nn.Module):
    """Stationary Wavelet Transform (Undecimated DWT).

    Provides shift-invariant wavelet decomposition using convolution
    at all scales without downsampling.
    """

    SUPPORTED_WAVELETS = ["db4", "db6", "sym4"]

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 4,
        mode: str = "symmetric",
    ):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

        self._init_filters()

    def _init_filters(self):
        """Initialize wavelet filters."""
        import pywt

        wavelet = pywt.Wavelet(self.wavelet)
        self.register_buffer("dec_lo", torch.tensor(wavelet.dec_lo))
        self.register_buffer("dec_hi", torch.tensor(wavelet.dec_hi))
        self.register_buffer("rec_lo", torch.tensor(wavelet.rec_lo))
        self.register_buffer("rec_hi", torch.tensor(wavelet.rec_hi))

    def forward(self, signal: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute stationary wavelet transform.

        Args:
            signal: Input signal

        Returns:
            Tuple of (approximation, list of details at each scale)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        device = signal.device
        approx = signal.float()
        details = []

        for level in range(self.level):
            approx, detail = self._swt_step(approx, level)
            details.append(approx)

        return approx, details

    def _swt_step(
        self, signal: torch.Tensor, level: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single level SWT step with upsampled filters."""
        upsample_factor = 2**level

        lo = self.dec_lo
        hi = self.dec_hi

        lo_upsampled = (
            F.interpolate(
                lo.unsqueeze(0).unsqueeze(0),
                scale_factor=upsample_factor,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        hi_upsampled = (
            F.interpolate(
                hi.unsqueeze(0).unsqueeze(0),
                scale_factor=upsample_factor,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        lo_upsampled = lo_upsampled / upsample_factor
        hi_upsampled = hi_upsampled / upsample_factor

        pad_len = len(lo_upsampled) // 2
        signal_padded = F.pad(signal, (pad_len, pad_len), mode=self.mode)

        approx = F.conv1d(
            signal_padded, lo_upsampled.unsqueeze(0).unsqueeze(0), padding=pad_len
        )
        detail = F.conv1d(
            signal_padded, hi_upsampled.unsqueeze(0).unsqueeze(0), padding=pad_len
        )

        return approx, detail


class DualTreeWaveletTransform(nn.Module):
    """Dual-Tree Complex Wavelet Transform.

    Provides nearly shift-invariant decomposition using two
    parallel wavelet trees.
    """

    def __init__(
        self,
        num_scales: int = 4,
        init_center_freq: float = 1.5,
    ):
        super().__init__()
        self.num_scales = num_scales

        self.tree1_wavelets = nn.ModuleList(
            [
                LearnableWavelet(init_center_freq=init_center_freq * (1.0 + 0.1 * i))
                for i in range(num_scales)
            ]
        )

        self.tree2_wavelets = nn.ModuleList(
            [
                LearnableWavelet(
                    init_center_freq=init_center_freq * (1.0 + 0.1 * i) + 0.5
                )
                for i in range(num_scales)
            ]
        )

    def forward(self, signal: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Compute dual-tree complex wavelet transform.

        Args:
            signal: Input signal

        Returns:
            Tuple of (approximation coefficients, list of complex detail coefficients)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size, length = signal.shape
        device = signal.device

        approx = signal.float()
        details = []

        for scale_idx in range(self.num_scales):
            wavelet1 = self.tree1_wavelets[scale_idx](length // (2**scale_idx) + 4)
            wavelet2 = self.tree2_wavelets[scale_idx](length // (2**scale_idx) + 4)

            conv1 = F.conv1d(
                F.pad(approx, (len(wavelet1) // 2, len(wavelet1) // 2)),
                wavelet1.real.unsqueeze(0).unsqueeze(0),
                padding=len(wavelet1) // 2,
            )
            conv2 = F.conv1d(
                F.pad(approx, (len(wavelet2) // 2, len(wavelet2) // 2)),
                wavelet2.real.unsqueeze(0).unsqueeze(0),
                padding=len(wavelet2) // 2,
            )

            complex_coeff = conv1 + 1j * conv2
            details.append(complex_coeff)

            approx = F.avg_pool1d(approx, kernel_size=2, stride=2)

        return approx, details

    def get_magnitude(
        self, signal: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get magnitude of complex coefficients."""
        approx, details = self.forward(signal)
        details_mag = [torch.abs(d) for d in details]
        return torch.abs(approx), details_mag


class WaveletReconstructionLoss(nn.Module):
    """Loss function for training wavelet-based networks.

    Combines reconstruction loss with scattering invariant loss.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        scattering_weight: float = 0.1,
        num_scales: int = 8,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.scattering_weight = scattering_weight

        self.scattering = WaveletScattering1D(num_scales=num_scales)

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        """Compute wavelet reconstruction loss.

        Args:
            reconstructed: Reconstructed signal
            target: Target signal
            original: Original (unreconstructed) signal

        Returns:
            Combined loss value
        """
        recon_loss = F.mse_loss(reconstructed, target)

        orig_scattering = self.scattering.get_invariant_features(original)
        recon_scattering = self.scattering.get_invariant_features(reconstructed)

        scattering_loss = F.mse_loss(recon_scattering, orig_scattering.detach())

        total_loss = (
            self.reconstruction_weight * recon_loss
            + self.scattering_weight * scattering_loss
        )

        return total_loss


class AdaptiveWaveletSynthesis(nn.Module):
    """Learnable wavelet synthesis for signal reconstruction.

    Learns to combine wavelet coefficients for reconstruction.
    """

    def __init__(
        self,
        num_scales: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_scales = num_scales

        self.combination_weights = nn.Parameter(torch.ones(num_scales))

        self.refinement = nn.Sequential(
            nn.Conv1d(num_scales, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Synthesize signal from wavelet coefficients.

        Args:
            coefficients: Wavelet coefficients of shape (batch, num_scales, length)

        Returns:
            Reconstructed signal
        """
        weights = torch.softmax(self.combination_weights, dim=0)

        weighted = coefficients * weights.view(1, -1, 1)

        output = self.refinement(weighted)

        return output.squeeze(1)

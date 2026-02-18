"""
Empirical Mode Decomposition (EMD)

Adaptive signal decomposition into Intrinsic Mode Functions (IMFs)
for nonlinear and non-stationary signal analysis.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EmpiricalModeDecomposition(nn.Module):
    """Empirical Mode Decomposition for adaptive signal decomposition.

    Decomposes signal into Intrinsic Mode Functions (IMFs) through
    iterative sifting process.
    """

    def __init__(
        self,
        max_iter: int = 100,
        tolerance: float = 0.05,
        num_imfs: Optional[int] = None,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.num_imfs = num_imfs

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Decompose signal into IMFs.

        Args:
            x: Input signal

        Returns:
            Tuple of (list of IMFs, residual)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        signal_length = x.shape[1]

        imfs = []
        residual = x.clone()

        max_imfs = self.num_imfs or 10

        for _ in range(max_imfs):
            imf, residual = self._extract_imf(residual)

            imfs.append(imf)

            if self._is_residual_flat(residual):
                break

        return imfs, residual

    def _extract_imf(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract single IMF through iterative sifting."""
        h = signal.clone()

        for _ in range(self.max_iter):
            upper = self._get_envelope(h, mode="upper")
            lower = self._get_envelope(h, mode="lower")

            mean_envelope = (upper + lower) / 2

            h_new = h - mean_envelope

            sd = torch.sum((h - h_new) ** 2) / (torch.sum(h**2) + 1e-8)

            if sd < self.tolerance:
                break

            h = h_new

        residual = signal - h

        return h, residual

    def _get_envelope(self, signal: torch.Tensor, mode: str = "upper") -> torch.Tensor:
        """Get upper or lower envelope using interpolation."""
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size = signal.shape[0]
        length = signal.shape[1]

        envelopes = []

        for b in range(batch_size):
            s = signal[b].cpu().numpy()

            if mode == "upper":
                peaks, _ = self._find_peaks(s)
                if len(peaks) < 2:
                    envelope = np.maximum.accumulate(s)
                else:
                    x_peaks = np.arange(length)[peaks]
                    envelope = np.interp(
                        np.arange(length),
                        x_peaks,
                        s[peaks],
                        left=s[peaks[0]],
                        right=s[peaks[-1]],
                    )
            else:
                peaks, _ = self._find_peaks(-s)
                if len(peaks) < 2:
                    envelope = np.minimum.accumulate(s)
                else:
                    x_peaks = np.arange(length)[peaks]
                    envelope = np.interp(
                        np.arange(length),
                        x_peaks,
                        s[peaks],
                        left=s[peaks[0]],
                        right=s[peaks[-1]],
                    )

            envelopes.append(
                torch.tensor(envelope, dtype=signal.dtype, device=signal.device)
            )

        return torch.stack(envelopes)

    def _find_peaks(self, signal: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Find peaks in signal."""
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(signal)
        return peaks, properties

    def _is_residual_flat(self, residual: torch.Tensor) -> bool:
        """Check if residual is mostly flat (monotonic or constant)."""
        diff = torch.diff(residual, dim=-1)
        variance = torch.var(diff)

        signal_variance = torch.var(residual)

        return variance < self.tolerance * signal_variance


class LearnableEMD(nn.Module):
    """EMD with learnable envelope parameters."""

    def __init__(
        self,
        num_imfs: int = 5,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_imfs = num_imfs

        self.envelope_net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

        self.imf_weights = nn.Parameter(torch.ones(num_imfs))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Apply learnable EMD decomposition."""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        length = x.shape[1]

        t = torch.linspace(0, 1, length, device=x.device).unsqueeze(0).unsqueeze(-1)

        imfs = []
        residual = x

        for i in range(self.num_imfs):
            imf, residual = self._extract_imf_fast(residual, t)
            imfs.append(imf)

        return imfs

    def _extract_imf_fast(
        self, signal: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast IMF extraction using learnable envelope."""
        h = signal

        for _ in range(20):
            mean_env = signal.mean(dim=-1, keepdim=True)

            h_new = h - mean_env

            h = h_new

        return h, signal - h


class EMDToIMF(nn.Module):
    """EMD layer that outputs fixed number of IMFs."""

    def __init__(
        self,
        num_imfs: int = 8,
        max_iter: int = 50,
    ):
        super().__init__()
        self.num_imfs = num_imfs
        self.max_iter = max_iter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose to fixed number of IMFs.

        Args:
            x: Input signal (batch, length) or (length,)

        Returns:
            IMFs of shape (batch, num_imfs, length)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        length = x.shape[1]

        emd = EmpiricalModeDecomposition(max_iter=self.max_iter, num_imfs=self.num_imfs)

        imfs_list = []

        for b in range(batch_size):
            signal = x[b : b + 1]
            imfs, _ = emd(signal)

            while len(imfs) < self.num_imfs:
                imfs.append(torch.zeros_like(imfs[0]))

            imfs_list.append(torch.cat(imfs[: self.num_imfs], dim=0))

        return torch.stack(imfs_list, dim=0)


class IMFSynthesis(nn.Module):
    """Synthesize signal from IMFs."""

    def __init__(self, num_imfs: int = 8):
        super().__init__()
        self.num_imfs = num_imfs

        self.imf_weights = nn.Parameter(torch.ones(num_imfs))

    def forward(self, imfs: torch.Tensor) -> torch.Tensor:
        """Reconstruct signal from IMFs.

        Args:
            imfs: IMFs of shape (batch, num_imfs, length)

        Returns:
            Reconstructed signal
        """
        weights = torch.softmax(self.imf_weights, dim=0)

        weighted_imfs = imfs * weights.view(1, -1, 1)

        return weighted_imfs.sum(dim=1)


class IMFSelectiveReconstruction(nn.Module):
    """Selectively reconstruct using subset of IMFs."""

    def __init__(self, num_imfs: int = 8):
        super().__init__()
        self.num_imfs = num_imfs

        self.selector = nn.Parameter(torch.ones(num_imfs) > 0)

    def forward(self, imfs: torch.Tensor) -> torch.Tensor:
        """Reconstruct using selected IMFs."""
        mask = self.selector.float().view(1, -1, 1)

        selected = imfs * mask

        return selected.sum(dim=1)


class IntrinsicModeFunctionFeatures(nn.Module):
    """Extract features from IMF decomposition."""

    def __init__(self, num_imfs: int = 5):
        super().__init__()
        self.num_imfs = num_imfs

        self.feature_proj = nn.Sequential(
            nn.Linear(num_imfs * 3, num_imfs * 2),
            nn.ReLU(),
            nn.Linear(num_imfs * 2, num_imfs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract IMF-based features.

        Args:
            x: Input signal

        Returns:
            Feature vector
        """
        emd = EMDToIMF(num_imfs=self.num_imfs)
        imfs = emd(x)

        features = []

        for i in range(self.num_imfs):
            imf = imfs[:, i, :]

            energy = (imf**2).mean(dim=-1)
            zero_crossings = self._count_zero_crossings(imf)
            mean = imf.mean(dim=-1)

            features.extend([energy, zero_crossings.float() / x.shape[-1], mean])

        feature_vector = torch.stack(features, dim=-1)

        return self.feature_proj(feature_vector)

    def _count_zero_crossings(self, x: torch.Tensor) -> torch.Tensor:
        """Count zero crossings."""
        sign_changes = ((x[:, :-1] * x[:, 1:]) < 0).sum(dim=-1)
        return sign_changes.float()


class EMDConvNet(nn.Module):
    """Convolutional network using IMF decomposition."""

    def __init__(
        self,
        num_classes: int = 10,
        num_imfs: int = 5,
    ):
        super().__init__()

        self.emd = EMDToIMF(num_imfs=num_imfs)

        self.conv = nn.Sequential(
            nn.Conv1d(num_imfs, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        imfs = self.emd(x)

        x = self.conv(imfs)

        x = x.squeeze(-1)

        return self.classifier(x)


class VariationalEMD(nn.Module):
    """Variational EMD with learnable decomposition."""

    def __init__(
        self,
        latent_dim: int = 16,
        num_imfs: int = 5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_imfs = num_imfs

        self.encoder = nn.Sequential(
            nn.Linear(num_imfs, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_imfs),
        )

        self.emd = EMDToIMF(num_imfs=num_imfs)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Variational EMD forward pass."""
        imfs = self.emd(x)

        imf_flat = imfs.mean(dim=-1)

        z_mean, z_logvar = self.encoder(imf_flat).chunk(2, dim=-1)

        z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_logvar)

        reconstructed_imfs = self.decoder(z)

        return reconstructed_imfs, z_mean, z_logvar


class EMDLSTM(nn.Module):
    """LSTM network operating on IMF decomposition."""

    def __init__(
        self,
        num_imfs: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 10,
    ):
        super().__init__()

        self.emd = EMDToIMF(num_imfs=num_imfs)

        self.lstm = nn.LSTM(
            input_size=num_imfs,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EMD-LSTM."""
        imfs = self.emd(x)

        imfs_seq = imfs.transpose(1, 2)

        lstm_out, _ = self.lstm(imfs_seq)

        last_output = lstm_out[:, -1, :]

        return self.classifier(last_output)


class EMDAttention(nn.Module):
    """Attention mechanism over IMFs."""

    def __init__(self, num_imfs: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.num_imfs = num_imfs

        self.attention = nn.Sequential(
            nn.Linear(num_imfs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, imfs: torch.Tensor) -> torch.Tensor:
        """Apply attention to IMFs.

        Args:
            imfs: IMFs of shape (batch, num_imfs, length)

        Returns:
            Attended output
        """
        batch_size = imfs.shape[0]

        imf_features = imfs.mean(dim=-1)

        attention_weights = self.attention(imf_features.transpose(1, 2))

        attended = (imfs * attention_weights.transpose(1, 2)).sum(dim=1)

        return attended

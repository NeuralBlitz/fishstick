"""
Climate Pattern Detection

Detection and analysis of climate patterns like ENSO, NAO, and other oscillations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy import signal, stats


@dataclass
class ClimatePattern:
    """Represents a detected climate pattern.

    Args:
        name: Pattern name (e.g., 'ENSO', 'NAO')
        phase: Current phase ('positive', 'negative', 'neutral')
        intensity: Pattern intensity
        confidence: Detection confidence
        timestamp: Detection time
    """

    name: str
    phase: str
    intensity: float
    confidence: float
    timestamp: datetime


class ENSODetector:
    """El Niño-Southern Oscillation (ENSO) detector.

    Monitors sea surface temperature anomalies in the equatorial Pacific
    to detect and classify ENSO phases.

    Args:
        region: Pacific region coordinates (lon_min, lon_max, lat_min, lat_max)
        threshold: Temperature anomaly threshold for El Niño/La Niña
    """

    def __init__(
        self,
        region: Tuple[float, float, float, float] = (190, 240, -5, 5),
        threshold: float = 0.5,
    ):
        self.region = region
        self.threshold = threshold

    def compute_sst_anomaly(
        self,
        sst_data: Tensor,
        climatology: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute sea surface temperature anomaly.

        Args:
            sst_data: SST data of shape (T, H, W)
            climatology: Optional climatology for comparison

        Returns:
            SST anomaly
        """
        if climatology is None:
            climatology = sst_data.mean(dim=0, keepdim=True)

        anomaly = sst_data - climatology
        return anomaly

    def detect(
        self,
        sst_anomaly: Tensor,
    ) -> ClimatePattern:
        """Detect ENSO phase from SST anomaly.

        Args:
            sst_anomaly: SST anomaly tensor

        Returns:
            ClimatePattern with ENSO classification
        """
        region_slice = self._get_region_slice(sst_anomaly)
        regional_anomaly = sst_anomaly[region_slice].mean().item()

        if regional_anomaly > self.threshold:
            phase = "positive"
            intensity = regional_anomaly
        elif regional_anomaly < -self.threshold:
            phase = "negative"
            intensity = abs(regional_anomaly)
        else:
            phase = "neutral"
            intensity = abs(regional_anomaly)

        confidence = min(1.0, abs(regional_anomaly) / (self.threshold * 2))

        return ClimatePattern(
            name="ENSO",
            phase=phase,
            intensity=intensity,
            confidence=confidence,
            timestamp=datetime.now(),
        )

    def _get_region_slice(self, data: Tensor) -> Tuple:
        T, H, W = data.shape
        h_slice = slice(H // 4, 3 * H // 4)
        w_slice = slice(W // 4, 3 * W // 4)
        return (slice(None), h_slice, w_slice)


class NAODetector:
    """North Atlantic Oscillation (NAO) detector.

    Detects NAO based on pressure difference between Iceland and Azores.

    Args:
        north_station: Northern station coordinates
        south_station: Southern station coordinates
        threshold: Pressure difference threshold
    """

    def __init__(
        self,
        north_station: Tuple[float, float] = (63.99, -22.6),
        south_station: Tuple[float, float] = (37.74, -25.67),
        threshold: float = 10.0,
    ):
        self.north_station = north_station
        self.south_station = south_station
        self.threshold = threshold

    def detect(
        self,
        pressure_north: Tensor,
        pressure_south: Tensor,
    ) -> ClimatePattern:
        """Detect NAO phase from pressure difference.

        Args:
            pressure_north: Surface pressure at northern station
            pressure_south: Surface pressure at southern station

        Returns:
            ClimatePattern with NAO classification
        """
        pressure_diff = (pressure_north - pressure_south).mean().item()

        if pressure_diff > self.threshold:
            phase = "positive"
            intensity = pressure_diff / 20.0
        elif pressure_diff < -self.threshold:
            phase = "negative"
            intensity = abs(pressure_diff) / 20.0
        else:
            phase = "neutral"
            intensity = abs(pressure_diff) / 20.0

        confidence = min(1.0, abs(pressure_diff) / (self.threshold * 2))

        return ClimatePattern(
            name="NAO",
            phase=phase,
            intensity=min(1.0, intensity),
            confidence=confidence,
            timestamp=datetime.now(),
        )


class ClimatePatternDetector:
    """Multi-pattern climate detector.

    Args:
        detectors: List of pattern detectors
    """

    def __init__(
        self,
        detectors: Optional[List] = None,
    ):
        if detectors is None:
            detectors = [
                ENSODetector(),
                NAODetector(),
            ]
        self.detectors = detectors

    def detect_all(
        self,
        data: Dict[str, Tensor],
    ) -> List[ClimatePattern]:
        """Detect all registered climate patterns.

        Args:
            data: Dictionary of climate data

        Returns:
            List of detected patterns
        """
        patterns = []

        for detector in self.detectors:
            if isinstance(detector, ENSODetector):
                pattern = detector.detect(
                    data.get("sst_anomaly", torch.zeros(10, 10, 10))
                )
            elif isinstance(detector, NAODetector):
                pattern = detector.detect(
                    data.get("pressure_north", torch.zeros(10)),
                    data.get("pressure_south", torch.zeros(10)),
                )
            else:
                continue

            patterns.append(pattern)

        return patterns


class ClimateAutoencoder(nn.Module):
    """Autoencoder for learning climate pattern representations.

    Args:
        input_channels: Number of input climate variables
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden dimensions
    """

    def __init__(
        self,
        input_channels: int = 5,
        latent_dim: int = 32,
        hidden_dims: List[int] = [128, 64],
    ):
        super().__init__()

        encoder_layers = []
        in_ch = input_channels
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_ch = hidden_dim

        encoder_layers.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_fc = nn.Linear(hidden_dims[-1], latent_dim)

        decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.ReLU(),
        )
        self.decoder_fc = decoder_fc

        decoder_layers = []
        hidden_dims_rev = list(reversed(hidden_dims[:-1])) + [input_channels]
        for i, hidden_dim in enumerate(hidden_dims_rev):
            is_last = i == len(hidden_dims_rev) - 1
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        hidden_dims[-1 - i] if i > 0 else hidden_dim,
                        hidden_dim if not is_last else input_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.Sigmoid() if is_last else nn.ReLU(),
                ]
            )

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.encoder_fc(h)

        h = self.decoder_fc(z)
        h = h.view(h.size(0), -1, 1, 1)
        h = h.expand(-1, -1, 8, 8)

        reconstruction = self.decoder(h)

        return z, reconstruction

    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.encoder_fc(h)

    def decode(self, z: Tensor) -> Tensor:
        h = self.decoder_fc(z)
        h = h.view(h.size(0), -1, 1, 1)
        h = h.expand(-1, -1, 8, 8)
        return self.decoder(h)


class PrincipalComponentAnalysis:
    """PCA for climate pattern extraction.

    Args:
        n_components: Number of principal components
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.mean: Optional[Tensor] = None
        self.components: Optional[Tensor] = None
        self.explained_variance: Optional[Tensor] = None

    def fit(self, data: Tensor) -> "PrincipalComponentAnalysis":
        """Fit PCA to climate data.

        Args:
            data: Input data of shape (N, C, H, W)

        Returns:
            self
        """
        B, C, H, W = data.shape

        flat = data.reshape(B, C * H * W)

        self.mean = flat.mean(dim=0, keepdim=True)
        centered = flat - self.mean

        cov = torch.mm(centered.t(), centered) / (B - 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components = eigenvectors[:, : self.n_components].t()
        self.explained_variance = eigenvalues[: self.n_components]

        return self

    def transform(self, data: Tensor) -> Tensor:
        """Transform data to principal components.

        Args:
            data: Input data

        Returns:
            Principal components
        """
        B, C, H, W = data.shape
        flat = data.reshape(B, C * H * W)

        centered = flat - self.mean

        return torch.mm(centered, self.components.t())

    def inverse_transform(self, components: Tensor) -> Tensor:
        """Inverse transform components to original space.

        Args:
            components: Principal components

        Returns:
            Reconstructed data
        """
        reconstructed = torch.mm(components, self.components)
        return reconstructed + self.mean


class EmpiricalOrthogonalFunction(nn.Module):
    """EOF analysis for climate data.

    Args:
        num_modes: Number of EOF modes to compute
    """

    def __init__(self, num_modes: int = 10):
        super().__init__()
        self.num_modes = num_modes

    def compute_eofs(self, data: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute Empirical Orthogonal Functions.

        Args:
            data: Climate data of shape (T, C, H, W)

        Returns:
            Tuple of (EOF patterns, PC time series, explained variance)
        """
        T, C, H, W = data.shape

        flat = data.reshape(T, C * H * W)

        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean

        cov = torch.mm(centered.t(), centered) / (T - 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eofs = eigenvectors[:, : self.num_modes].reshape(C, H, W, self.num_modes)
        eofs = eofs.permute(3, 0, 1, 2)

        pcs = torch.mm(centered, eigenvectors[:, : self.num_modes])

        explained_var = eigenvalues[: self.num_modes] / eigenvalues.sum()

        return eofs, pcs, explained_var


def compute_spatial_correlation(
    data1: Tensor,
    data2: Tensor,
) -> float:
    """Compute spatial correlation between two fields.

    Args:
        data1: First data field
        data2: Second data field

    Returns:
        Correlation coefficient
    """
    flat1 = data1.flatten()
    flat2 = data2.flatten()

    corr = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1].item()
    return corr


def detect_regime_shifts(
    time_series: Tensor,
    window_size: int = 12,
    threshold: float = 2.0,
) -> List[int]:
    """Detect regime shifts in time series.

    Args:
        time_series: Time series data
        window_size: Window for computing statistics
        threshold: Threshold in standard deviations

    Returns:
        List of indices where regime shifts occur
    """
    T = len(time_series)
    shifts = []

    rolling_mean = time_series.unfold(0, window_size, 1).mean(dim=1)
    rolling_std = time_series.unfold(0, window_size, 1).std(dim=1)

    for i in range(window_size, T):
        current = time_series[i].item()
        prev_mean = rolling_mean[i - window_size].item()
        prev_std = rolling_std[i - window_size].item()

        if abs(current - prev_mean) > threshold * prev_std:
            shifts.append(i)

    return shifts

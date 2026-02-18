"""
2D Wavelet Scattering Transform

Wavelet scattering transform for 2D signals (images) providing
translation-invariant feature extraction.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaborWavelet2D(nn.Module):
    """2D Gabor wavelet for image feature extraction."""

    def __init__(
        self,
        num_orientations: int = 8,
        num_scales: int = 4,
        sigma: float = 2.0,
    ):
        super().__init__()
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.sigma = sigma

        self.orientations = nn.Parameter(
            torch.linspace(0, np.pi, num_orientations), requires_grad=False
        )

    def forward(self, size: int, scale: float, orientation: float) -> torch.Tensor:
        """Generate 2D Gabor wavelet.

        Args:
            size: Wavelet spatial size
            scale: Scale factor
            orientation: Orientation in radians

        Returns:
            Complex wavelet of shape (size, size)
        """
        x = torch.arange(size, dtype=torch.float32)
        y = torch.arange(size, dtype=torch.float32)

        X, Y = torch.meshgrid(x, y, indexing="ij")

        X = X - size // 2
        Y = Y - size // 2

        X_rot = X * torch.cos(orientation) + Y * torch.sin(orientation)
        Y_rot = -X * torch.sin(orientation) + Y * torch.cos(orientation)

        scale_factor = scale * self.sigma

        envelope = torch.exp(-0.5 * (X_rot**2 + Y_rot**2) / (scale_factor**2))

        frequency = 1.0 / scale
        oscillation = torch.exp(1j * 2 * np.pi * frequency * X_rot / scale_factor)

        wavelet = envelope * oscillation

        return wavelet


class WaveletScattering2D(nn.Module):
    """2D Wavelet Scattering Transform.

    Provides translation-invariant feature extraction for images
    through successive wavelet convolutions and pooling.
    """

    def __init__(
        self,
        num_scales: int = 4,
        num_orientations: int = 8,
        image_size: int = 224,
        J: int = None,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.image_size = image_size
        self.J = J or num_scales

        self.wavelet = GaborWavelet2D(
            num_orientations=num_orientations,
            num_scales=num_scales,
        )

        self._init_wavelet_filters()

    def _init_wavelet_filters(self):
        """Initialize bank of wavelet filters."""
        filters = []

        for scale in range(1, self.num_scales + 1):
            for ori_idx in range(self.num_orientations):
                size = min(self.image_size // (2**scale), 32)
                size = size if size % 2 == 0 else size + 1

                orientation = float(self.wavelet.orientations[ori_idx])
                wavelet = self.wavelet(size, float(scale), orientation)

                filters.append(wavelet)

        self.register_buffer("filters", torch.stack(filters))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D wavelet scattering.

        Args:
            x: Input image of shape (batch, channels, height, width)

        Returns:
            Tuple of (order-0, order-1 scattering coefficients)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        batch_size, channels, height, width = x.shape

        x_gray = x.mean(dim=1, keepdim=True)

        S0 = F.avg_pool2d(x_gray, kernel_size=2**self.J)

        S1_coeffs = []

        for filter_idx in range(len(self.filters)):
            wavelet = self.filters[filter_idx]

            wavelet_real = wavelet.real.unsqueeze(0).unsqueeze(0)
            wavelet_imag = wavelet.imag.unsqueeze(0).unsqueeze(0)

            conv_real = F.conv2d(x_gray, wavelet_real, padding=wavelet.shape[-1] // 2)
            conv_imag = F.conv2d(x_gray, wavelet_imag, padding=wavelet.shape[-1] // 2)

            magnitude = torch.sqrt(conv_real**2 + conv_imag**2 + 1e-8)

            pooled = F.avg_pool2d(magnitude, kernel_size=2**self.J)

            S1_coeffs.append(pooled)

        S1 = torch.cat(S1_coeffs, dim=1)

        return S0, S1

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get flattened scattering features.

        Args:
            x: Input image

        Returns:
            Feature vector
        """
        S0, S1 = self.forward(x)

        S0_flat = S0.reshape(S0.shape[0], -1)
        S1_flat = S1.reshape(S1.shape[0], -1)

        return torch.cat([S0_flat, S1_flat], dim=-1)


class ScatteringResNet(nn.Module):
    """ResNet-style architecture using scattering features."""

    def __init__(
        self,
        num_classes: int = 1000,
        num_scales: int = 4,
        num_orientations: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.scatter = WaveletScattering2D(
            num_scales=num_scales,
            num_orientations=num_orientations,
        )

        num_features = num_scales * num_orientations + 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.scatter.get_features(x)
        return self.classifier(features)


class ConvScattering2D(nn.Module):
    """Convolutional scattering network combining wavelets and convnets."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        num_scales: int = 3,
        num_orientations: int = 6,
    ):
        super().__init__()

        self.scatter1 = WaveletScattering2D(
            num_scales=num_scales,
            num_orientations=num_orientations,
            J=num_scales,
        )

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(num_scales * num_orientations + 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through scattering-convnet."""
        S0, S1 = self.scatter1(x)

        x = torch.cat([S0, S1], dim=1)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.squeeze(-1).squeeze(-1)

        return self.classifier(x)


class InvariantScatteringLoss(nn.Module):
    """Loss for training scattering networks with invariance."""

    def __init__(
        self,
        invariance_weight: float = 0.1,
        num_scales: int = 4,
    ):
        super().__init__()
        self.invariance_weight = invariance_weight
        self.scatter = WaveletScattering2D(num_scales=num_scales)

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        original: torch.Tensor,
        transformed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification loss with invariance penalty.

        Args:
            output: Model predictions
            target: Ground truth labels
            original: Original images
            transformed: Transformed versions (rotated, translated, etc.)

        Returns:
            Combined loss
        """
        ce_loss = F.cross_entropy(output, target)

        orig_features = self.scatter.get_features(original)
        trans_features = self.scatter.get_features(transformed)

        invariance_loss = F.mse_loss(trans_features, orig_features.detach())

        total_loss = ce_loss + self.invariance_weight * invariance_loss

        return total_loss


class ScatteringBatchNorm(nn.Module):
    """Batch normalization for scattering coefficients."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch norm to scattering features."""
        if x.dim() == 3:
            batch_size, channels, features = x.shape
            x = x.reshape(batch_size, channels * features)
            x = self.bn(x)
            x = x.reshape(batch_size, channels, features)
        else:
            x = self.bn(x)
        return x


class WaveletAttention2D(nn.Module):
    """Attention mechanism over wavelet coefficients."""

    def __init__(
        self,
        num_scales: int = 4,
        num_orientations: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations

        total_features = num_scales * num_orientations

        self.attention = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, total_features),
            nn.Sigmoid(),
        )

    def forward(self, scattering_features: torch.Tensor) -> torch.Tensor:
        """Apply attention to scattering features.

        Args:
            scattering_features: Raw scattering coefficients

        Returns:
            Attended features
        """
        x = scattering_features.mean(dim=-1)

        weights = self.attention(x)
        weights = weights.unsqueeze(-1)

        return scattering_features * weights


class MultiscaleScattering(nn.Module):
    """Multi-scale scattering with learnable scales."""

    def __init__(
        self,
        scales: List[int] = [1, 2, 4, 8],
        num_orientations: int = 8,
    ):
        super().__init__()
        self.scales = scales
        self.num_orientations = num_orientations

        self.scatters = nn.ModuleList(
            [
                WaveletScattering2D(
                    num_scales=scale,
                    num_orientations=num_orientations,
                )
                for scale in scales
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute multi-scale scattering.

        Args:
            x: Input image

        Returns:
            List of scattering features at each scale
        """
        features = []

        for scatter in self.scatters:
            feat = scatter.get_features(x)
            features.append(feat)

        return features

    def concat_features(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate multi-scale features."""
        features = self.forward(x)
        return torch.cat(features, dim=-1)

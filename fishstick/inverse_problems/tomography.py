"""
Tomography Reconstruction Module

Implements tomography reconstruction algorithms including:
- Filtered back-projection (FBP)
- Algebraic Reconstruction Technique (ART)
- Simultaneous Iterative Reconstruction Technique (SIRT)
- Deep learning tomography
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TomographyReconstructor(nn.Module):
    """Base class for tomography reconstruction."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_angles: int = 180,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_angles = num_angles


class FilteredBackProjection(TomographyReconstructor):
    """Filtered Back-Projection (FBP) algorithm.

    Classic analytical reconstruction method for
    computed tomography.

    Solves: f(x,y) = integral of filtered projections

    Example:
        >>> fbp = FilteredBackProjection(image_size=(256, 256), num_angles=180)
        >>> reconstructed = fbp(sinograms)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_angles: int = 180,
        filter_type: str = "ram-lak",
    ):
        super().__init__(image_size, num_angles)
        self.filter_type = filter_type

    def forward(self, sinograms: torch.Tensor) -> torch.Tensor:
        """Reconstruct from sinograms using FBP.

        Args:
            sinograms: Sinogram data (batch, num_angles, num_detectors)

        Returns:
            Reconstructed images
        """
        if sinograms.dim() == 2:
            sinograms = sinograms.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = sinograms.shape[0]
        num_detectors = sinograms.shape[2]

        filtered_sinograms = self._apply_filter(sinograms)

        reconstruction = torch.zeros(
            batch_size, self.image_size[0], self.image_size[1], device=sinograms.device
        )

        angles = torch.linspace(0, np.pi, self.num_angles, device=sinograms.device)

        for b in range(batch_size):
            for i, angle in enumerate(angles):
                projection = filtered_sinograms[b, i, :]
                back_projected = self._back_project(projection, angle, num_detectors)
                reconstruction[b] += back_projected

        reconstruction = reconstruction * np.pi / self.num_angles

        if squeeze_output:
            reconstruction = reconstruction.squeeze(0)

        return torch.clamp(reconstruction, 0, reconstruction.max())

    def _apply_filter(self, sinograms: torch.Tensor) -> torch.Tensor:
        """Apply Fourier filter to projections."""
        num_detectors = sinograms.shape[2]

        filter_1d = self._create_filter(num_detectors)
        filter_1d = filter_1d.to(sinograms.device)

        filtered = torch.zeros_like(sinograms)

        for i in range(sinograms.shape[1]):
            projection = sinograms[:, i, :]

            projection_fft = torch.fft.fft(projection, dim=-1)

            filtered_fft = projection_fft * filter_1d.unsqueeze(0)

            filtered[:, i, :] = torch.fft.ifft(filtered_fft, dim=-1).real

        return filtered

    def _create_filter(self, num_detectors: int) -> torch.Tensor:
        """Create Fourier filter for FBP."""
        n = torch.arange(num_detectors)

        if self.filter_type == "ram-lak":
            filter_ = torch.zeros(num_detectors)
            filter_[0] = 0.25
            filter_[1::2] = -1 / (np.pi**2 * n[1::2] ** 2)
        elif self.filter_type == "shepp-logan":
            filter_ = torch.zeros(num_detectors)
            filter_[0] = 0.25
            n_vals = n[1::2].float()
            filter_[1::2] = (
                -1
                / (np.pi**2 * n_vals**2)
                * torch.sin(np.pi * n_vals)
                / (np.pi * n_vals)
            )
        elif self.filter_type == "cosine":
            filter_ = torch.zeros(num_detectors)
            filter_[0] = 0.25
            n_vals = n[1::2].float()
            filter_[1::2] = -1 / (np.pi**2 * n_vals**2) * torch.cos(np.pi * n_vals / 2)
        elif self.filter_type == "hamming":
            filter_ = torch.zeros(num_detectors)
            filter_[0] = 0.25
            n_vals = n[1::2].float()
            filter_[1::2] = (
                -1
                / (np.pi**2 * n_vals**2)
                * (0.54 + 0.46 * torch.cos(2 * np.pi * n_vals / num_detectors))
            )
        else:
            filter_ = torch.ones(num_detectors) * 0.5

        return filter_

    def _back_project(
        self,
        projection: torch.Tensor,
        angle: float,
        num_detectors: int,
    ) -> torch.Tensor:
        """Back-project a single projection."""
        h, w = self.image_size
        y, x = torch.meshgrid(
            torch.arange(h, device=projection.device),
            torch.arange(w, device=projection.device),
            indexing="ij",
        )

        x_centered = x - w / 2
        y_centered = y - h / 2

        detector_positions = torch.arange(num_detectors, device=projection.device)
        detector_positions = (
            (detector_positions - num_detectors / 2) * 2 * np.pi / num_detectors
        )

        rotated_x = x_centered * torch.cos(angle) + y_centered * torch.sin(angle)
        rotated_x_scaled = (rotated_x / (2 * np.pi) * num_detectors).long()

        rotated_x_scaled = torch.clamp(rotated_x_scaled, 0, num_detectors - 1)

        back_projection = projection[rotated_x_scaled]

        return back_projection


class ART(TomographyReconstructor):
    """Algebraic Reconstruction Technique.

    Iterative reconstruction method that solves
    the linear system of equations.

    Solves: Af = g

    Example:
        >>> art = ART(image_size=(256, 256), num_angles=180, num_iterations=10)
        >>> reconstructed = art(sinograms)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_angles: int = 180,
        num_iterations: int = 10,
        relaxation: float = 1.0,
    ):
        super().__init__(image_size, num_angles)
        self.num_iterations = num_iterations
        self.relaxation = relaxation

    def forward(self, sinograms: torch.Tensor) -> torch.Tensor:
        """Reconstruct using ART.

        Args:
            sinograms: Sinogram data

        Returns:
            Reconstructed image
        """
        if sinograms.dim() == 2:
            sinograms = sinograms.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        device = sinograms.device
        batch_size = sinograms.shape[0]

        image = torch.zeros(
            batch_size, self.image_size[0], self.image_size[1], device=device
        )

        angles = torch.linspace(0, np.pi, self.num_angles, device=device)

        for iteration in range(self.num_iterations):
            for i, angle in enumerate(angles):
                projections = sinograms[:, i, :]

                forward_proj = self._forward_project(image, angle)

                residual = projections - forward_proj

                correction = self._compute_correction(residual, angle)

                image = image + self.relaxation * correction

        if squeeze_output:
            image = image.squeeze(0)

        return torch.clamp(image, 0, image.max())

    def _forward_project(
        self,
        image: torch.Tensor,
        angle: float,
    ) -> torch.Tensor:
        """Forward projection at given angle."""
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        num_detectors = max(h, w)

        y, x = torch.meshgrid(
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device),
            indexing="ij",
        )

        x_centered = x - w / 2
        y_centered = y - h / 2

        rotated_x = x_centered * torch.cos(angle) + y_centered * torch.sin(angle)
        rotated_x_scaled = (rotated_x / (2 * np.pi) * num_detectors).long()
        rotated_x_scaled = torch.clamp(rotated_x_scaled, 0, num_detectors - 1)

        projections = torch.zeros(batch_size, num_detectors, device=image.device)

        for b in range(batch_size):
            projections[b] = (
                image[b, :, :]
                .reshape(-1)[rotated_x_scaled.reshape(-1)]
                .reshape(h, w)
                .sum(dim=0)
            )

        return projections

    def _compute_correction(
        self,
        residual: torch.Tensor,
        angle: float,
    ) -> torch.Tensor:
        """Compute ART correction."""
        batch_size = residual.shape[0]
        h, w = self.image_size
        num_detectors = residual.shape[1]

        correction = torch.zeros(batch_size, h, w, device=residual.device)

        y, x = torch.meshgrid(
            torch.arange(h, device=residual.device),
            torch.arange(w, device=residual.device),
            indexing="ij",
        )

        x_centered = x - w / 2
        y_centered = y - h / 2

        rotated_x = x_centered * torch.cos(angle) + y_centered * torch.sin(angle)
        rotated_x_scaled = (rotated_x / (2 * np.pi) * num_detectors).long()
        rotated_x_scaled = torch.clamp(rotated_x_scaled, 0, num_detectors - 1)

        for b in range(batch_size):
            correction[b] = residual[b, rotated_x_scaled]

        return correction / (w + 1e-10)


class SIRT(TomographyReconstructor):
    """Simultaneous Iterative Reconstruction Technique.

    Row-action method that updates all projections
    simultaneously.

    Example:
        >>> sirt = SIRT(image_size=(256, 256), num_angles=180, num_iterations=50)
        >>> reconstructed = sirt(sinograms)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_angles: int = 180,
        num_iterations: int = 50,
    ):
        super().__init__(image_size, num_angles)
        self.num_iterations = num_iterations

    def forward(self, sinograms: torch.Tensor) -> torch.Tensor:
        """Reconstruct using SIRT.

        Args:
            sinograms: Sinogram data

        Returns:
            Reconstructed image
        """
        if sinograms.dim() == 2:
            sinograms = sinograms.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        device = sinograms.device
        batch_size = sinograms.shape[0]

        image = torch.zeros(
            batch_size, self.image_size[0], self.image_size[1], device=device
        )

        angles = torch.linspace(0, np.pi, self.num_angles, device=device)

        for _ in range(self.num_iterations):
            corrections = torch.zeros_like(image)

            for angle in angles:
                forward_proj = self._forward_project(image, angle)
                residual = (
                    sinograms[:, int(angle / np.pi * (self.num_angles - 1)), :]
                    - forward_proj
                )
                correction = self._back_project(residual.unsqueeze(1), angle)
                corrections += correction

            corrections = corrections / self.num_angles
            image = image + corrections

        if squeeze_output:
            image = image.squeeze(0)

        return torch.clamp(image, 0, image.max())

    def _forward_project(
        self,
        image: torch.Tensor,
        angle: float,
    ) -> torch.Tensor:
        """Forward projection."""
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        num_detectors = max(h, w)

        y, x = torch.meshgrid(
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device),
            indexing="ij",
        )

        x_centered = x - w / 2
        y_centered = y - h / 2

        rotated_x = x_centered * torch.cos(angle) + y_centered * torch.sin(angle)
        rotated_x_scaled = (rotated_x / (2 * np.pi) * num_detectors).long()
        rotated_x_scaled = torch.clamp(rotated_x_scaled, 0, num_detectors - 1)

        projections = torch.zeros(batch_size, num_detectors, device=image.device)

        for b in range(batch_size):
            for j in range(num_detectors):
                projections[b, j] = image[b, :, :][rotated_x_scaled == j].sum()

        return projections

    def _back_project(
        self,
        projections: torch.Tensor,
        angle: float,
    ) -> torch.Tensor:
        """Back projection."""
        batch_size = projections.shape[0]
        h, w = self.image_size
        num_detectors = projections.shape[2]

        correction = torch.zeros(batch_size, h, w, device=projections.device)

        y, x = torch.meshgrid(
            torch.arange(h, device=projections.device),
            torch.arange(w, device=projections.device),
            indexing="ij",
        )

        x_centered = x - w / 2
        y_centered = y - h / 2

        rotated_x = x_centered * torch.cos(angle) + y_centered * torch.sin(angle)
        rotated_x_scaled = (rotated_x / (2 * np.pi) * num_detectors).long()
        rotated_x_scaled = torch.clamp(rotated_x_scaled, 0, num_detectors - 1)

        for b in range(batch_size):
            correction[b] = projections[b, 0, rotated_x_scaled]

        return correction


class DeepTomography(nn.Module):
    """Deep learning-based tomography reconstruction.

    U-Net based network for learning the
    reconstruction mapping.

    Example:
        >>> deep_tomo = DeepTomography(image_size=(256, 256), num_angles=180)
        >>> reconstructed = deep_tomo(sinograms)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_angles: int = 180,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_angles = num_angles

        self.encoder1 = nn.Sequential(
            nn.Conv2d(num_angles, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 2, hidden_channels, 4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, sinograms: torch.Tensor) -> torch.Tensor:
        """Reconstruct using deep learning.

        Args:
            sinograms: Sinogram data

        Returns:
            Reconstructed image
        """
        if sinograms.dim() == 2:
            sinograms = sinograms.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        x = sinograms.unsqueeze(1)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        b = self.bottleneck(e2)

        d2 = self.decoder2(b)
        d1 = self.decoder1(d2)

        output = self.output(d1)

        if squeeze_output:
            output = output.squeeze(0).squeeze(0)
        else:
            output = output.squeeze(1)

        return torch.clamp(output, 0, output.max())


class LearnedIterativeTomography(nn.Module):
    """Learned iterative tomography reconstruction.

    Unrolled optimization with learned iterations.

    Example:
        >>> learned_tomo = LearnedIterativeTomography(image_size=(256, 256), num_angles=180, num_iterations=5)
        >>> reconstructed = learned_tomo(sinograms)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_angles: int = 180,
        num_iterations: int = 5,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_angles = num_angles
        self.num_iterations = num_iterations

        self.iteration_modules = nn.ModuleList(
            [
                TomographyIterationBlock(num_angles, hidden_channels)
                for _ in range(num_iterations)
            ]
        )

    def forward(self, sinograms: torch.Tensor) -> torch.Tensor:
        """Reconstruct using learned iterative method.

        Args:
            sinograms: Sinogram data

        Returns:
            Reconstructed image
        """
        if sinograms.dim() == 2:
            sinograms = sinograms.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        image = torch.zeros(
            sinograms.shape[0],
            1,
            self.image_size[0],
            self.image_size[1],
            device=sinograms.device,
        )

        for module in self.iteration_modules:
            image = module(image, sinograms)

        if squeeze_output:
            image = image.squeeze(0).squeeze(0)
        else:
            image = image.squeeze(1)

        return torch.clamp(image, 0, image.max())


class TomographyIterationBlock(nn.Module):
    """Single iteration block for learned tomography."""

    def __init__(self, num_angles: int, hidden_channels: int):
        super().__init__()

        self.denoisier = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, image: torch.Tensor, sinograms: torch.Tensor) -> torch.Tensor:
        """Apply one iteration."""
        denoised = self.denoisier(image)
        return image + denoised


class MLEM(TomographyReconstructor):
    """Maximum Likelihood Expectation Maximization.

    Statistical reconstruction method for PET/CT.

    Example:
        >>> mlem = MLEM(image_size=(256, 256), num_angles=180, num_iterations=100)
        >>> reconstructed = mlem(sinograms)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_angles: int = 180,
        num_iterations: int = 100,
    ):
        super().__init__(image_size, num_angles)
        self.num_iterations = num_iterations

    def forward(self, sinograms: torch.Tensor) -> torch.Tensor:
        """Reconstruct using MLEM.

        Args:
            sinograms: Sinogram data

        Returns:
            Reconstructed image
        """
        if sinograms.dim() == 2:
            sinograms = sinograms.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        device = sinograms.device
        batch_size = sinograms.shape[0]

        image = torch.ones(
            batch_size, self.image_size[0], self.image_size[1], device=device
        )

        angles = torch.linspace(0, np.pi, self.num_angles, device=device)

        for _ in range(self.num_iterations):
            expected = torch.zeros_like(sinograms)

            for angle in angles:
                forward_proj = self._forward_project(image, angle)
                expected += forward_proj

            ratio = sinograms / (expected + 1e-10)

            back_proj = torch.zeros_like(image)
            for angle in angles:
                back_proj += self._back_project(
                    ratio[:, int(angle / np.pi * (self.num_angles - 1)), :].unsqueeze(
                        1
                    ),
                    angle,
                )

            image = image * (back_proj / self.num_angles + 1e-10)

        if squeeze_output:
            image = image.squeeze(0)

        return torch.clamp(image, 0, image.max())

    def _forward_project(
        self,
        image: torch.Tensor,
        angle: float,
    ) -> torch.Tensor:
        """Forward projection."""
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        num_detectors = max(h, w)

        y, x = torch.meshgrid(
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device),
            indexing="ij",
        )

        x_centered = x - w / 2
        y_centered = y - h / 2

        rotated_x = x_centered * torch.cos(angle) + y_centered * torch.sin(angle)
        rotated_x_scaled = (rotated_x / (2 * np.pi) * num_detectors).long()
        rotated_x_scaled = torch.clamp(rotated_x_scaled, 0, num_detectors - 1)

        projections = torch.zeros(batch_size, num_detectors, device=image.device)

        for b in range(batch_size):
            for j in range(num_detectors):
                projections[b, j] = image[b, :, :][rotated_x_scaled == j].sum()

        return projections

    def _back_project(
        self,
        projections: torch.Tensor,
        angle: float,
    ) -> torch.Tensor:
        """Back projection."""
        batch_size = projections.shape[0]
        h, w = self.image_size
        num_detectors = projections.shape[2]

        back_proj = torch.zeros(batch_size, h, w, device=projections.device)

        y, x = torch.meshgrid(
            torch.arange(h, device=projections.device),
            torch.arange(w, device=projections.device),
            indexing="ij",
        )

        x_centered = x - w / 2
        y_centered = y - h / 2

        rotated_x = x_centered * torch.cos(angle) + y_centered * torch.sin(angle)
        rotated_x_scaled = (rotated_x / (2 * np.pi) * num_detectors).long()
        rotated_x_scaled = torch.clamp(rotated_x_scaled, 0, num_detectors - 1)

        for b in range(batch_size):
            back_proj[b] = projections[b, 0, rotated_x_scaled]

        return back_proj


def create_sinogram(
    image: torch.Tensor,
    num_angles: int = 180,
) -> torch.Tensor:
    """Generate sinogram from image using forward projection.

    Args:
        image: Input image
        num_angles: Number of projection angles

    Returns:
        Sinogram
    """
    h, w = image.shape[-2:]
    num_detectors = max(h, w)

    angles = torch.linspace(0, np.pi, num_angles)
    sinogram = torch.zeros(
        image.shape[0] if image.dim() > 2 else 1, num_angles, num_detectors
    )

    fbp = FilteredBackProjection(image_size=(h, w), num_angles=num_angles)

    for i, angle in enumerate(angles):
        projection = fbp._forward_project(image, angle)
        if image.dim() > 2:
            sinogram[:, i, :] = projection
        else:
            sinogram[0, i, :] = projection

    return sinogram

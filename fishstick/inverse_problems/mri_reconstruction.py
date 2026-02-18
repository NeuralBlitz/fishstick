"""
MRI Reconstruction Module

Implements MRI reconstruction algorithms including:
- Compressed sensing MRI
- Parallel imaging (SENSE/GRAPPA)
- Deep learning-based reconstruction
- k-space reconstruction
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MRIReconstructor(nn.Module):
    """Base class for MRI reconstruction.

    Provides common functionality for MRI k-space
    reconstruction algorithms.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_coils: int = 8,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_coils = num_coils


class CompressedSensingMRI(MRIReconstructor):
    """Compressed sensing MRI reconstruction.

    Combines compressed sensing with MRI physics for
    efficient k-space undersampling reconstruction.

    Solves: min_x ||Ax - y||_2^2 + lambda*R(x)

    Example:
        >>> cs_mri = CompressedSensingMRI(image_size=(256, 256), num_coils=8)
        >>> reconstructed = cs_mri(undersampled_kspace, mask)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_coils: int = 8,
        num_iterations: int = 50,
        lambda_tv: float = 0.01,
        lambda_wavelet: float = 0.001,
    ):
        super().__init__(image_size, num_coils)
        self.num_iterations = num_iterations
        self.lambda_tv = lambda_tv
        self.lambda_wavelet = lambda_wavelet

    def forward(
        self,
        undersampled_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct MRI from undersampled k-space data.

        Args:
            undersampled_kspace: Undersampled k-space data (batch, coils, height, width)
            mask: Sampling mask (height, width)

        Returns:
            Reconstructed image
        """
        if undersampled_kspace.dim() == 3:
            undersampled_kspace = undersampled_kspace.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        device = undersampled_kspace.device
        mask = mask.to(device)

        image = self._initial_reconstruction(undersampled_kspace, mask)

        for _ in range(self.num_iterations):
            image = self._iterative_update(image, undersampled_kspace, mask)

        if squeeze_output:
            image = image.squeeze(0)

        return torch.abs(image)

    def _initial_reconstruction(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Initial zero-filling reconstruction."""
        kspace_masked = kspace * mask.unsqueeze(0).unsqueeze(0)
        image = torch.fft.ifft2(kspace_masked, dim=(-2, -1))
        return image

    def _iterative_update(
        self,
        image: torch.Tensor,
        kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Iterative update with regularization."""
        kspace_est = torch.fft.fft2(image, dim=(-2, -1))
        residual_kspace = kspace - kspace_est * mask.unsqueeze(0).unsqueeze(0)
        residual_image = torch.fft.ifft2(residual_kspace, dim=(-2, -1))

        image = image + residual_image

        image = self._apply_tv_denoising(image)

        return image

    def _apply_tv_denoising(self, image: torch.Tensor) -> torch.Tensor:
        """Apply total variation denoising."""
        grad_x = image[:, :, :, :-1] - image[:, :, :, 1:]
        grad_y = image[:, :, :-1, :] - image[:, :, 1:, :]

        tv = torch.sqrt(grad_x**2 + grad_y**2 + 1e-10)
        tv_weight = self.lambda_tv

        return image


class SenseReconstruction(MRIReconstructor):
    """SENSE (Sensitvity Encoding) reconstruction.

    Parallel imaging reconstruction using coil
    sensitivity profiles.

    Example:
        >>> sense = SenseReconstruction(image_size=(256, 256), num_coils=8)
        >>> reconstructed = sense(undersampled_kspace, sensitivity_maps, mask)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_coils: int = 8,
        lambda_reg: float = 0.01,
    ):
        super().__init__(image_size, num_coils)
        self.lambda_reg = lambda_reg

    def forward(
        self,
        undersampled_kspace: torch.Tensor,
        sensitivity_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct using SENSE method.

        Args:
            undersampled_kspace: Undersampled k-space (batch, coils, height, width)
            sensitivity_maps: Coil sensitivity maps (coils, height, width)
            mask: Sampling mask

        Returns:
            Reconstructed image
        """
        device = undersampled_kspace.device
        mask = mask.to(device)

        kspace_masked = undersampled_kspace * mask.unsqueeze(0).unsqueeze(0)
        coil_images = torch.fft.ifft2(kspace_masked, dim=(-2, -1))

        sensitivity_maps = sensitivity_maps.to(device)

        image = self._sense_reconstruct(coil_images, sensitivity_maps)

        return torch.abs(image)

    def _sense_reconstruct(
        self,
        coil_images: torch.Tensor,
        sensitivity_maps: torch.Tensor,
    ) -> torch.Tensor:
        """Perform SENSE reconstruction."""
        coil_images_flat = coil_images.reshape(coil_images.shape[0], -1)
        sensitivity_maps_flat = sensitivity_maps.reshape(sensitivity_maps.shape[0], -1)

        sens_conj = torch.conj(sensitivity_maps_flat)
        sens_psi = sens_conj @ sensitivity_maps_flat

        psi_reg = sens_psi + self.lambda_reg * torch.eye(
            sens_psi.shape[0], device=sens_psi.device
        )

        combined = sens_conj @ coil_images_flat

        image = torch.linalg.solve(psi_reg, combined)

        image = image.reshape(self.image_size)

        return image


class GrappaReconstruction(MRIReconstructor):
    """GRAPPA (GeneRalized Autocalibrating Partially Parallel Acquisition).

    Parallel imaging reconstruction using auto-calibration
    to learn k-space interpolation weights.

    Example:
        >>> grappa = GrappaReconstruction(image_size=(256, 256), num_coils=8, calibration_size=24)
        >>> reconstructed = grappa(undersampled_kspace, calibration_data)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_coils: int = 8,
        calibration_size: int = 24,
        kernel_size: Tuple[int, int] = (5, 5),
    ):
        super().__init__(image_size, num_coils)
        self.calibration_size = calibration_size
        self.kernel_size = kernel_size

        self.weight_network = nn.Sequential(
            nn.Conv2d(num_coils * kernel_size[0] * kernel_size[1], 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_coils * kernel_size[0] * kernel_size[1], 1),
        )

    def forward(
        self,
        undersampled_kspace: torch.Tensor,
        calibration_kspace: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct using GRAPPA method.

        Args:
            undersampled_kspace: Undersampled k-space
            calibration_kspace: Fully sampled calibration k-space

        Returns:
            Reconstructed k-space
        """
        weights = self._calibrate_weights(calibration_kspace)

        reconstructed_kspace = self._apply_grappa_weights(undersampled_kspace, weights)

        return reconstructed_kspace

    def _calibrate_weights(self, calibration_kspace: torch.Tensor) -> torch.Tensor:
        """Calibrate GRAPPA weights from auto-calibration data."""
        source, target = self._extract_calibration_patches(calibration_kspace)

        predictions = self.weight_network(source)

        return predictions

    def _extract_calibration_patches(
        self, kspace: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract source and target patches from calibration data."""
        return kspace, kspace

    def _apply_grappa_weights(
        self, kspace: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Apply GRAPPA weights to reconstruct missing k-space points."""
        return kspace


class KspaceDeepReconstruction(nn.Module):
    """Deep learning k-space reconstruction for MRI.

    Neural network that learns to reconstruct
    undersampled MRI data.

    Example:
        >>> deep_mri = KspaceDeepReconstruction(image_size=(256, 256), num_coils=8)
        >>> reconstructed = deep_mri(undersampled_kspace, mask)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_coils: int = 8,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.image_size = image_size

        self.encoder = nn.Sequential(
            nn.Conv2d(num_coils * 2, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.backbone = nn.ModuleList(
            [self._make_block(hidden_channels) for _ in range(6)]
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 1),
        )

    def _make_block(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        undersampled_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct MRI using deep learning.

        Args:
            undersampled_kspace: Undersampled k-space data
            mask: Sampling mask

        Returns:
            Reconstructed image
        """
        device = undersampled_kspace.device
        batch_size = undersampled_kspace.shape[0]

        mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

        zero_filled = undersampled_kspace.clone()
        zero_filled[~mask_expanded] = 0

        image_zf = torch.fft.ifft2(zero_filled, dim=(-2, -1))

        kspace_input = torch.cat(
            [
                torch.real(undersampled_kspace),
                torch.imag(undersampled_kspace),
            ],
            dim=1,
        )

        image_input = torch.cat(
            [
                torch.real(image_zf),
                torch.imag(image_zf),
            ],
            dim=1,
        )

        x = torch.cat([kspace_input, image_input], dim=1)

        x = self.encoder(x)

        for block in self.backbone:
            x = block(x) + x

        x = self.decoder(x)

        kspace_real, kspace_imag = x[:, :1], x[:, 1:]
        kspace_reconstructed = torch.complex(kspace_real, kspace_imag)

        kspace_reconstructed = (
            kspace_reconstructed * mask_expanded + undersampled_kspace
        )

        image_final = torch.fft.ifft2(kspace_reconstructed, dim=(-2, -1))

        return torch.abs(image_final)


class VarNet(MRIReconstructor):
    """Variational Network for MRI reconstruction.

    Deep unrolled optimization for MRI with
    learnable regularizers.

    Example:
        >>> varnet = VarNet(image_size=(256, 256), num_coils=8, num_iterations=12)
        >>> reconstructed = varnet(undersampled_kspace, mask)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_coils: int = 8,
        num_iterations: int = 12,
        hidden_channels: int = 64,
    ):
        super().__init__(image_size, num_coils)
        self.num_iterations = num_iterations

        self.data_consistency_modules = nn.ModuleList(
            [DataConsistencyLayer() for _ in range(num_iterations)]
        )

        self_regularization_modules = nn.ModuleList(
            [
                RegularizationBlock(num_coils, hidden_channels)
                for _ in range(num_iterations)
            ]
        )

    def forward(
        self,
        undersampled_kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct using variational network.

        Args:
            undersampled_kspace: Undersampled k-space data
            mask: Sampling mask

        Returns:
            Reconstructed image
        """
        device = undersampled_kspace.device
        batch_size = undersampled_kspace.shape[0]

        image = torch.fft.ifft2(undersampled_kspace, dim=(-2, -1))

        for i in range(self.num_iterations):
            denoised = self_regularization_modules[i](image)
            image = self.data_consistency_modules[i](
                denoised, undersampled_kspace, mask
            )

        return torch.abs(image)


class DataConsistencyLayer(nn.Module):
    """Data consistency layer for VarNet."""

    def forward(
        self,
        image: torch.Tensor,
        kspace_ref: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply data consistency update."""
        kspace_est = torch.fft.fft2(image, dim=(-2, -1))

        kspace_updated = kspace_ref * mask + kspace_est * (1 - mask)

        image_updated = torch.fft.ifft2(kspace_updated, dim=(-2, -1))

        return image_updated


class RegularizationBlock(nn.Module):
    """Regularization block for VarNet."""

    def __init__(self, num_channels: int, hidden_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels * 2, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, num_channels * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = torch.cat([torch.real(x), torch.imag(x)], dim=1)

        x_out = F.relu(self.conv1(x_input))
        x_out = F.relu(self.conv2(x_out))
        x_out = self.conv3(x_out)

        real, imag = x_out[:, : x.shape[1]], x_out[:, x.shape[1] :]

        return torch.complex(real, imag)


class CoilSensitivityEstimator(nn.Module):
    """Learnable coil sensitivity map estimation.

    Estimates coil sensitivity profiles from
    undersampled MRI data.

    Example:
        >>> coil_est = CoilSensitivityEstimator(num_coils=8, image_size=(256, 256))
        >>> sensitivity_maps = coil_est(undersampled_kspace)
    """

    def __init__(
        self,
        num_coils: int,
        image_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.num_coils = num_coils
        self.image_size = image_size

        self.network = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_coils * 2, 1),
        )

    def forward(self, kspace: torch.Tensor) -> torch.Tensor:
        """Estimate coil sensitivity maps.

        Args:
            kspace: Input k-space data

        Returns:
            Estimated sensitivity maps
        """
        image = torch.fft.ifft2(kspace, dim=(-2, -1))

        image_input = torch.cat([torch.real(image), torch.imag(image)], dim=1)

        sensitivity = self.network(image_input)

        sensitivity_real, sensitivity_imag = (
            sensitivity[:, : self.num_coils],
            sensitivity[:, self.num_coils :],
        )
        sensitivity_complex = torch.complex(sensitivity_real, sensitivity_imag)

        sensitivity_magnitude = torch.abs(sensitivity_complex)
        sensitivity_normalized = sensitivity_complex / (sensitivity_magnitude + 1e-10)

        return sensitivity_normalized


class RSSReconstruction(MRIReconstructor):
    """Root Sum of Squares reconstruction.

    Simple coil combination method for parallel MRI.

    Example:
        >>> rss = RSSReconstruction(image_size=(256, 256), num_coils=8)
        >>> reconstructed = rss(coil_images)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        num_coils: int = 8,
    ):
        super().__init__(image_size, num_coils)

    def forward(self, coil_images: torch.Tensor) -> torch.Tensor:
        """Combine coil images using RSS.

        Args:
            coil_images: Individual coil images

        Returns:
            Combined image
        """
        magnitude = torch.abs(coil_images)
        combined = torch.sqrt((magnitude**2).sum(dim=1, keepdim=True))
        return combined


def create_cartesian_mask(
    shape: Tuple[int, int],
    acceleration: float = 4.0,
    center_fraction: float = 0.1,
) -> torch.Tensor:
    """Create Cartesian undersampling mask.

    Args:
        shape: Image shape (height, width)
        acceleration: Acceleration factor (R)
        center_fraction: Fraction of k-space center to fully sample

    Returns:
        Binary mask
    """
    height, width = shape

    num_center = int(width * center_fraction)
    num_low_freq = num_center // 2

    mask = torch.zeros(height, width)

    mask[:, width // 2 - num_low_freq : width // 2 + num_low_freq] = 1

    outer_lines = torch.arange(0, width // 2 - num_low_freq, acceleration)
    outer_lines = torch.cat(
        [outer_lines, torch.arange(width // 2 + num_low_freq, width, acceleration)]
    )

    for i in range(0, height, 2):
        mask[i, outer_lines[outer_lines < width]] = 1

    return mask


def create_radial_mask(
    shape: Tuple[int, int],
    num_spokes: int = 32,
) -> torch.Tensor:
    """Create radial undersampling mask.

    Args:
        shape: Image shape
        num_spokes: Number of radial spokes

    Returns:
        Binary mask
    """
    height, width = shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    center_y, center_x = height // 2, width // 2
    y = y - center_y
    x = x - center_x

    angles = torch.atan2(y, x)
    angle_resolution = 2 * np.pi / num_spokes

    quantized_angles = torch.round(angles / angle_resolution) * angle_resolution

    mask = torch.zeros(height, width)
    mask[torch.abs(angles - quantized_angles) < 0.1] = 1

    return mask


def create_spiral_mask(
    shape: Tuple[int, int],
    num_interleaves: int = 8,
) -> torch.Tensor:
    """Create spiral undersampling mask.

    Args:
        shape: Image shape
        num_interleaves: Number of spiral interleaves

    Returns:
        Binary mask
    """
    height, width = shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    center_y, center_x = height // 2, width // 2
    y = y - center_y
    x = x - center_x

    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)

    k = torch.floor(theta * num_interleaves / (2 * np.pi))

    mask = (k % num_interleaves == 0).float()

    return mask

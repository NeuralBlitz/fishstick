"""
Synthetic Medical Image Generation

Generate synthetic CT and MRI volumes with realistic anatomy and pathologies
for training and testing purposes.
"""

from typing import Optional, Tuple, Dict, Any, List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def generate_synthetic_ct(
    shape: Tuple[int, int, int] = (128, 128, 128),
    body_range: Tuple[int, int] = (40, 60),
    hu_air: int = -1000,
    hu_water: int = 0,
    hu_bone: int = 700,
    add_noise: bool = True,
    noise_std: float = 10.0,
) -> Tensor:
    """Generate synthetic CT volume with approximate body structure.

    Args:
        shape: Volume shape (D, H, W)
        body_range: Range for body radius as percentage
        hu_air: HU value for air
        hu_water: HU value for soft tissue
        hu_bone: HU value for bone
        add_noise: Whether to add noise
        noise_std: Noise standard deviation

    Returns:
        Synthetic CT volume
    """
    d, h, w = shape

    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2

    radius = min(h, w) * random.uniform(*body_range) / 100

    body_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius**2

    volume = np.full((d, h, w), hu_air, dtype=np.float32)

    for z in range(d):
        slice_2d = np.where(body_mask, hu_water, hu_air)

        bone_y, bone_x = (
            center_y + random.randint(-radius // 3, radius // 3),
            center_x + random.randint(-radius // 3, radius // 3),
        )
        bone_radius = radius // 5

        bone_mask = ((x - bone_x) ** 2 + (y - bone_y) ** 2) <= bone_radius**2

        slice_2d = np.where(bone_mask, hu_bone, slice_2d)

        volume[z] = slice_2d

    if add_noise:
        noise = np.random.normal(0, noise_std, volume.shape)
        volume = volume + noise.astype(np.float32)

    return torch.from_numpy(volume)


def generate_synthetic_mri(
    shape: Tuple[int, int, int] = (128, 128, 128),
    intensity_range: Tuple[float, float] = (0.0, 1.0),
    add_bias_field: bool = True,
    add_noise: bool = True,
    noise_std: float = 0.02,
) -> Tensor:
    """Generate synthetic MRI volume.

    Args:
        shape: Volume shape (D, H, W)
        intensity_range: Range for tissue intensity
        add_bias_field: Whether to add bias field artifact
        add_noise: Whether to add Gaussian noise
        noise_std: Noise standard deviation

    Returns:
        Synthetic MRI volume
    """
    d, h, w = shape

    volume = (
        torch.rand(d, h, w) * (intensity_range[1] - intensity_range[0])
        + intensity_range[0]
    )

    if add_bias_field:
        bias = generate_bias_field(shape)
        volume = volume * bias

    if add_noise:
        noise = torch.randn_like(volume) * noise_std
        volume = volume + noise

    volume = torch.clamp(volume, 0, 1)

    return volume


def generate_bias_field(shape: Tuple[int, ...], sigma: float = 0.5) -> Tensor:
    """Generate smooth bias field artifact.

    Args:
        shape: Volume shape
        sigma: Smoothness parameter

    Returns:
        Bias field
    """
    from scipy.ndimage import gaussian_filter

    bias = np.random.randn(*shape)
    bias = gaussian_filter(bias, sigma)
    bias = np.exp(bias)

    return torch.from_numpy(bias.astype(np.float32))


def add_synthetic_lesion(
    volume: Tensor,
    lesion_type: str = "tumor",
    num_lesions: int = 1,
    size_range: Tuple[int, int] = (5, 15),
    intensity_shift: float = 0.3,
) -> Tensor:
    """Add synthetic lesions to volume.

    Args:
        volume: Input volume
        lesion_type: Type of lesion (tumor, hemorrhage, stroke)
        num_lesions: Number of lesions to add
        size_range: Size range for lesions
        intensity_shift: Intensity change relative to background

    Returns:
        Volume with lesions
    """
    result = volume.clone()

    d, h, w = volume.shape[-3:]

    for _ in range(num_lesions):
        z = random.randint(10, d - 10)
        y = random.randint(10, h - 10)
        x = random.randint(10, w - 10)

        size = random.randint(*size_range)

        coords = torch.meshgrid(
            torch.arange(d, device=volume.device),
            torch.arange(h, device=volume.device),
            torch.arange(w, device=volume.device),
            indexing="ij",
        )

        distances = (
            (coords[0] - z) ** 2 + (coords[1] - y) ** 2 + (coords[2] - x) ** 2
        ) ** 0.5

        mask = distances <= size

        if lesion_type == "tumor":
            intensity = -intensity_shift
        elif lesion_type == "hemorrhage":
            intensity = intensity_shift * 0.8
        elif lesion_type == "stroke":
            intensity = -intensity_shift * 0.5
        else:
            intensity = intensity_shift

        result = torch.where(mask, result + intensity, result)

    return result


class SyntheticVolumeGenerator(nn.Module):
    """Generate synthetic medical volumes with configurable properties.

    Example:
        >>> generator = SyntheticVolumeGenerator(modality='ct', shape=(128, 128, 128))
        >>> volume, mask = generator(return_mask=True)
    """

    def __init__(
        self,
        modality: str = "ct",
        shape: Tuple[int, int, int] = (128, 128, 128),
        num_classes: int = 3,
    ):
        super().__init__()
        self.modality = modality.lower()
        self.shape = shape
        self.num_classes = num_classes

    def forward(self, return_mask: bool = False) -> Tensor:
        """Generate synthetic volume.

        Args:
            return_mask: Whether to also return segmentation mask

        Returns:
            Synthetic volume, optionally with mask
        """
        if self.modality == "ct":
            volume = generate_synthetic_ct(
                self.shape,
                add_noise=True,
                noise_std=15.0,
            )
        elif self.modality == "mri":
            volume = generate_synthetic_mri(
                self.shape,
                add_bias_field=True,
                add_noise=True,
            )
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

        volume = volume.unsqueeze(0)

        if return_mask:
            mask = self._generate_mask()
            return volume, mask

        return volume

    def _generate_mask(self) -> Tensor:
        d, h, w = self.shape

        mask = torch.zeros(1, self.num_classes, d, h, w)

        for c in range(1, self.num_classes):
            num_regions = random.randint(1, 3)

            for _ in range(num_regions):
                z = random.randint(10, d - 10)
                y = random.randint(10, h - 10)
                x = random.randint(10, w - 10)

                size = random.randint(5, 15)

                coords = torch.meshgrid(
                    torch.arange(d),
                    torch.arange(h),
                    torch.arange(w),
                    indexing="ij",
                )

                distances = (
                    (coords[0] - z) ** 2 + (coords[1] - y) ** 2 + (coords[2] - x) ** 2
                ) ** 0.5

                region_mask = distances <= size

                mask[0, c] = torch.where(region_mask, 1.0, mask[0, c])

        return mask


class OrganSegmentationGenerator(nn.Module):
    """Generate synthetic organ segmentation data."""

    def __init__(
        self,
        shape: Tuple[int, int, int] = (128, 128, 128),
        organ_classes: Optional[List[str]] = None,
    ):
        super().__init__()
        self.shape = shape

        if organ_classes is None:
            organ_classes = ["background", "liver", "kidney", "spleen"]

        self.organ_classes = organ_classes

    def forward(self) -> Tensor:
        mask = torch.zeros(1, len(self.organ_classes), *self.shape)

        d, h, w = self.shape

        for c, organ in enumerate(self.organ_classes):
            if organ == "background":
                continue

            num_blobs = random.randint(1, 3)

            for _ in range(num_blobs):
                z = random.randint(d // 4, 3 * d // 4)
                y = random.randint(h // 4, 3 * h // 4)
                x = random.randint(w // 4, 3 * w // 4)

                size = random.randint(d // 8, d // 4)

                coords = torch.meshgrid(
                    torch.arange(d),
                    torch.arange(h),
                    torch.arange(w),
                    indexing="ij",
                )

                distances = (
                    (coords[0] - z) ** 2 + (coords[1] - y) ** 2 + (coords[2] - x) ** 2
                ) ** 0.5

                blob = distances <= size
                mask[0, c] = torch.where(blob, 1.0, mask[0, c])

        return mask

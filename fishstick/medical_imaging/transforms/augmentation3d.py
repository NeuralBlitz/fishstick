"""
3D Volume Augmentation Transforms

Elastic deformation, random flipping, rotation, and intensity
augmentation for volumetric medical images.
"""

from typing import Optional, Tuple, Union, List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class RandomVolumeFlip(nn.Module):
    """Random flip for 3D volumes.

    Flips volume along specified axes with given probability.
    """

    def __init__(
        self,
        axes: List[int] = [-1, -2, -3],
        p: float = 0.5,
    ):
        super().__init__()
        self.axes = axes
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        for ax in self.axes:
            if random.random() < self.p:
                x = torch.flip(x, dims=[ax])
        return x


class RandomVolumeRotate(nn.Module):
    """Random rotation for 3D volumes.

    Applies random rotation around specified axes.
    """

    def __init__(
        self,
        angle_range: Tuple[float, float] = (-15, 15),
        axes: List[int] = [-1, -2],
        p: float = 0.5,
        mode: str = "bilinear",
    ):
        super().__init__()
        self.angle_range = angle_range
        self.axes = axes
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        if x.ndim == 4:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        angle = random.uniform(*self.angle_range)
        angle_rad = np.deg2rad(angle)

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        theta = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
            dtype=x.dtype,
            device=x.device,
        )
        theta = theta.unsqueeze(0).repeat(x.size(0), 1, 1)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        rotated = F.grid_sample(x, grid, mode=self.mode, align_corners=False)

        if squeeze:
            rotated = rotated.squeeze(0)

        return rotated


class RandomVolumeElasticDeform(nn.Module):
    """Elastic deformation for 3D volumes.

    Applies random elastic deformation using dense displacement field.
    """

    def __init__(
        self,
        alpha: float = 20.0,
        sigma: float = 5.0,
        p: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        shape = x.shape[-3:]

        dx = self._generate_displacement_field(shape, self.alpha, self.sigma)
        dy = self._generate_displacement_field(shape, self.alpha, self.sigma)
        dz = self._generate_displacement_field(shape, self.alpha, self.sigma)

        if x.is_cuda:
            dx = dx.to(x.device)
            dy = dy.to(x.device)
            dz = dz.to(x.device)

        displacement = torch.stack([dx, dy, dz], dim=0)

        grid = self._make_grid(shape, x.device, x.dtype)

        new_grid = grid + displacement

        new_grid = new_grid.permute(1, 2, 3, 0).unsqueeze(0)

        x = x.unsqueeze(0) if x.ndim == 4 else x

        deformed = F.grid_sample(
            x,
            new_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        if deformed.size(0) == 1 and x.ndim == 4:
            deformed = deformed.squeeze(0)

        return deformed.squeeze(0) if x.ndim == 4 else deformed.squeeze(0)

    def _generate_displacement_field(
        self,
        shape: Tuple[int, int, int],
        alpha: float,
        sigma: float,
    ) -> Tensor:
        from scipy.ndimage import gaussian_filter

        u = np.random.randn(*shape) * alpha
        v = np.random.randn(*shape) * alpha
        w = np.random.randn(*shape) * alpha

        u = gaussian_filter(u, sigma)
        v = gaussian_filter(v, sigma)
        w = gaussian_filter(w, sigma)

        return torch.from_numpy(np.stack([u, v, w], axis=0)).float()

    def _make_grid(
        self,
        shape: Tuple[int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        d, h, w = shape

        x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        z = torch.linspace(-1, 1, d, device=device, dtype=dtype)

        grid = torch.stack(
            torch.meshgrid(x, y, z, indexing="xy"),
            dim=0,
        )

        return grid


class RandomIntensityScale(nn.Module):
    """Random intensity scaling for volumes.

    Multiplies intensities by a random scale factor.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.5,
    ):
        super().__init__()
        self.scale_range = scale_range
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        scale = random.uniform(*self.scale_range)
        return x * scale


class RandomIntensityShift(nn.Module):
    """Random intensity shift for volumes.

    Adds random offset to intensities.
    """

    def __init__(
        self,
        shift_range: Tuple[float, float] = (-0.1, 0.1),
        p: float = 0.5,
    ):
        super().__init__()
        self.shift_range = shift_range
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        shift = random.uniform(*self.shift_range)
        return x + shift


class RandomGaussianNoise(nn.Module):
    """Add Gaussian noise to volumes."""

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.01,
        p: float = 0.5,
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


class RandomGaussianBlur(nn.Module):
    """Apply Gaussian blur to volumes."""

    def __init__(
        self,
        sigma_range: Tuple[float, float] = (0.5, 2.0),
        p: float = 0.5,
    ):
        super().__init__()
        self.sigma_range = sigma_range
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        sigma = random.uniform(*self.sigma_range)

        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = self._gaussian_kernel(kernel_size, sigma, x.device, x.dtype)

        x = x.unsqueeze(0) if x.ndim == 4 else x

        padding = kernel_size // 2
        x = F.conv3d(x, kernel, padding=padding)

        return x.squeeze(0) if x.ndim == 4 else x

    def _gaussian_kernel(
        self,
        kernel_size: int,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        coords = (
            torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        )
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()

        kernel = g[:, None, None] * g[None, :, None] * g[None, None, :]

        kernel = kernel.unsqueeze(0)

        return kernel


class RandomGammaTransform(nn.Module):
    """Random gamma correction for volumes."""

    def __init__(
        self,
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
    ):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        gamma = random.uniform(*self.gamma_range)

        x_min = x.min()
        x_max = x.max()

        x_norm = (x - x_min) / (x_max - x_min + 1e-8)

        x_gamma = torch.pow(x_norm + 1e-8, gamma)

        return x_gamma * (x_max - x_min) + x_min


class VolumeAugmentationPipeline(nn.Module):
    """Pipeline for composing multiple volume augmentations."""

    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def add_transform(self, transform: nn.Module) -> None:
        self.transforms.append(transform)

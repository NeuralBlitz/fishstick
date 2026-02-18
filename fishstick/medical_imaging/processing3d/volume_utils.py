"""
Volume Processing Utilities

Resampling, cropping, padding, and statistics computation for 3D volumes.
"""

from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


@dataclass
class VolumeStatistics:
    """Statistics for a 3D volume."""

    mean: float
    std: float
    min: float
    max: float
    median: float
    percentiles: Dict[str, float]
    histogram: Optional[np.ndarray] = None
    histogram_bins: Optional[np.ndarray] = None


class VolumeResampler(nn.Module):
    """Resample 3D volumes to new spacing.

    Supports both isotropic and anisotropic resampling with
    various interpolation modes.
    """

    def __init__(
        self,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        target_shape: Optional[Tuple[int, int, int]] = None,
        order: int = 1,
        mode: str = "constant",
    ):
        super().__init__()
        self.target_spacing = target_spacing
        self.target_shape = target_shape
        self.order = order
        self.mode = mode

    def forward(
        self,
        volume: Tensor,
        current_spacing: Optional[Tuple[float, float, float]] = None,
    ) -> Tensor:
        """Resample volume.

        Args:
            volume: Input volume (C, D, H, W) or (D, H, W)
            current_spacing: Current voxel spacing

        Returns:
            Resampled volume
        """
        if self.target_spacing and current_spacing:
            return self._resample_spacing(volume, current_spacing)
        elif self.target_shape:
            return self._resample_shape(volume)
        return volume

    def _resample_spacing(
        self,
        volume: Tensor,
        current_spacing: Tuple[float, float, float],
    ) -> Tensor:
        scale_factors = [
            cs / ts for cs, ts in zip(current_spacing, self.target_spacing)
        ]

        if volume.ndim == 4:
            scale_factors = [1.0] + scale_factors
        else:
            scale_factors = scale_factors

        new_shape = [
            int(volume.shape[i] * scale_factors[i]) for i in range(volume.ndim)
        ]

        mode = "bilinear" if self.order == 1 else "nearest"

        if volume.ndim == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif volume.ndim == 4:
            volume = volume.unsqueeze(0)

        resized = F.interpolate(
            volume,
            size=new_shape[2:],
            mode=mode,
            align_corners=False if self.order == 1 else None,
        )

        return resized.squeeze(0) if volume.ndim == 4 else resized.squeeze(0).squeeze(0)

    def _resample_shape(self, volume: Tensor) -> Tensor:
        if volume.ndim == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif volume.ndim == 4:
            volume = volume.unsqueeze(0)

        mode = "trilinear" if volume.ndim == 5 else "bilinear"

        resized = F.interpolate(
            volume,
            size=self.target_shape,
            mode=mode,
            align_corners=False,
        )

        if volume.ndim == 4:
            return resized.squeeze(0)
        return resized.squeeze(0).squeeze(0)


class CropAndPad(nn.Module):
    """Crop and pad 3D volumes to target shape.

    Crops or pads volumes to achieve a target shape with
    configurable padding mode.
    """

    def __init__(
        self,
        target_shape: Tuple[int, int, int],
        pad_value: float = 0,
        crop_mode: str = "center",
    ):
        super().__init__()
        self.target_shape = target_shape
        self.pad_value = pad_value
        self.crop_mode = crop_mode

    def forward(self, volume: Tensor) -> Tensor:
        current_shape = volume.shape[-3:]

        if all(c == t for c, t in zip(current_shape, self.target_shape)):
            return volume

        volume = self._crop(volume, current_shape)
        volume = self._pad(volume, current_shape)

        return volume

    def _crop(self, volume: Tensor, current_shape: Tuple[int, int, int]) -> Tensor:
        crop_start = [0, 0, 0]

        if self.crop_mode == "center":
            for i in range(3):
                if current_shape[i] > self.target_shape[i]:
                    crop_start[i] = (current_shape[i] - self.target_shape[i]) // 2
        elif self.crop_mode == "random":
            for i in range(3):
                if current_shape[i] > self.target_shape[i]:
                    crop_start[i] = random.randint(
                        0,
                        current_shape[i] - self.target_shape[i],
                    )

        slices = [
            slice(crop_start[i], crop_start[i] + self.target_shape[i]) for i in range(3)
        ]

        if volume.ndim == 4:
            slices = [slice(None)] + slices
        elif volume.ndim == 5:
            slices = [slice(None), slice(None)] + slices

        return volume[slices]

    def _pad(self, volume: Tensor, current_shape: Tuple[int, int, int]) -> Tensor:
        pad_dims = []

        for i in range(3):
            dim_idx = volume.ndim - 3 + i
            if current_shape[i] < self.target_shape[i]:
                total_pad = self.target_shape[i] - current_shape[i]
                before = total_pad // 2
                after = total_pad - before
                pad_dims = [before, after] + pad_dims
            else:
                pad_dims = [0, 0] + pad_dims

        if len(pad_dims) > 0:
            volume = F.pad(volume, pad_dims, value=self.pad_value)

        return volume


def resample_volume(
    volume: Tensor,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int = 1,
) -> Tensor:
    """Resample volume to target spacing.

    Args:
        volume: Input volume
        current_spacing: Current voxel spacing (z, y, x)
        target_spacing: Target voxel spacing (z, y, x)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        Resampled volume
    """
    resampler = VolumeResampler(
        target_spacing=target_spacing,
        order=order,
    )
    return resampler(volume, current_spacing)


def compute_volume_statistics(
    volume: Tensor,
    percentiles: Tuple[float, float, float] = (1, 50, 99),
) -> VolumeStatistics:
    """Compute statistics for a volume.

    Args:
        volume: Input volume
        percentiles: Percentiles to compute

    Returns:
        VolumeStatistics object
    """
    flat = volume.flatten().float()

    percentiles_dict = {}
    for p in percentiles:
        percentiles_dict[f"p{int(p)}"] = float(torch.quantile(flat, p / 100))

    hist, bins = torch.histc(flat, bins=256).cpu().numpy(), None

    return VolumeStatistics(
        mean=float(flat.mean()),
        std=float(flat.std()),
        min=float(flat.min()),
        max=float(flat.max()),
        median=float(torch.median(flat)),
        percentiles=percentiles_dict,
        histogram=hist,
    )


class ZScoreNormalize(nn.Module):
    """Z-score normalization for volumes."""

    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        if self.mean is None:
            mean = x.mean()
        else:
            mean = self.mean

        if self.std is None:
            std = x.std() + 1e-8
        else:
            std = self.std

        return (x - mean) / std


class MinMaxNormalize(nn.Module):
    """Min-max normalization for volumes."""

    def __init__(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        output_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.output_range = output_range

    def forward(self, x: Tensor) -> Tensor:
        if self.min_val is None:
            min_val = x.min()
        else:
            min_val = self.min_val

        if self.max_val is None:
            max_val = x.max()
        else:
            max_val = self.max_val

        x_norm = (x - min_val) / (max_val - min_val + 1e-8)

        x_norm = (
            x_norm * (self.output_range[1] - self.output_range[0])
            + self.output_range[0]
        )

        return x_norm

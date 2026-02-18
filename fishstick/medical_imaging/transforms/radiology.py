"""
Radiology-specific Transforms

Window/level, HU normalization, and artifact simulation transforms.
"""

from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


WINDOW_PRESETS = {
    "brain": {"center": 40, "width": 80},
    "subdural": {"center": 75, "width": 250},
    "stroke": {"center": 40, "width": 40},
    "soft_tissue": {"center": 50, "width": 350},
    "bone": {"center": 300, "width": 1500},
    "lung": {"center": -600, "width": 1500},
    "liver": {"center": 60, "width": 150},
    "mediastinum": {"center": 50, "width": 350},
    "abdomen": {"center": 60, "width": 400},
    "kidney": {"center": 40, "width": 300},
}


class WindowLevelTransform(nn.Module):
    """Apply window/level (contrast/brightness) transform for CT images.

    Maps Hounsfield Units to displayable values based on window/level settings.

    Example:
        >>> transform = WindowLevelTransform(window_center=40, window_width=80)
        >>> transformed = transform(ct_volume)
    """

    def __init__(
        self,
        window_center: float = 40,
        window_width: float = 400,
        output_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.window_center = window_center
        self.window_width = window_width
        self.output_range = output_range

    def forward(self, x: Tensor) -> Tensor:
        """Apply window/level transform.

        Args:
            x: Input CT volume in HU

        Returns:
            Windowed volume
        """
        min_val = self.window_center - self.window_width / 2
        max_val = self.window_center + self.window_width / 2

        windowed = torch.clamp(x, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val)

        windowed = (
            windowed * (self.output_range[1] - self.output_range[0])
            + self.output_range[0]
        )

        return windowed

    def __repr__(self) -> str:
        return f"WindowLevelTransform(center={self.window_center}, width={self.window_width})"


class HUNormalize(nn.Module):
    """Normalize Hounsfield Unit values.

    Converts raw CT values to HU and optionally clips to physiological range.
    """

    def __init__(
        self,
        slope: float = 1.0,
        intercept: float = -1024,
        clip_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.slope = slope
        self.intercept = intercept
        self.clip_range = clip_range
        self.normalize = normalize

    def forward(self, x: Tensor) -> Tensor:
        hu = x * self.slope + self.intercept

        if self.clip_range:
            hu = torch.clamp(hu, self.clip_range[0], self.clip_range[1])

        if self.normalize:
            hu = (hu - hu.mean()) / (hu.std() + 1e-8)

        return hu


class CTBoneRemoval(nn.Module):
    """Remove bone from CT images using thresholding.

    Sets pixels above bone threshold to a specified value.
    """

    def __init__(
        self,
        threshold: float = 1200,
        fill_value: float = -1024,
    ):
        super().__init__()
        self.threshold = threshold
        self.fill_value = fill_value

    def forward(self, x: Tensor) -> Tensor:
        mask = x > self.threshold
        result = x.clone()
        result[mask] = self.fill_value
        return result


class MRNormalize(nn.Module):
    """Normalize MRI intensities.

    Supports multiple normalization strategies for MRI data.
    """

    def __init__(
        self,
        method: str = "zscore",
        percentiles: Optional[Tuple[float, float]] = (1, 99),
    ):
        super().__init__()
        self.method = method
        self.percentiles = percentiles

    def forward(self, x: Tensor) -> Tensor:
        if self.method == "zscore":
            return (x - x.mean()) / (x.std() + 1e-8)

        elif self.method == "minmax":
            min_val = x.min()
            max_val = x.max()
            if max_val > min_val:
                return (x - min_val) / (max_val - min_val)
            return x

        elif self.method == "percentile":
            if self.percentiles:
                p_min, p_max = self.percentiles

                x_flat = x.flatten()
                min_val = torch.quantile(x_flat, p_min / 100)
                max_val = torch.quantile(x_flat, p_max / 100)

                x_clipped = torch.clamp(x, min_val, max_val)

                return (x_clipped - min_val) / (max_val - min_val + 1e-8)

        return x


class RandomWindowLevel(nn.Module):
    """Apply random window/level transform.

    Randomly selects window presets or generates random window/level values.
    """

    def __init__(
        self,
        presets: Optional[List[str]] = None,
        width_range: Tuple[float, float] = (50, 500),
        center_range: Tuple[float, float] = (-100, 300),
        p: float = 0.5,
    ):
        super().__init__()

        if presets is None:
            self.presets = list(WINDOW_PRESETS.keys())
        else:
            self.presets = presets

        self.width_range = width_range
        self.center_range = center_range
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        if self.presets and random.random() > 0.3:
            preset = random.choice(self.presets)
            center = WINDOW_PRESETS[preset]["center"]
            width = WINDOW_PRESETS[preset]["width"]
        else:
            center = random.uniform(*self.center_range)
            width = random.uniform(*self.width_range)

        transform = WindowLevelTransform(center, width)
        return transform(x)


class RandomBiasField(nn.Module):
    """Add random bias field artifact to MRI.

    Simulates low-frequency intensity inhomogeneity common in MRI.
    """

    def __init__(
        self,
        num_coeffs: int = 3,
        sigma: float = 0.5,
        p: float = 0.5,
    ):
        super().__init__()
        self.num_coeffs = num_coeffs
        self.sigma = sigma
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        shape = x.shape

        coeffs = (
            torch.randn(self.num_coeffs, self.num_coeffs, self.num_coeffs) * self.sigma
        )

        if x.is_cuda:
            coeffs = coeffs.to(x.device)

        coeffs = F.interpolate(
            coeffs.unsqueeze(0),
            size=shape[-3:],
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        bias = torch.exp(coeffs)

        return x * bias


class RandomGhosting(nn.Module):
    """Add ghosting artifact to MRI.

    Simulates slice cross-talk ghosting artifact.
    """

    def __init__(
        self,
        num_ghosts: int = 4,
        intensity: float = 0.5,
        p: float = 0.5,
    ):
        super().__init__()
        self.num_ghosts = num_ghosts
        self.intensity = intensity
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        result = x.clone()

        dim = random.choice([-3, -2, -1])

        shift = random.randint(5, 20)

        slices = [slice(None)] * x.ndim
        for i in range(1, self.num_ghosts + 1):
            slices_ghost = slices.copy()
            slices_ghost[dim] = slice(i * shift, None)

            slices_orig = slices.copy()
            slices_orig[dim] = slice(None, -i * shift)

            result[slices_ghost] += x[slices_orig] * self.intensity / (i + 1)

        return result


class RandomSpike(nn.Module):
    """Add spike artifact to MRI or CT.

    Simulates RF interference spike artifacts.
    """

    def __init__(
        num_spikes: int = 1,
        intensity: float = 0.3,
        p: float = 0.3,
    ):
        super().__init__()
        self.num_spikes = num_spikes
        self.intensity = intensity
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        result = x.clone()

        shape = x.shape

        for _ in range(self.num_spikes):
            spike = torch.randn_like(x) * self.intensity

            if x.is_cuda:
                spike = spike.cuda()

            result += spike

        return result


class ComposeTransforms(nn.Module):
    """Compose multiple transforms."""

    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class OrganSpecificTransform(nn.Module):
    """Organ-specific window/level transform for CT.

    Automatically applies optimal window/level for specific organs.
    """

    def __init__(self, organ: str = "liver"):
        super().__init__()
        self.organ = organ.lower()

        if self.organ not in WINDOW_PRESETS:
            raise ValueError(f"Unknown organ: {organ}")

        preset = WINDOW_PRESETS[self.organ]
        self.transform = WindowLevelTransform(
            preset["center"],
            preset["width"],
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)

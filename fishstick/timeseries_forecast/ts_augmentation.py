"""
Time Series Augmentation Library.

Implements various augmentation techniques for time series data:
- Time warping
- Magnitude scaling
- Window slicing/wrapping
- Noise injection
- Permutation
- Augmentation composer

Example:
    >>> from fishstick.timeseries_forecast import (
    ...     TimeWarping,
    ...     MagnitudeScaling,
    ...     WindowSlice,
    ...     NoiseInjection,
    ...     TimeSeriesAugmenter,
    ... )
"""

from typing import Optional, List, Tuple, Callable, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from dataclasses import dataclass


class TimeSeriesAugmentation(ABC):
    """Abstract base class for time series augmentations.

    Args:
        p: Probability of applying augmentation
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Apply augmentation.

        Args:
            x: Input time series [B, L, D] or [L, D]

        Returns:
            Augmented time series
        """
        pass

    def _should_apply(self) -> bool:
        return torch.rand(1).item() < self.p


class TimeWarping(TimeSeriesAugmentation):
    """Time warping augmentation.

    Stretches or compresses the time axis using a smooth warp function.

    Args:
        p: Probability of applying augmentation
        sigma: Standard deviation of warp magnitude
        knot: Number of knot points for warping

    Example:
        >>> aug = TimeWarping(p=0.5, sigma=0.2)
        >>> x = torch.randn(32, 100, 7)
        >>> augmented = aug(x)
    """

    def __init__(
        self,
        p: float = 0.5,
        sigma: float = 0.2,
        knot: int = 4,
    ):
        super().__init__(p)
        self.sigma = sigma
        self.knot = knot

    def __call__(self, x: Tensor) -> Tensor:
        """Apply time warping.

        Args:
            x: Input [B, L, D] or [L, D]

        Returns:
            Warped time series
        """
        if not self._should_apply():
            return x

        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, L, D = x.shape

        warps = torch.randn(self.knot + 2) * self.sigma
        warps = torch.cumsum(warps, dim=0)
        warps = warps - warps[0]
        warps = warps / warps[-1]

        warp_points = torch.linspace(0, 1, self.knot + 2, device=x.device)
        warp_fn = torch.interp(
            torch.linspace(0, 1, L, device=x.device),
            warp_points,
            warps,
        )

        warped_indices = (warp_fn * (L - 1)).long().clamp(0, L - 1)

        x_warped = x.clone()
        for b in range(B):
            for d in range(D):
                x_warped[b, :, d] = x[b, warped_indices, d]

        if squeeze:
            x_warped = x_warped.squeeze(0)

        return x_warped


class MagnitudeScaling(TimeSeriesAugmentation):
    """Magnitude scaling augmentation.

    Scales the amplitude of the signal randomly.

    Args:
        p: Probability of applying augmentation
        sigma: Standard deviation of scaling factor
        min_scale: Minimum scaling factor
        max_scale: Maximum scaling factor

    Example:
        >>> aug = MagnitudeScaling(p=0.5, sigma=0.1)
        >>> x = torch.randn(32, 100, 7)
        >>> augmented = aug(x)
    """

    def __init__(
        self,
        p: float = 0.5,
        sigma: float = 0.1,
        min_scale: float = 0.9,
        max_scale: float = 1.1,
    ):
        super().__init__(p)
        self.sigma = sigma
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, x: Tensor) -> Tensor:
        """Apply magnitude scaling.

        Args:
            x: Input [B, L, D] or [L, D]

        Returns:
            Scaled time series
        """
        if not self._should_apply():
            return x

        scale = torch.randn(1).item() * self.sigma + 1.0
        scale = np.clip(scale, self.min_scale, self.max_scale)

        return x * scale


class WindowSlice(TimeSeriesAugmentation):
    """Window slicing augmentation.

    Extracts a random sub-window from the time series.

    Args:
        p: Probability of applying augmentation
        slice_ratio: Ratio of window size to original length
        mode: 'slice' (cut) or 'wrap' (wrap around)

    Example:
        >>> aug = WindowSlice(p=0.5, slice_ratio=0.8)
        >>> x = torch.randn(32, 100, 7)
        >>> augmented = aug(x)
    """

    def __init__(
        self,
        p: float = 0.5,
        slice_ratio: float = 0.8,
        mode: str = "slice",
    ):
        super().__init__(p)
        self.slice_ratio = slice_ratio
        self.mode = mode

    def __call__(self, x: Tensor) -> Tensor:
        """Apply window slicing.

        Args:
            x: Input [B, L, D] or [L, D]

        Returns:
            Sliced time series
        """
        if not self._should_apply():
            return x

        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, L, D = x.shape

        new_L = int(L * self.slice_ratio)
        start = torch.randint(0, L - new_L + 1, (1,)).item()

        if self.mode == "slice":
            x_sliced = x[:, start : start + new_L, :]
            x_sliced = F.interpolate(
                x_sliced.permute(0, 2, 1),
                size=L,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
        else:
            indices = torch.arange(start, start + new_L, device=x.device) % L
            x_sliced = x[:, indices, :]

        if squeeze:
            x_sliced = x_sliced.squeeze(0)

        return x_sliced


class NoiseInjection(TimeSeriesAugmentation):
    """Gaussian noise injection augmentation.

    Args:
        p: Probability of applying augmentation
        sigma: Standard deviation of noise
        mean: Mean of noise

    Example:
        >>> aug = NoiseInjection(p=0.5, sigma=0.05)
        >>> x = torch.randn(32, 100, 7)
        >>> augmented = aug(x)
    """

    def __init__(
        self,
        p: float = 0.5,
        sigma: float = 0.05,
        mean: float = 0.0,
    ):
        super().__init__(p)
        self.sigma = sigma
        self.mean = mean

    def __call__(self, x: Tensor) -> Tensor:
        """Apply noise injection.

        Args:
            x: Input [B, L, D] or [L, D]

        Returns:
            Noisy time series
        """
        if not self._should_apply():
            return x

        noise = torch.randn_like(x) * self.sigma + self.mean

        return x + noise


class Permutation(TimeSeriesAugmentation):
    """Random permutation augmentation.

    Randomly permutes segments of the time series.

    Args:
        p: Probability of applying augmentation
        n_segments: Number of segments to permute

    Example:
        >>> aug = Permutation(p=0.5, n_segments=5)
        >>> x = torch.randn(32, 100, 7)
        >>> augmented = aug(x)
    """

    def __init__(
        self,
        p: float = 0.5,
        n_segments: int = 5,
    ):
        super().__init__(p)
        self.n_segments = n_segments

    def __call__(self, x: Tensor) -> Tensor:
        """Apply permutation.

        Args:
            x: Input [B, L, D] or [L, D]

        Returns:
            Permuted time series
        """
        if not self._should_apply():
            return x

        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, L, D = x.shape

        segment_length = L // self.n_segments

        indices = torch.randperm(self.n_segments)

        segments = []
        for i in range(self.n_segments):
            start = i * segment_length
            end = start + segment_length if i < self.n_segments - 1 else L
            segments.append(x[:, start:end, :])

        reordered = [segments[i] for i in indices]
        x_permuted = torch.cat(reordered, dim=1)

        if squeeze:
            x_permuted = x_permuted.squeeze(0)

        return x_permuted


class MagnitudeWarping(TimeSeriesAugmentation):
    """Magnitude warping augmentation.

    Applies smooth warping to the amplitude using a random curve.

    Args:
        p: Probability of applying augmentation
        sigma: Standard deviation of warp magnitude
        knot: Number of knot points

    Example:
        >>> aug = MagnitudeWarping(p=0.5, sigma=0.2)
        >>> x = torch.randn(32, 100, 7)
        >>> augmented = aug(x)
    """

    def __init__(
        self,
        p: float = 0.5,
        sigma: float = 0.2,
        knot: int = 4,
    ):
        super().__init__(p)
        self.sigma = sigma
        self.knot = knot

    def __call__(self, x: Tensor) -> Tensor:
        """Apply magnitude warping.

        Args:
            x: Input [B, L, D] or [L, D]

        Returns:
            Warped time series
        """
        if not self._should_apply():
            return x

        B, L, D = x.shape

        warps = torch.randn(self.knot, D, device=x.device) * self.sigma
        warps = torch.cat([torch.ones(1, D, device=x.device), warps], dim=0)
        warps = torch.cumsum(warps, dim=0)

        warp_points = torch.linspace(0, 1, self.knot + 1, device=x.device)
        warp_fn = torch.interp(
            torch.linspace(0, 1, L, device=x.device).unsqueeze(-1),
            warp_points.unsqueeze(-1),
            warps,
        )

        return x * warp_fn


class RandomCutout(TimeSeriesAugmentation):
    """Random cutout augmentation.

    Randomly masks out a portion of the time series.

    Args:
        p: Probability of applying augmentation
        cutout_ratio: Ratio of time series to mask
        mode: 'zero' (replace with zeros) or 'mean' (replace with mean)

    Example:
        >>> aug = RandomCutout(p=0.5, cutout_ratio=0.2)
        >>> x = torch.randn(32, 100, 7)
        >>> augmented = aug(x)
    """

    def __init__(
        self,
        p: float = 0.5,
        cutout_ratio: float = 0.2,
        mode: str = "zero",
    ):
        super().__init__(p)
        self.cutout_ratio = cutout_ratio
        self.mode = mode

    def __call__(self, x: Tensor) -> Tensor:
        """Apply cutout.

        Args:
            x: Input [B, L, D] or [L, D]

        Returns:
            Cutout time series
        """
        if not self._should_apply():
            return x

        B, L, D = x.shape

        cutout_length = int(L * self.cutout_ratio)
        start = torch.randint(0, L - cutout_length + 1, (1,)).item()

        x_masked = x.clone()

        if self.mode == "zero":
            x_masked[:, start : start + cutout_length, :] = 0
        elif self.mode == "mean":
            mean_val = x.mean(dim=(0, 1), keepdim=True)
            x_masked[:, start : start + cutout_length, :] = mean_val

        return x_masked


class TimeSeriesAugmenter(nn.Module):
    """Composable time series augmentation pipeline.

    Args:
        augmentations: List of augmentation modules
        apply_all: Whether to apply all augmentations or sample

    Example:
        >>> augmentations = [
        ...     TimeWarping(p=0.3),
        ...     MagnitudeScaling(p=0.3),
        ...     NoiseInjection(p=0.3),
        ... ]
        >>> augmenter = TimeSeriesAugmenter(augmentations)
        >>> x = torch.randn(32, 100, 7)
        >>> augmented = augmenter(x)
    """

    def __init__(
        self,
        augmentations: List[TimeSeriesAugmentation],
        apply_all: bool = False,
    ):
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations)
        self.apply_all = apply_all

    def forward(self, x: Tensor) -> Tensor:
        """Apply augmentations.

        Args:
            x: Input [B, L, D] or [L, D]

        Returns:
            Augmented time series
        """
        if self.apply_all:
            for aug in self.augmentations:
                x = aug(x)
        else:
            for aug in self.augmentations:
                if torch.rand(1).item() < aug.p:
                    x = aug(x)

        return x


class RandAugment(nn.Module):
    """RandAugment-style augmentation for time series.

    Randomly selects and applies N augmentations with magnitude M.

    Args:
        n: Number of augmentations to apply
        m: Magnitude of augmentations
        augmentation_list: List of available augmentations

    Example:
        >>> augmentations = [TimeWarping, MagnitudeScaling, NoiseInjection]
        >>> rand_augment = RandAugment(n=2, m=0.5)
        >>> x = torch.randn(32, 100, 7)
        >>> augmented = rand_augment(x)
    """

    def __init__(
        self,
        n: int = 2,
        m: float = 0.5,
        augmentation_list: Optional[List[type]] = None,
    ):
        super().__init__()
        self.n = n
        self.m = m

        if augmentation_list is None:
            augmentation_list = [
                TimeWarping,
                MagnitudeScaling,
                NoiseInjection,
                Permutation,
                WindowSlice,
            ]

        self.augmentation_classes = augmentation_list
        self.augmentations = nn.ModuleList([aug(p=1.0) for aug in augmentation_list])

    def forward(self, x: Tensor) -> Tensor:
        """Apply random augmentations.

        Args:
            x: Input [B, L, D] or [L, D]

        Returns:
            Augmented time series
        """
        indices = torch.randperm(len(self.augmentations))[: self.n]

        for idx in indices:
            aug = self.augmentations[idx]
            if hasattr(aug, "sigma"):
                aug.sigma = aug.sigma * self.m
            x = aug(x)

        return x


def create_standard_augmentations(
    aug_types: Optional[List[str]] = None,
    p: float = 0.5,
) -> List[TimeSeriesAugmentation]:
    """Factory function to create standard augmentation set.

    Args:
        aug_types: List of augmentation types
        p: Base probability for each augmentation

    Returns:
        List of augmentation modules

    Example:
        >>> augs = create_standard_augmentations(['timewarp', 'scale', 'noise'])
    """
    if aug_types is None:
        aug_types = ["timewarp", "scale", "noise", "slice", "permute"]

    aug_dict = {
        "timewarp": TimeWarping,
        "scale": MagnitudeScaling,
        "noise": NoiseInjection,
        "slice": WindowSlice,
        "permute": Permutation,
        "magwarp": MagnitudeWarping,
        "cutout": RandomCutout,
    }

    augmentations = []
    for aug_type in aug_types:
        if aug_type in aug_dict:
            augmentations.append(aug_dict[aug_type](p=p))

    return augmentations


@dataclass
class AugmentationConfig:
    """Configuration for time series augmentation."""

    aug_type: str
    p: float
    sigma: float = 0.1
    knot: int = 4
    slice_ratio: float = 0.8
    n_segments: int = 5


class AugmentationScheduler:
    """Scheduler for applying augmentations during training.

    Args:
        augmenter: TimeSeriesAugmenter
        initial_p: Initial probability
        final_p: Final probability
        n_epochs: Number of epochs
    """

    def __init__(
        self,
        augmenter: TimeSeriesAugmenter,
        initial_p: float = 0.0,
        final_p: float = 1.0,
        n_epochs: int = 100,
    ):
        self.augmenter = augmenter
        self.initial_p = initial_p
        self.final_p = final_p
        self.n_epochs = n_epochs

    def step(self, epoch: int) -> None:
        """Update augmentation probability.

        Args:
            epoch: Current epoch
        """
        p = self.initial_p + (self.final_p - self.initial_p) * (epoch / self.n_epochs)
        p = min(p, self.final_p)

        for aug in self.augmenter.augmentations:
            aug.p = p

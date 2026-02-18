"""
Slice Extraction Utilities

Extract 2D slices from 3D volumes along different anatomical planes.
"""

from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


def extract_slices_axial(
    volume: Tensor,
    slice_indices: Optional[List[int]] = None,
    num_slices: Optional[int] = None,
) -> Tensor:
    """Extract axial (transverse) slices from 3D volume.

    Args:
        volume: Input volume (C, D, H, W) or (D, H, W)
        slice_indices: Specific slice indices to extract
        num_slices: Number of evenly spaced slices to extract

    Returns:
        Extracted slices (C, num_slices, H, W) or (num_slices, H, W)
    """
    is_4d = volume.ndim == 4

    if is_4d:
        c, d, h, w = volume.shape
    else:
        d, h, w = volume.shape

    if slice_indices is None:
        if num_slices is None:
            num_slices = d

        slice_indices = np.linspace(0, d - 1, num_slices, dtype=int)

    slices = []
    for idx in slice_indices:
        if is_4d:
            slices.append(volume[:, idx, :, :])
        else:
            slices.append(volume[idx, :, :])

    if is_4d:
        return torch.stack(slices, dim=1)
    else:
        return torch.stack(slices, dim=0)


def extract_slices_sagittal(
    volume: Tensor,
    slice_indices: Optional[List[int]] = None,
    num_slices: Optional[int] = None,
) -> Tensor:
    """Extract sagittal (lateral) slices from 3D volume.

    Args:
        volume: Input volume (C, D, H, W) or (D, H, W)
        slice_indices: Specific slice indices to extract
        num_slices: Number of evenly spaced slices to extract

    Returns:
        Extracted slices (C, num_slices, D, H) or (num_slices, D, H)
    """
    is_4d = volume.ndim == 4

    if is_4d:
        c, d, h, w = volume.shape
    else:
        d, h, w = volume.shape

    if slice_indices is None:
        if num_slices is None:
            num_slices = w

        slice_indices = np.linspace(0, w - 1, num_slices, dtype=int)

    slices = []
    for idx in slice_indices:
        if is_4d:
            slices.append(volume[:, :, :, idx])
        else:
            slices.append(volume[:, :, idx])

    if is_4d:
        return torch.stack(slices, dim=1)
    else:
        return torch.stack(slices, dim=0)


def extract_slices_coronal(
    volume: Tensor,
    slice_indices: Optional[List[int]] = None,
    num_slices: Optional[int] = None,
) -> Tensor:
    """Extract coronal slices from 3D volume.

    Args:
        volume: Input volume (C, D, H, W) or (D, H, W)
        slice_indices: Specific slice indices to extract
        num_slices: Number of evenly spaced slices to extract

    Returns:
        Extracted slices (C, num_slices, D, W) or (num_slices, D, W)
    """
    is_4d = volume.ndim == 4

    if is_4d:
        c, d, h, w = volume.shape
    else:
        d, h, w = volume.shape

    if slice_indices is None:
        if num_slices is None:
            num_slices = h

        slice_indices = np.linspace(0, h - 1, num_slices, dtype=int)

    slices = []
    for idx in slice_indices:
        if is_4d:
            slices.append(volume[:, :, idx, :])
        else:
            slices.append(volume[:, idx, :])

    if is_4d:
        return torch.stack(slices, dim=1)
    else:
        return torch.stack(slices, dim=0)


class MultiPlaneSliceExtractor(nn.Module):
    """Extract slices from multiple anatomical planes.

    Example:
        >>> extractor = MultiPlaneSliceExtractor(planes=['axial', 'sagittal'], num_slices=32)
        >>> axial_slices, sag_slices = extractor(volume)
    """

    def __init__(
        self,
        planes: List[str] = ["axial"],
        num_slices: Optional[Union[int, Dict[str, int]]] = None,
    ):
        super().__init__()
        self.planes = planes

        if isinstance(num_slices, int):
            num_slices = {p: num_slices for p in planes}

        self.num_slices = num_slices or {p: None for p in planes}

    def forward(
        self,
        volume: Tensor,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Extract slices from specified planes.

        Args:
            volume: Input 3D or 4D volume

        Returns:
            Dictionary of slices per plane or stacked tensor
        """
        results = {}

        for plane in self.planes:
            num_slices = self.num_slices.get(plane)

            if plane == "axial":
                results[plane] = extract_slices_axial(volume, num_slices=num_slices)
            elif plane == "sagittal":
                results[plane] = extract_slices_sagittal(volume, num_slices=num_slices)
            elif plane == "coronal":
                results[plane] = extract_slices_coronal(volume, num_slices=num_slices)
            else:
                raise ValueError(f"Unknown plane: {plane}")

        if len(results) == 1:
            return list(results.values())[0]

        return results


class RandomSliceExtractor(nn.Module):
    """Extract random slices from volume."""

    def __init__(
        self,
        plane: str = "axial",
        num_slices: int = 1,
        include_indices: bool = False,
    ):
        super().__init__()
        self.plane = plane
        self.num_slices = num_slices
        self.include_indices = include_indices

    def forward(
        self,
        volume: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, List[int]]]:
        if self.plane == "axial":
            max_idx = volume.shape[-3]
        elif self.plane == "sagittal":
            max_idx = volume.shape[-1]
        elif self.plane == "coronal":
            max_idx = volume.shape[-2]
        else:
            raise ValueError(f"Unknown plane: {self.plane}")

        indices = sorted(
            np.random.choice(max_idx, self.num_slices, replace=False).tolist()
        )

        if self.plane == "axial":
            slices = extract_slices_axial(volume, indices)
        elif self.plane == "sagittal":
            slices = extract_slices_sagittal(volume, indices)
        elif self.plane == "coronal":
            slices = extract_slices_coronal(volume, indices)

        if self.include_indices:
            return slices, indices

        return slices

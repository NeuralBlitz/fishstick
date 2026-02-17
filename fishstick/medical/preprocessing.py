"""
Medical Image Preprocessing
"""

from typing import Tuple, Optional
import torch
from torch import Tensor
import numpy as np


class MedicalImageLoader:
    """Load medical images in various formats (DICOM, NIfTI, etc.)."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def load_nifti(self, filepath: str) -> Tuple[Tensor, dict]:
        """Load NIfTI (.nii) file."""
        try:
            import nibabel as nib

            img = nib.load(filepath)
            data = img.get_fdata()
            affine = img.affine
            header = img.header

            tensor = torch.from_numpy(data).float()

            if self.normalize:
                tensor = self._normalize(tensor)

            metadata = {
                "affine": affine,
                "header": header,
                "shape": data.shape,
                "spacing": header.get_zooms(),
            }

            return tensor, metadata

        except ImportError:
            raise ImportError("nibabel is required. Install with: pip install nibabel")

    def load_dicom(self, directory: str) -> Tuple[Tensor, dict]:
        """Load DICOM series from directory."""
        try:
            import pydicom
            from pathlib import Path

            dicom_files = sorted(Path(directory).glob("*.dcm"))
            slices = []

            for file in dicom_files:
                ds = pydicom.dcmread(str(file))
                slices.append(ds.pixel_array)

            volume = np.stack(slices, axis=0)
            tensor = torch.from_numpy(volume).float()

            if self.normalize:
                tensor = self._normalize(tensor)

            metadata = {
                "shape": volume.shape,
                "spacing": getattr(ds, "PixelSpacing", [1.0, 1.0]),
            }

            return tensor, metadata

        except ImportError:
            raise ImportError("pydicom is required. Install with: pip install pydicom")

    def _normalize(self, tensor: Tensor) -> Tensor:
        """Normalize to zero mean and unit variance."""
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-8)


class NormalizeMedicalImage:
    """Normalization methods for medical images."""

    @staticmethod
    def z_score(tensor: Tensor) -> Tensor:
        """Z-score normalization."""
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-8)

    @staticmethod
    def min_max(tensor: Tensor) -> Tensor:
        """Min-max normalization to [0, 1]."""
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    @staticmethod
    def window_level(
        tensor: Tensor,
        window_center: float,
        window_width: float,
    ) -> Tensor:
        """Apply window/level transform (for CT images)."""
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2

        tensor = torch.clamp(tensor, min_val, max_val)
        return (tensor - min_val) / (max_val - min_val)

    @staticmethod
    def histogram_equalization(tensor: Tensor) -> Tensor:
        """Histogram equalization."""
        # Flatten and compute histogram
        flat = tensor.flatten().numpy()

        # Compute CDF
        hist, bins = np.histogram(flat, bins=256, range=(flat.min(), flat.max()))
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalize

        # Interpolate
        equalized = np.interp(flat, bins[:-1], cdf)

        return torch.from_numpy(equalized).reshape(tensor.shape)


class ResampleMedicalImage:
    """Resample medical images to new spacing."""

    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.target_spacing = target_spacing

    def __call__(
        self,
        image: Tensor,
        current_spacing: Tuple[float, float, float],
    ) -> Tensor:
        """Resample image to target spacing."""
        # Calculate new shape
        current_shape = image.shape[-3:]
        new_shape = [
            int(current_shape[i] * current_spacing[i] / self.target_spacing[i])
            for i in range(3)
        ]

        # Interpolate
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        resampled = torch.nn.functional.interpolate(
            image,
            size=new_shape,
            mode="trilinear",
            align_corners=False,
        )

        return resampled.squeeze(0).squeeze(0)


class RandomMedicalAugmentation:
    """Random augmentations for medical images."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Tensor) -> Tensor:
        """Apply random augmentations."""
        import random

        if random.random() < self.p:
            # Random flip
            if random.random() < 0.5:
                image = torch.flip(image, dims=[-1])

        if random.random() < self.p:
            # Random rotation (90 degree multiples)
            k = random.randint(0, 3)
            if image.ndim >= 2:
                image = torch.rot90(image, k=k, dims=[-2, -1])

        if random.random() < self.p:
            # Random intensity shift
            shift = torch.randn(1) * 0.1
            image = image + shift

        return image

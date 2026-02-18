"""
Medical Imaging Base Module

Provides base classes, enums, and core utilities for medical imaging.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any, Union, List
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class Modality(Enum):
    """Medical imaging modality types."""

    CT = auto()
    MRI = auto()
    XRAY = auto()
    ULTRASOUND = auto()
    PET = auto()
    SPECT = auto()
    MAMMOGRAPHY = auto()
    ENDOSCOPY = auto()
    DERMATOLOGY = auto()
    OPHTHALMOLOGY = auto()
    PATHOLOGY = auto()
    UNKNOWN = auto()


class BodyRegion(Enum):
    """Body region types for medical imaging."""

    HEAD = auto()
    NECK = auto()
    CHEST = auto()
    ABDOMEN = auto()
    PELVIS = auto()
    SPINE = auto()
    LIMBS = auto()
    HEART = auto()
    BRAIN = auto()
    LIVER = auto()
    KIDNEY = auto()
    BREAST = auto()
    PROSTATE = auto()
    UNKNOWN = auto()


@dataclass
class MedicalImageMetadata:
    """Metadata for medical images."""

    modality: Modality = Modality.UNKNOWN
    body_region: BodyRegion = BodyRegion.UNKNOWN
    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    series_id: Optional[str] = None
    slice_thickness: Optional[float] = None
    pixel_spacing: Optional[Tuple[float, float]] = None
    rows: int = 0
    columns: int = 0
    depth: int = 0
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    hu_min: Optional[float] = None
    hu_max: Optional[float] = None
    acquisition_date: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def get_modality_info(modality: Modality) -> Dict[str, Any]:
    """Get information about a specific modality.

    Args:
        modality: The medical imaging modality

    Returns:
        Dictionary with modality information
    """
    info = {
        Modality.CT: {
            "name": "Computed Tomography",
            "typical_spacing": (0.5, 0.5, 1.0),
            "hu_range": (-1024, 3071),
            "default_window": (40, 400),
        },
        Modality.MRI: {
            "name": "Magnetic Resonance Imaging",
            "typical_spacing": (1.0, 1.0, 1.0),
            "intensity_normalized": True,
            "sequences": ["T1", "T2", "FLAIR", "DWI", "T1CE"],
        },
        Modality.XRAY: {
            "name": "X-Ray",
            "typical_spacing": (0.1, 0.1, 1.0),
            "default_window": (127, 255),
        },
        Modality.ULTRASOUND: {
            "name": "Ultrasound",
            "typical_spacing": (0.5, 0.5, 1.0),
        },
        Modality.PET: {
            "name": "Positron Emission Tomography",
            "typical_spacing": (2.0, 2.0, 2.0),
            "suv_normalized": True,
        },
        Modality.MAMMOGRAPHY: {
            "name": "Mammography",
            "typical_spacing": (0.05, 0.05, 1.0),
        },
        Modality.PATHOLOGY: {
            "name": "Digital Pathology",
            "typical_spacing": (0.001, 0.001, 1.0),
        },
    }
    return info.get(modality, {"name": "Unknown Modality"})


class MedicalImageBase:
    """Base class for medical images.

    Provides common functionality for handling medical images
    including metadata, preprocessing, and validation.
    """

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray, Image.Image],
        metadata: Optional[MedicalImageMetadata] = None,
    ):
        """Initialize medical image.

        Args:
            data: Image data as tensor, numpy array, or PIL image
            metadata: Optional metadata for the image
        """
        self._data = None
        self._set_data(data)
        self.metadata = metadata or MedicalImageMetadata()

    def _set_data(self, data: Union[torch.Tensor, np.ndarray, Image.Image]) -> None:
        """Convert input data to tensor."""
        if isinstance(data, Image.Image):
            data = np.array(data)

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor, np.ndarray, or PIL.Image, got {type(data)}"
            )

        self._data = data.float()

    @property
    def data(self) -> torch.Tensor:
        """Get the image data tensor."""
        return self._data

    @data.setter
    def data(self, value: Union[torch.Tensor, np.ndarray]) -> None:
        """Set the image data."""
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        self._data = value.float()

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the image."""
        return tuple(self._data.shape)

    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return self._data.ndim

    @property
    def is_3d(self) -> bool:
        """Check if image is 3D volumetric."""
        return self.ndim >= 3

    @property
    def is_2d(self) -> bool:
        """Check if image is 2D."""
        return self.ndim == 2

    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self._data.numpy()

    def cpu(self) -> "MedicalImageBase":
        """Move to CPU."""
        self._data = self._data.cpu()
        return self

    def cuda(self, device: Optional[int] = None) -> "MedicalImageBase":
        """Move to CUDA."""
        if device is not None:
            self._data = self._data.cuda(device)
        else:
            self._data = self._data.cuda()
        return self

    def to(self, device: Union[int, str]) -> "MedicalImageBase":
        """Move to device."""
        self._data = self._data.to(device)
        return self

    def clone(self) -> "MedicalImageBase":
        """Create a copy of this image."""
        new_data = self._data.clone()
        new_metadata = MedicalImageMetadata(
            modality=self.metadata.modality,
            body_region=self.metadata.body_region,
            patient_id=self.metadata.patient_id,
            study_id=self.metadata.study_id,
            series_id=self.metadata.series_id,
            slice_thickness=self.metadata.slice_thickness,
            pixel_spacing=self.metadata.pixel_spacing,
            rows=self.metadata.rows,
            columns=self.metadata.columns,
            depth=self.metadata.depth,
            window_center=self.metadata.window_center,
            window_width=self.metadata.window_width,
            rescale_slope=self.metadata.rescale_slope,
            rescale_intercept=self.metadata.rescale_intercept,
            hu_min=self.metadata.hu_min,
            hu_max=self.metadata.hu_max,
            acquisition_date=self.metadata.acquisition_date,
            extra=self.metadata.extra.copy(),
        )
        return MedicalImageBase(new_data, new_metadata)

    def __repr__(self) -> str:
        return f"MedicalImageBase(shape={self.shape}, modality={self.metadata.modality.name})"


def validate_medical_image(
    data: torch.Tensor,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> Tuple[bool, str]:
    """Validate medical image data.

    Args:
        data: Image tensor to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values

    Returns:
        Tuple of (is_valid, message)
    """
    if data is None or data.numel() == 0:
        return False, "Empty image data"

    if not allow_nan and torch.isnan(data).any():
        return False, "Image contains NaN values"

    if not allow_inf and torch.isinf(data).any():
        return False, "Image contains infinite values"

    if min_value is not None and (data < min_value).any():
        return False, f"Image contains values below {min_value}"

    if max_value is not None and (data > max_value).any():
        return False, f"Image contains values above {max_value}"

    return True, "Valid"

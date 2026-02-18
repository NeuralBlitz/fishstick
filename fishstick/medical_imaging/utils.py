"""
Medical Imaging Utilities

I/O operations, preprocessing, visualization, and common utilities
for medical images.
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any, Callable

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

try:
    import SimpleITK as sitk

    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

try:
    import nibabel as nib

    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


def load_medical_image(
    path: Union[str, Path],
    modality: Optional[str] = None,
    normalize: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load a medical image from file.

    Supports NIfTI, DICOM, NRRD, and standard image formats.

    Args:
        path: Path to the medical image file
        modality: Optional modality hint (ct, mri, xray, etc.)
        normalize: Whether to normalize intensity values

    Returns:
        Tuple of (image tensor, metadata dict)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    metadata = {"path": str(path), "modality": modality}

    if suffix in [".nii", ".nii.gz"]:
        return _load_nifti(path, normalize, metadata)
    elif suffix in [".nrrd"]:
        return _load_nrrd(path, normalize, metadata)
    elif suffix in [".mha", ".mhd"]:
        return _load_mha(path, normalize, metadata)
    else:
        return _load_standard_image(path, normalize, metadata)


def _load_nifti(
    path: Path,
    normalize: bool,
    metadata: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load NIfTI image."""
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required for NIfTI support")

    img = nib.load(str(path))
    data = img.get_fdata()

    metadata.update(
        {
            "affine": img.affine,
            "header": dict(img.header),
            "voxel_size": img.header.get_zooms(),
        }
    )

    data = torch.from_numpy(data.astype(np.float32))

    if normalize:
        data = _normalize_tensor(data)

    return data, metadata


def _load_nrrd(
    path: Path,
    normalize: bool,
    metadata: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load NRRD image."""
    if not SITK_AVAILABLE:
        raise ImportError("SimpleITK is required for NRRD support")

    img = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(img)

    metadata.update(
        {
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "direction": img.GetDirection(),
        }
    )

    data = torch.from_numpy(data.astype(np.float32))

    if normalize:
        data = _normalize_tensor(data)

    return data, metadata


def _load_mha(
    path: Path,
    normalize: bool,
    metadata: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load MHA/MHD image."""
    return _load_nrrd(path, normalize, metadata)


def _load_standard_image(
    path: Path,
    normalize: bool,
    metadata: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load standard image (PNG, JPG, etc.)."""
    img = Image.open(path)
    data = np.array(img)

    if len(data.shape) == 2:
        data = data[np.newaxis, ...]
    elif len(data.shape) == 3:
        data = np.transpose(data, (2, 0, 1))

    data = torch.from_numpy(data.astype(np.float32))

    if normalize:
        data = _normalize_tensor(data)

    metadata["shape"] = data.shape

    return data, metadata


def _normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range."""
    min_val = x.min()
    max_val = x.max()
    if max_val > min_val:
        x = (x - min_val) / (max_val - min_val)
    return x


def save_medical_image(
    data: torch.Tensor,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    format: str = "nifti",
) -> None:
    """Save a medical image to file.

    Args:
        data: Image tensor (D, H, W) or (H, W)
        path: Output path
        metadata: Optional metadata dictionary
        format: Output format (nifti, nrrd, png, npy)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if format == "nifti":
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for NIfTI export")

        affine = metadata.get("affine") if metadata else np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(path))

    elif format == "nrrd":
        if not SITK_AVAILABLE:
            raise ImportError("SimpleITK is required for NRRD export")

        img = sitk.GetImageFromArray(data)
        if metadata:
            if "spacing" in metadata:
                img.SetSpacing(metadata["spacing"])
            if "origin" in metadata:
                img.SetOrigin(metadata["origin"])
        sitk.WriteImage(img, str(path))

    elif format == "png":
        if data.ndim == 3:
            data = data[0] if data.shape[0] == 1 else np.transpose(data, (1, 2, 0))
        data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        Image.fromarray(data).save(path)

    elif format == "npy":
        np.save(path, data)

    else:
        raise ValueError(f"Unknown format: {format}")


def compute_dicom_window(
    hu_value: float,
    window_center: float,
    window_width: float,
) -> float:
    """Compute windowed value for DICOM window/level.

    Args:
        hu_value: Hounsfield unit value
        window_center: Window center (level)
        window_width: Window width

    Returns:
        Windowed value in [0, 255]
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2

    if hu_value <= min_val:
        return 0.0
    elif hu_value >= max_val:
        return 255.0
    else:
        return ((hu_value - min_val) / window_width) * 255.0


def apply_window_level(
    image: torch.Tensor,
    window_center: Union[float, Tuple[float, ...]],
    window_width: Union[float, Tuple[float, ...]],
) -> torch.Tensor:
    """Apply window/level to CT or MRI image.

    Args:
        image: Input image tensor
        window_center: Window center(s)
        window_width: Window width(s)

    Returns:
        Windowed image(s)
    """
    if isinstance(window_center, (int, float)):
        window_center = (window_center,)
        window_width = (window_width,)

    if len(window_center) == 1 and image.ndim > 2:
        window_center = window_center * image.shape[0]
        window_width = window_width * image.shape[0]

    result = torch.zeros_like(image)

    for i, (wc, ww) in enumerate(zip(window_center, window_width)):
        min_val = wc - ww / 2
        max_val = wc + ww / 2

        slice_data = image[i] if i < image.shape[0] else image
        slice_data = torch.clamp(slice_data, min_val, max_val)
        slice_data = (slice_data - min_val) / ww
        result[i] = slice_data

    return result


def normalize_hu(
    image: torch.Tensor,
    slope: float = 1.0,
    intercept: float = -1024,
    clip_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Normalize CT image to Hounsfield Units.

    Args:
        image: Raw CT image
        slope: Rescale slope from DICOM
        intercept: Rescale intercept from DICOM
        clip_range: Optional HU range to clip to

    Returns:
        HU-normalized image
    """
    hu_image = image * slope + intercept

    if clip_range:
        hu_image = torch.clamp(hu_image, clip_range[0], clip_range[1])

    return hu_image


def get_bounding_box(
    mask: torch.Tensor,
    margin: int = 0,
) -> Tuple[Tuple[int, int], ...]:
    """Get bounding box of non-zero regions in mask.

    Args:
        mask: Binary mask tensor
        margin: Optional margin to add around bbox

    Returns:
        Tuple of (min, max) indices for each dimension
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    bbox = []
    for dim in range(mask.ndim):
        dims_sum = mask.sum(axis=tuple(i for i in range(mask.ndim) if i != dim))
        nonzero_idx = np.where(dims_sum > 0)[0]

        if len(nonzero_idx) > 0:
            min_idx = max(0, nonzero_idx[0] - margin)
            max_idx = min(mask.shape[dim], nonzero_idx[-1] + 1 + margin)
            bbox.append((int(min_idx), int(max_idx)))
        else:
            bbox.append((0, mask.shape[dim]))

    return tuple(bbox)


def crop_to_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    margin: int = 0,
    pad_value: float = 0,
) -> torch.Tensor:
    """Crop image to bounding box of mask.

    Args:
        image: Input image tensor
        mask: Binary mask tensor
        margin: Margin to add around mask
        pad_value: Value for padding if crop goes out of bounds

    Returns:
        Cropped image
    """
    bbox = get_bounding_box(mask, margin)

    slices = [slice(start, end) for start, end in bbox]

    if len(slices) < image.ndim:
        slices = slices + [slice(None)] * (image.ndim - len(slices))

    cropped = image[slices]

    return cropped


def resample_volume(
    image: torch.Tensor,
    current_spacing: Tuple[float, ...],
    target_spacing: Tuple[float, ...],
    order: int = 1,
) -> torch.Tensor:
    """Resample volumetric image to new spacing.

    Args:
        image: Input volume (D, H, W) or (C, D, H, W)
        current_spacing: Current voxel spacing
        target_spacing: Target voxel spacing
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        Resampled volume
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    if len(current_spacing) != len(target_spacing):
        raise ValueError("Spacing dimensions must match")

    scale_factor = tuple(cs / ts for cs, ts in zip(current_spacing, target_spacing))

    new_shape = tuple(
        int(s * sf) for s, sf in zip(image.shape[-len(scale_factor) :], scale_factor)
    )

    if image.ndim > len(scale_factor):
        new_shape = image.shape[: -len(scale_factor)] + new_shape

    from scipy.ndimage import zoom

    zoom_factors = [n / o for n, o in zip(new_shape, image.shape)]

    resampled = zoom(image, zoom_factors, order=order)

    return torch.from_numpy(resampled.astype(np.float32))


class MedicalImageVisualizer:
    """Visualization utilities for medical images.

    Provides methods for creating overlays, multi-plane views,
    and saving visualizations.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (15, 10),
        cmap: str = "gray",
        alpha: float = 0.5,
    ):
        """Initialize visualizer.

        Args:
            figsize: Default figure size
            cmap: Default colormap
            alpha: Overlay transparency
        """
        self.figsize = figsize
        self.cmap = cmap
        self.alpha = alpha

    def create_overlay(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mask: Union[torch.Tensor, np.ndarray],
        slice_idx: Optional[int] = None,
        axis: int = 0,
    ) -> np.ndarray:
        """Create image-mask overlay for a specific slice.

        Args:
            image: Input image
            mask: Segmentation mask
            slice_idx: Slice index (None for middle)
            axis: Axis to slice along

        Returns:
            RGB overlay image
        """
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        if slice_idx is None:
            slice_idx = image.shape[axis] // 2

        if axis == 0:
            img_slice = image[slice_idx]
            mask_slice = mask[slice_idx]
        elif axis == 1:
            img_slice = image[:, slice_idx, :]
            mask_slice = mask[:, slice_idx, :]
        else:
            img_slice = image[:, :, slice_idx]
            mask_slice = mask[:, :, slice_idx]

        if img_slice.ndim == 2:
            img_slice = (img_slice - img_slice.min()) / (
                img_slice.max() - img_slice.min() + 1e-8
            )
            img_rgb = np.stack([img_slice] * 3, axis=-1)
        else:
            img_rgb = img_slice

        mask_colors = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [255, 0, 255],
                [0, 255, 255],
            ]
        )

        overlay = img_rgb.copy()

        for i in range(1, int(mask_slice.max()) + 1):
            if i < len(mask_colors):
                mask_binary = (mask_slice == i).astype(float)
                for c in range(3):
                    overlay[:, :, c] = (
                        overlay[:, :, c] * (1 - mask_binary * self.alpha)
                        + mask_colors[i, c] / 255 * mask_binary * self.alpha
                    )

        return (overlay * 255).astype(np.uint8)

    def plot_mip(
        self,
        volume: Union[torch.Tensor, np.ndarray],
        axis: int = 0,
    ) -> np.ndarray:
        """Create Maximum Intensity Projection.

        Args:
            volume: 3D volume
            axis: Axis to project along

        Returns:
            2D MIP image
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()

        if axis == 0:
            mip = volume.max(axis=0)
        elif axis == 1:
            mip = volume.max(axis=1)
        else:
            mip = volume.max(axis=2)

        mip = (mip - mip.min()) / (mip.max() - mip.min() + 1e-8)
        return (mip * 255).astype(np.uint8)

    def save_slice(
        self,
        image: Union[torch.Tensor, np.ndarray],
        path: Union[str, Path],
        slice_idx: Optional[int] = None,
        axis: int = 0,
    ) -> None:
        """Save a single slice as image file.

        Args:
            image: Input image
            path: Output path
            slice_idx: Slice index
            axis: Axis to slice along
        """
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        if slice_idx is None:
            slice_idx = image.shape[axis] // 2

        if axis == 0:
            img_slice = image[slice_idx]
        elif axis == 1:
            img_slice = image[:, slice_idx, :]
        else:
            img_slice = image[:, :, slice_idx]

        img_slice = (img_slice - img_slice.min()) / (
            img_slice.max() - img_slice.min() + 1e-8
        )
        img_slice = (img_slice * 255).astype(np.uint8)

        Image.fromarray(img_slice).save(path)

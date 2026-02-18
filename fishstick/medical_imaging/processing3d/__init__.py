"""
3D Medical Image Processing Module

Volume resampling, cropping, slice extraction, and synthetic data generation.
"""

from fishstick.medical_imaging.processing3d.volume_utils import (
    VolumeResampler,
    CropAndPad,
    resample_volume,
    compute_volume_statistics,
    VolumeStatistics,
)

from fishstick.medical_imaging.processing3d.slice_extraction import (
    extract_slices_axial,
    extract_slices_sagittal,
    extract_slices_coronal,
    MultiPlaneSliceExtractor,
)

from fishstick.medical_imaging.processing3d.synthetic import (
    generate_synthetic_ct,
    generate_synthetic_mri,
    add_synthetic_lesion,
    SyntheticVolumeGenerator,
)

__all__ = [
    "VolumeResampler",
    "CropAndPad",
    "resample_volume",
    "compute_volume_statistics",
    "VolumeStatistics",
    "extract_slices_axial",
    "extract_slices_sagittal",
    "extract_slices_coronal",
    "MultiPlaneSliceExtractor",
    "generate_synthetic_ct",
    "generate_synthetic_mri",
    "add_synthetic_lesion",
    "SyntheticVolumeGenerator",
]

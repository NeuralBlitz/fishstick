# Medical Imaging

Medical image processing, segmentation, and analysis.

## Installation

```bash
pip install fishstick[medical]
```

## Overview

The `medical` module provides tools for medical imaging including segmentation models and image registration.

## Usage

```python
from fishstick.medical import UNet3D, VNet, MedicalImageLoader

# 3D segmentation
unet = UNet3D(
    in_channels=1,
    out_channels=1,
    base_channels=32
)
segmentation = unet(volume)

# Medical image loader
loader = MedicalImageLoader(
    path="data/scans/",
    transform=NormalizeMedicalImage()
)
```

## Models

| Model | Description |
|-------|-------------|
| `UNet3D` | 3D U-Net for volumetric segmentation |
| `VNet` | V-Net for medical image segmentation |

## Utilities

| Utility | Description |
|---------|-------------|
| `MedicalImageLoader` | Medical image dataset loader |
| `NormalizeMedicalImage` | Normalization for medical images |
| `ImageRegistration` | Image registration |

## Examples

See `examples/medical/` for complete examples.

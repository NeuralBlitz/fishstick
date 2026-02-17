# Computer Vision

Vision models, augmentations, and image processing.

## Installation

```bash
pip install fishstick[vision]
```

## Overview

The `vision` module provides computer vision models including Vision Transformers (ViT), object detection, and geometric augmentations.

## Usage

```python
from fishstick.vision import VisionTransformer, ObjectDetector, GeometricAugmentation

# Vision Transformer
vit = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12
)

# Object detection
detector = ObjectDetector(num_classes=80)
boxes, labels, scores = detector.predict(images)

# Augmentation
aug = GeometricAugmentation()
augmented = aug(images)
```

## Models

| Model | Description |
|-------|-------------|
| `VisionTransformer` | ViT (Vision Transformer) |
| `ViT` | Standard ViT |
| `DeiT` | Data-efficient Image Transformer |
| `SwinTransformer` | Swin Transformer |
| `CvT` | Convolutional Vision Transformer |
| `ObjectDetector` | Object detection model |

## Utilities

| Utility | Description |
|---------|-------------|
| `PatchEmbedding` | Patch embedding layer |
| `VisionTransformerBlock` | ViT transformer block |
| `GeometricAugmentation` | Geometric augmentations |
| `ImageAugmentationPipeline` | Augmentation pipeline |

## Examples

See `examples/vision/` for complete examples.

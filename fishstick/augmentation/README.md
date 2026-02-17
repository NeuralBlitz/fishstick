# Data Augmentation

State-of-the-art data augmentation techniques for images and other data types.

## Installation

```bash
pip install fishstick[augmentation]
```

## Overview

The `augmentation` module provides implementations of modern augmentation techniques for improving model generalization without requiring additional labeled data.

## Usage

```python
from fishstick.augmentation import MixUp, CutMix, RandAugment, AugmentationPipeline

# MixUp augmentation
mixed_images, mixed_labels = MixUp(alpha=0.2)(images, labels)

# CutMix augmentation
cut_images, cut_labels = CutMix(alpha=1.0)(images, labels)

# RandAugment
augmented = RandAugment(img_size=224, magnitude=9)(images)

# Compose multiple augmentations
pipeline = AugmentationPipeline([
    RandAugment(),
    CutMix(alpha=0.5),
])
```

## Available Augmentations

| Augmentation | Description |
|--------------|-------------|
| `CutOut` | Randomly mask square regions |
| `MixUp` | Linear interpolation between images and labels |
| `CutMix` | Cut and paste patches between images |
| `RandAugment` | Simplified AutoAugment |
| `TrivialAugmentWide` | RandAugment variant |
| `AugmentationPipeline` | Compose multiple augmentations |
| `MixupCutmixCollator` | Collator for mixed training |

## Examples

See `examples/augmentation/` for complete examples.

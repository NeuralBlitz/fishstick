# Multi-Modal Learning

Fusion, alignment, and encoding for multi-modal learning tasks.

## Installation

```bash
pip install fishstick
```

## Quick Start

```python
import torch
from fishstick.multimodal import (
    CrossModalAttention,
    MultiModalEncoder,
    EarlyFusion,
    LateFusion,
)

# Create a multi-modal encoder
encoder = MultiModalEncoder(
    image_dim=768,
    text_dim=768,
    audio_dim=768,
    output_dim=512,
)

# Process different modalities
image_features = torch.randn(1, 196, 768)
text_features = torch.randn(1, 512, 768)
audio_features = torch.randn(1, 100, 768)

# Fuse modalities
fused = encoder(image_features, text_features, audio_features)
```

## API Reference

### Fusion

| Class | Description |
|-------|-------------|
| `EarlyFusion` | Concatenate features early in the network |
| `LateFusion` | Combine predictions from separate encoders |
| `CrossModalAttention` | Cross-attention between modalities |
| `ModalityAlignment` | Align different modalities in common space |

### Encoders

| Class | Description |
|-------|-------------|
| `ImageEncoder` | Encode images to feature vectors |
| `TextEncoder` | Encode text to feature vectors |
| `AudioEncoder` | Encode audio to feature vectors |
| `MultiModalEncoder` | Combined encoder for multiple modalities |

## Examples

### Early Fusion

```python
from fishstick.multimodal import EarlyFusion, ImageEncoder, TextEncoder

image_enc = ImageEncoder(output_dim=512)
text_enc = TextEncoder(output_dim=512)

fusion = EarlyFusion(input_dims=[512, 512], output_dim=256)

image_feat = image_enc(image)
text_feat = text_enc(text)
fused = fusion([image_feat, text_feat])
```

### Cross-Modal Attention

```python
from fishstick.multimodal import CrossModalAttention

cross_attn = CrossModalAttention(
    query_dim=512,
    context_dim=768,
    num_heads=8,
)

# Query from one modality, attend to another
output = cross_attn(query=image_features, context=text_features)
```

### Late Fusion for Classification

```python
from fishstick.multimodal import LateFusion, ImageEncoder, TextEncoder

image_enc = ImageEncoder(output_dim=256)
text_enc = TextEncoder(output_dim=256)

fusion = LateFusion(input_dims=[256, 256], num_classes=10)

image_pred = image_enc(image)
text_pred = text_enc(text)
final_pred = fusion([image_pred, text_pred])
```

### Modality Alignment

```python
from fishstick.multimodal import ModalityAlignment

aligner = ModalityAlignment(
    source_dim=768,
    target_dim=512,
    temperature=0.1,
)

aligned = aligner(source_features, target_features)
```

# Self-Supervised Learning

Contrastive learning, masked autoencoders, and self-supervised pretraining.

## Installation

```bash
pip install fishstick[selfsupervised]
```

## Overview

The `selfsupervised` module provides implementations of popular self-supervised learning methods including SimCLR, BYOL, MAE, and others.

## Usage

```python
from fishstick.selfsupervised import SimCLR, BYOL, MAE

# SimCLR
simclr = SimCLR(
    encoder=encoder,
    projection_dim=128,
    temperature=0.1
)
loss = simclr(anchor, positive)

# BYOL
byol = BYOL(encoder, projection_dim=256)
loss = byol(online_view, target_view)

# Masked Autoencoder
mae = MAE(
    encoder=encoder,
    decoder=decoder,
    mask_ratio=0.75
)
loss = mae(images)
```

## Methods

| Method | Description |
|--------|-------------|
| `SimCLR` | Simple Contrastive Learning |
| `BYOL` | Bootstrap Your Own Latent |
| `SimSiam` | Siamese networks with stop-gradient |
| `MoCo` | Momentum Contrast |
| `MAE` | Masked Autoencoder |
| `SimMIM` | Simple Masked Image Modeling |
| `MaskedAutoencoder` | General masked autoencoder |
| `DeepInfoMax` | Deep InfoMax |
| `BarlowTwins` | Barlow Twins redundancy reduction |

## Loss Functions

| Loss | Description |
|------|-------------|
| `NT_XentLoss` | Normalized Temperature-scaled Cross Entropy |
| `SimSiamLoss` | SimSiam loss |
| `BYOLLoss` | BYOL loss |

## Examples

See `examples/selfsupervised/` for complete examples.

# Neural Rendering

NeRF, Instant NGP, and volume rendering for 3D scene reconstruction.

## Installation

```bash
pip install fishstick[rendering]
```

## Overview

The `rendering` module provides neural rendering implementations including Neural Radiance Fields (NeRF), Instant NGP, and volume rendering.

## Usage

```python
from fishstick.rendering import NeRF, InstantNGP, RayMarcher, VolumeRenderer

# NeRF model
nerf = NeRF(
    position_dim=63,
    direction_dim=27,
    hidden_dim=256
)

# Render ray
ray_marcher = RayMarcher()
rendered = ray_marcher(nerf, ray_origin, ray_direction)

# Instant NGP
ingp = InstantNGP(
    num_levels=16,
    max_resolution=1024,
    feature_dim=2
)
```

## Models

| Model | Description |
|-------|-------------|
| `NeRF` | Neural Radiance Fields |
| `InstantNGP` | Instant Neural Graphics Primitives |
| `DVRNetwork` | Differentiable Volume Rendering |

## Rendering

| Class | Description |
|-------|-------------|
| `RayMarcher` | Ray marching for volume rendering |
| `VolumeRenderer` | Volume rendering |
| `CameraPose` | Camera pose estimation |
| `HashGridEncoding` | Hash grid encoding |
| `GLOEmbeddings` | Generative latent optimizations |
| `SDFNetwork` | Signed Distance Function network |

## Examples

See `examples/rendering/` for complete examples.

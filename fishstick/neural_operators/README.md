# Neural Operators Module

Implementation of neural operators for learning mappings between function spaces.

## Overview

Neural operators learn solutions to PDEs and map between infinite-dimensional function spaces:

- **Fourier Neural Operator (FNO)**: FFT-based operator learning
- **DeepONet**: Deep operator network
- **Multi-scale**: Multi-resolution operator learning
- **PDE求解**: Pre-built PDE solvers

## Installation

```bash
pip install torch numpy scipy
```

## Quick Start

### Fourier Neural Operator

```python
import torch
from fishstick.neural_operators import FNO2d, SpectralConv2d

# FNO for 2D PDE solving
fno = FNO2d(
    in_channels=3,
    out_channels=1,
    hidden_channels=64,
    n_modes=(16, 16),
    n_layers=4
)

# Input: [batch, channels, x, y]
x = torch.randn(8, 3, 64, 64)
y = fno(x)  # [8, 1, 64, 64]
```

### DeepONet

```python
from fishstick.neural_operators import DeepONet, BranchNet, TrunkNet

# Branch net for encoding input functions
branch = BranchNet(
    input_dim=1,
    branch_dims=[64, 128, 256]
)

# Trunk net for encoding sensor locations
trunk = TrunkNet(
    sensor_dim=2,
    trunk_dims=[64, 128, 256]
)

# Combined DeepONet
deeponet = DeepONet(
    branch_net=branch,
    trunk_net=trunk,
    output_dim=1
)

# u: function values at sensors, y: sensor locations
u = torch.randn(32, 10)   # 10 sensors
y = torch.randn(32, 10, 2) # sensor locations
output = deeponet(u, y)
```

### Multi-scale Operator

```python
from fishstick.neural_operators import (
    MultiScaleOperator,
    WaveletTransform,
    AdaptiveMeshing
)

# Multi-scale operator
msop = MultiScaleOperator(
    n_scales=3,
    channels=64,
    wavelet='haar'
)

x = torch.randn(8, 64, 128, 128)
output = msop(x)
```

## API Reference

| Class | Description |
|-------|-------------|
| `FNO1d` | 1D Fourier Neural Operator |
| `FNO2d` | 2D Fourier Neural Operator |
| `FNO3d` | 3D Fourier Neural Operator |
| `DeepONet` | Deep Operator Network |
| `ONet` | Operator Network |
| `MultiScaleOperator` | Multi-scale operator |
| `GraphOperator` | Graph-based operator |

## References

- Li et al., "Fourier Neural Operator" (NeurIPS 2020)
- Lu et., "DeepONet" (Nature 2021)

## License

MIT License

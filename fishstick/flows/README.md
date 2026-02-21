# Normalizing Flows

A comprehensive implementation of normalizing flow models for density estimation and generative modeling. This module provides state-of-the-art flow-based generative models including RealNVP, Glow, and MAF.

## Overview

Normalizing flows are a class of generative models that learn to transform a simple base distribution (typically Gaussian) into a complex data distribution through a sequence of invertible transformations. This module implements:

- **RealNVP**: Real-valued Non-Volume Preserving flows with affine coupling layers
- **Glow**: Generative Flow with invertible 1×1 convolutions and ActNorm
- **MAF**: Masked Autoregressive Flow for density estimation
- **Conditional Flows**: Conditional normalizing flows for conditioned generation
- **MADE**: Masked Autoencoder for Distribution Estimation

## Installation

```bash
pip install fishstick
```

Or install from source:

```bash
git clone https://github.com/yourorg/fishstick.git
cd fishstick
pip install -e .
```

## Quick Start

```python
import torch
from fishstick.flows import RealNVP, Glow, MAF

# Create a RealNVP flow
flow = RealNVP(dim=10, n_coupling=8, hidden_dim=256)

# Sample from the flow
samples = flow.sample(n_samples=100, device='cuda')
print(f"Sampled shape: {samples.shape}")  # [100, 10]

# Compute log probability
x = torch.randn(100, 10, device='cuda')
log_prob = flow.log_prob(x)
print(f"Log probability shape: {log_prob.shape}")  # [100]

# Train the flow
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)

for epoch in range(100):
    optimizer.zero_grad()
    x = torch.randn(256, 10, device='cuda')  # Your data
    log_prob = flow.log_prob(x)
    loss = -log_prob.mean()  # Negative log likelihood
    loss.backward()
    optimizer.step()
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `CouplingLayer` | Affine coupling layer for RealNVP |
| `RealNVP` | Real-valued Non-Volume Preserving flow |
| `InvertibleLinear` | Invertible 1×1 convolution layer |
| `ActNorm` | Activation normalization (Glow) |
| `GlowStep` | Single Glow flow step |
| `Glow` | Complete Glow model |
| `MADE` | Masked Autoencoder for Distribution Estimation |
| `MAF` | Masked Autoregressive Flow |
| `ConditionalNormalizingFlow` | Conditional flow p(x\|y) |
| `ConditionalCouplingLayer` | Conditional coupling layer |

### RealNVP

```python
RealNVP(
    dim: int,                    # Data dimensionality
    n_coupling: int = 8,        # Number of coupling layers
    hidden_dim: int = 256,      # Hidden dimension for MLPs
    n_hidden: int = 2,          # Number of hidden layers
    base_dist: str = "gaussian"  # Base distribution type
)
```

**Methods:**
- `forward(x)` → `(z, logdet)`: Transform data to latent space
- `inverse(z)` → `(x, logdet)`: Transform latent to data space
- `sample(n_samples, device)` → `Tensor`: Sample from the flow
- `log_prob(x)` → `Tensor`: Compute log probability

### Glow

```python
Glow(
    dim: int,                    # Data dimensionality
    n_levels: int = 3,          # Number of flow levels
    n_steps_per_level: int = 4, # Steps per level
    hidden_dim: int = 256       # Hidden dimension
)
```

### MAF

```python
MAF(
    dim: int,                    # Data dimensionality
    n_mades: int = 5,           # Number of MADE layers
    hidden_dim: int = 256       # Hidden dimension
)
```

### Conditional Flow

```python
# Create conditional flow
flow = ConditionalNormalizingFlow(
    x_dim=10,       # Data dimension
    y_dim=5,        # Condition dimension
    hidden_dim=256,
    n_coupling=6
)

# Sample given condition
y = torch.randn(32, 5)  # Conditions
x = flow.sample(y)      # [32, 10]

# Compute conditional log probability
x = torch.randn(32, 10)
log_prob = flow.forward(x, y)
```

## Code Examples

### Training RealNVP on 2D Data

```python
import torch
import torch.nn as nn
from fishstick.flows import RealNVP
import numpy as np

# Generate 2D toy data (e.g., two moons)
def generate_moons(n_samples, noise=0.05):
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 0.5 - np.sin(np.linspace(0, np.pi, n_samples_in))
    
    data = np.array([
        np.append(outer_circ_x, inner_circ_x),
        np.append(outer_circ_y, inner_circ_y)
    ]).T + noise * np.random.randn(n_samples, 2)
    return torch.tensor(data, dtype=torch.float32)

# Create flow
flow = RealNVP(dim=2, n_coupling=6, hidden_dim=128)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

# Training loop
for epoch in range(1000):
    x = generate_moons(256)
    log_prob = flow.log_prob(x)
    loss = -log_prob.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Sample from trained flow
samples = flow.sample(n_samples=1000)
print(f"Generated samples shape: {samples.shape}")
```

### Using Glow for Image Generation

```python
import torch
from fishstick.flows import Glow

# Glow for small images (e.g., 4x4 with 3 channels)
dim = 4 * 4 * 3  # 48
flow = Glow(dim=dim, n_levels=2, n_steps_per_level=4)

# Train on image data
# images: [batch, 3, 4, 4]
images = torch.randn(32, 3, 4, 4)
x = images.view(32, -1)

log_prob = flow.log_prob(x)
loss = -log_prob.mean()

# Sample
samples = flow.sample(n_samples=16)
samples = samples.view(16, 3, 4, 4)
```

### Flow Composition

```python
import torch
from fishstick.flows import RealNVP, Glow, MAF

# Compose multiple flows
flow1 = RealNVP(dim=10, n_coupling=4)
flow2 = Glow(dim=10, n_levels=2, n_steps_per_level=2)

# Combined transformation (chaining flows)
x = torch.randn(32, 10)

# Apply flow1
z1, ld1 = flow1.forward(x)
# Apply flow2
z2, ld2 = flow2.forward(z1)

# Total log probability
log_prob = -0.5 * (z2**2).sum(dim=-1) + ld1 + ld2
```

## Custom Coupling Layers

```python
from fishstick.flows import CouplingLayer

# Custom coupling layer with different architecture
coupling = CouplingLayer(
    dim=10,
    hidden_dim=512,
    n_hidden=3,
    mask_type="alternating"  # or "half"
)

# Forward pass (inverse for sampling)
y, logdet = coupling(x)

# Inverse pass (forward for density estimation)
x_reconstructed, logdet_inv = coupling.inverse(y)
```

## Invertible Linear Layers

```python
from fishstick.flows import InvertibleLinear

# Create invertible linear layer
inv_linear = InvertibleLinear(dim=10)

# Forward
y, ld = inv_linear(x)

# Inverse
x_reconstructed, ld_inv = inv_linear.inverse(y)
```

## Loss Functions

The flow models use negative log-likelihood as the training objective:

```python
def nll_loss(flow, x):
    """Negative log-likelihood loss."""
    log_prob = flow.log_prob(x)
    return -log_prob.mean()
```

## Performance Tips

1. **Batch Size**: Larger batches provide better gradient estimates
2. **Hidden Dimensions**: Increase for more complex distributions
3. **Number of Flows**: More coupling layers = more expressivity but slower
4. **ActNorm**: Use Glow for better training stability on image data
5. **Memory**: Use `checkpoint` for large models

## Citation

If you use this code, please cite:

```bibtex
@article{Dinh2017RealNVP,
  title={RealNVP: Real-valued Non-Volume Preserving Flows},
  author={Dinh, Laurent and Sohl-Dickstein, Jascha and Bengio, Samy},
  journal={arXiv preprint arXiv:1605.08803},
  year={2017}
}

@article{Kingma2018Glow,
  title={Glow: Generative Flow with Invertible 1x1 Convolutions},
  author={Kingma, Diederik P and Dhariwal, Prafulla},
  journal={arXiv preprint arXiv:1807.03039},
  year={2018}
}

@article{Papamakarios2017MAF,
  title={Masked Autoregressive Flow for Density Estimation},
  author={Papamakarios, George and Pavlakos, Theo and Murray, Iain},
  journal={Advances in Neural Information Processing Systems},
  year={2017}
}
```

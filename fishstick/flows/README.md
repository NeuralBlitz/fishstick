# flows - Normalizing Flows Module

## Overview

The `flows` module implements normalizing flow models for density estimation and generative modeling, including RealNVP, Glow, MAF, and continuous normalizing flows.

## Purpose and Scope

This module enables:
- Density estimation via invertible transformations
- Generative modeling with exact likelihood
- Variational inference
- Conditional generation

## Key Classes and Functions

### Coupling Layers

#### `CouplingLayer`
Affine coupling layer for RealNVP.

```python
from fishstick.flows import CouplingLayer

layer = CouplingLayer(
    dim=784,
    hidden_dim=256,
    n_hidden=2,
    mask_type="alternating"
)

# Forward (sampling direction: z → x)
z, logdet = layer(x)

# Inverse (density estimation: x → z)
x, logdet = layer.inverse(z)
```

#### `ActNorm`
Activation normalization (from Glow).

```python
from fishstick.flows import ActNorm

actnorm = ActNorm(dim=784)
x, logdet = actnorm(x)
```

#### `InvertibleLinear`
1×1 invertible convolution.

```python
from fishstick.flows import InvertibleLinear

linear = InvertibleLinear(dim=64)
y, logdet = linear(x)
```

### Flow Models

#### `RealNVP`
Real-valued Non-Volume Preserving flow.

```python
from fishstick.flows import RealNVP

flow = RealNVP(
    dim=784,
    n_coupling=8,
    hidden_dim=256,
    n_hidden=2,
    base_dist="gaussian"
)

# Sample
samples = flow.sample(n_samples=100)

# Log probability
log_prob = flow.log_prob(x)
```

#### `Glow`
Glow: Generative Flow with Invertible 1×1 Convolutions.

```python
from fishstick.flows import Glow

glow = Glow(
    dim=784,
    n_levels=3,
    n_steps_per_level=4,
    hidden_dim=256
)

samples = glow.sample(100)
log_prob = glow.log_prob(x)
```

#### `MAF`
Masked Autoregressive Flow.

```python
from fishstick.flows import MAF

maf = MAF(
    dim=784,
    n_mades=5,
    hidden_dim=256
)

samples = maf.sample(100)
log_prob = maf.log_prob(x)
```

#### `MADE`
Masked Autoencoder for Distribution Estimation.

```python
from fishstick.flows import MADE

made = MADE(
    input_dim=784,
    hidden_dim=256,
    n_hidden=2,
    num_masks=1
)

mu, log_scale = made(x)
```

### Conditional Flows

#### `ConditionalNormalizingFlow`
Conditional normalizing flow p(x|y).

```python
from fishstick.flows import ConditionalNormalizingFlow

flow = ConditionalNormalizingFlow(
    x_dim=100,
    y_dim=10,
    hidden_dim=256,
    n_coupling=6
)

# Sample conditioned on y
x = flow.sample(y)

# Log probability
log_prob = flow.log_prob(x, y)
```

## Mathematical Background

### Normalizing Flows
Transform a simple distribution q_0 to a complex distribution q_K:
- z_K = f_K ∘ ... ∘ f_1(z_0)
- log q_K(z_K) = log q_0(z_0) - Σ log |det ∂f_k/∂z|

### Affine Coupling
y_1 = x_1
y_2 = x_2 ⊙ exp(s(x_1)) + t(x_1)

### Autoregressive Flows
Each dimension transformed conditioned on previous dimensions.

## Dependencies

- `torch`: PyTorch tensors
- `numpy`: Numerical operations

## Usage Examples

### Density Estimation

```python
from fishstick.flows import RealNVP
import torch

# Create flow
flow = RealNVP(dim=2, n_coupling=6, hidden_dim=64)

# Training data (e.g., two moons)
x_train = load_two_moons()

# Training
optimizer = torch.optim.Adam(flow.parameters())

for epoch in range(1000):
    log_prob = flow.log_prob(x_train)
    loss = -log_prob.mean()  # NLL
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Sample from trained flow
samples = flow.sample(1000)
```

### Variational Inference

```python
from fishstick.flows import RealNVP

# Flow as variational posterior
flow = RealNVP(dim=latent_dim, n_coupling=4)

# ELBO
def elbo(x):
    z = flow.sample(1)
    log_q = flow.log_prob(z)
    log_p = model.log_joint(x, z)
    return log_p - log_q
```

### Conditional Generation

```python
from fishstick.flows import ConditionalNormalizingFlow

flow = ConditionalNormalizingFlow(
    x_dim=784,  # Image
    y_dim=10,   # Class label
    hidden_dim=256
)

# Train: flow.log_prob(images, labels)
# Generate: flow.sample(label)
```

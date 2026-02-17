# neural_ode - Neural Ordinary Differential Equations Module

## Overview

The `neural_ode` module implements continuous-depth neural networks using ODE solvers, combining naturally with Hamiltonian dynamics for physics-informed learning.

## Purpose and Scope

This module enables:
- Continuous-depth neural networks
- Memory-efficient backpropagation via adjoint method
- Latent ODE models for time series
- Continuous normalizing flows (FFJORD)

## Key Classes and Functions

### Core Classes

#### `ODEFunction`
Learnable ODE dynamics function dz/dt = f(z, t, θ).

```python
from fishstick.neural_ode import ODEFunction

f = ODEFunction(
    dim=64,
    hidden_dim=200,
    n_layers=3,
    time_invariant=True
)

dzdt = f(t, z)
```

#### `NeuralODE`
Neural ODE layer with various integration methods.

```python
from fishstick.neural_ode import NeuralODE

ode = NeuralODE(
    odefunc=f,
    t_span=(0.0, 1.0),
    method="dopri5",
    rtol=1e-5,
    atol=1e-6,
    adjoint=True
)

z1 = ode(z0)  # Integrate from t0 to t1
```

### Extended Architectures

#### `AugmentedNeuralODE`
Augmented Neural ODE with higher-dimensional space.

```python
from fishstick.neural_ode import AugmentedNeuralODE

aug_ode = AugmentedNeuralODE(
    dim=64,
    augment_dim=10,
    hidden_dim=200
)

output = aug_ode(x)
```

#### `LatentODE`
Latent ODE model for time series.

```python
from fishstick.neural_ode import LatentODE

model = LatentODE(
    input_dim=10,
    latent_dim=20,
    encoder_hidden=200,
    decoder_hidden=200,
    ode_hidden=200
)

result = model(x)
# Returns: x_recon, z0, z1, mu, logvar

# Sample trajectory
trajectory = model.sample_trajectory(x0, t_span=(0, 10))
```

#### `SecondOrderNeuralODE`
Second-order Neural ODE for learning second-order dynamics.

```python
from fishstick.neural_ode import SecondOrderNeuralODE

ode2 = SecondOrderNeuralODE(dim=10, hidden_dim=200)
q1, p1 = ode2(q0, p0)  # Position and velocity
```

#### `ContinuousNormalizingFlow`
Continuous normalizing flow using FFJORD.

```python
from fishstick.neural_ode import ContinuousNormalizingFlow

cnf = ContinuousNormalizingFlow(
    dim=64,
    hidden_dim=200,
    n_layers=3
)

z, logdet = cnf(x)
x_recon = cnf.inverse(z)
```

### Factory Function

#### `create_neural_ode_model`
Factory function to create Neural ODE model.

```python
from fishstick.neural_ode import create_neural_ode_model

model = create_neural_ode_model(
    dim=64,
    hidden_dim=200,
    augment=False
)
```

## Mathematical Background

### Neural ODE
Forward pass: z(t1) = z(t0) + ∫_{t0}^{t1} f(z(t), t, θ) dt

### Adjoint Method
Memory-efficient backpropagation by solving:
dL/dθ = -∫_{t1}^{t0} a(t)^T ∂f/∂θ dt

where a(t) = ∂L/∂z(t)

## Dependencies

- `torch`: PyTorch tensors
- `torchdiffeq`: ODE solvers (required)

## Usage Examples

### Basic Neural ODE

```python
from fishstick.neural_ode import ODEFunction, NeuralODE

# Create ODE function
f = ODEFunction(dim=64, hidden_dim=128)

# Create Neural ODE layer
ode_layer = NeuralODE(
    odefunc=f,
    t_span=(0, 1),
    method="dopri5",
    adjoint=True
)

# Use as layer in network
class ODENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.ode = NeuralODE(ODEFunction(64, 128))
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.ode(x)
        return self.fc2(x)
```

### Latent ODE for Time Series

```python
from fishstick.neural_ode import LatentODE

model = LatentODE(input_dim=10, latent_dim=20)

# Train
result = model(observations)
loss = reconstruction_loss + kl_loss

# Forecast
future = model.sample_trajectory(x0, t_span=(0, 20))
```

# core - Core Types and Utilities Module

## Overview

The `core` module provides fundamental types, mathematical structures, and base classes used throughout fishstick. It defines core abstractions for manifolds, metrics, phase space states, and verification.

## Purpose and Scope

This module enables:
- Core type definitions for tensors and morphisms
- Statistical manifold with Fisher information geometry
- Metric tensor and connection structures
- Phase space state representations
- Conservation law tracking
- Verification certificates

## Key Classes and Functions

### Core Types (`types.py`)

#### `Shape`
Tensor shape with semantic labels.

```python
from fishstick.core import Shape

shape = Shape(dims=(784,), labels=("features",))
```

#### `Morphism`
Protocol for categorical morphisms.

```python
from fishstick.core import Morphism

def f(x: Tensor) -> Tensor:
    return x * 2

# f is a Morphism[Tensor, Tensor]
```

#### `MetricTensor`
Riemannian metric tensor g_ij.

```python
from fishstick.core import MetricTensor
import torch

metric = MetricTensor(torch.eye(3))
inverse = metric.inverse()
result = metric @ vector
```

#### `SymplecticForm`
Canonical symplectic form ω = dq ∧ dp.

```python
from fishstick.core import SymplecticForm

form = SymplecticForm(dim=2)
J = form.matrix  # [0, I; -I, 0]
```

#### `Connection`
Affine connection ∇ on a manifold.

```python
from fishstick.core import Connection

conn = Connection(christoffel=Gamma)
transported = conn.parallel_transport(v, along=path)
```

#### `ProbabilisticState`
State in statistical manifold with uncertainty.

```python
from fishstick.core import ProbabilisticState

state = ProbabilisticState(
    mean=torch.zeros(10),
    covariance=torch.eye(10),
    entropy=2.3
)

samples = state.sample(n=100)
```

#### `PhaseSpaceState`
State in Hamiltonian phase space (q, p).

```python
from fishstick.core import PhaseSpaceState

state = PhaseSpaceState(
    q=torch.randn(1, 3),  # Coordinates
    p=torch.randn(1, 3)   # Momenta
)

z = state.stack()  # Concatenate q and p
recovered = PhaseSpaceState.unstack(z)
```

#### `ConservationLaw`
Noether-derived conservation law.

```python
from fishstick.core import ConservationLaw, PhaseSpaceState

energy_law = ConservationLaw(
    name="energy",
    quantity_fn=lambda s: (s.p ** 2).sum() / 2,
    symmetry_group="time_translation"
)

is_conserved = energy_law.check(state_before, state_after)
```

#### `VerificationCertificate`
Cryptographic certificate of verified property.

```python
from fishstick.core import VerificationCertificate

cert = VerificationCertificate(
    property_name="robustness",
    is_verified=True,
    proof_hash="abc123",
    details={"epsilon": 0.1, "lipschitz_bound": 1.5}
)
```

#### `Module`
Base module with formal verification support.

```python
from fishstick.core import Module

class VerifiedLayer(Module):
    @property
    def lipschitz_constant(self) -> float:
        return 1.0  # Upper bound
    
    def forward(self, x: Tensor) -> Tensor:
        return x
    
    def verify_robustness(self, x, epsilon=0.1):
        # Returns VerificationCertificate
        return super().verify_robustness(x, epsilon)
```

### Statistical Manifold (`manifold.py`)

#### `StatisticalManifold`
Statistical manifold with Fisher-Rao metric.

```python
from fishstick.core import StatisticalManifold

manifold = StatisticalManifold(dim=10, alpha=0.0)

# Compute Fisher information
fisher = manifold.fisher_information(
    params=params,
    log_prob_fn=log_prob
)

# Natural gradient
nat_grad = manifold.natural_gradient(params, euclidean_grad, fisher)

# Geodesic path
path = manifold.geodesic(start=params, end=target)

# KL divergence
kl = manifold.kl_divergence(p_params, q_params, sample_fn)

# Wasserstein distance
w2 = manifold.wasserstein_distance(p_state, q_state)
```

#### `InformationGeometryLayer`
Neural network layer respecting information geometry.

```python
from fishstick.core import InformationGeometryLayer

layer = InformationGeometryLayer(in_features=784, out_features=256)
output = layer(x)

# Natural gradient update
layer.natural_gradient_step(loss, lr=0.01, damping=1e-4)
```

## Mathematical Background

### Fisher Information Geometry
The Fisher metric defines a Riemannian structure on parameter space:
- g_ij(θ) = E[∂_i log p · ∂_j log p]
- Natural gradient: G^{-1} ∇L

### Symplectic Geometry
Hamiltonian mechanics uses symplectic form ω:
- ω = dq ∧ dp
- Preserves phase space volume

### Conservation Laws
Noether's theorem: continuous symmetries → conserved quantities:
- Time translation → Energy
- Space translation → Momentum
- Rotation → Angular momentum

## Dependencies

- `torch`: PyTorch tensors and modules
- `numpy`: Numerical operations
- `dataclasses`: For data structures

## Usage Examples

### Hamiltonian Dynamics

```python
from fishstick.core import PhaseSpaceState, SymplecticForm
import torch

# Initialize state
state = PhaseSpaceState(
    q=torch.randn(1, 3),
    p=torch.randn(1, 3)
)

# Symplectic form for integration
omega = SymplecticForm(dim=3)
J = omega.matrix  # For symplectic integration
```

### Information Geometry Optimization

```python
from fishstick.core import StatisticalManifold

manifold = StatisticalManifold(dim=model_dim)

for x, y in dataloader:
    loss = model.loss(x, y)
    euclidean_grad = torch.autograd.grad(loss, model.parameters())
    
    # Use natural gradient for faster convergence
    nat_grad = manifold.natural_gradient(params, euclidean_grad)
    update_parameters(nat_grad)
```

### Verification

```python
from fishstick.core import Module, VerificationCertificate

class VerifiedNetwork(Module):
    @property
    def lipschitz_constant(self):
        return self._compute_lipschitz()
    
    def forward(self, x):
        return self._forward(x)

model = VerifiedNetwork()
cert = model.verify_robustness(x, epsilon=0.1)

if cert.is_verified:
    print(f"Model is robust to ε={epsilon} perturbations")
```

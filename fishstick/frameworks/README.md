# Frameworks Module

## Overview

Unified intelligence framework implementations combining categorical, geometric, thermodynamic, and verification approaches into complete neural architectures. Contains the core frameworks: UniIntelli, HSCA, UIA, UIF, UIS, and their variants.

## Purpose and Scope

- Complete unified intelligence architectures
- Composable framework building blocks
- Factory functions for model creation
- Integration of all fishstick modules

## Core Frameworks

### UniIntelli Framework

Categorical-Geometric-Thermodynamic synthesis with:
- Categorical Information Manifold (CIM)
- Sheaf-Optimized Attention (SOA)
- Thermodynamic Gradient Flow (TGF)
- Automated Formal Synthesis Pipeline (AFSP)

```python
from fishstick.frameworks import create_uniintelli

model = create_uniintelli(input_dim=784, output_dim=10)
output = model(x)

# Train with thermodynamic gradient flow
history = model.train_with_tgf(dataloader, n_epochs=10, lr=1e-3)
```

### HSCA Framework (Holo-Symplectic Cognitive Architecture)

Hamiltonian dynamics with sheaf-theoretic representations:
- Hamiltonian Sheaf Network (HSN)
- Symplectic integration for energy conservation
- RG pooling for hierarchical abstraction

```python
from fishstick.frameworks import create_hsca

model = create_hsca(input_dim=784, output_dim=10)
output = model(x)

# Verify energy conservation
verification = model.verify_conservation()
print(f"Energy violation: {verification['energy_violation']}")
```

### UIA Framework (Unified Intelligence Architecture)

Integrated architecture with:
- Categorical-Hamiltonian Neural Process (CHNP)
- RG-aware Autoencoder (RGA-AE)
- Sheaf-Theoretic Transformer (S-TF)
- Dependently-Typed Learner (DTL)

```python
from fishstick.frameworks import create_uia

model = create_uia(input_dim=784, output_dim=10, hidden_dim=256)
output = model(x)
```

### UIF Framework (Unified Intelligence Framework)

Four-layer architecture:
- Layer I: Category-Theoretic Composition Engine
- Layer II: Geometric & Topological Representation
- Layer III: Dynamical Inference via Variational Principles
- Layer IV: Verified Decision Logic via Type Theory

```python
from fishstick.frameworks import create_uif

model = create_uif(input_dim=784, output_dim=10, latent_dim=128)
output = model(x)
```

### UIS Framework (Unified Intelligence Synthesis)

Components:
- Categorical Quantum-Inspired Data Representation
- RG-Guided Deep Architecture
- Information-Geometric Optimization
- Neuro-Symbolic Causal Engine

```python
from fishstick.frameworks import create_uis

model = create_uis(input_dim=784, output_dim=10, n_rules=10)
output = model(x)
```

## Additional Frameworks

The module includes numerous variant frameworks:

| Framework | Description |
|-----------|-------------|
| ToposFormer | Topos-theoretic transformer architecture |
| CRLS | Categorical reinforcement learning system |
| SCIF | Stochastic causal inference framework |
| UIF variants | Specialized UIF configurations (I through V) |
| UIS variants | Specialized UIS configurations (J, N) |
| UIA variants | Specialized UIA configurations (K, M, O) |

```python
from fishstick.frameworks import (
    create_toposformer,
    create_crls,
    create_scif_z,
    create_uif_i,
    create_uis_j,
    create_uia_k
)

# Create specialized models
topos = create_toposformer(input_dim=784, output_dim=10)
crls = create_crls(input_dim=784, output_dim=10)
scif = create_scif_z(input_dim=784, output_dim=10)
```

## Architectural Components

### HamiltonianSheafLayer

Symplectic integration with energy conservation.

```python
from fishstick.frameworks.hsca import HamiltonianSheafLayer

layer = HamiltonianSheafLayer(dim=128, dt=0.01)
q_new, p_new = layer(q, p)

# Energy conservation loss
loss = layer.energy_conservation_loss(q_before, p_before, q_after, p_after)
```

### CategoricalInformationManifold

Higher-categorical space for neural architectures.

```python
from fishstick.frameworks.uniintelli import CategoricalInformationManifold

manifold = CategoricalInformationManifold(dim=128, n_levels=3)
manifold.add_object(data_sheaf)
manifold.add_morphism(functor)

# Compose morphisms along path
composed = manifold.compose_path([0, 1, 2])
```

### QuantumInspiredRepresentation

Complex-valued quantum-inspired encoding.

```python
from fishstick.frameworks.uis import QuantumInspiredRepresentation

rep = QuantumInspiredRepresentation(input_dim=100, latent_dim=64)
z_complex = rep(x)  # Complex tensor
z_real = rep.to_real(z_complex)  # Real concatenation
```

### NeuroSymbolicEngine

Neuro-symbolic reasoning with rule attention.

```python
from fishstick.frameworks.uis import NeuroSymbolicEngine

engine = NeuroSymbolicEngine(dim=64, n_rules=10)
reasoned = engine(query)
```

## Dependencies

All frameworks integrate components from:
- `fishstick.categorical` - Category theory structures
- `fishstick.geometric` - Geometric representations
- `fishstick.dynamics` - Hamiltonian and thermodynamic dynamics
- `fishstick.verification` - Dependently-typed verification
- `fishstick.rg` - Renormalization group autoencoders
- `fishstick.sheaf` - Sheaf attention mechanisms

## Usage Examples

### Complete Training Pipeline

```python
from fishstick.frameworks import create_uniintelli
import torch
import torch.nn.functional as F

model = create_uniintelli(input_dim=784, output_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
```

### Thermodynamic Training

```python
from fishstick.frameworks import create_uniintelli

model = create_uniintelli(input_dim=784, output_dim=10)
history = model.train_with_tgf(
    dataloader,
    n_epochs=10,
    lr=1e-3
)

print(f"Final efficiency: {history['efficiency'][-1]}")
```

### Energy Conservation Verification

```python
from fishstick.frameworks import create_hsca

model = create_hsca(input_dim=784, output_dim=10)
output = model(x)

# Check energy conservation
verification = model.verify_conservation()
if verification['verified']:
    print("Energy conservation verified")
```

### Multi-Framework Comparison

```python
from fishstick.frameworks import (
    create_uniintelli,
    create_hsca,
    create_uia,
    create_uif,
    create_uis
)

frameworks = {
    'UniIntelli': create_uniintelli(784, 10),
    'HSCA': create_hsca(784, 10),
    'UIA': create_uia(784, 10),
    'UIF': create_uif(784, 10),
    'UIS': create_uis(784, 10)
}

for name, model in frameworks.items():
    output = model(x)
    print(f"{name}: {output.shape}")
```

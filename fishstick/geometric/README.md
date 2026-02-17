# geometric - Geometric Deep Learning Module

## Overview

The `geometric` module provides sheaf-theoretic data representations and Fisher information geometry for geometric deep learning.

## Purpose and Scope

This module enables:
- Data sheaf structures for local-to-global consistency
- Sheaf cohomology computation
- Fisher information metric computation
- Natural gradient descent optimization

## Key Classes and Functions

### Sheaf Structures (`sheaf.py`)

#### `DataSheaf`
Data sheaf on a topological space.

```python
from fishstick.geometric import DataSheaf

sheaf = DataSheaf(
    open_cover=[[0, 1, 2], [2, 3, 4], [4, 5, 0]],
    stalk_dim=16,
    restriction_type="learned"
)

# Set local sections
sheaf.set_section(0, torch.randn(16))

# Restrict data
restricted = sheaf.restrict(from_patch=0, to_patch=1, data=data)

# Compute cohomology
betti_1, obstruction = sheaf.compute_cohomology()

# Consistency loss
loss = sheaf.consistency_loss()
```

#### `SheafCohomology`
Compute sheaf cohomology groups.

```python
from fishstick.geometric import SheafCohomology

cohomology = SheafCohomology(sheaf)
betti = cohomology.compute_betti(k=1)
obstruction = cohomology.obstruction_vector()
```

#### `SheafLayer`
Neural network layer enforcing sheaf consistency.

```python
from fishstick.geometric import SheafLayer

layer = SheafLayer(
    n_patches=10,
    feature_dim=64,
    lambda_cohomology=0.1
)

updated, loss = layer(patch_features)
```

### Fisher Geometry (`fisher.py`)

#### `FisherInformationMetric`
Fisher information metric g_ij(θ).

```python
from fishstick.geometric import FisherInformationMetric

fisher = FisherInformationMetric(damping=1e-4, ema_decay=0.99)

# Compute from log probabilities
metric = fisher.compute(log_probs, params)

# Monte Carlo estimate
metric = fisher.monte_carlo_estimate(model, data, n_samples=100)
```

#### `NaturalGradient`
Natural gradient descent optimizer.

```python
from fishstick.geometric import NaturalGradient

optimizer = NaturalGradient(
    params=model.parameters(),
    lr=0.01,
    damping=1e-4
)

loss = model.loss(x, y)
optimizer.step(loss)
```

#### `NaturalGradientOptimizer`
PyTorch optimizer implementing natural gradient.

```python
from fishstick.geometric import NaturalGradientOptimizer

optimizer = NaturalGradientOptimizer(
    model.parameters(),
    lr=0.01,
    damping=1e-4,
    ema_decay=0.99
)
```

#### `KFAC`
Kronecker-Factored Approximate Curvature.

```python
from fishstick.geometric import KFAC

optimizer = KFAC(
    model.parameters(),
    lr=0.01,
    damping=1e-4,
    factor_decay=0.95
)
```

## Mathematical Background

### Sheaf Theory
A sheaf F assigns data F(U) to open sets U with restriction maps:
- ρ_UU = id (identity)
- ρ_VW ∘ ρ_UV = ρ_UW (composition)

Sheaf cohomology H^k(X,F) measures obstruction to gluing local sections.

### Fisher Information
The Fisher metric g_ij = E[∂_i log p · ∂_j log p] defines a Riemannian structure on parameter space.

## Dependencies

- `torch`: PyTorch tensors
- `numpy`: Numerical operations
- `scipy`: Sparse matrix operations

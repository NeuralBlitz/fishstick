# Manifold Learning

Dimensionality reduction and manifold learning algorithms.

## Installation

```bash
pip install fishstick[manifold_learning]
```

## Overview

The `manifold_learning` module provides implementations of manifold learning algorithms including Isomap, LLE, and diffusion maps.

## Usage

```python
from fishstick.manifold_learning import Isomap, LLE, DiffusionMap

# Isomap
isomap = Isomap(n_neighbors=10, n_components=2)
embedding = isomap.fit_transform(X)

# Locally Linear Embedding
lle = LLE(n_neighbors=10, n_components=2)
embedding = lle.fit_transform(X)

# Diffusion Maps
dm = DiffusionMap(n_components=10, alpha=0.5)
embedding = dm.fit_transform(X)
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| `Isomap` | Isomap for nonlinear dimensionality reduction |
| `LandmarkIsomap` | Landmark Isomap for large datasets |
| `IsomapLayer` | Differentiable Isomap layer |
| `LLE` | Locally Linear Embedding |
| `ModifiedLLE` | Modified LLE |
| `HessianLLE` | Hessian LLE |
| `LTSA` | Local Tangent Space Alignment |
| `DiffusionMap` | Diffusion maps |
| `KernelPCA` | Kernel PCA |
| `HessianEigenmaps` | Hessian eigenmaps |

## Operators

| Class | Description |
|-------|-------------|
| `LaplaceBeltramiOperator` | Laplace-Beltrami operator |
| `ManifoldRegularizationLoss` | Manifold regularization loss |

## Examples

See `examples/manifold_learning/` for complete examples.

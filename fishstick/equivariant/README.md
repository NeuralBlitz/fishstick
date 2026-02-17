# equivariant - Equivariant Neural Networks Module

## Overview

The `equivariant` module provides E(n) and SE(3) equivariant neural network layers for processing point clouds, molecular systems, and 3D data while preserving geometric symmetries.

## Purpose and Scope

This module enables:
- SE(3)-equivariant message passing
- E(3)-equivariant convolutions
- Equivariant transformers for geometric data
- Molecular and crystal property prediction

## Key Classes and Functions

### Equivariant Layers

#### `SE3EquivariantLayer`
SE(3)-equivariant layer for point clouds.

```python
from fishstick.equivariant import SE3EquivariantLayer

layer = SE3EquivariantLayer(
    in_features=64,
    out_features=64,
    hidden_dim=128,
    edge_dim=16
)

# Forward pass
features_new, coords_new = layer(
    features=features,    # [n_nodes, in_features]
    coords=coords,        # [n_nodes, 3]
    edge_index=edge_index,  # [2, n_edges]
    edge_attr=edge_attr   # [n_edges, edge_dim]
)
```

#### `SE3Transformer`
SE(3)-equivariant transformer.

```python
from fishstick.equivariant import SE3Transformer

transformer = SE3Transformer(
    feature_dim=128,
    num_heads=8,
    hidden_dim=64
)

output = transformer(features, coords, edge_index)
```

#### `E3Conv`
E(3)-equivariant convolution.

```python
from fishstick.equivariant import E3Conv

conv = E3Conv(
    in_channels=64,
    out_channels=128,
    hidden_channels=32
)

output = conv(features, coords, edge_index)
```

### Basis Functions

#### `RadialBasisFunctions`
Radial basis functions for encoding distances.

```python
from fishstick.equivariant import RadialBasisFunctions

rbf = RadialBasisFunctions(
    n_rbf=50,
    cutoff=10.0,
    rbf_type="gaussian"
)

rbf_features = rbf(distances)  # [n_edges, n_rbf]
```

#### `SphericalBasisLayer`
Spherical harmonics for angular features.

```python
from fishstick.equivariant import SphericalBasisLayer

spherical = SphericalBasisLayer(l_max=2)
harmonics = spherical(direction_vectors)  # [n_edges, n_harmonics]
```

### Complete Networks

#### `EquivariantPointCloudNetwork`
Complete SE(3)-equivariant network.

```python
from fishstick.equivariant import EquivariantPointCloudNetwork

model = EquivariantPointCloudNetwork(
    in_features=5,       # Atom types
    hidden_features=128,
    out_features=1,      # Energy
    n_layers=6
)

output = model(features, coords, edge_index)
```

#### `MolecularGraphNetwork`
Network for molecular property prediction.

```python
from fishstick.equivariant import MolecularGraphNetwork

model = MolecularGraphNetwork(
    node_feature_dim=10,
    edge_feature_dim=4,
    hidden_dim=128,
    num_layers=6,
    num_tasks=1,
    readout="mean"
)

energy = model(x, pos, edge_index, edge_attr, batch)
```

#### `CrystalGraphNetwork`
Graph network for crystalline materials.

```python
from fishstick.equivariant import CrystalGraphNetwork

model = CrystalGraphNetwork(
    node_dim=10,
    hidden_dim=128,
    num_layers=4,
    max_neighbors=12
)

property_pred = model(x, pos, lattice, edge_index)
```

#### `EquivariantMolecularEnergy`
Predict molecular energy with SE(3) invariance.

```python
from fishstick.equivariant import EquivariantMolecularEnergy

model = EquivariantMolecularEnergy(
    n_atom_types=100,
    hidden_dim=128,
    n_layers=6
)

energy = model(atomic_numbers, coords, edge_index)
```

#### `TetrisNetwork`
E(3)-invariant network for Tetris classification.

```python
from fishstick.equivariant import TetrisNetwork

model = TetrisNetwork(num_classes=8)
logits = model(features, coords, edge_index, batch)
```

## Mathematical Background

### Equivariance
A function f is G-equivariant if:
- f(g·x) = g·f(x) for all g ∈ G

SE(3) equivariance ensures predictions are invariant to rotations and translations.

### Message Passing
Equivariant message passing:
- Messages computed from invariant distances
- Coordinate updates computed equivariantly

## Dependencies

- `torch`: PyTorch tensors
- `numpy`: Numerical operations
- `torch_geometric` (optional): Graph neural network utilities

## Usage Examples

### Molecular Property Prediction

```python
from fishstick.equivariant import MolecularGraphNetwork
import torch

model = MolecularGraphNetwork(
    node_feature_dim=10,
    edge_feature_dim=4,
    hidden_dim=128,
    num_layers=4
)

# Training
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    pred = model(
        batch.x,
        batch.pos,
        batch.edge_index,
        batch.edge_attr,
        batch.batch
    )
    
    loss = F.mse_loss(pred, batch.y)
    loss.backward()
    optimizer.step()
```

### Point Cloud Classification

```python
from fishstick.equivariant import EquivariantPointCloudNetwork

model = EquivariantPointCloudNetwork(
    in_features=1,
    hidden_features=64,
    out_features=10,
    n_layers=4
)

# Classify point cloud
logits = model(features, coords, edge_index)
```

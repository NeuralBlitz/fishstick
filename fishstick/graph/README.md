# graph - Graph Neural Networks Module

## Overview

The `graph` module provides geometric graph neural networks with equivariance, sheaf structure, and support for molecular and crystalline systems.

## Purpose and Scope

This module enables:
- Equivariant message passing on graphs
- Sheaf-based graph convolutions
- Geometric graph transformers
- Molecular and crystal graph networks

## Key Classes and Functions

### Message Passing

#### `EquivariantMessagePassing`
E(n)-equivariant message passing layer.

```python
from fishstick.graph import EquivariantMessagePassing

layer = EquivariantMessagePassing(
    node_dim=64,
    edge_dim=16,
    hidden_dim=128
)

features_out, pos_out = layer(
    x=features,
    pos=positions,
    edge_index=edge_index,
    edge_attr=edge_features
)
```

#### `SheafGraphConv`
Graph convolution with sheaf structure.

```python
from fishstick.graph import SheafGraphConv

conv = SheafGraphConv(
    in_channels=64,
    out_channels=128,
    stalk_dim=16
)

output = conv(x, edge_index, stalk_features)
```

### Transformers

#### `GeometricGraphTransformer`
Transformer architecture for geometric graphs.

```python
from fishstick.graph import GeometricGraphTransformer

model = GeometricGraphTransformer(
    node_dim=64,
    edge_dim=16,
    hidden_dim=128,
    num_heads=8,
    num_layers=4
)

output = model(x, pos, edge_index, edge_attr)
```

#### `GeometricTransformerLayer`
Single geometric transformer layer.

```python
from fishstick.graph import GeometricTransformerLayer

layer = GeometricTransformerLayer(
    in_dim=64,
    out_dim=128,
    edge_dim=16,
    num_heads=8
)
```

### Specialized Networks

#### `MolecularGraphNetwork`
Network for molecular property prediction.

```python
from fishstick.graph import MolecularGraphNetwork

model = MolecularGraphNetwork(
    node_feature_dim=10,
    edge_feature_dim=4,
    hidden_dim=128,
    num_layers=4,
    num_tasks=1,
    readout="mean"
)

pred = model(x, pos, edge_index, edge_attr, batch)
```

#### `CrystalGraphNetwork`
Graph network for crystalline materials.

```python
from fishstick.graph import CrystalGraphNetwork

model = CrystalGraphNetwork(
    node_dim=10,
    hidden_dim=128,
    num_layers=4,
    max_neighbors=12
)

pred = model(x, pos, lattice, edge_index)
```

#### `RiemannianGraphConv`
Graph convolution on Riemannian manifolds.

```python
from fishstick.graph import RiemannianGraphConv

conv = RiemannianGraphConv(
    in_channels=64,
    out_channels=128,
    manifold_dim=3,
    curvature=-1.0  # Hyperbolic
)
```

### Data Structures

#### `GeometricEdge`
Geometric edge with distance and direction.

```python
from fishstick.graph import GeometricEdge

edge = GeometricEdge(
    distance=1.5,
    direction=torch.randn(3),
    features=torch.randn(16)
)
```

## Dependencies

- `torch`: PyTorch tensors
- `torch_geometric`: Graph neural network utilities (optional)

## Usage Examples

### Molecular Property Prediction

```python
from fishstick.graph import MolecularGraphNetwork

model = MolecularGraphNetwork(
    node_feature_dim=10,
    edge_feature_dim=4,
    hidden_dim=128
)

for batch in dataloader:
    pred = model(
        batch.x,
        batch.pos,
        batch.edge_index,
        batch.edge_attr,
        batch.batch
    )
    loss = F.mse_loss(pred, batch.y)
```

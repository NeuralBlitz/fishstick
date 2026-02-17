# Topological Data Analysis

Persistent homology, Vietoris-Rips complex, and Mapper algorithm.

## Installation

```bash
pip install fishstick[topology]
```

## Overview

The `topology` module provides tools for topological data analysis (TDA) including persistent homology and Mapper algorithm.

## Usage

```python
from fishstick.topology import PersistentHomology, VietorisRipsComplex, Mapper

# Compute persistent homology
ph = PersistentHomology(max_dimension=2)
diagram = ph.fit_transform(point_cloud)

# Vietoris-Rips complex
vr = VietorisRipsComplex(threshold=0.5)
complex = vr.build(point_cloud)

# Mapper algorithm
mapper = Mapper(n_cubes=10, overlap=0.5)
simplified = mapper.fit_transform(data)
```

## Core Classes

| Class | Description |
|-------|-------------|
| `PersistentHomology` | Compute persistent homology |
| `PersistenceDiagram` | Persistence diagram |
| `VietorisRipsComplex` | Vietoris-Rips simplicial complex |
| `Mapper` | Mapper algorithm for data simplification |

## Features

| Class | Description |
|-------|-------------|
| `TopologicalFeatures` | Extract topological features |
| `PersistentEntropy` | Compute persistent entropy |
| `BettiCurve` | Betti curve computation |

## Losses

| Loss | Description |
|------|-------------|
| `PersistentHomologyLoss` | TDA-based loss function |
| `DiagramDistanceLoss` | Diagram distance loss |

## Examples

See `examples/topology/` for complete examples.

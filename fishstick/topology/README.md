# Topological Data Analysis (TDA) Module

Comprehensive implementation of topological data analysis methods for extracting geometric and topological features from data.

## Overview

This module provides state-of-the-art tools for:

- **Persistent Homology**: Multi-scale topological feature detection
- **Mapper Algorithm**: Topological simplification and clustering
- **Vietoris-Rips Complex**: Simplicial complex construction
- **Topological Losses**: Geometric deep learning regularization
- **TDA Layers**: Differentiable topological operations
- **TDA Kernels**: Kernel methods for persistence diagrams
- **Multi-Scale Analysis**: Adaptive scale selection

## Installation

```bash
# Core dependencies (included in fishstick)
pip install torch numpy scipy

# Optional: for faster homology computations
pip install gudhi ripser
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

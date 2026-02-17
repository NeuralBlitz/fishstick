# algtopo - Algebraic Topology Module

## Overview

The `algtopo` module provides implementations of algebraic topology concepts for topological data analysis (TDA) and topological deep learning. It includes simplicial complexes, homology/cohomology theory, and persistent homology for analyzing the shape and structure of data.

## Purpose and Scope

This module enables:
- Construction and manipulation of simplicial complexes
- Computation of homology and cohomology groups
- Persistent homology for topological feature extraction
- Topological analysis of point clouds and images
- Integration with neural networks via persistent homology layers

## Key Classes and Functions

### Simplicial Complexes (`homology.py`)

#### `Simplex`
Fundamental building block in algebraic topology.

```python
from fishstick.algtopo import Simplex

# Create a 2-simplex (triangle)
s = Simplex(vertices=(0, 1, 2))
print(s.dimension)  # 2
print(s.faces())    # All proper faces
```

#### `SimplicialComplex`
Collection of simplices closed under faces.

```python
from fishstick.algtopo import SimplicialComplex

# Build a complex
complex = SimplicialComplex()
complex.add_simplex((0, 1, 2))  # Add triangle with all faces

# Compute Betti numbers
betti = complex.homology(dim=2)
print(betti)  # {0: 1, 1: 0, 2: 0} - connected, no holes
```

**Key Methods:**
- `add_simplex(vertices)`: Add simplex and all its faces
- `boundary_operator(n)`: Compute boundary matrix ∂_n
- `homology(dim)`: Compute Betti numbers up to dimension n

#### `Chain` and `Boundary`
Formal sums of simplices and boundary operations.

#### `BettiNumbers`
Topological invariants counting holes of each dimension:
- β_0: Number of connected components
- β_1: Number of 1-dimensional holes (loops)
- β_2: Number of 2-dimensional voids

### Cohomology (`cohomology.py`)

#### `Cocycle`
A cochain with coboundary zero.

#### `Coboundary`
Coboundary operator δ: C^n → C^{n+1}.

#### `CohomologyGroup`
Computes cohomology groups H^n(X; G).

#### `CupProduct`
Computes cup product in cohomology - a ring structure on cohomology.

#### `DeRhamComplex`
De Rham complex for smooth manifolds with:
- Exterior derivative d: Ω^k → Ω^{k+1}
- Hodge Laplacian Δ = dd* + d*d
- Harmonic forms

#### `MorseTheory`
Analyzes topology via critical points of a function:
- `critical_points()`: Find points where gradient = 0
- `morse_index()`: Count negative eigenvalues of Hessian

### Persistent Homology (`persistent.py`)

#### `VietorisRipsComplex`
Construct Vietoris-Rips complex from point cloud.

```python
from fishstick.algtopo import VietorisRipsComplex
import torch

points = torch.randn(100, 3)  # 100 points in 3D
vr = VietorisRipsComplex(points, threshold=0.5, max_dimension=2)
```

#### `filtration`
Compute filtration of simplicial complexes at different scales.

#### `persistence_diagram`
Compute persistence diagram from point cloud.

```python
from fishstick.algtopo import persistence_diagram

diagram = persistence_diagram(points, max_dimension=1, max_scale=1.0)
# Returns {dim: [(birth, death), ...], ...}
```

#### `bottleneck_distance` and `wasserstein_distance`
Distance metrics between persistence diagrams for comparing topological features.

#### `PersistentHomology` (nn.Module)
Neural network layer for persistent homology.

```python
from fishstick.algtopo import PersistentHomology

ph_layer = PersistentHomology(max_dimension=1, max_scale=1.0)
diagrams = ph_layer(point_cloud)  # batch of point clouds
loss = ph_layer.topological_loss(point_cloud)
```

#### `Barcode` and `Landscape`
Alternative representations of persistence diagrams.

## Mathematical Background

### Homology
Homology groups H_n(X) measure n-dimensional "holes" in a space:
- Computed as H_n = Ker(∂_n) / Im(∂_{n+1})
- Betti numbers β_n = rank(H_n)

### Cohomology
Dual to homology, cohomology groups H^n(X) have additional algebraic structure:
- Cup product gives cohomology a ring structure
- De Rham cohomology uses differential forms

### Persistent Homology
Tracks how topological features persist across scales:
- Birth: scale at which feature appears
- Death: scale at which feature disappears
- Long-lived features are typically significant

## Dependencies

- `torch`: PyTorch tensors and neural network modules
- `numpy`: Numerical operations
- `scipy`: Distance computations and linear algebra

## Usage Examples

### Basic Topological Analysis

```python
from fishstick.algtopo import SimplicialComplex, BettiNumbers

# Create a torus-like structure
complex = SimplicialComplex()
# Add simplices to form a torus...
betti = BettiNumbers(complex).compute()
# Torus has β_0=1, β_1=2, β_2=1
```

### Persistent Homology Pipeline

```python
from fishstick.algtopo import persistence_diagram, bottleneck_distance
import torch

# Two point clouds
points1 = torch.randn(100, 2)
points2 = torch.randn(100, 2) + 1.0

# Compute persistence diagrams
dgm1 = persistence_diagram(points1, max_dimension=1)
dgm2 = persistence_diagram(points2, max_dimension=1)

# Compare topologies
dist = bottleneck_distance(dgm1[1], dgm2[1])
```

### Integration with Neural Networks

```python
import torch
from fishstick.algtopo import PersistentHomology

class TopologicalAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_dim)
        )
        self.ph = PersistentHomology(max_dimension=1)
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        
        # Topological regularization
        topo_loss = self.ph.topological_loss(z)
        
        return recon, topo_loss
```

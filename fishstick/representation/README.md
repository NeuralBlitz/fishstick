# Representation Theory

Lie algebras, group representations, and Weyl groups.

## Installation

```bash
pip install fishstick[representation]
```

## Overview

The `representation` module provides tools for representation theory including Lie algebras, Lie groups, and group representations.

## Usage

```python
from fishstick.representation import LieAlgebra, LieGroup, so3, su2

# Create Lie algebra
so3_algebra = so3()
print(so3_algebra.dimension)  # 3

# Lie group operations
so3_group = LieGroup(so3_algebra)
exp_map = so3_group.exp(tangent_vector)
log_map = so3_group.log(element)

# Weyl group
from fishstick.representation import WeylGroup, RootSystem
root_system = RootSystem("A2")
weyl = WeylGroup(root_system)
```

## Lie Algebras

| Algebra | Description |
|---------|-------------|
| `LieAlgebra` | Base Lie algebra |
| `su2` | su(2) algebra |
| `so3` | so(3) algebra |
| `sl2c` | sl(2, C) algebra |

## Groups and Representations

| Class | Description |
|-------|-------------|
| `LieGroup` | Lie group |
| `GroupRepresentation` | Group representation |
| `IrreducibleRepresentation` | Irreducible representation |

## Weyl Groups

| Class | Description |
|-------|-------------|
| `WeylGroup` | Weyl group |
| `RootSystem` | Root system |
| `WeightLattice` | Weight lattice |
| `DynkinDiagram` | Dynkin diagram |

## Examples

See `examples/representation/` for complete examples.

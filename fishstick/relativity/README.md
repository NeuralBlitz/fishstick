# Relativity

Special and general relativity computations.

## Installation

```bash
pip install fishstick[relativity]
```

## Overview

The `relativity` module provides utilities for special and general relativity calculations including Lorentz transformations, metrics, and geodesics.

## Usage

```python
from fishstick.relativity import LorentzTransformation, FourVector, MinkowskiMetric

# Lorentz boost
boost = LorentzTransformation(velocity=0.5)
four_vector = FourVector([1, 0, 0, 0])
boosted = boost.apply(four_vector)

# Minkowski metric
metric = MinkowskiMetric()
interval = metric.spacetime_interval(event1, event2)

# Schwarzschild metric
from fishstick.relativity import SchwarzschildMetric
schwarzschild = SchwarzschildMetric(mass=1.0)
```

## Special Relativity

| Class | Description |
|-------|-------------|
| `LorentzTransformation` | Lorentz transformation |
| `Boost` | Lorentz boost |
| `Rotation` | Spatial rotation |
| `FourVector` | Four-vector |
| `MinkowskiMetric` | Minkowski metric |
| `ProperTime` | Proper time calculation |

## General Relativity

| Class | Description |
|-------|-------------|
| `SchwarzschildMetric` | Schwarzschild metric |
| `KerrMetric` | Kerr metric |
| `GeodesicEquation` | Geodesic equation solver |
| `RelativisticParticle` | Relativistic particle |

## Additional Classes

| Class | Description |
|-------|-------------|
| `SpacetimeInterval` | Spacetime interval |
| `LightCone` | Light cone |
| `Causality` | Causality analysis |
| `EnergyMomentum` | Energy-momentum 4-vector |

## Examples

See `examples/relativity/` for complete examples.

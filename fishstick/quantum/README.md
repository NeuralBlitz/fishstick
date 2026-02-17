# Quantum Computing

Quantum circuits, gates, and tensor networks.

## Installation

```bash
pip install fishstick[quantum]
```

## Overview

The `quantum` module provides quantum computing primitives including quantum gates, circuits, and tensor network representations.

## Usage

```python
from fishstick.quantum import QuantumCircuit, Hadamard, CNOT, RX, RY, RZ

# Create quantum circuit
qc = QuantumCircuit(3)
qc.add_gate(Hadamard(), [0])
qc.add_gate(CNOT(), [0, 1])
qc.add_gate(RX(theta=0.5), [2])

# Run simulation
result = qc.simulate()

# Tensor networks
from fishstick.quantum import TensorNetwork, MPS, TTN

tn = MPS(num_sites=10, bond_dim=32)
```

## Quantum Gates

| Gate | Description |
|------|-------------|
| `Hadamard` | Hadamard gate |
| `CNOT` | Controlled-NOT gate |
| `RX`, `RY`, `RZ` | Rotation gates |
| `PauliX`, `PauliY`, `PauliZ` | Pauli gates |

## Tensor Networks

| Class | Description |
|-------|-------------|
| `TensorNetwork` | Base tensor network |
| `MPS` | Matrix Product State |
| `TTN` | Tree Tensor Network |
| `PEPS` | Projected Entangled Pair State |

## Embeddings

| Class | Description |
|-------|-------------|
| `QuantumEmbedding` | Base quantum embedding |
| `AmplitudeEmbedding` | Amplitude encoding |
| `AngleEmbedding` | Angle encoding |

## Examples

See `examples/quantum/` for complete examples.

# Quantum Machine Learning

Quantum neural networks and quantum kernels for machine learning.

## Installation

```bash
pip install fishstick[quantum_ml]
```

## Overview

The `quantum_ml` module provides quantum computing extensions for machine learning including quantum neural networks, quantum kernels, and quantum attention mechanisms.

## Usage

```python
from fishstick.quantum_ml import QuantumConv1DLayer, QuantumLSTMCell, QuantumKernel

# Quantum convolution
qconv = QuantumConv1DLayer(in_channels=3, out_channels=16, kernel_size=3)

# Quantum LSTM
qlstm = QuantumLSTMCell(input_size=64, hidden_size=128)

# Quantum kernel
qkernel = FidelityQuantumKernel(feature_dim=64)
K = qkernel.compute_kernel(X_train, X_test)
```

## Quantum Layers

| Layer | Description |
|-------|-------------|
| `QuantumConv1DLayer` | 1D quantum convolution |
| `QuantumConv2DLayer` | 2D quantum convolution |
| `QuantumRecurrentLayer` | Quantum RNN |
| `QuantumLSTMCell` | Quantum LSTM cell |
| `QuantumGRUCell` | Quantum GRU cell |

## Quantum Kernels

| Kernel | Description |
|--------|-------------|
| `QuantumKernel` | Base quantum kernel |
| `FidelityQuantumKernel` | Fidelity-based quantum kernel |
| `ProjectedQuantumKernel` | Projected quantum kernel |

## Quantum Classifiers/Regressors

| Class | Description |
|-------|-------------|
| `VariationalQuantumClassifier` | VQC for classification |
| `VariationalQuantumRegressor` | VQR for regression |
| `QuantumAttention` | Quantum attention mechanism |
| `QuantumMultiHeadAttention` | Multi-head quantum attention |

## Examples

See `examples/quantum_ml/` for complete examples.

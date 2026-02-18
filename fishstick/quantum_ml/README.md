# Quantum Machine Learning Extensions

Quantum machine learning modules for the fishstick AI framework, providing quantum neural network layers, kernel methods, variational circuits, attention mechanisms, and integration with fishstick's core modules.

## Installation

Requires fishstick core. Install with:
```bash
pip install fishstick
```

## Overview

The `quantum_ml` module provides:

- **Quantum Neural Network Layers**: Quantum convolutional and recurrent layers
- **Quantum Kernel Methods**: Fidelity-based and projected quantum kernels
- **Variational Circuits**: Quantum classifiers and regressors
- **Quantum Attention**: Quantum attention mechanisms for transformers
- **Integration**: Integration with geometric and dynamics modules

## Usage

```python
import torch
from fishstick.quantum_ml import (
    QuantumConv1DLayer,
    QuantumLSTMCell,
    FidelityQuantumKernel,
    VariationalQuantumClassifier,
    QuantumAttention,
    QuantumTransformerLayer,
    QuantumGeometricIntegration,
)
from fishstick.core.types import PhaseSpaceState

# Quantum convolution
qconv = QuantumConv1DLayer(in_channels=3, out_channels=16, kernel_size=3)
x = torch.randn(2, 3, 32)
out = qconv(x)

# Quantum LSTM
qlstm = QuantumLSTMCell(input_size=64, hidden_size=128)
h, c = qlstm(x)

# Quantum kernel
kernel = FidelityQuantumKernel(n_qubits=8)
K = kernel(x1, x2)

# Quantum classifier
clf = VariationalQuantumClassifier(n_qubits=8, n_classes=10)
out = clf(x)

# Quantum attention
attn = QuantumAttention(embed_dim=64, n_qubits=8, n_heads=4)
out, weights = attn(x, x, x)

# Geometric integration
geo = QuantumGeometricIntegration(embed_dim=64, n_qubits=8)
result = geo(x)

# Phase space encoder
ps_enc = QuantumPhaseSpaceEncoder(config_dim=4, n_qubits=8)
state = PhaseSpaceState(q=torch.randn(2, 4), p=torch.randn(2, 4))
encoded = ps_enc(state)
```

## Modules

### quantum_layers.py
- `QuantumConv1DLayer`: 1D quantum convolution
- `QuantumConv2DLayer`: 2D quantum convolution  
- `QuantumRecurrentLayer`: Quantum RNN cell
- `QuantumLSTMCell`: Quantum LSTM cell
- `QuantumGRUCell`: Quantum GRU cell
- `QuantumBatchNorm`: Quantum-inspired batch normalization
- `QuantumDropout`: Quantum-inspired dropout

### quantum_kernels.py
- `QuantumKernel`: Base quantum kernel class
- `FidelityQuantumKernel`: Fidelity-based quantum kernel
- `ProjectedQuantumKernel`: Projected quantum kernel
- `QuantumKernelPCA`: Kernel PCA with quantum kernels
- `QuantumKernelSVM`: SVM with quantum kernels
- `QuantumKernelRidge`: Ridge regression with quantum kernels

### variational_circuits.py
- `ParameterizedQuantumCircuit`: Parameterized quantum circuit
- `QuantumEmbeddingCircuit`: Data embedding circuit
- `VariationalQuantumClassifier`: VQC for classification
- `VariationalQuantumRegressor`: VQR for regression
- `QuantumCircuitLearner`: Generic circuit learner

### quantum_attention.py
- `QuantumAttention`: Quantum attention mechanism
- `QuantumMultiHeadAttention`: Multi-head quantum attention
- `QuantumSelfAttention`: Quantum self-attention layer
- `QuantumCrossAttention`: Quantum cross-attention
- `QuantumTransformerLayer`: Complete transformer layer
- `QuantumTransformer`: Full transformer model

### integration.py
- `QuantumMetricTensor`: Quantum-inspired metric tensor
- `QuantumGeometricIntegration`: Integration with geometric modules
- `QuantumDynamicsIntegration`: Integration with dynamics modules
- `QuantumManifoldProjection`: Manifold projection
- `QuantumPhaseSpaceEncoder`: Phase space to quantum state
- `QuantumSymplecticMap`: Symplectic map
- `QuantumConnection`: Affine connection on quantum manifold

## Integration with Fishstick

The `quantum_ml` module integrates with:
- `fishstick.core.types`: PhaseSpaceState, MetricTensor
- `fishstick.geometric`: FisherInformationMetric (optional)
- `fishstick.dynamics`: Hamiltonian dynamics

## Type Hints

All modules use full type hints for IDE support and static analysis.

## License

Part of the fishstick AI framework.

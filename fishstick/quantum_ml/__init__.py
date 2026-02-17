"""
Quantum Machine Learning Extensions for fishstick.

This module provides quantum computing extensions for machine learning,
including quantum neural network layers, kernel methods, variational circuits,
and attention mechanisms.
"""

from .quantum_layers import (
    QuantumConv1DLayer,
    QuantumConv2DLayer,
    QuantumRecurrentLayer,
    QuantumLSTMCell,
    QuantumGRUCell,
    QuantumBatchNorm,
    QuantumDropout,
)

from .quantum_kernels import (
    QuantumKernel,
    FidelityQuantumKernel,
    ProjectedQuantumKernel,
    QuantumKernelPCA,
    QuantumKernelSVM,
    QuantumKernelRidge,
)

from .variational_circuits import (
    VariationalQuantumClassifier,
    VariationalQuantumRegressor,
    QuantumCircuitLearner,
    ParameterizedQuantumCircuit,
    QuantumEmbeddingCircuit,
    EntanglingStrategy,
)

from .quantum_attention import (
    QuantumAttention,
    QuantumMultiHeadAttention,
    QuantumTransformerLayer,
    QuantumSelfAttention,
    QuantumCrossAttention,
)

from .integration import (
    QuantumGeometricIntegration,
    QuantumDynamicsIntegration,
    QuantumManifoldProjection,
    QuantumPhaseSpaceEncoder,
    QuantumSymplecticMap,
    QuantumMetricTensor,
)

__all__ = [
    # Quantum layers
    "QuantumConv1DLayer",
    "QuantumConv2DLayer",
    "QuantumRecurrentLayer",
    "QuantumLSTMCell",
    "QuantumGRUCell",
    "QuantumBatchNorm",
    "QuantumDropout",
    # Quantum kernels
    "QuantumKernel",
    "FidelityQuantumKernel",
    "ProjectedQuantumKernel",
    "QuantumKernelPCA",
    "QuantumKernelSVM",
    "QuantumKernelRidge",
    # Variational circuits
    "VariationalQuantumClassifier",
    "VariationalQuantumRegressor",
    "QuantumCircuitLearner",
    "ParameterizedQuantumCircuit",
    "QuantumEmbeddingCircuit",
    "EntanglingStrategy",
    # Quantum attention
    "QuantumAttention",
    "QuantumMultiHeadAttention",
    "QuantumTransformerLayer",
    "QuantumSelfAttention",
    "QuantumCrossAttention",
    # Integration
    "QuantumGeometricIntegration",
    "QuantumDynamicsIntegration",
    "QuantumManifoldProjection",
    "QuantumPhaseSpaceEncoder",
    "QuantumSymplecticMap",
    "QuantumMetricTensor",
]

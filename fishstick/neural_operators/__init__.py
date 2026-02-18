"""
Neural Operators Module for fishstick AI Framework.

Comprehensive implementations of neural operator architectures for learning
mappings between infinite-dimensional function spaces.

Modules:
    - fno: Fourier Neural Operators
    - deeponet: Deep Operator Networks
    - pde: PDE-related neural networks
    - graph_operator: Graph Neural Operators
    - neural_ode_integrators: Neural ODE solvers
    - multiscale: Multi-scale operator methods
    - base_operator: Base classes and interfaces
    - operator_utils: Common utilities and helpers
"""

from .fno import (
    FNO1d,
    FNO2d,
    FNO3d,
    SpectralConv,
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
    AdaptiveFNO2d,
)
from .deeponet import (
    DeepONet,
    BranchNet,
    TrunkNet,
    DeepONetCartesian,
    DeepONetDistributed,
    NodewiseDeepONet,
    DeepONetEnsemble,
    ConvolutionalDeepONet,
    AttentionDeepONet,
)
from .pde import PdeNet, PodNet, NeuralGalerkin, PdeNetResidual
from .graph_operator import (
    MessagePassingOperator,
    SpectralGraphConv,
    GraphPoolingOperator,
    GraphUnpoolingOperator,
    GraphAttentionOperator,
    PointCloudOperator,
    MeshOperator,
    GraphNeuralOperatorBlock,
)
from .neural_ode_integrators import (
    NeuralODEFunction,
    RungeKuttaIntegrator,
    AdamsBashforthIntegrator,
    AdaptiveStepIntegrator,
    SymplecticIntegrator,
    HamiltonianNeuralNetwork,
    ContinuousNormalizingFlow,
    LatentODEFunc,
    NeuralODEDecoder,
)
from .multiscale import (
    WaveletTransform1D,
    WaveletNeuralOperator,
    MultiScaleAttention,
    MultigridNeuralOperator,
    HierarchicalOperatorBlock,
    AdaptiveScaleOperator,
    ScaleSeparationModule,
    MultiResolutionDeepONet,
    FractalNeuralOperator,
    OperatorInterpolation,
)
from .base_operator import (
    OperatorConfig,
    BaseNeuralOperator,
    FunctionToFunctionOperator,
    TimeSeriesOperator,
    OperatorOutput,
    OperatorLoss,
    OperatorL2Loss,
    OperatorRelativeLoss,
    OperatorValidationMonitor,
    OperatorDataset,
    Collator,
    OperatorTrainer,
    FourierFeatures,
    PositionalEncoding,
    DomainTransformer,
    IntegralTransform,
    KernelIntegration,
)
from .operator_utils import (
    GridGenerator,
    SensorSampler,
    FunctionGenerator,
    PDEOperator,
    OperatorNormalizer,
    BoundaryCondition,
    QuadratureRule,
    LossScheduler,
    AttentionOperator,
    ResidualBlock,
    OperatorEmbedding,
    SpectralPooling1D,
    DomainPadding,
    ComplexMLP,
    OperatorEnsemble,
    OperatorMetrics,
    OperatorEvaluator,
)

__all__ = [
    # Fourier Neural Operators
    "FNO1d",
    "FNO2d",
    "FNO3d",
    "SpectralConv",
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralConv3d",
    "AdaptiveFNO2d",
    # DeepONet
    "DeepONet",
    "BranchNet",
    "TrunkNet",
    "DeepONetCartesian",
    "DeepONetDistributed",
    "NodewiseDeepONet",
    "DeepONetEnsemble",
    "ConvolutionalDeepONet",
    "AttentionDeepONet",
    # PDE Networks
    "PdeNet",
    "PodNet",
    "NeuralGalerkin",
    "PdeNetResidual",
    # Graph Operators
    "MessagePassingOperator",
    "SpectralGraphConv",
    "GraphPoolingOperator",
    "GraphUnpoolingOperator",
    "GraphAttentionOperator",
    "PointCloudOperator",
    "MeshOperator",
    "GraphNeuralOperatorBlock",
    # Neural ODE Integrators
    "NeuralODEFunction",
    "RungeKuttaIntegrator",
    "AdamsBashforthIntegrator",
    "AdaptiveStepIntegrator",
    "SymplecticIntegrator",
    "HamiltonianNeuralNetwork",
    "ContinuousNormalizingFlow",
    "LatentODEFunc",
    "NeuralODEDecoder",
    # Multi-scale Operators
    "WaveletTransform1D",
    "WaveletNeuralOperator",
    "MultiScaleAttention",
    "MultigridNeuralOperator",
    "HierarchicalOperatorBlock",
    "AdaptiveScaleOperator",
    "ScaleSeparationModule",
    "MultiResolutionDeepONet",
    "FractalNeuralOperator",
    "OperatorInterpolation",
    # Base Classes
    "OperatorConfig",
    "BaseNeuralOperator",
    "FunctionToFunctionOperator",
    "TimeSeriesOperator",
    "OperatorOutput",
    "OperatorLoss",
    "OperatorL2Loss",
    "OperatorRelativeLoss",
    "OperatorValidationMonitor",
    "OperatorDataset",
    "Collator",
    "OperatorTrainer",
    "FourierFeatures",
    "PositionalEncoding",
    "DomainTransformer",
    "IntegralTransform",
    "KernelIntegration",
    # Utilities
    "GridGenerator",
    "SensorSampler",
    "FunctionGenerator",
    "PDEOperator",
    "OperatorNormalizer",
    "BoundaryCondition",
    "QuadratureRule",
    "LossScheduler",
    "AttentionOperator",
    "ResidualBlock",
    "OperatorEmbedding",
    "SpectralPooling1D",
    "DomainPadding",
    "ComplexMLP",
    "OperatorEnsemble",
    "OperatorMetrics",
    "OperatorEvaluator",
]

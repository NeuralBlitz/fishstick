"""
Advanced Optimization Module

A comprehensive collection of advanced optimization techniques:
- Second-order optimization methods (K-FAC, ESGD, Shampoo)
- Physics-inspired learning rate schedulers
- Gradient accumulation strategies
- Adaptive momentum methods
- Meta-learning optimization primitives

This module provides state-of-the-art optimization tools extending beyond
standard approaches, including second-order methods, physics-based schedulers,
and meta-learning techniques.

Submodules:
- second_order: Second-order optimization methods
- physics_schedulers: Physics-inspired LR schedulers
- gradient_accumulation: Advanced gradient accumulation
- adaptive_momentum: Adaptive momentum methods
- meta_learning: Meta-learning optimization primitives
"""

from .second_order import (
    KFACOptimizer,
    ESGDOptimizer,
    ShampooOptimizer,
    SecondOrderInfo,
)

from .physics_schedulers import (
    ThermodynamicAnnealingScheduler,
    HamiltonianDynamicsScheduler,
    QuantumTunnelingScheduler,
    RiemannianGradientFlowScheduler,
    AdaptiveCurvatureScheduler,
    CyclicOscillationScheduler,
    StochasticWeightAveragingScheduler,
)

from .gradient_accumulation import (
    GradientAccumulator,
    AdaptiveGradientAccumulator,
    GradientAccumulatorWithScaling,
    MultiScaleGradientAccumulator,
    GradientCentralization,
)

from .adaptive_momentum import (
    AdaptiveMomentumOptimizer,
    HeavyBallOptimizer,
    NesterovMomentumOptimizer,
    VarianceReducedMomentum,
    PadagradMomentum,
    ElasticMomentum,
)

from .meta_learning import (
    LearnedOptimizer,
    MetaLearnedOptimizer,
    MAMLOptimizerStep,
    ReptileOptimizer,
    MetaLearningRateScheduler,
    NeuralOptimizerLayer,
    MetaGradientAccumulator,
    FastGradientOptimizer,
    MetaSGDOptimizer,
)


__all__ = [
    # Second-order optimization
    "KFACOptimizer",
    "ESGDOptimizer",
    "ShampooOptimizer",
    "SecondOrderInfo",
    # Physics-inspired schedulers
    "ThermodynamicAnnealingScheduler",
    "HamiltonianDynamicsScheduler",
    "QuantumTunnelingScheduler",
    "RiemannianGradientFlowScheduler",
    "AdaptiveCurvatureScheduler",
    "CyclicOscillationScheduler",
    "StochasticWeightAveragingScheduler",
    # Gradient accumulation
    "GradientAccumulator",
    "AdaptiveGradientAccumulator",
    "GradientAccumulatorWithScaling",
    "MultiScaleGradientAccumulator",
    "GradientCentralization",
    # Adaptive momentum
    "AdaptiveMomentumOptimizer",
    "HeavyBallOptimizer",
    "NesterovMomentumOptimizer",
    "VarianceReducedMomentum",
    "PadagradMomentum",
    "ElasticMomentum",
    # Meta-learning optimization
    "LearnedOptimizer",
    "MetaLearnedOptimizer",
    "MAMLOptimizerStep",
    "ReptileOptimizer",
    "MetaLearningRateScheduler",
    "NeuralOptimizerLayer",
    "MetaGradientAccumulator",
    "FastGradientOptimizer",
    "MetaSGDOptimizer",
]


__version__ = "0.1.0"

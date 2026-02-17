"""
fishstick AutoML Module

Neural Architecture Search and hyperparameter optimization.
"""

from fishstick.automl.search import (
    NASearch,
    RandomSearch,
    GridSearch,
    Hyperband,
    SearchSpace,
    Choice,
    Uniform,
    LogUniform,
    Conditional,
)

from fishstick.automl.nas import (
    # Search Space
    Operation,
    ConvOperation,
    PoolingOperation,
    ActivationOperation,
    SkipOperation,
    NoneOperation,
    SearchSpace as NASSearchSpace,
    # Architecture
    LayerSpec,
    CellSpec,
    Architecture,
    ArchitectureBuilder,
    # Evaluator
    ArchitectureEvaluator,
    # NAS Algorithms
    NeuralArchitectureSearch,
    RandomNAS,
    EvolutionaryNAS,
    # Convenience Functions
    create_default_search_space,
    search_architecture,
    estimate_model_complexity,
    build_model_from_architecture,
)

__all__ = [
    # Search (hyperopt)
    "NASearch",
    "RandomSearch",
    "GridSearch",
    "Hyperband",
    # Space (hyperopt)
    "SearchSpace",
    "Choice",
    "Uniform",
    "LogUniform",
    "Conditional",
    # NAS Operations
    "Operation",
    "ConvOperation",
    "PoolingOperation",
    "ActivationOperation",
    "SkipOperation",
    "NoneOperation",
    "NASSearchSpace",
    # NAS Architecture
    "LayerSpec",
    "CellSpec",
    "Architecture",
    "ArchitectureBuilder",
    # NAS Evaluator
    "ArchitectureEvaluator",
    # NAS Algorithms
    "NeuralArchitectureSearch",
    "RandomNAS",
    "EvolutionaryNAS",
    # Convenience Functions
    "create_default_search_space",
    "search_architecture",
    "estimate_model_complexity",
    "build_model_from_architecture",
]

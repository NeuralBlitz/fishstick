"""
fishstick Architecture Search Module

Neural Architecture Search (NAS) and Meta-Learning tools including:
- Search space definitions (DARTS, NB201, MobileNet, ResNet)
- Architecture controllers (DARTS, ProxylessNAS, FBNet)
- Super-net training utilities
- MAML and Reptile meta-learning algorithms
- Architecture performance predictors
- Cost models (FLOPs, latency, memory)
"""

from .search_space import (
    # Operation types
    OperationType,
    OperationSpec,
    EdgeSpec,
    CellSpec,
    ArchitectureSpec,
    # Search spaces
    SearchSpace,
    DARTSearchSpace,
    NB201SearchSpace,
    MobileNetSearchSpace,
    ResNetSearchSpace,
    # Utilities
    create_search_space,
    get_operation_info,
)

from .controller import (
    # Controllers
    ArchitectureController,
    ProxylessNASController,
    FBNetController,
    RandomNASController,
    # Optimization
    ArchitectureSearchOptimizer,
    ControllerState,
    # Factory
    create_controller,
)

from .supernet import (
    # Operations
    MixedOperation,
    ZeroOperation,
    SkipConnection,
    MixedCell,
    # Super-net
    SuperNet,
    SuperNetTrainer,
    SuperNetConfig,
    build_supernet_from_architecture,
    create_supernet,
)

from .maml import (
    # MAML
    MAML,
    FirstOrderMAML,
    MAMLPlus,
    ImplicitMAML,
    # Training
    MetaLearner,
    MetaLearningConfig,
    MetaLearningTrainer,
    # Task utilities
    Task,
    create_few_shot_task,
    omniglot_task_sampler,
)

from .reptile import (
    # Reptile
    Reptile,
    ReptileWithWeightDecay,
    FOMAML,
    # Training
    ReptileConfig,
    ReptileTrainer,
    # Utilities
    reptile_outer_step,
    simple_task_sampler,
)

from .predictor import (
    # Predictors
    PerformancePredictor,
    LinearPredictor,
    NeuralPredictor,
    EnsemblePredictor,
    # Cost models
    CostModel,
    # Factory
    create_predictor,
    load_nasbench_data,
)

__all__ = [
    # Search space - Operations
    "OperationType",
    "OperationSpec",
    "EdgeSpec",
    "CellSpec",
    "ArchitectureSpec",
    # Search space - Classes
    "SearchSpace",
    "DARTSearchSpace",
    "NB201SearchSpace",
    "MobileNetSearchSpace",
    "ResNetSearchSpace",
    # Search space - Utilities
    "create_search_space",
    "get_operation_info",
    # Controller
    "ArchitectureController",
    "ProxylessNASController",
    "FBNetController",
    "RandomNASController",
    "ArchitectureSearchOptimizer",
    "ControllerState",
    "create_controller",
    # Super-net
    "MixedOperation",
    "ZeroOperation",
    "SkipConnection",
    "MixedCell",
    "SuperNet",
    "SuperNetTrainer",
    "SuperNetConfig",
    "build_supernet_from_architecture",
    "create_supernet",
    # MAML
    "MAML",
    "FirstOrderMAML",
    "MAMLPlus",
    "ImplicitMAML",
    "MetaLearner",
    "MetaLearningConfig",
    "MetaLearningTrainer",
    "Task",
    "create_few_shot_task",
    "omniglot_task_sampler",
    # Reptile
    "Reptile",
    "ReptileWithWeightDecay",
    "FOMAML",
    "ReptileConfig",
    "ReptileTrainer",
    "reptile_outer_step",
    "simple_task_sampler",
    # Predictor
    "PerformancePredictor",
    "LinearPredictor",
    "NeuralPredictor",
    "EnsemblePredictor",
    "CostModel",
    "create_predictor",
    "load_nasbench_data",
]

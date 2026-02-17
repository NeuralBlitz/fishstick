"""
Meta-Learning Module

MAML, ProtoNet, Reptile, MatchingNetworks, RelationNetworks, and related
few-shot learning methods with comprehensive utilities for task sampling,
inner loop optimization, and evaluation.
"""

# Meta-learning algorithms
from .learning import (
    MAML,
    Reptile,
    PrototypicalNetworks,
    MatchingNetworks,
    RelationNetworks,
    RelationModule,
)

# Few-shot learning utilities
from .learning import (
    Task,
    Episode,
    MetaLearningState,
    TaskSampler,
    EpisodeDataset,
    create_episode,
    split_support_query,
)

# Meta-learners
from .learning import (
    MetaLearner,
    GradientBasedMetaLearner,
    MetricBasedMetaLearner,
    MemoryAugmentedMetaLearner,
)

# Inner loop optimization
from .learning import (
    InnerLoopOptimizer,
    MetaOptimizer,
    learnable_learning_rates,
)

# Task loaders
from .learning import (
    OmniglotTaskLoader,
    MiniImageNetTaskLoader,
    CIFAR_FSTaskLoader,
    CustomTaskLoader,
)

# Evaluation functions
from .learning import (
    evaluate_few_shot,
    cross_domain_evaluation,
    meta_train,
    meta_validate,
    meta_test,
)

# Utility encoders and helpers
from .learning import (
    CNNEncoder,
    ResNetEncoder,
    compute_prototypical_accuracy,
    compute_confidence_interval,
)

# For backward compatibility
from .learning import PrototypicalNetworks as ProtoNet

__all__ = [
    # Meta-learning algorithms
    "MAML",
    "Reptile",
    "PrototypicalNetworks",
    "ProtoNet",
    "MatchingNetworks",
    "RelationNetworks",
    "RelationModule",
    # Data structures
    "Task",
    "Episode",
    "MetaLearningState",
    # Few-shot utilities
    "TaskSampler",
    "EpisodeDataset",
    "create_episode",
    "split_support_query",
    # Meta-learners
    "MetaLearner",
    "GradientBasedMetaLearner",
    "MetricBasedMetaLearner",
    "MemoryAugmentedMetaLearner",
    # Inner loop optimization
    "InnerLoopOptimizer",
    "MetaOptimizer",
    "learnable_learning_rates",
    # Task loaders
    "OmniglotTaskLoader",
    "MiniImageNetTaskLoader",
    "CIFAR_FSTaskLoader",
    "CustomTaskLoader",
    # Evaluation
    "evaluate_few_shot",
    "cross_domain_evaluation",
    "meta_train",
    "meta_validate",
    "meta_test",
    # Utilities
    "CNNEncoder",
    "ResNetEncoder",
    "compute_prototypical_accuracy",
    "compute_confidence_interval",
]

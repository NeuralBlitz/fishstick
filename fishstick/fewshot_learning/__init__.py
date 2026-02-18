"""
Few-Shot Learning Module for Fishstick AI Framework.

Comprehensive few-shot and meta-learning implementations including:
- MAML variants (MAML, FOMAML, ANIL, BOIL, MetaSGD)
- Prototypical Networks
- Relation Networks
- Matching Networks
- Episode generation utilities
- Training and evaluation tools
- Encoder networks
"""

from .types import (
    FewShotTask,
    MetaBatch,
    AdaptationResult,
    EvaluationResult,
    TrainingState,
    MetaLearningConfig,
    DistanceMetric,
    EpisodeType,
)

from .config import (
    MAMLConfig,
    PrototypicalConfig,
    RelationNetworkConfig,
    MatchingNetworkConfig,
    EpisodeConfig,
    TrainingConfig,
    EvaluatorConfig,
    DEFAULT_MAML_CONFIG,
    DEFAULT_PROTONET_CONFIG,
    DEFAULT_RELATION_CONFIG,
    DEFAULT_MATCHING_CONFIG,
    DEFAULT_EPISODE_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_EVALUATOR_CONFIG,
    create_optimizer,
    create_scheduler,
)

from .maml import MAML, MetaSGD

from .fomaml import FOMAML

from .anil import ANIL, BOIL

from .protonet import (
    PrototypicalNetworks,
    SoftPrototypicalNetworks,
    VariationalPrototypicalNetworks,
    compute_prototypical_accuracy,
)

from .relationnet import (
    RelationNetwork,
    DeepRelationNetwork,
    MultiScaleRelationNetwork,
    AttentionRelationNetwork,
)

from .matchingnet import (
    MatchingNetwork,
    FullContextMatchingNetwork,
    ConvolutionalMatchingNetwork,
    ImprintedWeights,
)

from .reptile import Reptile, MetaLearningBaseline

from .episode_generator import (
    EpisodeGenerator,
    TaskSampler,
    NWayKShotSampler,
    TransductiveTaskSampler,
    DomainShiftSampler,
    create_episode_loader,
)

from .episodic_trainer import (
    EpisodicTrainer,
    FewShotEvaluator,
    compute_confidence_interval,
)

from .encoders import (
    CNNEncoder,
    ResNetEncoder,
    FourLayerCNN,
    SixLayerCNN,
    ConvBlock,
    get_encoder,
)


__all__ = [
    # Types
    "FewShotTask",
    "MetaBatch",
    "AdaptationResult",
    "EvaluationResult",
    "TrainingState",
    "MetaLearningConfig",
    "DistanceMetric",
    "EpisodeType",
    # Config
    "MAMLConfig",
    "PrototypicalConfig",
    "RelationNetworkConfig",
    "MatchingNetworkConfig",
    "EpisodeConfig",
    "TrainingConfig",
    "EvaluatorConfig",
    "DEFAULT_MAML_CONFIG",
    "DEFAULT_PROTONET_CONFIG",
    "DEFAULT_RELATION_CONFIG",
    "DEFAULT_MATCHING_CONFIG",
    "DEFAULT_EPISODE_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_EVALUATOR_CONFIG",
    "create_optimizer",
    "create_scheduler",
    # MAML variants
    "MAML",
    "MetaSGD",
    "FOMAML",
    "ANIL",
    "BOIL",
    # Prototypical Networks
    "PrototypicalNetworks",
    "SoftPrototypicalNetworks",
    "VariationalPrototypicalNetworks",
    "compute_prototypical_accuracy",
    # Relation Networks
    "RelationNetwork",
    "DeepRelationNetwork",
    "MultiScaleRelationNetwork",
    "AttentionRelationNetwork",
    # Matching Networks
    "MatchingNetwork",
    "FullContextMatchingNetwork",
    "ConvolutionalMatchingNetwork",
    "ImprintedWeights",
    # Other algorithms
    "Reptile",
    "MetaLearningBaseline",
    # Episode generation
    "EpisodeGenerator",
    "TaskSampler",
    "NWayKShotSampler",
    "TransductiveTaskSampler",
    "DomainShiftSampler",
    "create_episode_loader",
    # Training
    "EpisodicTrainer",
    "FewShotEvaluator",
    "compute_confidence_interval",
    # Encoders
    "CNNEncoder",
    "ResNetEncoder",
    "FourLayerCNN",
    "SixLayerCNN",
    "ConvBlock",
    "get_encoder",
]

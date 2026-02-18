"""
Metric Learning Module for Fishstick

Comprehensive metric learning implementations including:
- Contrastive learning losses (NT-Xent, NPair, SupCon)
- Triplet mining strategies
- Hard negative sampling
- Learnable distance functions
- Metric-based few-shot learning
- Evaluation metrics
"""

from fishstick.metric_learning.base import (
    MetricSpace,
    EuclideanMetric,
    CosineMetric,
    ManhattanMetric,
    MahalanobisMetric,
    LearnableDistance,
    AttentionDistance,
    compute_distance_matrix,
    compute_similarity_matrix,
)

from fishstick.metric_learning.losses import (
    NTXentLoss,
    NPairLoss,
    SupConLoss,
    ProtoNCELoss,
    CircleLoss,
    TripletMarginLoss,
    MultiSimilarityLoss,
    ContrastiveLoss,
)

from fishstick.metric_learning.triplet_mining import (
    TripletMiner,
    RandomTripletMiner,
    HardNegativeMiner,
    SemihardNegativeMiner,
    DistanceWeightedMiner,
    AngularTripletMiner,
    BatchAllTripletMiner,
    BatchHardTripletMiner,
    NPairMiner,
    create_triplet_miner,
)

from fishstick.metric_learning.negative_sampling import (
    NegativeSampler,
    RandomNegativeSampler,
    SemihardNegativeSampler,
    HardestNegativeSampler,
    DistanceWeightedNegativeSampler,
    CurriculumNegativeSampler,
    InformedNegativeSampler,
    BatchNegativeSampler,
    create_negative_sampler,
)

from fishstick.metric_learning.learnable_distance import (
    LearnableEuclidean,
    LearnableMahalanobis,
    NeuralDistance,
    BilinearDistance,
    AttentionDistance,
    HyperbolicDistance,
    LearnableMetric,
    LearnableBregmanDistance,
    GaussianKernelDistance,
    CosineLearnableDistance,
    create_learnable_distance,
)

from fishstick.metric_learning.fewshot import (
    PrototypicalNetwork,
    RelationNetwork,
    MatchingNetwork,
    FEAT,
    MAMLFewShot,
    FewShotClassifier,
    EpisodicSampler,
)

from fishstick.metric_learning.evaluation import (
    recall_at_k,
    mean_recall_at_k,
    normalized_mutual_information,
    clustering_accuracy,
    f1_score_metric,
    adjusted_rand_index,
    mean_average_precision,
    precision_at_k,
    compute_distance_matrix,
    retrieval_metrics,
    clustering_metrics,
    evaluate_retrieval,
    evaluate_clustering,
    MetricTracker,
)


__all__ = [
    "MetricSpace",
    "EuclideanMetric",
    "CosineMetric",
    "ManhattanMetric",
    "MahalanobisMetric",
    "LearnableDistance",
    "AttentionDistance",
    "compute_distance_matrix",
    "compute_similarity_matrix",
    "NTXentLoss",
    "NPairLoss",
    "SupConLoss",
    "ProtoNCELoss",
    "CircleLoss",
    "TripletMarginLoss",
    "MultiSimilarityLoss",
    "ContrastiveLoss",
    "TripletMiner",
    "RandomTripletMiner",
    "HardNegativeMiner",
    "SemihardNegativeMiner",
    "DistanceWeightedMiner",
    "AngularTripletMiner",
    "BatchAllTripletMiner",
    "BatchHardTripletMiner",
    "NPairMiner",
    "create_triplet_miner",
    "NegativeSampler",
    "RandomNegativeSampler",
    "SemihardNegativeSampler",
    "HardestNegativeSampler",
    "DistanceWeightedNegativeSampler",
    "CurriculumNegativeSampler",
    "InformedNegativeSampler",
    "BatchNegativeSampler",
    "create_negative_sampler",
    "LearnableEuclidean",
    "LearnableMahalanobis",
    "NeuralDistance",
    "BilinearDistance",
    "AttentionDistance",
    "HyperbolicDistance",
    "LearnableMetric",
    "LearnableBregmanDistance",
    "GaussianKernelDistance",
    "CosineLearnableDistance",
    "create_learnable_distance",
    "PrototypicalNetwork",
    "RelationNetwork",
    "MatchingNetwork",
    "FEAT",
    "MAMLFewShot",
    "FewShotClassifier",
    "EpisodicSampler",
    "recall_at_k",
    "mean_recall_at_k",
    "normalized_mutual_information",
    "clustering_accuracy",
    "f1_score_metric",
    "adjusted_rand_index",
    "mean_average_precision",
    "precision_at_k",
    "retrieval_metrics",
    "clustering_metrics",
    "evaluate_retrieval",
    "evaluate_clustering",
    "MetricTracker",
]

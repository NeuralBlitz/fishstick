"""
Active Learning Module

Comprehensive active learning implementations including query strategies,
uncertainty estimation, batch selection, and evaluation tools.

This module provides state-of-the-art active learning algorithms for efficient
labeling and model training with minimal annotation cost.
"""

# Import all components from learning.py
from .learning import (
    # Base classes
    QueryStrategy,
    UncertaintyEstimator,
    BatchQueryStrategy,
    QueryResult,
    ActiveLearningState,
    # Query strategies
    UncertaintySampling,
    MarginSampling,
    EntropySampling,
    RandomSampling,
    ClusterBasedSampling,
    DensityWeightedSampling,
    # Uncertainty estimation
    MCDropoutUncertainty,
    EnsembleUncertainty,
    BayesianUncertainty,
    EvidentialUncertainty,
    # Batch active learning
    BatchBALD,
    CoreSet,
    BADGE,
    BatchActive,
    GreedyBatch,
    # Diversity sampling
    KCenterSampling,
    KMeansSampling,
    RepresentativeSampling,
    DiversityAwareSampling,
    AdversarialSampling,
    # Expected model change
    EGL,
    BALD,
    VariationRatio,
    InformationGain,
    # Pool and stream
    PoolBasedAL,
    StreamBasedAL,
    MembershipQuerySynthesis,
    # Multi-task
    MultiTaskAL,
    TransferActive,
    DomainAdaptiveAL,
    # Evaluation
    LearningCurve,
    AnnotationCost,
    ALBenchmark,
    ALVisualization,
    # Integration
    ActiveDataset,
    ActiveTrainer,
    ActiveLoop,
    # Factories
    create_query_strategy,
    create_batch_strategy,
    create_uncertainty_estimator,
    # Utilities
    compute_query_diversity,
    compute_coverage,
    active_learning_summary,
)

__all__ = [
    # Base classes
    "QueryStrategy",
    "UncertaintyEstimator",
    "BatchQueryStrategy",
    "QueryResult",
    "ActiveLearningState",
    # Query strategies
    "UncertaintySampling",
    "MarginSampling",
    "EntropySampling",
    "RandomSampling",
    "ClusterBasedSampling",
    "DensityWeightedSampling",
    # Uncertainty estimation
    "MCDropoutUncertainty",
    "EnsembleUncertainty",
    "BayesianUncertainty",
    "EvidentialUncertainty",
    # Batch active learning
    "BatchBALD",
    "CoreSet",
    "BADGE",
    "BatchActive",
    "GreedyBatch",
    # Diversity sampling
    "KCenterSampling",
    "KMeansSampling",
    "RepresentativeSampling",
    "DiversityAwareSampling",
    "AdversarialSampling",
    # Expected model change
    "EGL",
    "BALD",
    "VariationRatio",
    "InformationGain",
    # Pool and stream
    "PoolBasedAL",
    "StreamBasedAL",
    "MembershipQuerySynthesis",
    # Multi-task
    "MultiTaskAL",
    "TransferActive",
    "DomainAdaptiveAL",
    # Evaluation
    "LearningCurve",
    "AnnotationCost",
    "ALBenchmark",
    "ALVisualization",
    # Integration
    "ActiveDataset",
    "ActiveTrainer",
    "ActiveLoop",
    # Factories
    "create_query_strategy",
    "create_batch_strategy",
    "create_uncertainty_estimator",
    # Utilities
    "compute_query_diversity",
    "compute_coverage",
    "active_learning_summary",
]

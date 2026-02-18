"""
fishstick.recommenders
======================

Comprehensive recommender system tools for the fishstick framework.

This module provides implementations of:
- Collaborative Filtering: Memory-based and model-based approaches
- Matrix Factorization: SVD, NMF, ALS, Bayesian MF
- Neural Recommenders: NCF, DeepFM, Wide&Deep, AutoRec
- Session-based: GRU4Rec, SASRec
- Evaluation: Ranking, rating, and business metrics

Key Classes:
-----------
- CollaborativeFiltering: Base class for collaborative methods
- MatrixFactorization: Abstract base for MF models
- NeuralRecommender: Neural network-based recommenders
- SessionRecommender: Session-based recommendation models
- RecommenderMetrics: Evaluation metrics for recommenders
"""

from .base import (
    InteractionMatrix,
    UserItemDataset,
    SparseInteractionMatrix,
    TrainTestSplit,
    RecommenderBase,
)

from .collaborative import (
    UserBasedCF,
    ItemBasedCF,
    CosineSimilarity,
    PearsonCorrelation,
    JaccardSimilarity,
    kNNCollaborativeFiltering,
)

from .matrix_factorization import (
    SVDRecommender,
    NMFRecommender,
    ALSRecommender,
    BayesianMatrixFactorization,
    SVDPlusPlus,
    BiasedMF,
    WeightedRegularizedMF,
)

from .neural_recommenders import (
    NeuMF,
    DeepFM,
    WideAndDeep,
    AutoRec,
    FactorizationMachine,
    DeepFactorizationMachine,
    WideComponent,
    DeepComponent,
    EmbeddingLayer,
)

from .session_based import (
    GRU4Rec,
    SASRec,
    SessionGRU,
    SelfAttentiveSequential,
    NextItemPredictor,
    SessionEncoder,
)

from .evaluation import (
    RecommenderMetrics,
    RankingMetrics,
    RatingMetrics,
    DiversityMetrics,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    mrr_at_k,
    rmse,
    mae,
    hit_rate,
    coverage,
    diversity,
    novelty,
    OnlineEvaluator,
    OfflineEvaluator,
)

__all__ = [
    # Base
    "InteractionMatrix",
    "UserItemDataset",
    "SparseInteractionMatrix",
    "TrainTestSplit",
    "RecommenderBase",
    # Collaborative Filtering
    "UserBasedCF",
    "ItemBasedCF",
    "CosineSimilarity",
    "PearsonCorrelation",
    "JaccardSimilarity",
    "kNNCollaborativeFiltering",
    # Matrix Factorization
    "SVDRecommender",
    "NMFRecommender",
    "ALSRecommender",
    "BayesianMatrixFactorization",
    "SVDPlusPlus",
    "BiasedMF",
    "WeightedRegularizedMF",
    # Neural Recommenders
    "NeuMF",
    "DeepFM",
    "WideAndDeep",
    "AutoRec",
    "FactorizationMachine",
    "DeepFactorizationMachine",
    "WideComponent",
    "DeepComponent",
    "EmbeddingLayer",
    # Session-based
    "GRU4Rec",
    "SASRec",
    "SessionGRU",
    "SelfAttentiveSequential",
    "NextItemPredictor",
    "SessionEncoder",
    # Evaluation
    "RecommenderMetrics",
    "RankingMetrics",
    "RatingMetrics",
    "DiversityMetrics",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "rmse",
    "mae",
    "hit_rate",
    "coverage",
    "diversity",
    "novelty",
    "OnlineEvaluator",
    "OfflineEvaluator",
]

__version__ = "0.1.0"

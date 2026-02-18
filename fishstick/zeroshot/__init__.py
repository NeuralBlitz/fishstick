"""
Zero-Shot Learning (ZSL) Module for Fishstick

This module provides comprehensive implementations of state-of-the-art
zero-shot learning methods, including embedding-based approaches, generative
models, semantic embeddings, and evaluation metrics.

Example usage:
    >>> from fishstick.zeroshot import ALE, ZSLConfig
    >>> config = ZSLConfig(feature_dim=2048, semantic_dim=300, num_classes=50)
    >>> model = ALE(config)
    >>> # Train and evaluate
    >>> model.fit(train_loader, val_loader)
"""

from .learning import (
    # Base classes
    ZSLConfig,
    ZSLBase,
    # Embedding-based methods
    DeViSE,
    ALE,
    SJE,
    LatEm,
    ESZSL,
    SYNC,
    SAE,
    # Generative methods
    ZSLGAN,
    FCLSWGAN,
    CycleWCL,
    LisGAN,
    FREE,
    LsrGAN,
    # Semantic embeddings
    AttributeVectors,
    WordEmbeddings,
    SentenceEmbeddings,
    ClassDescription,
    HierarchicalEmbeddings,
    # Generalized ZSL
    CalibratedStacking,
    RelationNet,
    DeepEmbedding,
    GPZSL,
    FVAE,
    # Feature augmentation
    FeatureGenerating,
    SemanticAugmentation,
    MixupZSL,
    FeatureRefinement,
    # Transductive ZSL
    QuasiFullySupervised,
    DomainAdaptationZSL,
    SemiSupervisedZSL,
    SelfSupervisedZSL,
    # Evaluation
    ZSLAccuracy,
    HarmonicMean,
    AUSUC,
    PerClassMetrics,
    # Few-shot ZSL
    FSLZSL,
    GeneralizedFSLZSL,
    EpisodeZSL,
    # Utilities
    create_zsl_dataset,
    split_seen_unseen,
    compute_class_similarity,
    evaluate_zsl_model,
)

__all__ = [
    # Base classes
    "ZSLConfig",
    "ZSLBase",
    # Embedding-based methods
    "DeViSE",
    "ALE",
    "SJE",
    "LatEm",
    "ESZSL",
    "SYNC",
    "SAE",
    # Generative methods
    "ZSLGAN",
    "FCLSWGAN",
    "CycleWCL",
    "LisGAN",
    "FREE",
    "LsrGAN",
    # Semantic embeddings
    "AttributeVectors",
    "WordEmbeddings",
    "SentenceEmbeddings",
    "ClassDescription",
    "HierarchicalEmbeddings",
    # Generalized ZSL
    "CalibratedStacking",
    "RelationNet",
    "DeepEmbedding",
    "GPZSL",
    "FVAE",
    # Feature augmentation
    "FeatureGenerating",
    "SemanticAugmentation",
    "MixupZSL",
    "FeatureRefinement",
    # Transductive ZSL
    "QuasiFullySupervised",
    "DomainAdaptationZSL",
    "SemiSupervisedZSL",
    "SelfSupervisedZSL",
    # Evaluation
    "ZSLAccuracy",
    "HarmonicMean",
    "AUSUC",
    "PerClassMetrics",
    # Few-shot ZSL
    "FSLZSL",
    "GeneralizedFSLZSL",
    "EpisodeZSL",
    # Utilities
    "create_zsl_dataset",
    "split_seen_unseen",
    "compute_class_similarity",
    "evaluate_zsl_model",
]

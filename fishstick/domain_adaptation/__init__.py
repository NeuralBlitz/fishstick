"""
Domain Adaptation & Transfer Learning Module for Fishstick.

This module provides comprehensive domain adaptation techniques for transferring
knowledge from source to target domains. Includes adversarial methods, moment
matching, domain-specific batch normalization, and transfer learning utilities.

Submodules:
    adversarial: Adversarial domain adaptation methods (DANN, ADDA, MCDAN)
    moment_matching: Moment matching methods (MMDA, DeepJDOT)
    batch_norm: Domain-specific batch normalization (AdaBN, DBN, SHiP)
    transfer_utils: Transfer learning utilities
    domain_confusion: Domain confusion losses (MMD, CMD, DDC)
    components: Reusable components for domain adaptation
    trainer: Training utilities and evaluation

Example:
    >>> from fishstick.domain_adaptation import DANN, TransferLearner
    >>> dann = DANN(feature_dim=512, num_classes=10)
    >>> learner = TransferLearner(source_model=model, target_data=target_loader)
    >>> learner.fine_tune(epochs=100)
"""

from __future__ import annotations

from fishstick.domain_adaptation.adversarial import (
    ADDA,
    DANN,
    MCDAN,
    CDAN,
    GradientReversalLayer,
    GradientReversal,
    DomainDiscriminator,
)
from fishstick.domain_adaptation.moment_matching import (
    MMDA,
    DeepJDOT,
    CMD,
    MomentMatchingLoss,
    KornetovLoss,
)
from fishstick.domain_adaptation.batch_norm import (
    AdaBN,
    DomainBatchNorm,
    SHiP,
    AdaptiveBatchNorm,
    BatchNormAdapter,
)
from fishstick.domain_adaptation.transfer_utils import (
    TransferLearner,
    FineTuner,
    FeatureExtractor,
    ProgressiveFineTuning,
    LwF,
    TransferabilityEstimator,
)
from fishstick.domain_adaptation.domain_confusion import (
    MMDLoss,
    CMMDissimilarity,
    DDC,
    CORAL,
    DomainConfusionLoss,
)
from fishstick.domain_adaptation.components import (
    DomainClassifier,
    FeatureExtractorNetwork,
    EncoderDecoder,
    AttentionDomainAdaptation,
)
from fishstick.domain_adaptation.trainer import (
    DATrainer,
    DomainAdaptationEvaluator,
    compute_domain_accuracy,
    compute_hscore,
)

__all__ = [
    # Adversarial
    "DANN",
    "ADDA",
    "MCDAN",
    "CDAN",
    "GradientReversalLayer",
    "GradientReversal",
    "DomainDiscriminator",
    # Moment Matching
    "MMDA",
    "DeepJDOT",
    "CMD",
    "MomentMatchingLoss",
    "KornetovLoss",
    # Batch Norm
    "AdaBN",
    "DomainBatchNorm",
    "SHiP",
    "AdaptiveBatchNorm",
    "BatchNormAdapter",
    # Transfer Learning
    "TransferLearner",
    "FineTuner",
    "FeatureExtractor",
    "ProgressiveFineTuning",
    "LwF",
    "TransferabilityEstimator",
    # Domain Confusion
    "MMDLoss",
    "CMMDissimilarity",
    "DDC",
    "CORAL",
    "DomainConfusionLoss",
    # Components
    "DomainClassifier",
    "FeatureExtractorNetwork",
    "EncoderDecoder",
    "AttentionDomainAdaptation",
    # Trainer
    "DATrainer",
    "DomainAdaptationEvaluator",
    "compute_domain_accuracy",
    "compute_hscore",
]

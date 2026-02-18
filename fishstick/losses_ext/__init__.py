"""
Advanced Loss Functions Extension

Comprehensive collection of advanced loss functions for deep learning,
including focal losses, label smoothing, contrastive losses,
adversarial losses, and custom regression losses.

Modules:
    - focal: Focal loss variants for class imbalance handling
    - label_smoothing: Various label smoothing implementations
    - contrastive: Advanced contrastive learning losses
    - adversarial: GAN and adversarial training losses
    - regression: Robust regression loss functions

Example:
    >>> from fishstick.losses_ext import FocalLoss, SupConLoss, WGAN_GPLoss
    >>> focal = FocalLoss(alpha=0.25, gamma=2.0)
    >>> supcon = SupConLoss(temperature=0.07)
    >>> wgan_gp = WGAN_GPLoss(lambda_gp=10.0)
"""

from typing import Optional, List, Tuple

from .focal import (
    FocalLoss,
    BinaryFocalLoss,
    AdaptiveFocalLoss,
    ClassBalancedFocalLoss,
    PolynomialFocalLoss,
)

from .label_smoothing import (
    LabelSmoothingCrossEntropy,
    ConfidenceAwareSmoothing,
    ClassWeightedSmoothing,
    AdaptiveSmoothing,
    KnowledgeDistillationSmoothing,
    MixupSmoothing,
)

from .contrastive import (
    SupConLoss,
    ProtoNCELoss,
    MultiViewContrastiveLoss,
    InstanceContrastiveLoss,
    ClusterContrastiveLoss,
    NTNContrastiveLoss,
    AngularContrastiveLoss,
)

from .adversarial import (
    WGAN_GPLoss,
    HingeLossGAN,
    LeastSquaresGANLoss,
    DistributionMatchingLoss,
    GradientPenaltyLoss,
    SpectralNormPenalty,
    RelativisticAverageLoss,
    ConsistencyLoss,
    MultiScaleLoss,
    FeatureMatchingLoss,
    PerceptualAdversarialLoss,
)

from .regression import (
    HuberLoss,
    AdaptiveHuberLoss,
    LogCoshLoss,
    QuantileLoss,
    MultiQuantileLoss,
    TweedieLoss,
    PseudoHuberLoss,
    CauchyLoss,
    SmoothL1Loss,
    GaussianNLLLoss,
    LaplaceNLLLoss,
    CompositeLoss,
    TrimmedLoss,
)


__all__ = [
    "FocalLoss",
    "BinaryFocalLoss",
    "AdaptiveFocalLoss",
    "ClassBalancedFocalLoss",
    "PolynomialFocalLoss",
    "LabelSmoothingCrossEntropy",
    "ConfidenceAwareSmoothing",
    "ClassWeightedSmoothing",
    "AdaptiveSmoothing",
    "KnowledgeDistillationSmoothing",
    "MixupSmoothing",
    "SupConLoss",
    "ProtoNCELoss",
    "MultiViewContrastiveLoss",
    "InstanceContrastiveLoss",
    "ClusterContrastiveLoss",
    "NTNContrastiveLoss",
    "AngularContrastiveLoss",
    "WGAN_GPLoss",
    "HingeLossGAN",
    "LeastSquaresGANLoss",
    "DistributionMatchingLoss",
    "GradientPenaltyLoss",
    "SpectralNormPenalty",
    "RelativisticAverageLoss",
    "ConsistencyLoss",
    "MultiScaleLoss",
    "FeatureMatchingLoss",
    "PerceptualAdversarialLoss",
    "HuberLoss",
    "AdaptiveHuberLoss",
    "LogCoshLoss",
    "QuantileLoss",
    "MultiQuantileLoss",
    "TweedieLoss",
    "PseudoHuberLoss",
    "CauchyLoss",
    "SmoothL1Loss",
    "GaussianNLLLoss",
    "LaplaceNLLLoss",
    "CompositeLoss",
    "TrimmedLoss",
]


__version__ = "0.1.0"

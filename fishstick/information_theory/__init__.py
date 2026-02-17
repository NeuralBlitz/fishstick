"""
fishstick Information Theory Module

Comprehensive tools for information-theoretic analysis in machine learning:
- Entropy estimators (Shannon, Renyi, Tsallis, differential)
- Mutual information estimators (KNN, kernel, variational, InfoNCE)
- Information bottleneck implementations
- Entropy-based loss functions
- Compression-aware training
- Channel capacity estimators
"""

from fishstick.information_theory.entropy import (
    shannon_entropy,
    differential_entropy,
    renyi_entropy,
    tsallis_entropy,
    conditional_entropy,
    joint_entropy,
    entropy_rate,
    sample_entropy,
    approximate_entropy,
    bispectrum_entropy,
    EntropyEstimator,
)

from fishstick.information_theory.mutual_info import (
    mutual_information,
    knn_mi_estimator,
    kernel_mi_estimator,
    info_nce,
    variational_mi_bound,
    mine_estimator,
    conditional_mutual_information,
    total_correlation,
    pairwise_mutual_information,
    MutualInformationEstimator,
)

from fishstick.information_theory.bottleneck import (
    InformationBottleneck,
    VariationalInformationBottleneck,
    LagrangianIB,
    ConditionalInformationBottleneck,
    DeepInfoMax,
    soft_ib_loss,
    IBParameters,
)

from fishstick.information_theory.losses import (
    InfoNCE,
    BarlowTwinsLoss,
    EntropyRegularizedCrossEntropy,
    InformationGainLoss,
    RedundancyPenalty,
    CompressionAwareLoss,
    ConditionalEntropyLoss,
    MutualInformationMaximizationLoss,
    TripletInfoNCE,
    ClusterSeparationLoss,
    EntropyPenalizedBCE,
    spectral_entropy,
    gini_entropy,
)

from fishstick.information_theory.compression import (
    RateDistortionLoss,
    EntropyBottleneck,
    NeuralCompressionEncoder,
    NeuralCompressionDecoder,
    LearnedCompressionModel,
    QuantizationAwareTraining,
    BitAllocationOptimizer,
    AdaptiveEntropyCoding,
    CompressionRegularizer,
    RateDistortionScheduler,
    SSIMLoss,
    CompressionMetrics,
)

from fishstick.information_theory.channel_capacity import (
    ChannelCapacityEstimator,
    InformationRateEstimator,
    RateDistortionCurve,
    BlahutArimoto,
    NeuralChannel,
    ChannelCapacityLoss,
    InformationBottleneckChannel,
    GSNCapacity,
    TransferEntropyEstimator,
    compute_channel_snr,
    compute_shannon_limit,
    mutual_information_rate,
    ChannelMetrics,
)

__all__ = [
    "shannon_entropy",
    "differential_entropy",
    "renyi_entropy",
    "tsallis_entropy",
    "conditional_entropy",
    "joint_entropy",
    "entropy_rate",
    "sample_entropy",
    "approximate_entropy",
    "bispectrum_entropy",
    "EntropyEstimator",
    "mutual_information",
    "knn_mi_estimator",
    "kernel_mi_estimator",
    "info_nce",
    "variational_mi_bound",
    "mine_estimator",
    "conditional_mutual_information",
    "total_correlation",
    "pairwise_mutual_information",
    "MutualInformationEstimator",
    "InformationBottleneck",
    "VariationalInformationBottleneck",
    "LagrangianIB",
    "ConditionalInformationBottleneck",
    "DeepInfoMax",
    "soft_ib_loss",
    "IBParameters",
    "InfoNCE",
    "BarlowTwinsLoss",
    "EntropyRegularizedCrossEntropy",
    "InformationGainLoss",
    "RedundancyPenalty",
    "CompressionAwareLoss",
    "ConditionalEntropyLoss",
    "MutualInformationMaximizationLoss",
    "TripletInfoNCE",
    "ClusterSeparationLoss",
    "EntropyPenalizedBCE",
    "spectral_entropy",
    "gini_entropy",
    "RateDistortionLoss",
    "EntropyBottleneck",
    "NeuralCompressionEncoder",
    "NeuralCompressionDecoder",
    "LearnedCompressionModel",
    "QuantizationAwareTraining",
    "BitAllocationOptimizer",
    "AdaptiveEntropyCoding",
    "CompressionRegularizer",
    "RateDistortionScheduler",
    "SSIMLoss",
    "CompressionMetrics",
    "ChannelCapacityEstimator",
    "InformationRateEstimator",
    "RateDistortionCurve",
    "BlahutArimoto",
    "NeuralChannel",
    "ChannelCapacityLoss",
    "InformationBottleneckChannel",
    "GSNCapacity",
    "TransferEntropyEstimator",
    "compute_channel_snr",
    "compute_shannon_limit",
    "mutual_information_rate",
    "ChannelMetrics",
]

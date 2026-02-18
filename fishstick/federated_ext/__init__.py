"""
fishstick Federated Learning Extensions

Advanced federated learning tools for the fishstick AI framework including:
- Federated averaging algorithms (FedAvg, FedAdam, FedNova, SCAFFOLD, FedDyn)
- Client sampling strategies (Random, Round-Robin, FedCS, Oort, Power-of-Choice)
- Communication compression (Top-K, Random-K, QSGD, SignSGD, Error-Feedback)
- Heterogeneous data handling (non-IID partitioning, balanced aggregation)
- Federated evaluation (accuracy, fairness, privacy-preserving metrics)
- Advanced aggregation strategies (FedNova, FedProx, FedDyn, Async)

References:
- McMahan et al. (2017): "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Li et al. (2020): "Federated Optimization in Heterogeneous Networks"
- Reddi et al. (2021): "Adaptive Federated Optimization"
- Karimireddy et al. (2020): "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
- Wang et al. (2021): "Federated Learning with Non-IID Data"
- Hsu et al. (2019): "Measuring the Effects of Non-Identical Data Distribution"
- Nishio & Yonetani (2019): "Client Selection for Federated Learning with Heterogeneous Resources"
- Lai et al. (2021): "Oort: Efficient Federated Learning via Guided Client Selection"
- Alistarh et al. (2017): "QSGD: Randomized Quantization for Communication-Efficient FL"
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List

try:
    from .averaging import (
        AveragingMethod,
        AveragingConfig,
        BaseAveraging,
        FedAvgAveraging,
        FedAvgMomentumAveraging,
        FedAdamAveraging,
        FedOptAveraging,
        ScaffoldAveraging,
        FedNovaAveraging,
        FedDynAveraging,
        create_averaging_strategy,
    )

    _AVERAGING_AVAILABLE = True
except ImportError as e:
    _AVERAGING_AVAILABLE = False
    _AVERAGING_ERROR = str(e)

try:
    from .sampling import (
        SamplingStrategy,
        SamplingConfig,
        ClientInfo,
        BaseSamplingStrategy,
        RandomSamplingStrategy,
        RoundRobinSamplingStrategy,
        FedCSSamplingStrategy,
        OortSamplingStrategy,
        PowerOfChoiceSamplingStrategy,
        BanditSamplingStrategy,
        create_sampling_strategy,
    )

    _SAMPLING_AVAILABLE = True
except ImportError as e:
    _SAMPLING_AVAILABLE = False
    _SAMPLING_ERROR = str(e)

try:
    from .compression import (
        CompressionMethod,
        CompressionConfig,
        BaseCompressor,
        TopKCompressor,
        RandomKCompressor,
        QSGDCompressor,
        SignSGDCompressor,
        EFSignCompressor,
        NoCompression,
        create_compressor,
        GradientCompressor,
    )

    _COMPRESSION_AVAILABLE = True
except ImportError as e:
    _COMPRESSION_AVAILABLE = False
    _COMPRESSION_ERROR = str(e)

try:
    from .heterogeneity import (
        HeterogeneityType,
        HeterogeneityConfig,
        BaseDataPartitioner,
        DirichletPartitioner,
        ShardPartitioner,
        QuantitySkewPartitioner,
        FeatureSkewPartitioner,
        BalancedAggregator,
        LocalAdaptation,
        create_partitioner,
    )

    _HETEROGENEITY_AVAILABLE = True
except ImportError as e:
    _HETEROGENEITY_AVAILABLE = False
    _HETEROGENEITY_ERROR = str(e)

try:
    from .evaluation import (
        MetricType,
        EvaluationConfig,
        ClientMetrics,
        FederatedMetrics,
        BaseEvaluator,
        StandardEvaluator,
        PrivacyPreservingEvaluator,
        PersonalizedEvaluator,
        ConvergenceDiagnostics,
        create_evaluator,
    )

    _EVALUATION_AVAILABLE = True
except ImportError as e:
    _EVALUATION_AVAILABLE = False
    _EVALUATION_ERROR = str(e)

try:
    from .aggregation import (
        AggregationStrategy,
        AggregationConfig,
        BaseAggregationStrategy,
        FedNovaStrategy,
        FedProxStrategy,
        FedDynStrategy,
        AsyncAggregationStrategy,
        AdaptiveAggregationStrategy,
        create_aggregation_strategy,
    )

    _AGGREGATION_AVAILABLE = True
except ImportError as e:
    _AGGREGATION_AVAILABLE = False
    _AGGREGATION_ERROR = str(e)

__all__ = [
    # Averaging
    "AveragingMethod",
    "AveragingConfig",
    "FedAvgAveraging",
    "FedAvgMomentumAveraging",
    "FedAdamAveraging",
    "FedOptAveraging",
    "ScaffoldAveraging",
    "FedNovaAveraging",
    "FedDynAveraging",
    "create_averaging_strategy",
    # Sampling
    "SamplingStrategy",
    "SamplingConfig",
    "ClientInfo",
    "RandomSamplingStrategy",
    "RoundRobinSamplingStrategy",
    "FedCSSamplingStrategy",
    "OortSamplingStrategy",
    "PowerOfChoiceSamplingStrategy",
    "BanditSamplingStrategy",
    "create_sampling_strategy",
    # Compression
    "CompressionMethod",
    "CompressionConfig",
    "TopKCompressor",
    "RandomKCompressor",
    "QSGDCompressor",
    "SignSGDCompressor",
    "EFSignCompressor",
    "NoCompression",
    "create_compressor",
    "GradientCompressor",
    # Heterogeneity
    "HeterogeneityType",
    "HeterogeneityConfig",
    "DirichletPartitioner",
    "ShardPartitioner",
    "QuantitySkewPartitioner",
    "FeatureSkewPartitioner",
    "BalancedAggregator",
    "LocalAdaptation",
    "create_partitioner",
    # Evaluation
    "MetricType",
    "EvaluationConfig",
    "ClientMetrics",
    "FederatedMetrics",
    "StandardEvaluator",
    "PrivacyPreservingEvaluator",
    "PersonalizedEvaluator",
    "ConvergenceDiagnostics",
    "create_evaluator",
    # Aggregation
    "AggregationStrategy",
    "AggregationConfig",
    "FedNovaStrategy",
    "FedProxStrategy",
    "FedDynStrategy",
    "AsyncAggregationStrategy",
    "AdaptiveAggregationStrategy",
    "create_aggregation_strategy",
]

__version__ = "0.1.0"

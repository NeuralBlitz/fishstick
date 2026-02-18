"""
Anomaly Detection Module.

A comprehensive collection of anomaly detection algorithms including:
- Statistical anomaly detectors
- One-class classification
- Autoencoder-based detection
- Isolation forest variants
- Time series anomaly detection
- Ensemble methods
- Deep learning methods
- Streaming/online detection

Author: Fishstick Team
"""

from .statistical import (
    BaseStatisticalDetector,
    ZScoreDetector,
    IQRDetector,
    GrubbsDetector,
    ChiSquareDetector,
    MahalanobisDetector,
    AdjustedBoxplotDetector,
    GeneralizedESDDetector,
    DetectionResult,
)

from .autoencoder import (
    BaseAutoencoderDetector,
    VanillaAutoencoder,
    DenoisingAutoencoder,
    VariationalAutoencoder,
    ContractiveAutoencoder,
    SparseAutoencoder,
    AutoencoderResult,
)

from .oneclass import (
    BaseOneClassClassifier,
    KernelOneClassSVM,
    SVDDClassifier,
    KernelPCAOneClass,
    GaussianProcessOneClass,
    DeepOneClass,
    OneClassResult,
)

from .isolation_forest import (
    BaseIsolationForest,
    RandomizedBinaryTree,
    IsolationForestDetector,
    KernelizedIsolationForest,
    DeepIsolationForest,
    StreamingIsolationForest,
    EnsembleIsolationForest,
    AdaptiveIsolationForest,
    IsolationForestResult,
)

from .time_series import (
    BaseTimeSeriesDetector,
    StatisticalTimeSeriesDetector,
    LSTMAnomalyDetector,
    TransformerAnomalyDetector,
    SeasonalDecompositionDetector,
    ChangePointDetector,
    SpectralAnomalyDetector,
    KSigmaDetector,
    TimeSeriesResult,
)

from .ensemble import (
    BaseEnsembleDetector,
    VotingAnomalyEnsemble,
    StackingAnomalyEnsemble,
    ScoreAggregationEnsemble,
    DynamicEnsembleSelector,
    BaggingAnomalyDetector,
    AdaptiveWeightedEnsemble,
    UncertaintyAwareEnsemble,
    EnsembleResult,
)

from .deep_learning import (
    BaseDeepAnomalyDetector,
    GANomalyDetector,
    OneClassRNN,
    MemoryAugmentedAutoencoder,
    AttentionAnomalyDetector,
    DeviationNetwork,
    OneClassContrastive,
    DAGMMDetector,
    DeepLearningResult,
)

from .streaming import (
    BaseStreamingDetector,
    SlidingWindowDetector,
    DriftAdaptiveDetector,
    AdaptiveThresholdDetector,
    CumulativeSumDetector,
    ExponentialWeightedDetector,
    HalfSpaceTreesDetector,
    LodaDetector,
    ReservoirSamplingDetector,
    StreamingResult,
)

__version__ = "1.0.0"

__all__ = [
    # Statistical
    "BaseStatisticalDetector",
    "ZScoreDetector",
    "IQRDetector",
    "GrubbsDetector",
    "ChiSquareDetector",
    "MahalanobisDetector",
    "AdjustedBoxplotDetector",
    "GeneralizedESDDetector",
    "DetectionResult",
    # Autoencoder
    "BaseAutoencoderDetector",
    "VanillaAutoencoder",
    "DenoisingAutoencoder",
    "VariationalAutoencoder",
    "ContractiveAutoencoder",
    "SparseAutoencoder",
    "AutoencoderResult",
    # One-class
    "BaseOneClassClassifier",
    "KernelOneClassSVM",
    "SVDDClassifier",
    "KernelPCAOneClass",
    "GaussianProcessOneClass",
    "DeepOneClass",
    "OneClassResult",
    # Isolation Forest
    "BaseIsolationForest",
    "RandomizedBinaryTree",
    "IsolationForestDetector",
    "KernelizedIsolationForest",
    "DeepIsolationForest",
    "StreamingIsolationForest",
    "EnsembleIsolationForest",
    "AdaptiveIsolationForest",
    "IsolationForestResult",
    # Time Series
    "BaseTimeSeriesDetector",
    "StatisticalTimeSeriesDetector",
    "LSTMAnomalyDetector",
    "TransformerAnomalyDetector",
    "SeasonalDecompositionDetector",
    "ChangePointDetector",
    "SpectralAnomalyDetector",
    "KSigmaDetector",
    "TimeSeriesResult",
    # Ensemble
    "BaseEnsembleDetector",
    "VotingAnomalyEnsemble",
    "StackingAnomalyEnsemble",
    "ScoreAggregationEnsemble",
    "DynamicEnsembleSelector",
    "BaggingAnomalyDetector",
    "AdaptiveWeightedEnsemble",
    "UncertaintyAwareEnsemble",
    "EnsembleResult",
    # Deep Learning
    "BaseDeepAnomalyDetector",
    "GANomalyDetector",
    "OneClassRNN",
    "MemoryAugmentedAutoencoder",
    "AttentionAnomalyDetector",
    "DeviationNetwork",
    "OneClassContrastive",
    "DAGMMDetector",
    "DeepLearningResult",
    # Streaming
    "BaseStreamingDetector",
    "SlidingWindowDetector",
    "DriftAdaptiveDetector",
    "AdaptiveThresholdDetector",
    "CumulativeSumDetector",
    "ExponentialWeightedDetector",
    "HalfSpaceTreesDetector",
    "LodaDetector",
    "ReservoirSamplingDetector",
    "StreamingResult",
]

"""
fishstick Time Series Forecasting Module

Advanced time series forecasting tools including:
- Transformer-based forecasters (Informer, Autoformer)
- Probabilistic forecasting models
- Multi-horizon prediction heads
- Time series augmentation
- Anomaly detection for time series

Example:
    >>> from fishstick.timeseries_forecast import (
    ...     Informer,
    ...     Autoformer,
    ...     QuantileForecaster,
    ...     TimeSeriesDataset,
    ...     ForecastingTrainer,
    ... )
"""

from fishstick.timeseries_forecast.transformer_forecasters import (
    Informer,
    Autoformer,
    TransformerForecaster,
    TimeSeriesTransformerConfig,
    create_transformer_forecaster,
    FullAttention,
    ProbSparseAttention,
    AutoCorrelation,
    SeriesDecomposition,
)

from fishstick.timeseries_forecast.probabilistic_forecasters import (
    QuantileForecaster,
    DeepVariationalForecaster,
    GaussianProcessForecaster,
    EnsembleProbabilisticForecaster,
    MixtureDensityForecaster,
    QuantileLoss,
    ProbabilisticLosses,
    ForecastWithUncertainty,
    create_probabilistic_forecaster,
)

from fishstick.timeseries_forecast.multi_horizon_heads import (
    MultiHorizonHead,
    DirectMultiHorizonHead,
    RecursiveMultiHorizonHead,
    DirRecMultiHorizonHead,
    MultiResolutionHead,
    AttentionMultiHorizonHead,
    ResidualMultiHorizonHead,
    StrategySelector,
    MultiHorizonForecaster,
    create_multihorizon_head,
)

from fishstick.timeseries_forecast.ts_augmentation import (
    TimeSeriesAugmentation,
    TimeWarping,
    MagnitudeScaling,
    WindowSlice,
    NoiseInjection,
    Permutation,
    MagnitudeWarping,
    RandomCutout,
    TimeSeriesAugmenter,
    RandAugment,
    create_standard_augmentations,
    AugmentationConfig,
    AugmentationScheduler,
)

from fishstick.timeseries_forecast.anomaly_detection import (
    AnomalyDetector,
    StatisticalAnomalyDetector,
    IsolationForestDetector,
    LSTMAutoencoderDetector,
    OneClassSVMDetector,
    EnsembleAnomalyDetector,
    TimeSeriesAnomalyDetector,
    AnomalyDetectionResult,
    AnomalyScoreMode,
    create_anomaly_detector,
)

from fishstick.timeseries_forecast.data_utils import (
    TimeSeriesDataset,
    SlidingWindowDataset,
    TemporalEmbedding,
    PositionalEncoding,
    TimeSeriesMasker,
    TimeSeriesScaler,
    TimeSeriesDataLoader,
    create_sequences,
    temporal_train_test_split,
    DataConfig,
    TimeSeriesCollator,
    create_time_series_loader,
)

from fishstick.timeseries_forecast.training import (
    ForecastingMetrics,
    EarlyStopping,
    LearningRateScheduler,
    CheckpointManager,
    ForecastingTrainer,
    create_trainer,
)

__all__ = [
    # Transformer Forecasters
    "Informer",
    "Autoformer",
    "TransformerForecaster",
    "TimeSeriesTransformerConfig",
    "create_transformer_forecaster",
    "FullAttention",
    "ProbSparseAttention",
    "AutoCorrelation",
    "SeriesDecomposition",
    # Probabilistic Forecasters
    "QuantileForecaster",
    "DeepVariationalForecaster",
    "GaussianProcessForecaster",
    "EnsembleProbabilisticForecaster",
    "MixtureDensityForecaster",
    "QuantileLoss",
    "ProbabilisticLosses",
    "ForecastWithUncertainty",
    "create_probabilistic_forecaster",
    # Multi-horizon Heads
    "MultiHorizonHead",
    "DirectMultiHorizonHead",
    "RecursiveMultiHorizonHead",
    "DirRecMultiHorizonHead",
    "MultiResolutionHead",
    "AttentionMultiHorizonHead",
    "ResidualMultiHorizonHead",
    "StrategySelector",
    "MultiHorizonForecaster",
    "create_multihorizon_head",
    # Augmentation
    "TimeSeriesAugmentation",
    "TimeWarping",
    "MagnitudeScaling",
    "WindowSlice",
    "NoiseInjection",
    "Permutation",
    "MagnitudeWarping",
    "RandomCutout",
    "TimeSeriesAugmenter",
    "RandAugment",
    "create_standard_augmentations",
    "AugmentationConfig",
    "AugmentationScheduler",
    # Anomaly Detection
    "AnomalyDetector",
    "StatisticalAnomalyDetector",
    "IsolationForestDetector",
    "LSTMAutoencoderDetector",
    "OneClassSVMDetector",
    "EnsembleAnomalyDetector",
    "TimeSeriesAnomalyDetector",
    "AnomalyDetectionResult",
    "AnomalyScoreMode",
    "create_anomaly_detector",
    # Data Utils
    "TimeSeriesDataset",
    "SlidingWindowDataset",
    "TemporalEmbedding",
    "PositionalEncoding",
    "TimeSeriesMasker",
    "TimeSeriesScaler",
    "TimeSeriesDataLoader",
    "create_sequences",
    "temporal_train_test_split",
    "DataConfig",
    "TimeSeriesCollator",
    "create_time_series_loader",
    # Training
    "ForecastingMetrics",
    "EarlyStopping",
    "LearningRateScheduler",
    "CheckpointManager",
    "ForecastingTrainer",
    "create_trainer",
]

__version__ = "0.1.0"

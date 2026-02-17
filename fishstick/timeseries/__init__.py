"""
fishstick Time Series Module

Advanced time series forecasting and analysis tools.
"""

from fishstick.timeseries.models import (
    TemporalConvolutionalNetwork,
    TransformerTimeSeries,
    LSTMForecaster,
    GRUForecaster,
    WaveNet,
    TimeSeriesModel,
)
from fishstick.timeseries.analysis import (
    StationarityTest,
    SeasonalDecompose,
    AutocorrelationAnalysis,
    FourierTransform,
    WaveletTransform,
)
from fishstick.timeseries.preprocessing import (
    TimeSeriesScaler,
    SlidingWindow,
    HolidayFeatures,
    CalendarFeatures,
)
from fishstick.timeseries.forecasting import (
    BaseForecaster,
    ForecastingTrainer,
    LSTMForecaster,
    TransformerForecaster,
    NBeatsForecaster,
    DeepARForecaster,
    FeatureEngineer,
    TimeSeriesMetrics,
    TimeSeriesDataset,
    create_sequences,
    temporal_train_test_split,
    TimeSeriesScaler as ForecastingScaler,
    ScalerType,
    EnsembleForecaster,
    create_forecaster,
)

__all__ = [
    # Models
    "TemporalConvolutionalNetwork",
    "TransformerTimeSeries",
    "LSTMForecaster",
    "GRUForecaster",
    "WaveNet",
    "TimeSeriesModel",
    # Analysis
    "StationarityTest",
    "SeasonalDecompose",
    "AutocorrelationAnalysis",
    "FourierTransform",
    "WaveletTransform",
    # Preprocessing
    "TimeSeriesScaler",
    "SlidingWindow",
    "HolidayFeatures",
    "CalendarFeatures",
    # Forecasting
    "BaseForecaster",
    "ForecastingTrainer",
    "TransformerForecaster",
    "NBeatsForecaster",
    "DeepARForecaster",
    "FeatureEngineer",
    "TimeSeriesMetrics",
    "TimeSeriesDataset",
    "create_sequences",
    "temporal_train_test_split",
    "ForecastingScaler",
    "ScalerType",
    "EnsembleForecaster",
    "create_forecaster",
]

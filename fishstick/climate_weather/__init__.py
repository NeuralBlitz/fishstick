"""
fishstick Climate & Weather Module

Comprehensive climate and weather modeling tools including:
- Weather forecasting models (LSTM, Transformer, Graph-based)
- Climate pattern detection (ENSO, NAO, etc.)
- Spatio-temporal modeling
- Extreme event prediction
- Data assimilation (Kalman filters, 4D-Var)
- Evaluation metrics
"""

from fishstick.climate_weather.types import (
    CoordinateSystem,
    VerticalLevelType,
    VariableType,
    ClimateDataSpec,
    WeatherState,
    ClimateIndex,
    ForecastOutput,
    ExtremeEvent,
    AssimilationObservation,
    GridSpec,
    create_lat_lon_grid,
    normalize_coordinates,
    geo_distance,
)

from fishstick.climate_weather.data_utils import (
    ClimateDataset,
    ClimateDataLoader,
    WeatherStandardScaler,
    create_weather_patches,
    add_calendar_features,
    compute_climatology,
    compute_anomaly,
    temporal_train_test_split,
    load_era5_sample,
)

from fishstick.climate_weather.weather_forecaster import (
    BaseWeatherForecaster,
    LSTMWeatherForecaster,
    TransformerWeatherForecaster,
    ConvLSTMWeatherForecaster,
    ThreeDimensionalWeatherModel,
    WeatherForecasterConfig,
    create_weather_forecaster,
)

from fishstick.climate_weather.graph_weather import (
    GraphWeatherEncoder,
    GraphWeatherProcessor,
    GraphWeatherLayer,
    MultiMeshWeatherModel,
    SpatialGraphTransformer,
    create_icosphere_grid,
    create_lat_lon_graph,
    GraphCastModel,
    GraphCastLayer,
)

from fishstick.climate_weather.climate_patterns import (
    ClimatePattern,
    ENSODetector,
    NAODetector,
    ClimatePatternDetector,
    ClimateAutoencoder,
    PrincipalComponentAnalysis,
    EmpiricalOrthogonalFunction,
    compute_spatial_correlation,
    detect_regime_shifts,
)

from fishstick.climate_weather.spatiotemporal import (
    SpatioTemporalEncoder,
    TemporalConvolutionalNetwork,
    ConvGRUClimate,
    ConvLSTMClimate,
    AttentionSpatioTemporal,
    UNetClimate,
    LatentClimateModel,
    ClimateDynamicsPredictor,
)

from fishstick.climate_weather.extreme_events import (
    ExtremeEventPrediction,
    ExtremeEventClassifier,
    HeatWaveDetector,
    TropicalCycloneTracker,
    ExtremeValueModel,
    SevereStormPredictor,
    ProbabilityOfPrecipitation,
    compute_return_period,
    compute_exceedance_probability,
)

from fishstick.climate_weather.data_assimilator import (
    AssimilationState,
    KalmanFilter,
    ExtendedKalmanFilter,
    EnsembleKalmanFilter,
    ObservationOperator,
    FourDimensionalVar,
    ParticleFilter,
    compute_innovation,
    compute_analysis_increment,
)

from fishstick.climate_weather.climate_metrics import (
    compute_rmse,
    compute_mae,
    compute_mse,
    compute_correlation,
    compute_anomaly_correlation,
    compute_mean_absolute_percentage_error,
    compute_forecast_skill,
    compute_heidke_skill_score,
    compute_equitable_threat_score,
    compute_brier_score,
    compute_brier_skill_score,
    compute_roc_auc,
    compute_energy_score,
    compute_continuous_ranked_probability_score,
    compute_spatial_correlation,
    compute_pattern_correlation,
    compute_rmse_by_variable,
    compute_latitude_weighted_rmse,
    ClimateMetrics,
)

__all__ = [
    # Types
    "CoordinateSystem",
    "VerticalLevelType",
    "VariableType",
    "ClimateDataSpec",
    "WeatherState",
    "ClimateIndex",
    "ForecastOutput",
    "ExtremeEvent",
    "AssimilationObservation",
    "GridSpec",
    "create_lat_lon_grid",
    "normalize_coordinates",
    "geo_distance",
    # Data utilities
    "ClimateDataset",
    "ClimateDataLoader",
    "WeatherStandardScaler",
    "create_weather_patches",
    "add_calendar_features",
    "compute_climatology",
    "compute_anomaly",
    "temporal_train_test_split",
    "load_era5_sample",
    # Weather forecasting
    "BaseWeatherForecaster",
    "LSTMWeatherForecaster",
    "TransformerWeatherForecaster",
    "ConvLSTMWeatherForecaster",
    "ThreeDimensionalWeatherModel",
    "WeatherForecasterConfig",
    "create_weather_forecaster",
    # Graph weather
    "GraphWeatherEncoder",
    "GraphWeatherProcessor",
    "GraphWeatherLayer",
    "MultiMeshWeatherModel",
    "SpatialGraphTransformer",
    "create_icosphere_grid",
    "create_lat_lon_graph",
    "GraphCastModel",
    "GraphCastLayer",
    # Climate patterns
    "ClimatePattern",
    "ENSODetector",
    "NAODetector",
    "ClimatePatternDetector",
    "ClimateAutoencoder",
    "PrincipalComponentAnalysis",
    "EmpiricalOrthogonalFunction",
    "compute_spatial_correlation",
    "detect_regime_shifts",
    # Spatio-temporal
    "SpatioTemporalEncoder",
    "TemporalConvolutionalNetwork",
    "ConvGRUClimate",
    "ConvLSTMClimate",
    "AttentionSpatioTemporal",
    "UNetClimate",
    "LatentClimateModel",
    "ClimateDynamicsPredictor",
    # Extreme events
    "ExtremeEventPrediction",
    "ExtremeEventClassifier",
    "HeatWaveDetector",
    "TropicalCycloneTracker",
    "ExtremeValueModel",
    "SevereStormPredictor",
    "ProbabilityOfPrecipitation",
    "compute_return_period",
    "compute_exceedance_probability",
    # Data assimilation
    "AssimilationState",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "EnsembleKalmanFilter",
    "ObservationOperator",
    "FourDimensionalVar",
    "ParticleFilter",
    "compute_innovation",
    "compute_analysis_increment",
    # Metrics
    "compute_rmse",
    "compute_mae",
    "compute_mse",
    "compute_correlation",
    "compute_anomaly_correlation",
    "compute_mean_absolute_percentage_error",
    "compute_forecast_skill",
    "compute_heidke_skill_score",
    "compute_equitable_threat_score",
    "compute_brier_score",
    "compute_brier_skill_score",
    "compute_roc_auc",
    "compute_energy_score",
    "compute_continuous_ranked_probability_score",
    "compute_spatial_correlation",
    "compute_pattern_correlation",
    "compute_rmse_by_variable",
    "compute_latitude_weighted_rmse",
    "ClimateMetrics",
]

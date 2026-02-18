# TODO: Climate & Weather Modeling Module for fishstick

## Phase 1: Core Infrastructure
- [ ] 1.1 Create directory structure `/home/runner/workspace/fishstick/climate_weather/`
- [ ] 1.2 Create `__init__.py` with exports
- [ ] 1.3 Create base types and utilities module

## Phase 2: Weather Forecasting Models
- [ ] 2.1 `weather_forecaster.py` - LSTM/Transformer-based weather prediction
- [ ] 2.2 `graph_weather.py` - Graph neural network for weather modeling (inspired by GraphCast)
- [ ] 2.3 `ensemble_weather.py` - Multi-model ensemble for weather prediction

## Phase 3: Climate Pattern Detection
- [ ] 3.1 `climate_patterns.py` - Climate pattern recognition (ENSO, NAO, etc.)
- [ ] 3.2 `climate_indices.py` - Climate index computation
- [ ] 3.3 `anomaly_detector.py` - Climate anomaly detection

## Phase 4: Spatio-Temporal Modeling
- [ ] 4.1 `spatiotemporal.py` - Spatio-temporal convolutional models
- [ ] 4.2 `climate_encoder.py` - Latent climate representation learning
- [ ] 4.3 `downscaling.py` - Statistical/dynamical downscaling

## Phase 5: Extreme Event Prediction
- [ ] 5.1 `extreme_events.py` - Extreme weather event prediction
- [ ] 5.2 `heat_waves.py` - Heat wave detection and prediction
- [ ] 5.3 `tropical_cyclones.py` - Tropical cyclone tracking

## Phase 6: Data Assimilation
- [ ] 6.1 `data_assimilator.py` - Kalman filter and variants for data assimilation
- [ ] 6.2 `observation_ops.py` - Observation operators for assimilation
- [ ] 6.3 `reanalysis.py` - Reanalysis data tools

## Phase 7: Utilities & Integration
- [ ] 7.1 `climate_metrics.py` - Weather/climate-specific evaluation metrics
- [ ] 7.2 `data_utils.py` - Data loading and preprocessing utilities
- [ ] 7.3 Update main `__init__.py` to include climate_weather module

## Implementation Priority:
1. First: Core types + data_utils (Prerequisites)
2. Second: Weather forecasting models (Primary use case)
3. Third: Climate pattern detection (Analysis)
4. Fourth: Spatio-temporal modeling (Advanced)
5. Fifth: Extreme events (Critical applications)
6. Sixth: Data assimilation (Integration)
7. Seventh: Metrics + finalize exports

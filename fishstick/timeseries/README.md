# Time Series

Time series forecasting and analysis.

## Installation

```bash
pip install fishstick[timeseries]
```

## Overview

The `timeseries` module provides time series forecasting models and analysis tools.

## Usage

```python
from fishstick.timeseries import TemporalConvolutionalNetwork, LSTMForecaster, WaveNet

# TCN
tcn = TCN(
    input_size=1,
    output_size=1,
    num_channels=[64, 128, 256]
)

# LSTM Forecaster
lstm = LSTMForecaster(
    input_dim=10,
    hidden_dim=128,
    num_layers=2
)
forecast = lstm(input_sequence)

# N-BEATS
from fishstick.timeseries import NBeatsForecaster
nbeats = NBeatsForecaster(backcast_length=100, forecast_length=10)
```

## Models

| Model | Description |
|-------|-------------|
| `TemporalConvolutionalNetwork` | TCN |
| `TransformerTimeSeries` | Transformer for time series |
| `LSTMForecaster` | LSTM-based forecaster |
| `GRUForecaster` | GRU-based forecaster |
| `WaveNet` | WaveNet for time series |
| `NBeatsForecaster` | N-BEATS |
| `DeepARForecaster` | DeepAR |

## Analysis

| Class | Description |
|-------|-------------|
| `StationarityTest` | Test for stationarity |
| `SeasonalDecompose` | Seasonal decomposition |
| `AutocorrelationAnalysis` | ACF/PACF analysis |
| `SlidingWindow` | Sliding window transform |

## Examples

See `examples/timeseries/` for complete examples.

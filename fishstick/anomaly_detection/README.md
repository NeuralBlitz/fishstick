# Anomaly Detection Module

Comprehensive collection of anomaly detection algorithms for identifying outliers in data.

## Overview

This module provides state-of-the-art anomaly detection methods:

- **Statistical**: Z-score, IQR, Mahalanobis, Chi-square
- **One-Class**: OC-SVM, SVDD, Deep One-Class
- **Autoencoder-based**: VAE, Denoising, Contractive
- **Isolation Forest**: Variants for high-dimensional data
- **Time Series**: Temporal anomaly detection
- **Deep Learning**: Neural network-based detection
- **Ensemble**: Combining multiple detectors
- **Streaming**: Online anomaly detection

## Installation

```bash
pip install torch numpy scipy scikit-learn
```

## Quick Start

### Statistical Detection

```python
import numpy as np
from fishstick.anomaly_detection import ZScoreDetector, IQRDetector

# Z-score detection
zscore = ZScoreDetector(threshold=3.0)
scores = zscore.fit_predict(data)

# IQR detection
iqr = IQRDetector(k=1.5)
anomalies = iqr.fit_predict(data)
```

### Autoencoder-Based Detection

```python
import torch
from fishstick.anomaly_detection import VariationalAutoencoder

# VAE-based anomaly detector
detector = VariationalAutoencoder(
    input_dim=784,
    latent_dim=32,
    hidden_dims=[256, 128]
)

# Train on normal data
detector.fit(normal_data, epochs=50)

# Detect anomalies
anomaly_scores = detector.predict(test_data)
anomalies = anomaly_scores > threshold
```

### Isolation Forest

```python
from fishstick.anomaly_detection import IsolationForest

# Isolation Forest
iforest = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    max_samples=256
)

# Fit and predict
iforest.fit(train_data)
predictions = iforest.predict(test_data)
scores = iforest.score_samples(test_data)
```

### One-Class SVM

```python
from fishstick.anomaly_detection import KernelOneClassSVM

# One-Class SVM
ocsvm = KernelOneClassSVM(
    kernel='rbf',
    nu=0.1,
    gamma='scale'
)

ocsvm.fit(normal_data)
predictions = ocsvm.predict(test_data)
```

### Time Series Anomaly Detection

```python
from fishstick.anomaly_detection import TimeSeriesAnomalyDetector

# Time series detection
detector = TimeSeriesAnomalyDetector(
    method='lstm',
    window_size=50,
    threshold=0.95
)

# Train on normal time series
detector.fit(normal_timeseries)

# Detect anomalies
scores = detector.predict(test_timeseries)
```

### Ensemble Detection

```python
from fishstick.anomaly_detection import EnsembleAnomalyDetector

# Combine multiple detectors
ensemble = EnsembleAnomalyDetector(
    detectors=[
        IsolationForest(),
        OneClassSVM(),
        AutoencoderDetector()
    ],
    aggregation='average'
)

# Train all detectors
ensemble.fit(normal_data)

# Get ensemble scores
scores = ensemble.score(test_data)
```

## API Reference

### Statistical Detectors

| Class | Description |
|-------|-------------|
| `ZScoreDetector` | Z-score based detection |
| `IQRDetector` | Interquartile range |
| `GrubbsDetector` | Grubbs test for outliers |
| `MahalanobisDetector` | Mahalanobis distance |
| `ChiSquareDetector` | Chi-square test |

### Autoencoder Methods

| Class | Description |
|-------|-------------|
| `VanillaAutoencoder` | Standard AE |
| `DenoisingAutoencoder` | Denoising AE |
| `VariationalAutoencoder` | VAE-based |
| `ContractiveAutoencoder` | Contractive AE |

### One-Class Classification

| Class | Description |
|-------|-------------|
| `KernelOneClassSVM` | One-class SVM |
| `SVDD` | Support Vector Data Description |
| `DeepOneClass` | Deep one-class classification |

### Isolation Forest

| Class | Description |
|-------|-------------|
| `IsolationForest` | Standard isolation forest |
| `ExtendedIsolationForest` | Extended version |
| `KernelIsolationForest` | Kernel-based |

### Time Series

| Class | Description |
|-------|-------------|
| `TimeSeriesAnomalyDetector` | Base class |
| `LSTMDetector` | LSTM-based |
| `TransformerDetector` | Transformer-based |

### Ensemble

| Class | Description |
|-------|-------------|
| `EnsembleAnomalyDetector` | Combine multiple detectors |
| `WeightedEnsemble` | Learned weights |
| `BoostingDetector` | Boosting approach |

## Examples

```python
# Full pipeline example
from fishstick.anomaly_detection import (
    IsolationForest,
    VariationalAutoencoder,
    EnsembleAnomalyDetector,
    DetectionResult
)

# Create ensemble
ensemble = EnsembleAnomalyDetector([
    IsolationForest(n_estimators=100),
    VariationalAutoencoder(input_dim=64),
])

# Detect
result = ensemble.fit_predict(data)
print(f"Found {result.n_anomalies} anomalies")
```

## References

- Liu et al., "Isolation Forest" (ICDM 2008)
- Schölkopf et al., "Support Vector Method for Novelty Detection" (1999)
- Ruff et al., "Deep One-Class Classification" (ICML 2018)

## License

MIT License - see project root for details.

# Anomaly Detection

Anomaly and outlier detection algorithms.

## Installation

```bash
pip install fishstick[anomaly]
```

## Overview

The `anomaly` module provides anomaly detection algorithms for identifying outliers in data.

## Usage

```python
from fishstick.anomaly import AnomalyDetector

detector = AnomalyDetector(threshold=0.95)
is_anomaly = detector.predict(sample)
```

## Components

See source code for available detection methods.

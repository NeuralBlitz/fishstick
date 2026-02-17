# Uncertainty Quantification

Uncertainty estimation and out-of-distribution detection.

## Installation

```bash
pip install fishstick[uncertainty]
```

## Overview

The `uncertainty` module provides tools for uncertainty quantification in neural networks including ensemble methods, Bayesian approaches, and OOD detection.

## Usage

```python
from fishstick.uncertainty import MCAlphaDropout, EnsembleUncertainty, BayesianNN

# Monte Carlo Dropout
mc_dropout = MCAlphaDropout(model, n_samples=100)
predictions, uncertainty = mc_dropout.predict(x)

# Ensemble
ensemble = EnsembleUncertainty(models=[model1, model2, model3])
pred, uncertainty = ensemble.predict(x)

# OOD Detection
from fishstick.uncertainty import MaxSoftmaxOODDetector, EnergyOODDetector
ood_detector = MaxSoftmaxOODDetector(model)
is_ood = ood_detector.detect(x)
```

## Uncertainty Methods

| Method | Description |
|--------|-------------|
| `MCAlphaDropout` | Monte Carlo Dropout |
| `EnsembleUncertainty` | Ensemble-based uncertainty |
| `BayesianNN` | Bayesian neural network |

## OOD Detection

| Detector | Description |
|----------|-------------|
| `MaxSoftmaxOODDetector` | Maximum softmax probability |
| `EnergyOODDetector` | Energy score |
| `MahalanobisOODDetector` | Mahalanobis distance |

## Calibration

| Calibrator | Description |
|------------|-------------|
| `TemperatureScaledClassifier` | Temperature scaling |
| `DirichletCalibrator` | Dirichlet calibration |
| `ConformalPredictor` | Conformal prediction |

## Examples

See `examples/uncertainty/` for complete examples.

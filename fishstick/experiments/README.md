# Experiment Tracking

Utilities for tracking experiments, logging metrics, and comparing results.

## Installation

```bash
pip install fishstick[experiments]
```

## Overview

The `experiments` module provides a unified interface for experiment tracking and management, supporting various backends.

## Usage

```python
from fishstick.experiments import Experiment, ExperimentTracker

# Create experiment
exp = Experiment(
    name="my_experiment",
    config={"lr": 0.001, "batch_size": 32},
    tags=["baseline", "v1"]
)

# Log metrics
exp.log_metric("train_loss", 0.5, step=100)
exp.log_metric("val_accuracy", 0.85, step=100)

# Log artifacts
exp.log_artifact("model.pt", model.state_dict())

# Use tracker
tracker = ExperimentTracker()
tracker.log_experiment(exp)
```

## Classes

| Class | Description |
|-------|-------------|
| `Experiment` | Single experiment with config and metrics |
| `ExperimentTracker` | Manager for multiple experiments |

## Examples

See `examples/experiments/` for complete examples.

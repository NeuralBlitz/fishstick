# Hyperparameter Optimization

Hyperparameter search and optimization utilities.

## Installation

```bash
pip install fishstick[hyperopt]
```

## Overview

The `hyperopt` module provides various hyperparameter optimization strategies including grid search, random search, Bayesian optimization, and early stopping.

## Usage

```python
from fishstick.hyperopt import BayesianOptimization, GridSearch, RandomSearch

# Define search space
space = {
    "lr": (1e-4, 1e-2),
    "batch_size": [16, 32, 64, 128],
    "hidden_dim": [64, 128, 256],
}

# Bayesian optimization
optimizer = BayesianOptimization(
    objective=train_fn,
    search_space=space,
    n_trials=50
)

best_params = optimizer.optimize()
```

## Optimizers

| Optimizer | Description |
|-----------|-------------|
| `GridSearch` | Exhaustive grid search |
| `RandomSearch` | Random search |
| `BayesianOptimization` | Bayesian optimization with Gaussian Processes |
| `Hyperband` | Early stopping with successive halving |
| `PopulationBasedTraining` | PBT with evolution |

## Classes

| Class | Description |
|-------|-------------|
| `SearchSpace` | Define hyperparameter search space |
| `Trial` | Single optimization trial |
| `ResultAnalyzer` | Analyze optimization results |

## Examples

See `examples/hyperopt/` for complete examples.

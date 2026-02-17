# automl - Automated Machine Learning Module

## Overview

The `automl` module provides neural architecture search (NAS) and hyperparameter optimization algorithms for automatic model tuning.

## Purpose and Scope

This module enables:
- Automated hyperparameter optimization
- Neural architecture search
- Efficient search strategies (Random, Grid, Hyperband, Bayesian)
- Configurable search spaces

## Key Classes and Functions

### Search Space (`search.py`)

#### `SearchSpace`
Container for hyperparameter definitions.

```python
from fishstick.automl import SearchSpace, Choice, Uniform, LogUniform

space = SearchSpace()
space.params = {
    "learning_rate": LogUniform(1e-4, 1e-1),
    "hidden_dim": Choice([64, 128, 256, 512]),
    "num_layers": Choice([2, 3, 4, 5]),
    "dropout": Uniform(0.0, 0.5),
}

# Sample a configuration
config = space.sample()
# {"learning_rate": 0.003, "hidden_dim": 128, "num_layers": 3, "dropout": 0.2}
```

#### Parameter Types

##### `Choice`
Categorical choice parameter.

```python
from fishstick.automl import Choice

optimizer = Choice(["adam", "sgd", "rmsprop"])
value = optimizer.sample()  # Randomly picks one
```

##### `Uniform`
Continuous uniform parameter.

```python
from fishstick.automl import Uniform

dropout = Uniform(0.0, 0.5)
value = dropout.sample()  # Uniform random in [0.0, 0.5]
```

##### `LogUniform`
Log-uniform parameter for scales.

```python
from fishstick.automl import LogUniform

lr = LogUniform(1e-5, 1e-1)
value = lr.sample()  # Log-uniform in [1e-5, 1e-1]
```

##### `Conditional`
Conditional parameter based on other values.

```python
from fishstick.automl import Conditional

# Only use momentum if optimizer is sgd
momentum = Conditional(
    condition=lambda config: config["optimizer"] == "sgd",
    value=Uniform(0.0, 0.99)
)
```

### Search Algorithms

#### `RandomSearch`
Simple random search over the space.

```python
from fishstick.automl import RandomSearch, SearchSpace

def objective(config):
    # Train model with config and return score
    model = create_model(**config)
    score = train_and_evaluate(model)
    return score

search = RandomSearch(
    space=space,
    objective_fn=objective,
    maximize=True
)

result = search.search(n_trials=100)
print(result["best_config"])
print(result["best_score"])
```

#### `GridSearch`
Exhaustive grid search.

```python
from fishstick.automl import GridSearch

search = GridSearch(
    space=space,
    objective_fn=objective,
    maximize=True,
    points_per_dim=3  # For continuous params
)

result = search.search()
```

#### `Hyperband`
Early stopping for efficient search.

```python
from fishstick.automl import Hyperband

def objective_with_budget(config, budget):
    # budget = number of training epochs or iterations
    model = create_model(**config)
    score = train_for_epochs(model, epochs=budget)
    return score

search = Hyperband(
    space=space,
    objective_fn=objective_with_budget,
    maximize=True,
    max_iter=81,  # Maximum budget
    eta=3         # Reduction factor
)

result = search.search()
```

#### `BayesianOptimization`
Gaussian process-based optimization.

```python
from fishstick.automl import BayesianOptimization

search = BayesianOptimization(
    space=space,
    objective_fn=objective,
    maximize=True,
    n_initial_points=5  # Random points before GP
)

result = search.search(n_trials=100)
```

### Trial Management

#### `Trial`
Represents a single search trial.

```python
from fishstick.automl.search import Trial

trial = Trial(
    config={"lr": 0.001, "hidden_dim": 128},
    score=0.95,
    metrics={"accuracy": 0.95, "loss": 0.1},
    status="completed",
    train_time=120.5
)
```

#### `NASearch`
Base class for search algorithms.

## Dependencies

- `torch`: For model creation and training
- `numpy`: Numerical operations

## Usage Examples

### Complete AutoML Pipeline

```python
import torch
from fishstick.automl import (
    SearchSpace, Choice, Uniform, LogUniform,
    RandomSearch
)
from fishstick.frameworks import create_uniintelli

# Define search space
space = SearchSpace()
space.params = {
    "hidden_dim": Choice([64, 128, 256]),
    "num_layers": Choice([2, 3, 4]),
    "learning_rate": LogUniform(1e-4, 1e-2),
    "dropout": Uniform(0.0, 0.5),
}

# Define objective
def objective(config):
    model = create_uniintelli(
        input_dim=784,
        output_dim=10,
        hidden_dim=config["hidden_dim"],
        n_layers=config["num_layers"],
        dropout=config["dropout"]
    )
    
    # Train for a few epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(5):
        # ... training loop ...
        pass
    
    return validation_accuracy

# Run search
search = RandomSearch(space, objective, maximize=True)
result = search.search(n_trials=50)

print(f"Best config: {result['best_config']}")
print(f"Best score: {result['best_score']}")
```

### Hyperband for Efficient Search

```python
from fishstick.automl import Hyperband

def objective_with_budget(config, budget):
    """Train with budget = number of epochs."""
    model = create_model(**config)
    
    for epoch in range(budget):
        train_one_epoch(model)
    
    return evaluate(model)

search = Hyperband(
    space=space,
    objective_fn=objective_with_budget,
    max_iter=27,  # Max epochs
    eta=3
)

result = search.search()
```

### Analyzing Results

```python
# After search completes
for trial in search.trials:
    if trial.status == "completed":
        print(f"Config: {trial.config}")
        print(f"Score: {trial.score}")
        print(f"Time: {trial.train_time:.1f}s")
```

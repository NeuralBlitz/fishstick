# Active Learning

Active learning strategies for efficient data labeling.

## Overview

This module provides implementations of various active learning query strategies to minimize labeling effort while maximizing model performance.

## Query Strategies

### Uncertainty-Based
```python
from fishstick.active_learning import (
    UncertaintyStrategy,
    EntropyStrategy,
    MarginStrategy,
    LeastConfidenceStrategy,
)

# Entropy-based selection
strategy = EntropyStrategy(model)
query_indices = strategy.query(unlabeled_data, n_samples=100)
```

### Diversity-Based
```python
from fishstick.active_learning import (
    DiversityStrategy,
    BADGEStrategy,
    CoreSetStrategy,
)

# Core-set selection
strategy = CoreSetStrategy(model)
query_indices = strategy.query(unlabeled_data, labeled_data, n_samples=100)
```

### Advanced
```python
from fishstick.active_learning import (
    VAALStrategy,
    ExpectedModelChangeStrategy,
    AdversarialDeepFoolingStrategy,
)

# Variational Adversarial Active Learning
strategy = VAALStrategy(model, latent_dim=32)
query_indices = strategy.query(unlabeled_data, n_samples=100)
```

## Evaluation

```python
from fishstick.active_learning import (
    ActiveLearningEvaluator,
    simulate_labeling,
    compute_learning_curve,
    compute_label_efficiency,
)

# Simulate active learning process
evaluator = ActiveLearningEvaluator(model, strategy)
learning_curve = evaluator.evaluate(
    initial_data,
    unlabeled_pool,
    n_iterations=10,
    n_queries=100
)

# Compute metrics
efficiency = compute_label_efficiency(learning_curve)
```

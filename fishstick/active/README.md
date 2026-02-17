# Active Learning

Query strategies for efficient labeling and data selection.

## Installation

```bash
pip install fishstick[active]
```

## Overview

The `active` module provides implementations of various active learning strategies for efficiently selecting the most informative data points for labeling. This reduces the amount of labeled data required to train high-quality models.

## Query Strategies

### Uncertainty Sampling

```python
from fishstick.active import UncertaintySampling, EntropySampling, MarginSampling

# Create uncertainty sampler
sampler = UncertaintySampling(model, X_pool, y_pool)

# Query most uncertain samples
indices = sampler.query(n_samples=10)
```

### Bayesian Active Learning (BALD)

```python
from fishstick.active import BALD

bald_sampler = BALD(model, X_pool, y_pool)
indices = bald_sampler.query(n_samples=10)
```

### CoreSet

```python
from fishstick.active import CoreSet

core_set = CoreSet(model, X_pool, y_pool)
indices = core_set.query(n_samples=10, batch_size=16)
```

## Available Strategies

| Strategy | Description |
|----------|-------------|
| `UncertaintySampling` | Select samples with highest prediction uncertainty |
| `MarginSampling` | Select samples where difference between top-2 predictions is small |
| `EntropySampling` | Select samples with highest entropy in predictions |
| `BALD` | Bayesian Active Learning by Disagreement |
| `CoreSet` | Greedy selection based on feature space coverage |

## Examples

See `examples/active_learning/` for complete examples.

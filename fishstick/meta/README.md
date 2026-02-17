# Meta-Learning

MAML, ProtoNet, Reptile, and few-shot learning methods.

## Installation

```bash
pip install fishstick[meta]
```

## Overview

The `meta` module provides implementations of popular meta-learning algorithms for few-shot learning tasks.

## Usage

```python
from fishstick.meta import MAML, PrototypicalNetworks, Reptile

# MAML for few-shot learning
maml = MAML(
    model=model,
    inner_lr=0.01,
    inner_steps=5,
    outer_lr=0.001
)

# Meta-training
meta_train(model, tasks, num_epochs=100)

# Prototypical Networks
proto_net = PrototypicalNetworks(embedding_dim=64)
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| `MAML` | Model-Agnostic Meta-Learning |
| `Reptile` | First-order meta-learning |
| `PrototypicalNetworks` | Prototypical networks for few-shot classification |
| `MatchingNetworks` | Matching networks |
| `RelationNetworks` | Relation networks |

## Supporting Classes

| Class | Description |
|-------|-------------|
| `Task` | Few-shot task definition |
| `Episode` | Training episode |
| `TaskSampler` | Sample tasks for meta-training |
| `EpisodeDataset` | Dataset that generates episodes |

## Functions

| Function | Description |
|----------|-------------|
| `meta_train` | Meta-training loop |
| `meta_validate` | Meta-validation |
| `meta_test` | Meta-testing / few-shot evaluation |
| `evaluate_few_shot` | Evaluate few-shot classification |

## Examples

See `examples/meta_learning/` for complete examples.

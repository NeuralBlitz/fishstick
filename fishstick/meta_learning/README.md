# Meta-Learning Module

A comprehensive meta-learning framework implementing state-of-the-art few-shot learning algorithms including MAML, Prototypical Networks, and Relation Networks.

## Overview

This module provides implementations of popular meta-learning algorithms for few-shot learning tasks, where the goal is to quickly adapt to new tasks with minimal training examples.

### Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| **MAML** | Model-Agnostic Meta-Learning - learns a good initialization |
| **FOMAML** | First-Order MAML - computationally efficient variant |
| **Reptile** | Simplified meta-learning algorithm |
| **MetaSGD** | Meta-learner that learns step sizes |
| **MetaLSTM** | LSTM-based meta-learner |
| **PrototypicalNetwork** | Prototypical Networks for few-shot classification |
| **RelationNetwork** | Learns a distance metric for few-shot learning |

## Quick Start

### MAML for Few-Shot Learning

```python
import torch
from fishstick.meta_learning import MAML

# Define your model
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Create MAML meta-learner
model = SimpleCNN(num_classes=5)
maml = MAML(model, lr=0.001, first_order=False)

# Meta-training
for episode in range(1000):
    # Sample tasks (5-way, 5-shot)
    support_x, support_y = ...  # 5 samples per class
    query_x, query_y = ...      # 15 samples per class
    
    # Inner loop: adapt to task
    adapted_model = maml.adapt(support_x, support_y)
    
    # Outer loop: evaluate on query
    loss = maml.meta_loss(adapted_model, query_x, query_y)
    maml.meta_step(loss)
```

### Prototypical Networks

```python
from fishstick.meta_learning import PrototypicalNetwork

# Create prototypical network
model = PrototypicalNetwork(
    encoder=torch.nn.Linear(64, 128),
    hidden_dim=128,
    num_classes=5
)

# Few-shot classification
prototypes = model.get_prototypes(support_x, support_y)
predictions = model.classify(query_x, prototypes)
```

## API Reference

### MAML

```python
MAML(
    model: nn.Module,
    lr: float = 0.001,           # Inner loop learning rate
    first_order: bool = False,   # Use FOMAML approximation
    num_inner_steps: int = 5     # Number of inner loop steps
)
```

| Method | Description |
|--------|-------------|
| `adapt(support_x, support_y)` | Adapt model to a task |
| `meta_loss(adapted_model, query_x, query_y)` | Compute meta-gradients |
| `meta_step(loss)` | Perform meta-update |

### FOMAML

```python
FOMAML(
    model: nn.Module,
    lr: float = 0.001,
    num_inner_steps: int = 5
)
```

### Reptile

```python
Reptile(
    model: nn.Module,
    lr: float = 0.001,
    inner_steps: int = 5
)
```

### PrototypicalNetwork

```python
PrototypicalNetwork(
    encoder: nn.Module,      # Feature encoder
    hidden_dim: int,         # Hidden dimension
    num_classes: int         # Number of classes
)
```

| Method | Description |
|--------|-------------|
| `forward(x)` | Forward pass |
| `get_prototypes(support_x, support_y)` | Compute class prototypes |
| `classify(query_x, prototypes)` | Classify query samples |

### RelationNetwork

```python
RelationNetwork(
    encoder: nn.Module,
    relation_module: nn.Module,
    hidden_dim: int = 128
)
```

## Examples

### N-way K-shot Classification

```python
import torch
from fishstick.meta_learning import MAML, PrototypicalNetwork

# N-way K-shot setting
N_WAY = 5
K_SHOT = 5
QUERY_SIZE = 15

def sample_episode(dataset, n_way, k_shot, query_size):
    """Sample a few-shot episode."""
    classes = torch.randperm(len(dataset.classes))[:n_way]
    support_x, support_y, query_x, query_y = [], [], [], []
    
    for i, cls in enumerate(classes):
        cls_data = dataset.get_class(cls)
        indices = torch.randperm(len(cls_data))[:k_shot + query_size]
        
        support_idx, query_idx = indices[:k_shot], indices[k_shot:]
        support_x.append(cls_data[support_idx])
        query_x.append(cls_data[query_idx])
        support_y.extend([i] * k_shot)
        query_idx = [i] * query_size
    
    return (torch.cat(support_x), torch.tensor(support_y),
            torch.cat(query_x), torch.tensor(query_y))

# Train with MAML
maml = MAML(model, lr=0.01)
for episode in range(10000):
    support_x, support_y, query_x, query_y = sample_episode(...)
    adapted = maml.adapt(support_x, support_y)
    loss = maml.meta_loss(adapted, query_x, query_y)
    maml.meta_step(loss)
```

## Installation Requirements

```bash
pip install torch numpy
```

## References

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- [Prototypical Networks for Few-Shot Learning](https://arxiv.org/abs/1703.05175)
- [Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025)

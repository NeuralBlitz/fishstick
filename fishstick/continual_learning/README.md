# Continual Learning Module

Comprehensive implementation of state-of-the-art continual learning methods for preventing catastrophic forgetting in neural networks.

## Overview

This module provides tools for training neural networks that can learn continuously from new data without forgetting previously learned knowledge. It implements major approaches including:

- **Experience Replay**: Store and replay past experiences
- **Elastic Weight Consolidation (EWC)**: Regularization-based protection
- **Progressive Networks**: Architecture expansion
- **Task-Agnostic Learning**: Methods without explicit task boundaries

## Installation

```bash
# Core dependencies (already included in fishstick)
pip install torch numpy
```

## Quick Start

### Basic Continual Learning Setup

```python
import torch
import torch.nn as nn
from fishstick.continual_learning import (
    ContinualTrainer,
    EWCRegularizer,
    ExperienceReplay,
    ForgettingMeasure
)

# Define your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Initialize EWC regularizer
ewc = EWCRegularizer(model, importance=1000)

# Initialize experience replay buffer
replay = ExperienceReplay(max_size=1000)

# Training loop for task
train eacher = ContinualTrainer(model, optimizer=torch.optim.Adam(model.parameters()))

for task_id, (train_loader, test_loader) in enumerate(task_loaders):
    # Train on current task with EWC
    for epoch in range(10):
        for batch in train_loader:
            inputs, labels = batch
            
            # EWC penalty from previous tasks
            ewc_penalty = ewc.compute_penalty()
            
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels) + ewc_penalty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Store important parameters for EWC
    ewc.register_task(model)
    
    # Add samples to replay buffer
    for batch in train_loader:
        replay.add(batch)
```

### Using Elastic Weight Consolidation

```python
from fishstick.continual_learning import OnlineEWC

# Online EWC for efficient continual learning
ewc = OnlineEWC(
    model=model,
    importance=1000,
    gamma=0.95  # EWC gamma parameter
)

# Training loop
for task_id, dataloader in enumerate(task_dataloaders):
    for epoch in range(10):
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            
            # Add EWC penalty
            if task_id > 0:
                ewc_loss = ewc.penalty()
                loss = loss + ewc_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Register task after training
    ewc.register_task_parameters()
```

### Experience Replay with Prioritization

```python
from fishstick.continual_learning import PrioritizedReplayBuffer

# Prioritized replay buffer
replay = PrioritizedReplayBuffer(
    max_size=5000,
    alpha=0.6,  # Priority exponent
    beta=0.4    # Importance sampling exponent
)

# Add experiences
for inputs, labels in dataloader:
    replay.add((inputs, labels))

# Sample with importance sampling
for _):
    batch, in range(10 weights, indices = replay.sample(batch_size=32, beta=0.4)
    
    outputs = model(batch[0])
    loss = nn.functional.cross_entropy(outputs, batch[1])
    
    # Apply importance sampling weights
    (weights * loss).mean().backward()
```

### Progressive Neural Networks

```python
from fishstick.continual_learning import ProgressiveNeuralNetwork

# Create progressive network for adding new tasks
pnn = ProgressiveNeuralNetwork(
    input_dim=784,
    hidden_dim=256,
    output_dim=10,
    num_tasks=5
)

# Add new task column
pnn.add_task_column()

# Train on task
for epoch in range(10):
    for inputs, labels in task_loader:
        outputs = pnn(inputs, task_id=current_task)
        loss = nn.functional.cross_entropy(outputs, labels)
        loss.backward()
```

### Gradient Episodic Memory (GEM)

```python
from fishstick.continual_learning import GradientEpisodicMemory

# GEM with episodic memory
gem = GradientEpisodicMemory(
    model,
    memory_size=1000,
    n_tasks=10
)

for task_id, dataloader in enumerate(task_dataloaders):
    for inputs, labels in dataloader:
        # Store in episodic memory
        gem.store_buffer(inputs, labels, task_id)
        
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)
        
        # Compute gradient projection for previous tasks
        gem_loss = gem.compute_continuation_loss()
        loss = loss + 0.1 * gem_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Available Methods

### Experience Replay

| Class | Description |
|-------|-------------|
| `ExperienceReplay` | Basic experience replay buffer |
| `ReservoirBuffer` | Reservoir sampling for fixed-size buffer |
| `WeightedReplayBuffer` | Replay with learned importance weights |
| `PrioritizedReplayBuffer` | Prioritized Experience Replay (PER) |
| `GenerativeReplayBuffer` | Replay with generated samples |

### Regularization Methods

| Class | Description |
|-------|-------------|
| `EWCRegularizer` | Elastic Weight Consolidation |
| `OnlineEWC` | Memory-efficient online EWC |
| `DiagonalEWC` | Diagonal approximation of Fisher |
| `SynapticIntelligence` | SI: Learning what not to forget |
| `MemoryAwareSynapses` | MAS: Importance-based protection |

### Progressive Architectures

| Class | Description |
|-------|-------------|
| `ProgressiveNeuralNetwork` | Networks that grow with new tasks |
| `PackNet` | PackNet: prune and freeze for tasks |
| `HardAttentionTask` | HAT: hard attention mechanisms |

### Task-Agnostic Learning

| Class | Description |
|-------|-------------|
| `TaskAgnosticContinualLearner` | Base class for task-agnostic methods |
| `OnlineContinualLearner` | Streaming learning setup |
| `StreamingLearner` | True online learning |

### Evaluation

```python
from fishstick.continual_learning import (
    ContinualMetrics,
    ForgettingMeasure,
    AverageAccuracy,
    BackwardTransfer
)

# Compute continual learning metrics
metrics = ContinualMetrics()

for task_id, test_loader in enumerate(test_loaders):
    accuracy = AverageAccuracy.compute(model, test_loader)
    forgetting = ForgettingMeasure.compute(model, task_id, test_loader)
    
    metrics.update(task_id, accuracy, forgetting)

print(f"Average Accuracy: {metrics.avg_accuracy}")
print(f"Average Forgetting: {metrics.avg_forgetting}")
```

## API Reference

### ContinualTrainer

```python
ContinualTrainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    gradient_clip: float = 1.0
)
```

### EWCRegularizer

```python
EWCRegularizer(
    model: nn.Module,
    importance: float = 1000.0,
    decay_factor: float = 0.95
)
```

## Examples

See `examples/continual_learning_example.py` for a complete example with multiple tasks.

## References

- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS 2017)
- Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning" (NeurIPS 2017)
- Chaudhry et al., "Efficient Lifelong Learning with A-GEM" (ICLR 2019)
- Mallya & Lazebnik, "PackNet: Adding Multiple Tasks to a Single Network" (CVPR 2018)

## License

MIT License - see project root for details.

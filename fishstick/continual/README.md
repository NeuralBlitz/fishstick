# Continual Learning

Methods for learning incrementally without catastrophic forgetting.

## Installation

```bash
pip install fishstick[continual]
```

## Overview

The `continual` module implements various continual learning methods that enable models to learn from new data while retaining knowledge from previous tasks.

## Usage

```python
from fishstick.continual import EWC, PackNet, MemoryReplay, GEM

# Elastic Weight Consolidation
ewc = EWC(model, importance=1000)
ewc.penalty(model, previous_params)

# Memory Replay
replay = MemoryReplay(model, memory_size=1000)
replay.store(task_data)
replay.sample()

# Gradient Episodic Memory
gem = GEM(model, memory_size=500, gradient_steps=1)
gem.remember()
```

## Available Methods

| Method | Description |
|--------|-------------|
| `EWC` | Elastic Weight Consolidation |
| `PackNet` | Prune and retain approach |
| `ProgressiveNeuralNetwork` | Progressive neural networks for new tasks |
| `MemoryReplay` | Experience replay buffer |
| `GEM` | Gradient Episodic Memory |
| `LwF` | Learning without Forgetting |
| `AGEM` | Averaged Gradient Episodic Memory |
| `PackMemory` | PackNet-style memory management |

## Examples

See `examples/continual_learning/` for complete examples.

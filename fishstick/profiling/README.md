# Model Profiling

Profiling, debugging, and analysis tools for neural networks.

## Installation

```bash
pip install fishstick[profiling]
```

## Overview

The `profiling` module provides utilities for profiling model performance, analyzing weights, detecting dead neurons, and debugging training issues.

## Usage

```python
from fishstick.profiling import ModelProfiler, GradientChecker, DeadNeuronDetector

# Profile model
profiler = ModelProfiler(model)
profiler.profile(input_data)

# Check gradients
checker = GradientChecker(model)
gradients_ok = checker.check_gradients(data)

# Detect dead neurons
detector = DeadNeuronDetector(model)
dead_neurons = detector.detect(data)
```

## Tools

| Tool | Description |
|------|-------------|
| `ModelProfiler` | Profile model inference time and memory |
| `GradientChecker` | Check gradient flow and values |
| `DeadNeuronDetector` | Identify dead neurons |
| `WeightAnalyzer` | Analyze weight distributions |
| `TrainingDebugger` | Debug training issues |

## Utility Functions

| Function | Description |
|----------|-------------|
| `measure_time` | Measure execution time |
| `profile_memory_usage` | Profile memory consumption |

## Examples

See `examples/profiling/` for complete examples.

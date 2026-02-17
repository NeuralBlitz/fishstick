# benchmarks - Benchmark Suite Module

## Overview

The `benchmarks` module provides standardized benchmarks for evaluating unified intelligence frameworks, including physics-informed learning, symbolic reasoning, and verification tasks.

## Purpose and Scope

This module enables:
- Framework performance comparison
- Conservation law verification
- Energy conservation benchmarks
- Formal verification benchmarks

## Key Classes and Functions

### Datasets

#### `SyntheticPhysicsDataset`
Generates trajectories from physical systems for testing conservation laws.

```python
from fishstick.benchmarks import SyntheticPhysicsDataset

# Harmonic oscillator
dataset = SynthenticPhysicsDataset(
    n_samples=1000,
    seq_len=50,
    system="harmonic",  # or "pendulum"
    noise=0.01
)

# Returns (q, p) trajectories
for traj_in, traj_out in dataloader:
    # traj_in: [batch, seq_len-1, 2]
    # traj_out: [batch, seq_len-1, 2]
```

#### `SyntheticClassificationDataset`
Synthetic classification data for framework testing.

```python
from fishstick.benchmarks import SyntheticClassificationDataset

dataset = SyntheticClassificationDataset(
    n_samples=1000,
    input_dim=784,
    n_classes=10
)
```

### Benchmark Runner

#### `BenchmarkRunner`
Orchestrates all benchmarks and collects results.

```python
from fishstick.benchmarks import BenchmarkRunner

runner = BenchmarkRunner(device="cuda")
results = runner.run_all(n_epochs=5, batch_size=32)
runner.print_summary()
```

**Key Methods:**
- `run_all()`: Run complete benchmark suite
- `benchmark_frameworks()`: Compare all 6 frameworks (UniIntelli, HSCA, UIA, SCIF, UIF, UIS)
- `benchmark_conservation()`: Test energy conservation
- `benchmark_verification()`: Test verification capabilities
- `print_summary()`: Display results

### Benchmark Results

Results are stored in a dictionary:

```python
{
    "UniIntelli": {
        "final_loss": 0.123,
        "training_time": 45.2,
        "n_parameters": 1000000,
        "loss_history": [2.1, 1.5, 0.8, 0.3, 0.123]
    },
    "EnergyConservation": {
        "mean_violation": 1e-6,
        "max_violation": 1e-5
    },
    "Verification": {
        "robustness_verified": True,
        "lipschitz_bound": 1.5
    }
}
```

### Convenience Functions

#### `run_benchmarks`
Quick function to run all benchmarks.

```python
from fishstick.benchmarks import run_benchmarks

results = run_benchmarks(n_epochs=5, batch_size=32)
```

## Dependencies

- `torch`: PyTorch for model training
- `numpy`: Numerical computations
- `fishstick.frameworks`: All 6 frameworks
- `fishstick.dynamics`: Hamiltonian neural networks
- `fishstick.verification`: Verification pipeline

## Usage Examples

### Complete Benchmark Run

```python
from fishstick.benchmarks import BenchmarkRunner

runner = BenchmarkRunner(device="cuda")
results = runner.run_all(n_epochs=10, batch_size=64)

# Access specific results
print(f"UniIntelli loss: {results['UniIntelli']['final_loss']:.4f}")
print(f"Energy violation: {results['EnergyConservation']['mean_violation']:.6f}")
```

### Custom Benchmark

```python
from fishstick.benchmarks import BenchmarkRunner, SyntheticClassificationDataset
from torch.utils.data import DataLoader

runner = BenchmarkRunner(device="cuda")

# Create custom dataset
dataset = SyntheticClassificationDataset(n_samples=500, input_dim=784)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Run just framework benchmarks
runner.benchmark_frameworks(n_epochs=5, batch_size=32)
runner.print_summary()
```

### Extending Benchmarks

```python
from fishstick.benchmarks import BenchmarkRunner

class CustomBenchmarkRunner(BenchmarkRunner):
    def benchmark_custom(self):
        """Add custom benchmark."""
        print("Running custom benchmark...")
        # Your benchmark code
        self.results["Custom"] = {"score": 0.95}
        
runner = CustomBenchmarkRunner()
runner.run_all()
runner.benchmark_custom()
runner.print_summary()
```

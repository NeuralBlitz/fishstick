# Causal Inference

Causal inference, structural causal models, and treatment effect estimation.

## Installation

```bash
pip install fishstick[causal_inference]
```

## Overview

The `causal_inference` module provides tools for causal inference including structural causal models, causal discovery, and treatment effect estimation.

## Usage

```python
from fishstick.causal_inference import AdditiveSCM, DoCalculus, CATEEstimator

# Structural causal model
scm = AdditiveSCM(n_variables=3, hidden_dim=16)
scm.fit(data)

# Do-calculus
do_calc = DoCalculus(scm)
effect = do_calc.do(x=1, y=?)
```

## Models

| Model | Description |
|-------|-------------|
| `AdditiveSCM` | Additive structural causal model |
| `NonlinearSCM` | Nonlinear SCM |
| `LinearSCM` | Linear SCM |

## Discovery

| Class | Description |
|-------|-------------|
| `PCAlgorithm` | PC algorithm for causal discovery |
| `NOTEARS` | DAG learning via continuous optimization |

## Estimation

| Class | Description |
|-------|-------------|
| `CATEEstimator` | Conditional average treatment effect |
| `TreatmentEffectEstimator` | Treatment effect estimation |
| `CounterfactualEngine` | Counterfactual reasoning |

## Examples

See `examples/causal_inference/` for complete examples.

# causal - Causal Inference Module

## Overview

The `causal` module provides tools for causal inference, structural causal models, causal discovery, and counterfactual reasoning.

## Purpose and Scope

This module enables:
- Structural Causal Model (SCM) implementation
- Do-calculus for interventions
- Counterfactual reasoning
- Causal discovery algorithms (PC, NOTEARS)
- Causal effect estimation (IV, propensity scores)

## Key Classes and Functions

### Causal Graphs

#### `CausalGraph`
Represents a causal DAG.

```python
from fishstick.causal import CausalGraph
import numpy as np

adjacency = np.array([
    [0, 1, 1],  # X1 -> X2, X1 -> X3
    [0, 0, 1],  # X2 -> X3
    [0, 0, 0]   # X3 is effect
])

graph = CausalGraph(
    n_nodes=3,
    adjacency=adjacency,
    node_names=["Treatment", "Mediator", "Outcome"]
)

# Get causal relationships
parents = graph.parents(2)    # [0, 1]
children = graph.children(0)  # [1, 2]
is_dag = graph.is_dag()       # True
```

### Structural Causal Models

#### `StructuralEquation`
Neural structural equation X_i = f_i(PA_i, ε_i).

```python
from fishstick.causal import StructuralEquation

eq = StructuralEquation(
    n_parents=2,
    noise_dim=1,
    hidden_dim=64,
    nonlinearity="mlp"  # or "additive"
)

# Compute node value
value = eq(parents, noise)
```

#### `StructuralCausalModel`
Complete SCM with multiple structural equations.

```python
from fishstick.causal import StructuralCausalModel, CausalGraph

# Define causal graph
graph = CausalGraph(n_nodes=3, adjacency=adj)

# Create SCM
scm = StructuralCausalModel(graph, hidden_dim=64)

# Sample observational data
samples = scm.forward()

# Intervention: do(X_0 = value)
intervened = scm.do_calculus(intervention_node=0, value=torch.ones(100, 1))

# Counterfactual
counterfactual = scm.counterfactual(
    evidence={0: observed_value},
    intervention=(1, new_value)
)
```

### Causal Discovery

#### `CausalDiscovery.pc_algorithm`
PC algorithm for causal discovery.

```python
from fishstick.causal import CausalDiscovery

adjacency = CausalDiscovery.pc_algorithm(
    data=observational_data,
    alpha=0.05,
    independence_test="fisher"
)
```

#### `CausalDiscovery.notears`
NOTEARS: Gradient-based DAG learning.

```python
adjacency = CausalDiscovery.notears(
    data=observational_data,
    lambda1=0.1,
    max_iter=100
)
```

### Effect Estimation

#### `InstrumentalVariableEstimator`
Estimate causal effects using instrumental variables.

```python
from fishstick.causal import InstrumentalVariableEstimator

iv = InstrumentalVariableEstimator(method="2sls")
effect = iv.fit(
    Z=instrument,    # Instrument
    X=treatment,     # Treatment
    Y=outcome        # Outcome
)
```

#### `PropensityScoreMatching`
Estimate ATE using propensity scores.

```python
from fishstick.causal import PropensityScoreMatching

psm = PropensityScoreMatching()
psm.fit_propensity(covariates, treatment)
ate = psm.estimate_ate(covariates, treatment, outcome)
```

#### `DoublyRobustEstimator`
Combines propensity score and outcome modeling.

```python
from fishstick.causal import DoublyRobustEstimator

dr = DoublyRobustEstimator(
    propensity_model=propensity_net,
    outcome_model=outcome_net
)
ate = dr.estimate_ate(covariates, treatment, outcome)
```

### Causal VAE

#### `CausalVAE`
Variational autoencoder with causal structure.

```python
from fishstick.causal import CausalVAE

vae = CausalVAE(
    input_dim=100,
    causal_dim=10,
    noise_dim=5,
    hidden_dim=256
)

# Encode and decode
result = vae(x)
# Returns x_recon, mu, logvar, z_causal, z_noise

# Counterfactual generation
counterfactual = vae.intervene(x, intervention={0: 1.0})
```

## Mathematical Background

### Structural Causal Models
- X_i = f_i(PA_i, ε_i) where PA_i are parents
- Interventions: do(X=x) cuts incoming edges
- Counterfactuals: three steps (abduction, action, prediction)

### Do-Calculus
Three rules for manipulating interventional distributions:
1. Insertion/deletion of observations
2. Action/observation exchange
3. Action insertion/deletion

### Identification
- Backdoor criterion
- Front-door criterion
- Instrumental variables

## Dependencies

- `torch`: PyTorch for neural networks
- `numpy`: Numerical operations
- `scipy`: Linear algebra

## Usage Examples

### Complete Causal Analysis

```python
import torch
from fishstick.causal import (
    CausalGraph, StructuralCausalModel,
    CausalDiscovery, PropensityScoreMatching
)

# Step 1: Discover causal structure
data = load_observational_data()  # [n_samples, n_features]
adjacency = CausalDiscovery.notears(data, lambda1=0.1)

# Step 2: Build causal graph
graph = CausalGraph(n_nodes=data.shape[1], adjacency=adjacency)

# Step 3: Create SCM and sample
scm = StructuralCausalModel(graph)
samples = scm.forward()

# Step 4: Estimate causal effect
psm = PropensityScoreMatching()
ate = psm.estimate_ate(covariates, treatment, outcome)
print(f"Average Treatment Effect: {ate:.3f}")
```

### Intervention Study

```python
from fishstick.causal import StructuralCausalModel, CausalGraph

# Create SCM
graph = CausalGraph(n_nodes=3, adjacency=adj)
scm = StructuralCausalModel(graph)

# Observational distribution
obs = scm.forward()

# Intervention: do(Treatment = 1)
intervened = scm.do_calculus(
    intervention_node=0,
    value=torch.ones(100, 1)
)

# Compare distributions
print(f"Obs mean: {obs[:, -1].mean()}")
print(f"Intervened mean: {intervened[:, -1].mean()}")
```

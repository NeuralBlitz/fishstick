# Probabilistic Module

## Overview

Probabilistic neural networks and Bayesian deep learning implementations for uncertainty quantification. Provides tools for epistemic and aleatoric uncertainty estimation, calibrated predictions, and variational inference.

## Purpose and Scope

- Bayesian neural networks with variational inference
- Monte Carlo Dropout for uncertainty estimation
- Deep ensembles for model uncertainty
- Evidential deep learning
- Conformal prediction for calibrated intervals

## Key Classes and Functions

### BayesianLinear

Bayesian linear layer with learned weight distributions using the reparameterization trick.

```python
from fishstick.probabilistic import BayesianLinear
import torch

# Create Bayesian layer
layer = BayesianLinear(in_features=128, out_features=64, prior_sigma=1.0)

# Forward pass with sampled weights
output = layer(x, sample=True)

# Get MAP estimate (no sampling)
output_map = layer(x, sample=False)

# KL divergence for regularization
kl = layer.kl_divergence()
```

### MCDropout

Monte Carlo Dropout for epistemic uncertainty estimation.

```python
from fishstick.probabilistic import MCDropout

dropout = MCDropout(p=0.5)

# Dropout applied at both train and test time
output = dropout(x)  # Always stochastic
```

### DeepEnsemble

Deep ensemble for uncertainty quantification.

```python
from fishstick.probabilistic import DeepEnsemble

# Create ensemble of models
ensemble = DeepEnsemble(
    model_class=MyModel,
    n_models=5,
    input_dim=784,
    output_dim=10
)

# Predict with uncertainty
mean, uncertainty = ensemble(x)

# Full uncertainty decomposition
mean, aleatoric, epistemic = ensemble.predict_with_uncertainty(x)
```

### EvidentialLayer

Evidential deep learning for uncertainty quantification using Normal-Inverse-Gamma distribution.

```python
from fishstick.probabilistic import EvidentialLayer

layer = EvidentialLayer(in_features=128, out_features=1)

# Get evidential parameters
params = layer(x)
# params = {'gamma': mean, 'nu': dof, 'alpha': shape, 'beta': scale}

# Compute loss
nll = layer.nig_nll(y, params)
reg = layer.nig_reg(y, params)
loss = nll + 0.01 * reg
```

### ConformalPredictor

Conformal prediction for calibrated uncertainty intervals.

```python
from fishstick.probabilistic import ConformalPredictor

# Create predictor
predictor = ConformalPredictor(model, alpha=0.05)

# Calibrate on held-out data
predictor.calibrate(x_cal, y_cal)

# Predict with guaranteed coverage intervals
mean, (lower, upper) = predictor.predict(x_test)
```

### BayesianNeuralNetwork

Complete Bayesian neural network with ELBO training.

```python
from fishstick.probabilistic import BayesianNeuralNetwork

# Create BNN
bnn = BayesianNeuralNetwork(
    input_dim=784,
    hidden_dims=[256, 128],
    output_dim=10,
    prior_sigma=1.0
)

# ELBO loss for training
loss = bnn.elbo_loss(x, y, n_samples=3, beta=1.0)

# Predict with uncertainty
mean, uncertainty = bnn.predict_with_uncertainty(x, n_samples=100)
```

### StochasticVariationalGP

Scalable Gaussian Process using inducing points.

```python
from fishstick.probabilistic import StochasticVariationalGP

gp = StochasticVariationalGP(input_dim=10, num_inducing=100)

# Predict with uncertainty
mean, var = gp(x)
```

## Mathematical Background

### Variational Inference

Approximate posterior q(w) with learned distribution:

```
KL(q(w)||p(w)) = E_q[log q(w)] - E_q[log p(w)]
ELBO = E_q[log p(y|x,w)] - KL(q(w)||p(w))
```

### Monte Carlo Dropout

Approximate Bayesian inference using dropout at test time:

```
p(y|x) ≈ (1/T) Σ p(y|x, w_t), w_t ~ Dropout
```

### Evidential Deep Learning

Model uncertainty via higher-order distributions:

```
y ~ N(γ, σ²), where (γ, σ²) ~ NIG(γ, ν, α, β)
```

### Conformal Prediction

Distribution-free prediction sets with guaranteed coverage:

```
P(Y ∈ Ĉ(X)) ≥ 1 - α
```

## Uncertainty Types

| Type | Source | Method |
|------|--------|--------|
| Epistemic | Model uncertainty | MC Dropout, Ensembles, BNN |
| Aleatoric | Data noise | Variational Layer, Evidential |
| Total | Combined | Full Bayesian inference |

## Dependencies

- `torch` - Neural network operations
- `torch.distributions` - Probability distributions
- `numpy` - Numerical operations

## Usage Examples

### Training a Bayesian Neural Network

```python
from fishstick.probabilistic import BayesianNeuralNetwork
import torch.nn.functional as F

bnn = BayesianNeuralNetwork(784, [256, 128], 10)
optimizer = torch.optim.Adam(bnn.parameters())

for x, y in dataloader:
    optimizer.zero_grad()
    loss = bnn.elbo_loss(x, y, n_samples=3, beta=0.1)
    loss.backward()
    optimizer.step()
```

### Uncertainty-Aware Prediction

```python
from fishstick.probabilistic import DeepEnsemble

ensemble = DeepEnsemble(MyModel, n_models=5, input_dim=100, output_dim=1)
mean, aleatoric, epistemic = ensemble.predict_with_uncertainty(x)

# High uncertainty threshold
uncertain = epistemic > threshold
```

### Calibrated Prediction Intervals

```python
from fishstick.probabilistic import ConformalPredictor

# Split data for calibration
x_train, x_cal, y_train, y_cal = train_test_split(X, y)

# Train model
model.train(x_train, y_train)

# Calibrate
predictor = ConformalPredictor(model, alpha=0.1)
predictor.calibrate(x_cal, y_cal)

# Get 90% coverage intervals
pred, (lower, upper) = predictor.predict(x_test)
```

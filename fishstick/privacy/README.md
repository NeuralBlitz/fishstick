# Privacy Module

Differential privacy and secure learning tools for building privacy-preserving machine learning systems.

## Overview

This module provides comprehensive tools for:

- **Differential Privacy (DP)**: Add noise to protect individual data points
- **Privacy Accounting**: Track privacy budget (ε, δ)
- **DP-SGD**: Differentially private stochastic gradient descent
- **Privacy Amplification**: Improve privacy through subsampling
- **Secure Aggregation**: Protect model updates in federated learning

## Installation

```bash
# Core dependencies (included in fishstick)
pip install torch numpy scipy
```

## Quick Start

### Basic DP Training

```python
import torch
import torch.nn as nn
from fishstick.privacy import PrivacyEngine

# Define model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Initialize privacy engine
privacy = PrivacyEngine(
    model,
    epsilon=8.0,      # Privacy budget
    delta=1e-5,      # Failure probability
    max_grad_norm=1.0  # Gradient clipping norm
)

# Train with DP
optimizer = privacy.make_optimizer(torch.optim.SGD, lr=0.01)

for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        
        inputs, labels = batch
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)
        
        loss.backward()
        optimizer.clip_grad()
        optimizer.step()
    
    # Get privacy budget spent
    epsilon = privacy.get_epsilon()
    print(f"Epoch {epoch}: ε = {epsilon:.2f}")
```

### Custom DP-SGD

```python
from fishstick.privacy import DPSGD, GaussianMechanism

# Configure DP training
dp_config = DPConfig(
    epsilon=2.0,
    delta=1e-5,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
    batch_size=256
)

# Create DP optimizer
optimizer = DPSGD(
    model.parameters(),
    lr=0.01,
    config=dp_config
)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        
        loss = compute_loss(model, batch)
        loss.backward()
        
        optimizer.clip_grad()
        optimizer.step()
```

### Privacy Accounting

```python
from fishstick.privacy import RDPAccountant, PrivacyBudget

# Rényi DP accounting
accountant = RDPAccountant(
    noise_multiplier=1.1,
    sample_rate=0.01  # batch_size / dataset_size
)

# After each step
for step in range(10000):
    accountant.step()
    
    # Get privacy spent
    epsilon, delta = accountant.get_privacy()
    if step % 100 == 0:
        print(f"Step {step}: ε = {epsilon:.2f}, δ = {delta:.2e}")
```

### Privacy Amplification

```python
from fishstick.privacy import SubsampleAmplifier, amplify_privacy

# Amplify privacy through subsampling
amplifier = SubsampleAmplifier(
    sample_rate=0.01,
    amplifier_type="poisson"
)

# Get amplified epsilon
base_epsilon = 2.0
base_delta = 1e-5

amplified_epsilon = amplifier.amplify(base_epsilon)
print(f"Original ε: {base_epsilon}, Amplified ε: {amplified_epsilon}")
```

### Secure Aggregation

```python
from fishstick.privacy import SecureAggregationProtocol, AdditiveSecretSharing

# Setup secure aggregation
sec_agg = SecureAggregationProtocol(
    n_clients=10,
    threshold=7,  # Need 7 clients to reconstruct
    bit_length=32
)

# Client-side: create secret shares
client_id = 0
model_update = get_model_update()  # Your gradient

shares = sec_agg.share(client_id, model_update)

# Server-side: aggregate shares
aggregated = sec_agg.aggregate(shares)
```

## API Reference

### Noise Mechanisms

| Class | Description |
|-------|-------------|
| `GaussianMechanism` | Gaussian noise addition |
| `LaplaceMechanism` | Laplace noise |
| `ExponentialMechanism` | Exponential mechanism |
| `NoiseConfig` | Configuration |

### Privacy Accounting

| Class | Description |
|-------|-------------|
| `RDPAccountant` | Rényi DP accounting |
| `BasicAccountant` | Basic composition |
| `PrivacyBudget` | Track (ε, δ) |

### Clipping

| Class | Description |
|-------|-------------|
| `StaticClipper` | Fixed norm clipping |
| `AdaptiveClipper` | Adaptive clipping |
| `PerLayerClipper` | Per-layer clipping |

### Training

| Class | Description |
|-------|-------------|
| `DPSGD` | DP-SGD optimizer |
| `PrivacyEngine` | High-level training API |
| `DPTrainer` | DP training loop |

### Secure Aggregation

| Class | Description |
|-------|-------------|
| `SecureAggregationProtocol` | Protocol for secure agg |
| `AdditiveSecretSharing` | Secret sharing |
| `ThresholdCryptography` | Threshold crypto |

## Examples

### Federated Learning with DP

```python
from fishstick.privacy import FederatedPrivacyEngine

# Federated DP training
privacy = FederatedPrivacyEngine(
    epsilon=1.0,
    delta=1e-6,
    max_grad_norm=1.0
)

# Each client trains locally
for client in clients:
    client_updates = []
    
    for epoch in epochs:
        # Local training
        loss = train_local(client.model, client.data)
        gradient = compute_gradient(client.model)
        
        # Clip and noise
        clipped = privacy.clip(gradient)
        noised = privacy.add_noise(clipped)
        
        client_updates.append(noisy_update)
    
    # Secure aggregation
    aggregated = secure_aggregate(client_updates)
```

## References

- Abadi et al., "Deep Learning with Differential Privacy" (CCS 2016)
- Mironov, "Rényi Differential Privacy" (CSF 2017)
- Bonawitz et al., "Practical Secure Aggregation" (CCS 2017)

## License

MIT License - see project root for details.

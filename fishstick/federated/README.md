# Federated Learning

Federated learning algorithms and utilities for distributed training.

## Installation

```bash
pip install fishstick[federated]
```

## Overview

The `federated` module provides implementations of federated learning algorithms that enable training across distributed devices while keeping data localized.

## Usage

```python
from fishstick.federated import FederatedClient, FederatedServer, FedAvg

# Create federated server
server = FederatedServer(
    model=model,
    strategy=FedAvg(),
    clients_per_round=10
)

# Add clients
for client_data in client_datasets:
    client = FederatedClient(model=model, data=client_data)
    server.add_client(client)

# Run federated training
server.train(num_rounds=100)
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| `FedAvg` | Federated Averaging |
| `FedProx` | FedProx with proximal regularization |
| `FedNova` | Federated Normalized Averaging |

## Advanced Features

| Feature | Description |
|---------|-------------|
| `ScaffoldStrategy` | Stochastic Controlled Averaging for Federated Learning |
| `SecureAggregationStrategy` | Privacy-preserving aggregation |
| `DifferentialPrivacy` | DP guarantees in federated settings |
| `DataPartitioner` | Partition data across clients |

## Examples

See `examples/federated/` for complete examples.

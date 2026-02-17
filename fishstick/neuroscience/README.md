# Neuroscience

Spiking neural networks, neuron models, and neural coding.

## Installation

```bash
pip install fishstick[neuroscience]
```

## Overview

The `neuroscience` module provides neuron models, spiking neural network layers, synaptic plasticity mechanisms, brain-inspired attention, and neural coding schemes.

## Usage

```python
import torch
from fishstick.neuroscience import (
    LeakyIntegrateAndFire,
    HodgkinHuxley,
    SpikingDense,
    STDP,
    NeuralAttention,
    GridCellEncoder,
)

# Leaky Integrate-and-Fire neuron
lif = LeakyIntegrateAndFire(n_neurons=100)
spikes, state = lif(torch.randn(2, 100))

# Hodgkin-Huxley model
hh = HodgkinHuxley(n_neurons=50)
spikes, state = hh.forward(torch.randn(2, 50))

# Spiking neural network layer
snn = SpikingDense(in_features=784, out_features=256)
spikes, state = snn(torch.randn(2, 784))

# STDP plasticity
stdp = STDP(n_synapses=100)
weights = stdp(torch.randn(2, 100), torch.randn(2, 50))

# Grid cell encoding
grid = GridCellEncoder()
response = grid(torch.rand(2, 2))
```

## Neuron Models

| Model | Description |
|-------|-------------|
| `LeakyIntegrateAndFire` | LIF neuron with membrane dynamics |
| `HodgkinHuxley` | Full biophysical HH model |
| `Izhikevich` | Simplified realistic model |
| `AdaptiveLIF` | LIF with spike-frequency adaptation |

## Layers

| Class | Description |
|-------|-------------|
| `SpikingDense` | Fully connected spiking layer |
| `SpikingConv2d` | Convolutional spiking layer |
| `LiquidStateMachine` | Reservoir computing network |
| `SpikingAttention` | Attention in spiking networks |

## Plasticity

| Class | Description |
|-------|-------------|
| `STDP` | Spike-Timing-Dependent Plasticity |
| `OjaRule` | Normalized Hebbian learning |
| `BCMPlasticity` | Bienenstock-Cooper-Munro rule |
| `HomeostaticPlasticity` | Activity regulation |
| `TripletSTDP` | Extended triplet STDP |

## Attention

| Class | Description |
|-------|-------------|
| `NeuralAttention` | Cortical-inspired attention |
| `WinnerTakeAllAttention` | Competitive attention |
| `DivisiveNormalization` | Normalization from visual cortex |
| `FeedbackAttention` | Top-down feedback attention |

## Coding

| Class | Description |
|-------|-------------|
| `RateEncoder` | Firing rate encoding |
| `PoissonEncoder` | Stochastic spike generation |
| `TemporalEncoder` | Latency/temporal coding |
| `PopulationEncoder` | Population vector coding |
| `GridCellEncoder` | Grid cell spatial encoding |
| `DeltaEncoder` | Change-based encoding |

"""Neuroscience-inspired computational models.

This module provides implementations of brain-inspired neural network
components including spiking neurons, plasticity mechanisms, attention,
and neural coding schemes.

Submodules:
    neurons: Neuron models (LIF, Hodgkin-Huxley, Izhikevich)
    spiking_layers: Spiking neural network layers
    plasticity: Synaptic plasticity (STDP, Oja, BCM)
    attention: Brain-inspired attention mechanisms
    coding: Neural coding implementations
"""

from .neurons import (
    LeakyIntegrateAndFire,
    HodgkinHuxley,
    Izhikevich,
    AdaptiveLIF,
)

from .spiking_layers import (
    SpikingDense,
    SpikingConv2d,
    LiquidStateMachine,
    SpikingAttention,
    ThresholdDependentPlasticity,
)

from .plasticity import (
    STDP,
    OjaRule,
    BCMPlasticity,
    HomeostaticPlasticity,
    TripletSTDP,
    VoltageBasedSTDP,
)

from .attention import (
    NeuralAttention,
    WinnerTakeAllAttention,
    DivisiveNormalization,
    NormalizedAttention,
    FeedbackAttention,
    PredictiveAttention,
)

from .coding import (
    RateEncoder,
    PoissonEncoder,
    TemporalEncoder,
    PopulationEncoder,
    MixedCodeEncoder,
    LatentPopulationEncoder,
    DeltaEncoder,
    GridCellEncoder,
)

__all__ = [
    # Neurons
    "LeakyIntegrateAndFire",
    "HodgkinHuxley",
    "Izhikevich",
    "AdaptiveLIF",
    # Spiking Layers
    "SpikingDense",
    "SpikingConv2d",
    "LiquidStateMachine",
    "SpikingAttention",
    "ThresholdDependentPlasticity",
    # Plasticity
    "STDP",
    "OjaRule",
    "BCMPlasticity",
    "HomeostaticPlasticity",
    "TripletSTDP",
    "VoltageBasedSTDP",
    # Attention
    "NeuralAttention",
    "WinnerTakeAllAttention",
    "DivisiveNormalization",
    "NormalizedAttention",
    "FeedbackAttention",
    "PredictiveAttention",
    # Coding
    "RateEncoder",
    "PoissonEncoder",
    "TemporalEncoder",
    "PopulationEncoder",
    "MixedCodeEncoder",
    "LatentPopulationEncoder",
    "DeltaEncoder",
    "GridCellEncoder",
]

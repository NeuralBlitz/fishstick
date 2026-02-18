"""
Normalizing Flows Extensions for fishstick AI Framework.

This module provides comprehensive implementations of normalizing flow architectures:
- Neural Spline Flows: Rational quadratic spline transformations
- FFJORD: Continuous normalizing flows with ODE solvers
- Masked Autoregressive Flows: MAF and MADE implementations
- Coupling Layers: Affine, additive, spline, and conditional variants
- Flow-based Density Estimation: RealNVP, Glow, Flow-based VAE

References:
- Neural Spline Flows: Durkan et al. (2019)
- FFJORD:Grathohl et al. (2019)
- MAF: Papamakarios et al. (2017)
- RealNVP: Dinh et al. (2017)
- Glow: Kingma & Dhariwal (2018)
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn

from .neural_spline_flows import (
    RationalQuadraticSpline,
    NeuralSplineFlow,
    SplineCouplingLayer,
    ConditionalNeuralSplineFlow,
    BatchNormFlow,
    ActNorm,
)

from .ffjord import (
    FFJORDDynamics,
    FFJORD,
    ConditionalFFJORD,
    TraceEstimator,
    FFJORDSolver,
    NeuralODE,
)

from .masked_autoregressive_flow import (
    MADE,
    InverseAutoregressiveTransform,
    MAF,
    Permutation,
    MADETransformer,
    MaskedLinear,
    AutoregressiveNetwork,
    FlowHead,
)

from .coupling_layers import (
    AffineCouplingLayer,
    AdditiveCouplingLayer,
    NeuralSplineCouplingLayer,
    ConditionalCouplingLayer,
    MaskGenerator,
    Conv1x1Coupling,
    ChannelMaskCoupling,
    SqueezeTransform,
)

from .density_estimation import (
    RealNVP,
    SqueezeLayer,
    Glow,
    InvertibleConv1x1,
    InvertibleDownsample,
    FlowStep,
    ChannelCoupling,
    FlowBasedVAE,
    InvertibleMLP,
    DensityEstimator,
)

__all__ = [
    "RationalQuadraticSpline",
    "NeuralSplineFlow",
    "SplineCouplingLayer",
    "ConditionalNeuralSplineFlow",
    "BatchNormFlow",
    "ActNorm",
    "FFJORDDynamics",
    "FFJORD",
    "ConditionalFFJORD",
    "TraceEstimator",
    "FFJORDSolver",
    "NeuralODE",
    "MADE",
    "InverseAutoregressiveTransform",
    "MAF",
    "Permutation",
    "MADETransformer",
    "MaskedLinear",
    "AutoregressiveNetwork",
    "FlowHead",
    "AffineCouplingLayer",
    "AdditiveCouplingLayer",
    "NeuralSplineCouplingLayer",
    "ConditionalCouplingLayer",
    "MaskGenerator",
    "Conv1x1Coupling",
    "ChannelMaskCoupling",
    "SqueezeTransform",
    "RealNVP",
    "SqueezeLayer",
    "Glow",
    "InvertibleConv1x1",
    "InvertibleDownsample",
    "FlowStep",
    "ChannelCoupling",
    "FlowBasedVAE",
    "InvertibleMLP",
    "DensityEstimator",
]

_FLOWS_AVAILABLE = True

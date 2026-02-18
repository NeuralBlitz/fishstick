"""
Neural Network Components Module for fishstick.

Advanced neural network building blocks including:
- Normalization layers (FilterResponseSync, EvoNorm, SwitchNorm)
- Activation functions (SiLU, Swish, GELU, Snake)
- Initialization strategies (Kaiming, Xavier, Orthogonal)
- Skip connections (Residual, Dense, StochasticDepth)
- Dropout variants (SpatialDropout, DropBlock, ConcreteDropout)

This module provides implementations of modern neural network components
used in state-of-the-art architectures like Transformers, ResNets, and DenseNets.
"""

from typing import Tuple

import torch
from torch import Tensor, nn

from fishstick.nn_components.filter_response_sync import FilterResponseSync
from fishstick.nn_components.evo_norm import (
    EvoNormB0,
    EvoNormS0,
    EvoNormB,
    EvoNormS,
    create_evo_norm,
)
from fishstick.nn_components.switch_norm import (
    SwitchNorm,
    SwitchNorm1d,
    SwitchNorm3d,
    create_switch_norm,
)
from fishstick.nn_components.activations import (
    SiLU,
    Swish,
    GELU,
    Snake,
    SnakeBeta,
    silu,
    gelu,
    snake,
    get_activation,
)
from fishstick.nn_components.initializers import (
    kaiming_normal_,
    kaiming_uniform_,
    xavier_normal_,
    xavier_uniform_,
    orthogonal_,
    delta_orthogonal_,
    KaimingNormal,
    KaimingUniform,
    XavierNormal,
    XavierUniform,
    Orthogonal,
    DeltaOrthogonal,
    initialize_module,
)
from fishstick.nn_components.skip_connections import (
    ResidualBlock,
    PreActResidualBlock,
    DenseBlock,
    TransitionLayer,
    StochasticDepth,
    DropPath,
    ScaledResidualBlock,
    create_skip_connection,
)
from fishstick.nn_components.dropout_variants import (
    SpatialDropout,
    SpatialDropout1d,
    SpatialDropout3d,
    DropBlock,
    DropBlock1d,
    DropBlock2d,
    DropBlock3d,
    ConcreteDropout,
    AutoDropout,
    create_dropout,
)

__all__ = [
    # Normalization Layers
    "FilterResponseSync",
    "EvoNormB0",
    "EvoNormS0",
    "EvoNormB",
    "EvoNormS",
    "create_evo_norm",
    "SwitchNorm",
    "SwitchNorm1d",
    "SwitchNorm3d",
    "create_switch_norm",
    # Activations
    "SiLU",
    "Swish",
    "GELU",
    "Snake",
    "SnakeBeta",
    "silu",
    "gelu",
    "snake",
    "get_activation",
    # Initializers
    "kaiming_normal_",
    "kaiming_uniform_",
    "xavier_normal_",
    "xavier_uniform_",
    "orthogonal_",
    "delta_orthogonal_",
    "KaimingNormal",
    "KaimingUniform",
    "XavierNormal",
    "XavierUniform",
    "Orthogonal",
    "DeltaOrthogonal",
    "initialize_module",
    # Skip Connections
    "ResidualBlock",
    "PreActResidualBlock",
    "DenseBlock",
    "TransitionLayer",
    "StochasticDepth",
    "DropPath",
    "ScaledResidualBlock",
    "create_skip_connection",
    # Dropout Variants
    "SpatialDropout",
    "SpatialDropout1d",
    "SpatialDropout3d",
    "DropBlock",
    "DropBlock1d",
    "DropBlock2d",
    "DropBlock3d",
    "ConcreteDropout",
    "AutoDropout",
    "create_dropout",
]

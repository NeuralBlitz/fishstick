"""
Weight initialization strategies for neural networks.

This module provides various initialization methods:
- Kaiming (He) initialization: Optimized for ReLU family activations
- Xavier (Glorot) initialization: Optimized for Sigmoid/Tanh
- Orthogonal initialization: Preserves gradient flow in recurrent networks
- Delta-Orthogonal initialization: For convolutional networks

Proper initialization is crucial for training deep networks and
can significantly impact convergence speed and final performance.
"""

from typing import Optional, Union, Callable
import math
import torch
import torch.nn as nn
from torch import Tensor


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
) -> Tensor:
    """
    Kaiming (He) normal initialization.

    Initializes weights using a normal distribution with variance
    2 / fan_in (for ReLU-like activations). Particularly effective for
    networks using ReLU, Leaky ReLU, PReLU, and similar activations.

    Reference: https://arxiv.org/abs/1502.01852

    Args:
        tensor: Tensor to initialize
        a: Negative slope for LeakyReLU (default: 0.0)
        mode: 'fan_in' or 'fan_out' (default: 'fan_in')
        nonlinearity: Non-linear activation function (default: 'relu')

    Returns:
        Initialized tensor

    Example:
        >>> linear = nn.Linear(512, 256)
        >>> kaiming_normal_(linear.weight, nonlinearity='relu')
    """
    if isinstance(tensor, nn.Parameter):
        tensor = tensor.data

    fan = _calculate_fan(tensor, mode)

    if nonlinearity == "relu" or nonlinearity == "leaky_relu":
        gain = math.sqrt(2.0 / (1 + a**2))
    else:
        gain = 1.0

    std = gain / math.sqrt(fan)

    with torch.no_grad():
        return tensor.normal_(0, std)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity: str = "relu",
) -> Tensor:
    """
    Kaiming (He) uniform initialization.

    Initializes weights using a uniform distribution within [-limit, limit]
    where limit = sqrt(6 / fan_in) for ReLU-like activations.

    Args:
        tensor: Tensor to initialize
        a: Negative slope for LeakyReLU (default: 0.0)
        mode: 'fan_in' or 'fan_out' (default: 'fan_in')
        nonlinearity: Non-linear activation function (default: 'relu')

    Returns:
        Initialized tensor
    """
    if isinstance(tensor, nn.Parameter):
        tensor = tensor.data

    fan = _calculate_fan(tensor, mode)

    if nonlinearity == "relu" or nonlinearity == "leaky_relu":
        gain = math.sqrt(2.0 / (1 + a**2))
    else:
        gain = 1.0

    std = gain / math.sqrt(fan)
    limit = std * math.sqrt(3.0)

    with torch.no_grad():
        return tensor.uniform_(-limit, limit)


def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
) -> Tensor:
    """
    Xavier (Glorot) normal initialization.

    Initializes weights using a normal distribution with variance
    2 / (fan_in + fan_out). Best for networks using Sigmoid or Tanh activations.

    Reference: http://proceedings.mlr.press/v9/glorot10a.html

    Args:
        tensor: Tensor to initialize
        gain: Scaling factor (default: 1.0)

    Returns:
        Initialized tensor

    Example:
        >>> linear = nn.Linear(512, 256)
        >>> xavier_normal_(linear.weight)
    """
    if isinstance(tensor, nn.Parameter):
        tensor = tensor.data

    fan_in, fan_out = _calculate_fan_in_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))

    with torch.no_grad():
        return tensor.normal_(0, std)


def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
) -> Tensor:
    """
    Xavier (Glorot) uniform initialization.

    Initializes weights using a uniform distribution within [-limit, limit]
    where limit = sqrt(6 / (fan_in + fan_out)).

    Args:
        tensor: Tensor to initialize
        gain: Scaling factor (default: 1.0)

    Returns:
        Initialized tensor
    """
    if isinstance(tensor, nn.Parameter):
        tensor = tensor.data

    fan_in, fan_out = _calculate_fan_in_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    limit = std * math.sqrt(3.0)

    with torch.no_grad():
        return tensor.uniform_(-limit, limit)


def orthogonal_(
    tensor: Tensor,
    gain: float = 1.0,
) -> Tensor:
    """
    Orthogonal initialization.

    Initializes weights as an orthogonal matrix. This preserves the norm
    of gradients during backpropagation, making it particularly useful
    for recurrent neural networks (RNNs, LSTMs, GRUs).

    Reference: https://arxiv.org/abs/1312.6120

    Args:
        tensor: Tensor to initialize (must be at least 2D)
        gain: Scaling factor (default: 1.0)

    Returns:
        Initialized tensor

    Example:
        >>> lstm = nn.LSTM(256, 256, batch_first=True)
        >>> for name, param in lstm.named_parameters():
        ...     if 'weight' in name:
        ...         orthogonal_(param)
    """
    if isinstance(tensor, nn.Parameter):
        tensor = tensor.data

    if tensor.dim() < 2:
        raise ValueError("Orthogonal initialization requires at least 2D tensor")

    rows = tensor.size(0)
    cols = tensor.numel() // rows

    if rows < cols:
        q, r = torch.linalg.qr(torch.randn(rows, cols))
        tensor.view(rows, cols).copy_(q @ torch.diag(torch.sign(torch.diag(r))))
    else:
        q, r = torch.linalg.qr(torch.randn(cols, rows).T)
        tensor.view(rows, cols).copy_(q)

    tensor.mul_(gain)
    return tensor


def delta_orthogonal_(
    tensor: Tensor,
    gain: float = 1.0,
) -> Tensor:
    """
    Delta-Orthogonal initialization for convolutional networks.

    Creates orthogonal filters in the Fourier domain. Particularly effective
    for very deep convolutional networks as it maintains gradient propagation.

    Reference: https://arxiv.org/abs/1805.10415

    Args:
        tensor: Convolutional weight tensor (4D: out_ch, in_ch, kH, kW)
        gain: Scaling factor (default: 1.0)

    Returns:
        Initialized tensor
    """
    if isinstance(tensor, nn.Parameter):
        tensor = tensor.data

    if tensor.dim() != 4:
        raise ValueError("Delta-orthogonal requires 4D convolution weights")

    out_ch, in_ch, kH, kW = tensor.shape

    if out_ch >= in_ch:
        orthogonal_(tensor.view(out_ch, -1), gain=gain)
    else:
        fan_in = in_ch * kH * kW
        std = gain * math.sqrt(2.0 / fan_in)
        with torch.no_grad():
            tensor.normal_(0, std)

    return tensor


def _calculate_fan(tensor: Tensor, mode: str = "fan_in") -> int:
    """Calculate fan for Kaiming initialization."""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan requires at least 2D tensor")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)

    if dimensions > 2:
        receptive_field = tensor[0, 0].numel()
    else:
        receptive_field = 1

    if mode == "fan_in":
        return num_input_fmaps * receptive_field
    elif mode == "fan_out":
        return num_output_fmaps * receptive_field
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _calculate_fan_in_fan_out(tensor: Tensor) -> tuple:
    """Calculate fan_in and fan_out for Xavier initialization."""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan requires at least 2D tensor")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)

    if dimensions > 2:
        receptive_field = tensor[0, 0].numel()
    else:
        receptive_field = 1

    fan_in = num_input_fmaps * receptive_field
    fan_out = num_output_fmaps * receptive_field

    return fan_in, fan_out


class KaimingNormal:
    """
    Kaiming normal initializer.

    Usage:
        >>> init = KaimingNormal(a=0.0, mode='fan_in', nonlinearity='relu')
        >>> model.apply(init)
    """

    def __init__(
        self,
        a: float = 0.0,
        mode: str = "fan_in",
        nonlinearity: str = "relu",
    ) -> None:
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def __call__(self, tensor: Tensor) -> Tensor:
        return kaiming_normal_(
            tensor,
            a=self.a,
            mode=self.mode,
            nonlinearity=self.nonlinearity,
        )


class KaimingUniform:
    """Kaiming uniform initializer."""

    def __init__(
        self,
        a: float = 0.0,
        mode: str = "fan_in",
        nonlinearity: str = "relu",
    ) -> None:
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def __call__(self, tensor: Tensor) -> Tensor:
        return kaiming_uniform_(
            tensor,
            a=self.a,
            mode=self.mode,
            nonlinearity=self.nonlinearity,
        )


class XavierNormal:
    """Xavier normal initializer."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain

    def __call__(self, tensor: Tensor) -> Tensor:
        return xavier_normal_(tensor, gain=self.gain)


class XavierUniform:
    """Xavier uniform initializer."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain

    def __call__(self, tensor: Tensor) -> Tensor:
        return xavier_uniform_(tensor, gain=self.gain)


class Orthogonal:
    """Orthogonal initializer."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain

    def __call__(self, tensor: Tensor) -> Tensor:
        return orthogonal_(tensor, gain=self.gain)


class DeltaOrthogonal:
    """Delta-orthogonal initializer for convolutions."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain

    def __call__(self, tensor: Tensor) -> Tensor:
        return delta_orthogonal_(tensor, gain=self.gain)


def initialize_module(
    module: nn.Module, init_type: str = "kaiming_normal", **kwargs
) -> nn.Module:
    """
    Initialize all parameters in a module.

    Args:
        module: Module to initialize
        init_type: Initialization method ('kaiming_normal', 'kaiming_uniform',
                  'xavier_normal', 'xavier_uniform', 'orthogonal', 'delta_orthogonal')
        **kwargs: Arguments passed to the initializer

    Returns:
        Initialized module

    Example:
        >>> model = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        >>> initialize_module(model, 'xavier_normal')
    """
    initializers = {
        "kaiming_normal": KaimingNormal(**kwargs),
        "kaiming_uniform": KaimingUniform(**kwargs),
        "xavier_normal": XavierNormal(**kwargs),
        "xavier_uniform": XavierUniform(**kwargs),
        "orthogonal": Orthogonal(**kwargs),
        "delta_orthogonal": DeltaOrthogonal(**kwargs),
    }

    if init_type not in initializers:
        raise ValueError(
            f"Unknown init_type: {init_type}. Choose from {list(initializers.keys())}"
        )

    return initializers[init_type]

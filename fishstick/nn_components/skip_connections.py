"""
Skip connection implementations for neural networks.

This module provides various skip connection strategies:
- Residual connections: Classic identity skip connections
- Dense connections: DenseNet-style feature concatenation
- Stochastic depth: Randomly drop layers during training

Skip connections help with gradient flow in deep networks and can improve
training stability and performance.
"""

from typing import Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlock(nn.Module):
    """
    Standard residual block with skip connection.

    Implements: y = F(x) + x where F is the residual function.
    The residual function typically consists of two or more convolutions
    with normalization and activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (default: 1)
        downsample: Optional downsample layer for matching dimensions
        activation: Activation function (default: nn.ReLU)
        norm: Normalization layer (default: nn.BatchNorm2d)

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C', H', W') where C' = out_channels

    Example:
        >>> block = ResidualBlock(64, 64)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = block(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm is None:
            norm = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = norm(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm(out_channels)
        self.activation = activation()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class PreActResidualBlock(nn.Module):
    """
    Pre-activation residual block.

    Applies normalization and activation before convolutions.
    Often more stable for very deep networks.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (default: 1)
        downsample: Optional downsample layer
        activation: Activation function (default: nn.ReLU)
        norm: Normalization layer

    Reference: https://arxiv.org/abs/1603.05027
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        activation: Callable[[], nn.Module] = nn.ReLU,
        norm: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm is None:
            norm = nn.BatchNorm2d

        self.bn1 = norm(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = norm(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.activation = activation()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.activation(out)

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out += identity

        return out


class DenseBlock(nn.Module):
    """
    Dense block for DenseNet-style connections.

    Each layer receives feature maps from all preceding layers as input,
    promoting feature reuse and gradient flow.

    Args:
        in_channels: Number of input channels
        growth_rate: Growth rate (channels added per layer)
        num_layers: Number of layers in the block
        bn_size: Bottleneck size for 1x1 convolutions (default: 4)
        activation: Activation function (default: nn.ReLU)

    Reference: https://arxiv.org/abs/1608.06993

    Example:
        >>> block = DenseBlock(64, growth_rate=32, num_layers=4)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = block(x)  # 64 + 4*32 = 192 channels
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_layers: int,
        bn_size: int = 4,
        activation: Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = self._make_layer(
                in_channels + i * growth_rate, growth_rate, bn_size, activation
            )
            self.layers.append(layer)

    def _make_layer(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        activation: Callable[[], nn.Module],
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation(),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            activation(),
            nn.Conv2d(
                bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = [x]

        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)

        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """
    Transition layer for DenseNet.

    Reduces dimensions between dense blocks using pooling and
    optionally a 1x1 convolution.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        pooling: Pooling type ('avg' or 'max', default: 'avg')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling: str = "avg",
    ) -> None:
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        if pooling == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.conv(x)
        x = self.pool(x)
        return x


class StochasticDepth(nn.Module):
    """
    Stochastic depth layer.

    Randomly drops the layer during training by connecting input directly
    to output. This helps with training very deep networks by reducing
    the effective depth during training.

    Reference: https://arxiv.org/abs/1603.09382

    Args:
        drop_prob: Probability of dropping the layer (default: 0.0)
        survival_rate: Alternative to drop_prob (1 - drop_prob)
        mode: 'row' or 'batch' (default: 'row')

    Example:
        >>> sd = StochasticDepth(drop_prob=0.2)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = sd(x)  # 20% chance of returning x directly
    """

    def __init__(
        self,
        drop_prob: float = 0.0,
        survival_rate: Optional[float] = None,
        mode: str = "row",
    ) -> None:
        super().__init__()

        if survival_rate is not None:
            self.drop_prob = 1.0 - survival_rate
        else:
            self.drop_prob = drop_prob

        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        if self.mode == "row":
            keep_prob = 1.0 - self.drop_prob
            batch_size = x.size(0)

            mask = torch.empty(batch_size, 1, 1, 1, device=x.device)
            mask.bernoulli_(keep_prob)

            return x / keep_prob * mask
        else:
            if torch.rand(1).item() > self.drop_prob:
                return x
            else:
                return x * 0.0

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}, mode={self.mode}"


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    Similar to StochasticDepth but operates on individual samples
    in a batch.

    Reference: https://arxiv.org/abs/1707.01083 (as used in EfficientNet, etc.)

    Args:
        drop_prob: Probability of dropping the path

    Example:
        >>> drop_path = DropPath(drop_prob=0.2)
        >>> x = torch.randn(4, 64, 32, 32)  # batch of 4
        >>> output = drop_path(x)
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()

        output = x.div(keep_prob) * random_tensor
        return output

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class ScaledResidualBlock(nn.Module):
    """
    Scaled residual block with learnable scaling factor.

    Adds a learnable scalar to control the contribution of the
    residual connection. Useful for very deep networks.

    Reference: https://arxiv.org.org/abs/1904.04971

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        scale: Initial scaling factor (default: 1.0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.scale = nn.Parameter(torch.tensor(scale))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        residual = self.residual(x)

        out = self.scale * residual + identity
        return self.activation(out)


def create_skip_connection(
    connection_type: str, in_channels: int, out_channels: int, **kwargs
) -> nn.Module:
    """
    Factory function to create skip connections.

    Args:
        connection_type: Type of skip connection ('residual', 'preact_residual',
                       'dense', 'stochastic_depth', 'drop_path', 'scaled_residual')
        in_channels: Number of input channels
        out_channels: Number of output channels
        **kwargs: Additional arguments

    Returns:
        Skip connection module
    """
    if connection_type == "residual":
        return ResidualBlock(in_channels, out_channels, **kwargs)
    elif connection_type == "preact_residual":
        return PreActResidualBlock(in_channels, out_channels, **kwargs)
    elif connection_type == "stochastic_depth":
        return StochasticDepth(**kwargs)
    elif connection_type == "drop_path":
        return DropPath(**kwargs)
    elif connection_type == "scaled_residual":
        return ScaledResidualBlock(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown connection type: {connection_type}")

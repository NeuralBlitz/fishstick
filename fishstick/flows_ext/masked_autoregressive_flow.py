"""
Masked Autoregressive Flows Implementation.

Implements MAF (Masked Autoregressive Flow) and MADE (Masked Autoencoder
for Density Estimation) as described in:
- Papamakarios et al. (2017) "Masked Autoregressive Flow for Density Estimation"
- Germain et al. (2015) "MADE: Masked Autoencoder for Distribution Estimation"

This module provides:
- MADE network architecture
- MAF (Masked Autoregressive Flow)
- Inverse autoregressive transforms
- TAN (Transformer Autoregressive Network) variant
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class MADE(nn.Module):
    """
    Masked Autoencoder for Density Estimation (MADE).

    Autoregressive neural network with masked connections to ensure
    that output at position i only depends on inputs at positions < i.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden dimensions
        num_masks: Number of unique masks to cycle through
        activation: Activation function
        bias: Whether to use bias in linear layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        num_masks: int = 1,
        activation: str = "relu",
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 256]
        self.num_masks = num_masks
        self.activation_name = activation

        self._init_masks()
        self._build_network(bias)

    def _init_masks(self) -> None:
        """Initialize masks for autoregressive ordering."""
        self.masks = []

        for mask_idx in range(self.num_masks):
            mask = torch.zeros(self.input_dim, self.input_dim)

            hidden_size = self.hidden_dims[0]
            degrees = torch.arange(self.input_dim)

            if mask_idx > 0:
                degrees = torch.roll(degrees, mask_idx)

            for i in range(self.input_dim):
                mask[i, : int(degrees[i] + 1)] = 1.0

            self.masks.append(mask)

        self.masks = nn.ParameterList([m.clone() for m in self.masks])

        for i in range(len(self.hidden_dims) - 1):
            mask = torch.zeros(self.hidden_dims[i + 1], self.hidden_dims[i])

            degrees = torch.arange(self.hidden_dims[i])
            degrees = torch.roll(degrees, mask_idx + 1)

            for j in range(self.hidden_dims[i + 1]):
                mask[j, : int(degrees[j % self.hidden_dims[i]] + 1)] = 1.0

            self.masks.append(mask)

        final_mask = torch.zeros(self.input_dim, self.hidden_dims[-1])
        degrees = torch.arange(self.hidden_dims[-1])

        for i in range(self.input_dim):
            final_mask[i, : int(degrees[i % self.hidden_dims[-1]] + 1)] = 1.0

        self.masks.append(final_mask)

    def _build_network(self, bias: bool) -> None:
        """Build the MADE network."""
        self.layers = nn.ModuleList()

        in_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            in_dim = hidden_dim

        self.layers.append(nn.Linear(in_dim, self.input_dim * 2, bias=bias))

        self.activation = self._get_activation(self.activation_name)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(
        self,
        x: Tensor,
        mask_id: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through MADE.

        Args:
            x: Input tensor [batch, input_dim]
            mask_id: Which mask to use

        Returns:
            Tuple of (mean, log_scale)
        """
        if mask_id is None:
            mask_id = self.mask_id

        mask_offset = 0

        h = x
        for i, layer in enumerate(self.layers[:-1]):
            mask = self.masks[mask_offset + i]
            h = layer(h)
            h = h * mask
            h = self.activation(h)

        mask_offset = len(self.hidden_dims)
        output = self.layers[-1](h)
        output = output * self.masks[mask_offset]

        mean, log_scale = output.chunk(2, dim=-1)

        log_scale = torch.tanh(log_scale)

        return mean, log_scale

    def set_mask(self, mask_id: int) -> None:
        """Set the current mask."""
        self.mask_id = mask_id % self.num_masks


class InverseAutoregressiveTransform(nn.Module):
    """
    Inverse Autoregressive Transform.

    Transforms samples from a base distribution (e.g., Gaussian)
    to a target distribution autoregressively.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_layers: Number of autoregressive layers
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(nn.Linear(in_dim, hidden_dim))

        self.mean_layer = nn.Linear(hidden_dim, input_dim)
        self.log_scale_layer = nn.Linear(hidden_dim, input_dim)

        self.activation = self._get_activation(activation)

        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def _init_weights(self) -> None:
        """Initialize weights."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.zeros_(self.mean_layer.bias)

        nn.init.xavier_uniform_(self.log_scale_layer.weight)
        nn.init.zeros_(self.log_scale_layer.bias)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply inverse autoregressive transform.

        Args:
            x: Input (noise) [batch, input_dim]
            context: Context for conditioning [batch, input_dim]

        Returns:
            Tuple of (transformed, log_det)
        """
        h = context

        for layer in self.layers:
            h = layer(h)
            h = self.activation(h)

        mean = self.mean_layer(h)
        log_scale = self.log_scale_layer(h)
        log_scale = torch.tanh(log_scale)

        output = mean + x * torch.exp(log_scale)

        log_det = log_scale.sum(dim=-1)

        return output, log_det


class MAF(nn.Module):
    """
    Masked Autoregressive Flow.

    Stacks multiple Inverse Autoregressive Transforms for flexible
    density estimation.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_layers: Number of flow layers
        num_blocks: Number of blocks per layer
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_blocks: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            for _ in range(num_blocks):
                self.layers.append(
                    InverseAutoregressiveTransform(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_layers=2,
                        activation=activation,
                    )
                )

        self.permute = nn.ModuleList(
            [Permutation(input_dim) for _ in range(num_layers - 1)]
        )

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply MAF flow.

        Args:
            x: Input tensor [batch, input_dim]
            inverse: If True, apply inverse transform

        Returns:
            Tuple of (output, log_det)
        """
        if inverse:
            return self._inverse(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass: data -> noise."""
        log_det = torch.zeros(x.shape[0], device=x.device)

        context = x

        for i, layer in enumerate(self.layers):
            x, ld = layer(x, context)
            log_det = log_det + ld

            if i < len(self.permute):
                x, ld = self.permute[i](x)
                log_det = log_det + ld

        return x, log_det

    def _inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Inverse pass: noise -> data (autoregressive)."""
        log_det = torch.zeros(z.shape[0], device=z.device)

        x = torch.zeros_like(z)

        for i in range(self.input_dim):
            context = x[:, :i] if i > 0 else x[:, :0]

            x_i, ld_i = self.layers[0](
                z[:, i : i + 1],
                torch.cat([z[:, :i], context], dim=-1) if i > 0 else z[:, :0],
            )

            x[:, i : i + 1] = x_i
            log_det = log_det + ld_i

        return x, log_det

    def sample(self, num_samples: int, device: str = "cpu") -> Tensor:
        """
        Generate samples from MAF.

        Args:
            num_samples: Number of samples
            device: Device

        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.input_dim, device=device)

        x = torch.zeros_like(z)

        for i in range(self.input_dim):
            if i == 0:
                context = z[:, :0]
            else:
                context = x[:, : i - 1]

            x_i, _ = self.layers[0](
                z[:, i : i + 1],
                torch.cat([z[:, :i], context], dim=-1) if i > 0 else z[:, :0],
            )

            x[:, i : i + 1] = x_i

        return x

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log probability.

        Args:
            x: Input samples

        Returns:
            Log probabilities
        """
        z, log_det = self.forward(x, inverse=False)
        log_prob = -0.5 * (z**2).sum(dim=-1)
        log_prob = log_prob - 0.5 * self.input_dim * math.log(2 * math.pi)
        return log_prob + log_det


class Permutation(nn.Module):
    """
    Permutation layer for flow.

    Args:
        input_dim: Input dimension
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer("perm", torch.randperm(input_dim))
        self.register_buffer("inv_perm", torch.argsort(self.perm))

    def forward(
        self,
        x: Tensor,
        inverse: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Apply permutation."""
        if inverse:
            return x[:, self.inv_perm], torch.zeros(x.shape[0], device=x.device)
        return x[:, self.perm], torch.zeros(x.shape[0], device=x.device)


class MADETransformer(nn.Module):
    """
    Transformer-based Autoregressive Network (TAN).

    Uses self-attention for autoregressive density estimation.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_bins: Number of bins for discretization (if using)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_bins: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins

        self.input_embedding = nn.Linear(1, hidden_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_projection = nn.Linear(hidden_dim, 2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through TAN.

        Args:
            x: Input [batch, input_dim]

        Returns:
            Tuple of (mean, log_scale)
        """
        batch_size = x.shape[0]

        x = x.unsqueeze(-1)

        h = self.input_embedding(x)

        h = h + self.pos_embedding[:, : self.input_dim, :]

        mask = torch.triu(
            torch.ones(self.input_dim, self.input_dim, device=x.device), diagonal=1
        ).bool()

        h = self.transformer(h, mask=mask)

        output = self.output_projection(h)

        mean = output[..., 0]
        log_scale = output[..., 1]
        log_scale = torch.tanh(log_scale)

        return mean, log_scale


class MaskedLinear(nn.Module):
    """
    Masked linear layer for autoregressive models.

    Args:
        in_features: Input features
        out_features: Output features
        mask: Binary mask
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: Tensor,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("mask", mask)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with masked weights."""
        return F.linear(x, self.weight * self.mask, self.bias)


class AutoregressiveNetwork(nn.Module):
    """
    Generic autoregressive network for density estimation.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden dimensions
        activation: Activation function
        use_made: Whether to use MADE architecture
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        use_made: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim

        if use_made:
            self.net = MADE(
                input_dim=input_dim,
                hidden_dims=hidden_dims or [256, 256],
            )
        else:
            layers = []
            in_dim = input_dim

            for hidden_dim in hidden_dims or [256, 256]:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(self._get_activation(activation))
                in_dim = hidden_dim

            layers.append(nn.Linear(in_dim, input_dim * 2))

            self.net = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Input [batch, input_dim]

        Returns:
            Tuple of (mean, log_scale)
        """
        if isinstance(self.net, MADE):
            return self.net(x)

        output = self.net(x)
        mean, log_scale = output.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale)

        return mean, log_scale

    def sample(self, context: Tensor, num_samples: int = 1) -> Tensor:
        """
        Sample autoregressively.

        Args:
            context: Context tensor
            num_samples: Number of samples to generate

        Returns:
            Samples
        """
        samples = []

        for i in range(self.input_dim):
            if i == 0:
                input_slice = context[:, :0]
            else:
                input_slice = torch.cat([context[:, :i]] + samples[:i], dim=1)

            mean_i, log_scale_i = self.forward(input_slice)

            epsilon = torch.randn(num_samples, 1, device=context.device)
            sample_i = mean_i + epsilon * torch.exp(log_scale_i)

            samples.append(sample_i)

        return torch.cat(samples, dim=-1)


class FlowHead(nn.Module):
    """
    Flow head for transforming to base distribution.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim * 2),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Transform to base distribution."""
        params = self.net(x)
        mean, log_scale = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale)

        return mean, log_scale

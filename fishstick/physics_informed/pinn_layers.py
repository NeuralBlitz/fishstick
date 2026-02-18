"""
Specialized Layers for Physics-Informed Neural Networks
=======================================================

Provides neural network layers designed for PINNs, including:
- Fourier feature layers
- Sinusoidal representations
- Periodic and wavelet features
- Domain-adaptive layers
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class FourierFeatures(nn.Module):
    """
    Random Fourier Features for representing high-frequency functions.

    Based on "Fourier Features Let Networks Learn High Frequency Functions"
    (Tancik et al., 2020).

    Args:
        in_features: Input dimension
        out_features: Number of Fourier features (output dimension / 2)
        scale: Scale factor for frequencies
        learnable: If True, frequencies are learnable
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if learnable:
            self.B = nn.Parameter(
                torch.randn(in_features, out_features) * scale, requires_grad=True
            )
        else:
            self.register_buffer("B", torch.randn(in_features, out_features) * scale)

    def forward(self, x: Tensor) -> Tensor:
        """
        Transform input through random Fourier features.

        Args:
            x: Input [batch, in_features]

        Returns:
            Fourier features [batch, out_features * 2]
        """
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SinusoidalRepresentation(nn.Module):
    """
    Sinusoidal representation learning layer.

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension
        n_harmonics: Number of harmonic frequencies
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        n_harmonics: int = 4,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_harmonics = n_harmonics

        self.freqs = nn.Parameter(
            torch.arange(1, n_harmonics + 1).float(), requires_grad=True
        )

        self.net = nn.Sequential(
            nn.Linear(in_features * n_harmonics * 2, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, hidden_features),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply sinusoidal encoding.

        Args:
            x: Input [batch, in_features]

        Returns:
            Encoded features [batch, hidden_features]
        """
        x_scaled = x.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        x_encoded = torch.cat([torch.sin(x_scaled), torch.cos(x_scaled)], dim=-1)

        x_flat = x_encoded.view(x.size(0), -1)
        return self.net(x_flat)


class PeriodicFeatures(nn.Module):
    """
    Learnable periodic features with learnable frequencies.

    Args:
        in_features: Input dimension
        n_periodic: Number of periodic components
        learnable_freqs: If True, frequencies are learnable
    """

    def __init__(
        self,
        in_features: int,
        n_periodic: int = 4,
        learnable_freqs: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_periodic = n_periodic

        if learnable_freqs:
            freqs = torch.randn(in_features, n_periodic) * 2
            self.freqs = nn.Parameter(freqs)
        else:
            freqs = torch.arange(1, n_periodic + 1).float()
            freqs = freqs.view(1, -1).expand(in_features, -1)
            self.register_buffer("freqs", freqs)

        self.linear = nn.Linear(in_features * n_periodic * 2, in_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply periodic encoding.

        Args:
            x: Input [batch, in_features]

        Returns:
            Encoded features [batch, in_features]
        """
        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.n_periodic)
        angles = x_expanded * self.freqs.unsqueeze(0)

        periodic = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        periodic = periodic.view(x.size(0), -1)

        return self.linear(periodic) + x


class WaveletFeatures(nn.Module):
    """
    Wavelet feature extraction layer.

    Uses wavelet basis functions for multi-resolution representation.

    Args:
        in_features: Input dimension
        n_wavelets: Number of wavelet scales
        n_translations: Number of translations per scale
    """

    def __init__(
        self,
        in_features: int,
        n_wavelets: int = 4,
        n_translations: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_wavelets = n_wavelets
        self.n_translations = n_translations

        self.scales = nn.Parameter(
            torch.tensor([2.0**i for i in range(n_wavelets)]), requires_grad=False
        )

        self.translations = nn.Parameter(
            torch.linspace(-1, 1, n_translations).unsqueeze(0).expand(in_features, -1),
            requires_grad=False,
        )

        self.output_dim = in_features * n_wavelets * n_translations

    def _morlet_wavelet(
        self,
        x: Tensor,
        scale: Tensor,
        translation: Tensor,
    ) -> Tensor:
        """Compute Morlet wavelet at given scale and translation."""
        x_centered = (x.unsqueeze(-1) - translation) / scale.unsqueeze(-2)
        envelope = torch.exp(-(x_centered**2) / 2)
        oscillation = torch.cos(5 * x_centered)
        return envelope * oscillation

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute wavelet features.

        Args:
            x: Input [batch, in_features]

        Returns:
            Wavelet features [batch, in_features * n_wavelets * n_translations]
        """
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(-1).expand(-1, -1, self.n_translations)

        wavelet_responses = []
        for scale in self.scales:
            scale_tensor = torch.full((self.in_features,), scale.item())
            response = self._morlet_wavelet(x_expanded, scale_tensor, self.translations)
            wavelet_responses.append(response)

        wavelet_stack = torch.stack(wavelet_responses, dim=-1)
        wavelet_stack = wavelet_stack.view(batch_size, -1)

        return wavelet_stack


class DomainAdaptiveLayer(nn.Module):
    """
    Domain adaptation layer for transfer learning across domains.

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension
        domain_count: Number of domains to adapt between
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        domain_count: int = 2,
    ):
        super().__init__()
        self.domain_count = domain_count

        self.feature_transform = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.Tanh(),
        )

        self.domain_embeddings = nn.Parameter(
            torch.randn(domain_count, hidden_features) * 0.1
        )

        self.domain_attention = nn.MultiheadAttention(
            hidden_features, num_heads=4, batch_first=True
        )

    def forward(
        self,
        x: Tensor,
        domain_id: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Transform features with domain adaptation.

        Args:
            x: Input features [batch, in_features]
            domain_id: Domain indices [batch] (optional)

        Returns:
            Adapted features [batch, hidden_features]
        """
        features = self.feature_transform(x)

        if domain_id is not None:
            domain_embeds = self.domain_embeddings[domain_id]

            features_expanded = features.unsqueeze(1)
            domain_expanded = domain_embeds.unsqueeze(1)

            attended, _ = self.domain_attention(
                features_expanded, domain_expanded, domain_expanded
            )

            features = features + attended.squeeze(1)

        return features


class ResidualBlockPINN(nn.Module):
    """
    Residual block tailored for PINN architectures.

    Args:
        dim: Feature dimension
        hidden_dim: Hidden dimension
        activation: Activation function
        use_norm: Whether to use layer normalization
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "tanh",
        use_norm: bool = True,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim * 4

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm1 = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_norm else nn.Identity()

        self.activation = self._get_activation(activation)

        if dim != hidden_dim:
            self.shortcut = nn.Linear(dim, hidden_dim)
        else:
            self.shortcut = nn.Identity()

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.Tanh())

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input [batch, dim]

        Returns:
            Output [batch, dim]
        """
        identity = x

        out = self.fc1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.fc2(out)
        out = self.norm2(out)

        if isinstance(self.shortcut, nn.Linear):
            identity = self.shortcut(identity)

        out = out + identity
        out = self.activation(out)

        return out


class HighwayNetwork(nn.Module):
    """
    Highway network for information flow in deep PINNs.

    Based on "Highway Networks" (Srivastava et al., 2015).

    Args:
        in_features: Input dimension
        num_layers: Number of highway layers
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int = 4,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.gates = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.Linear(in_features, in_features))
            self.gates.append(nn.Linear(in_features * 2, in_features))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through highway network.

        Args:
            x: Input [batch, dim]

        Returns:
            Output [batch, dim]
        """
        for layer, gate in zip(self.layers, self.gates):
            transform = torch.tanh(layer(x))

            carry = torch.sigmoid(gate(torch.cat([x, transform], dim=-1)))

            x = transform * carry + x * (1 - carry)

        return x


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention for PINN.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply scaled dot-product attention.

        Args:
            query: Query [batch, seq_len, embed_dim]
            key: Key [batch, seq_len, embed_dim]
            value: Value [batch, seq_len, embed_dim]
            mask: Optional attention mask

        Returns:
            Attention output [batch, seq_len, embed_dim]
        """
        batch_size = query.size(0)

        Q = (
            self.q_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )

        return self.out_proj(attn_output)


class PINNEncoder(nn.Module):
    """
    Encoder for spatial-temporal PINN inputs.

    Args:
        spatial_dim: Spatial dimension (1, 2, or 3)
        temporal: Whether to include time
        hidden_dim: Hidden dimension
        use_fourier: Whether to use Fourier features
    """

    def __init__(
        self,
        spatial_dim: int = 1,
        temporal: bool = True,
        hidden_dim: int = 128,
        use_fourier: bool = True,
    ):
        super().__init__()

        input_dim = spatial_dim + (1 if temporal else 0)

        if use_fourier:
            self.fourier = FourierFeatures(input_dim, hidden_dim // 4, scale=1.0)
            encoder_input = (hidden_dim // 4) * 2
        else:
            self.fourier = None
            encoder_input = input_dim

        self.net = nn.Sequential(
            nn.Linear(encoder_input, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """
        Encode spatial-temporal inputs.

        Args:
            x: Spatial coordinates [batch, spatial_dim]
            t: Time [batch] (optional)

        Returns:
            Encoded features [batch, hidden_dim]
        """
        if t is not None:
            inputs = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        else:
            inputs = x

        if self.fourier is not None:
            inputs = self.fourier(inputs)

        return self.net(inputs)


class PINNDecoder(nn.Module):
    """
    Decoder for PINN outputs.

    Args:
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        n_layers: Number of decoder layers
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 1,
        n_layers: int = 2,
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Decode hidden features to output.

        Args:
            x: Hidden features [batch, hidden_dim]

        Returns:
            Output [batch, output_dim]
        """
        return self.net(x)

"""
Model Building Blocks for Audio Source Separation

Provides reusable neural network components for building source separation
models including convolutional blocks, transformer blocks, LSTM blocks,
and attention mechanisms.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation.

    Standard building block for encoder/decoder networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        norm_type: str = "gLN",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.norm_type = norm_type
        if norm_type == "gLN":
            self.norm = GlobalLayerNorm(out_channels)
        elif norm_type == "cLN":
            self.norm = ChannelLayerNorm(out_channels)
        elif norm_type == "bn":
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()

        self.activation = get_activation(activation)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)

        if self.dropout:
            x = self.dropout(x)

        return x


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization.

    Normalizes across both channel and time dimensions.
    """

    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.gamma = nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global layer norm.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Normalized tensor
        """
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        x = x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)

        return x


class ChannelLayerNorm(nn.Module):
    """Channel-wise Layer Normalization.

    Normalizes across time for each channel independently.
    """

    def __init__(self, num_channels: int, eps: float = 1e-8):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.gamma = nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel layer norm.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Normalized tensor
        """
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        x = x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)

        return x


class TransformerBlock(nn.Module):
    """Transformer block for audio processing.

    Multi-head self-attention with feedforward network.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        d_ff = d_ff or d_model * 4

        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Attention mask

        Returns:
            Output tensor
        """
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class LSTMBlock(nn.Module):
    """LSTM block for sequential audio processing.

    Bidirectional LSTM with optional attention.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_size)

        Returns:
            Tuple of (output, (hidden_state, cell_state))
        """
        output, (hidden, cell) = self.lstm(x)

        return output, (hidden, cell)


class AttentionBlock(nn.Module):
    """Attention block for feature weighting.

    Applies attention over time or frequency dimensions.
    """

    def __init__(
        self,
        dim: int,
        attention_type: str = "scaled_dot",
        num_heads: int = 8,
    ):
        super().__init__()

        self.dim = dim
        self.attention_type = attention_type
        self.num_heads = num_heads

        if attention_type == "scaled_dot":
            self.attention = ScaledDotProductAttention(dim, num_heads)
        elif attention_type == "additive":
            self.attention = AdditiveAttention(dim)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention.

        Args:
            x: Query tensor
            context: Optional context tensor

        Returns:
            Attended tensor
        """
        return self.attention(x, context)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention.

    Standard attention mechanism from "Attention is All You Need".
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply scaled dot-product attention.

        Args:
            query: Query tensor
            key: Key tensor (if None, use query)
            value: Value tensor (if None, use key)

        Returns:
            Attended tensor
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size = query.size(0)

        Q = (
            self.q_linear(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.k_linear(key)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.v_linear(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended = torch.matmul(attention_weights, V)

        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.out_linear(attended)


class AdditiveAttention(nn.Module):
    """Additive attention mechanism.

    Bahdanau attention with feedforward alignment scoring.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        hidden_dim = hidden_dim or d_model

        self.W_a = nn.Linear(d_model, hidden_dim)
        self.W_b = nn.Linear(d_model, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply additive attention.

        Args:
            query: Query tensor
            key: Key tensor (if None, use query)

        Returns:
            Attended tensor
        """
        if key is None:
            key = query

        a = self.W_a(query)
        b = self.W_b(key)

        scores = self.v(torch.tanh(a.unsqueeze(2) + b.unsqueeze(1))).squeeze(-1)

        attention_weights = F.softmax(scores, dim=-1)

        attended = torch.bmm(attention_weights, key)

        return attended


class DualPathBlock(nn.Module):
    """Dual Path Processing Block.

    Processes audio in both intra-chunk and inter-chunk paths,
    useful for dual-path RNN architectures.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        chunk_size: int = 100,
        norm_type: str = "gLN",
    ):
        super().__init__()

        self.chunk_size = chunk_size

        self.intra_chunk = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            get_activation("relu"),
            nn.Conv1d(hidden_channels, in_channels, 1),
        )

        self.inter_chunk = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            get_activation("relu"),
            nn.Conv1d(hidden_channels, in_channels, 1),
        )

        if norm_type == "gLN":
            self.norm = GlobalLayerNorm(in_channels)
        else:
            self.norm = ChannelLayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual path processing.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Processed tensor
        """
        batch, channels, length = x.shape

        num_chunks = length // self.chunk_size
        if length % self.chunk_size != 0:
            num_chunks += 1
            padding = num_chunks * self.chunk_size - length
            x = F.pad(x, (0, padding))

        x = x.reshape(batch, channels, num_chunks, self.chunk_size)
        x = x.permute(0, 2, 1, 3)

        intra_out = self.intra_chunk(x.reshape(-1, channels, self.chunk_size))
        intra_out = intra_out.reshape(batch, num_chunks, channels, self.chunk_size)

        x_agg = x.sum(dim=-1)
        inter_out = self.inter_chunk(x_agg.permute(0, 2, 1))
        inter_out = inter_out.permute(0, 2, 1).unsqueeze(-1)

        out = intra_out + inter_out
        out = out.permute(0, 2, 1, 3).reshape(batch, channels, -1)
        out = out[:, :, :length]

        return self.norm(
            out
            + x[:, :, :, :length].permute(0, 2, 1, 3).reshape(batch, channels, length)
        )


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block.

    Channel attention mechanism for adaptive feature recalibration.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Recalibrated tensor
        """
        b, c, _ = x.size()

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)

        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection.

    Standard residual connection for deeper networks.
    """

    def __init__(
        self,
        module: nn.Module,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block.

        Args:
            x: Input tensor

        Returns:
            Output with residual
        """
        residual = x
        out = self.module(x)

        if self.dropout:
            out = self.dropout(out)

        return out + residual


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution.

    Efficient convolution with channel-wise and point-wise operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_type: str = "gLN",
        activation: str = "relu",
    ):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

        if norm_type == "gLN":
            self.norm = GlobalLayerNorm(out_channels)
        else:
            self.norm = ChannelLayerNorm(out_channels)

        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise separable convolution.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)

        return x


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.

    Args:
        name: Activation name

    Returns:
        Activation module
    """
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(),
        "prelu": nn.PReLU(),
        "selu": nn.SELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(dim=1),
        "none": nn.Identity(),
    }

    return activations.get(name.lower(), nn.ReLU())


class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block.

    Dilated causal convolution for temporal modeling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.norm = GlobalLayerNorm(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply TCN block.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Output tensor
        """
        residual = self.residual(x)

        x = self.conv1(x)
        x = x[:, :, : residual.shape[-1]]

        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = x[:, :, : residual.shape[-1]]

        x = self.norm(x + residual)

        return x

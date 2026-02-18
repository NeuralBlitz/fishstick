"""
Attention Mechanisms for Machine Translation

Specialized attention mechanisms for MT including:
- Bahdanau (additive) attention
- Luong (multiplicative) attention
- Multi-head attention
- Convolutional attention
- Linearized attention
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) attention mechanism.

    Computes attention as:
        score(s_t, h_i) = v^T * tanh(W_a * s_t + U_a * h_i)

    Args:
        hidden_dim: Decoder hidden dimension
        encoder_output_dim: Encoder output dimension
        attention_dim: Attention intermediate dimension
    """

    def __init__(
        self,
        hidden_dim: int,
        encoder_output_dim: int,
        attention_dim: int = 512,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.attention_dim = attention_dim

        self.W_a = nn.Linear(encoder_output_dim, attention_dim)
        self.U_a = nn.Linear(hidden_dim, attention_dim)
        self.V_a = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        hidden: Tensor,
        encoder_outputs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention weights and context.

        Args:
            hidden: Decoder hidden state [batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_dim]
            mask: Optional mask [batch_size, src_len]

        Returns:
            context: Context vector [batch_size, encoder_output_dim]
            attention_weights: Attention weights [batch_size, src_len]
        """
        src_len = encoder_outputs.size(1)

        hidden_expanded = hidden.unsqueeze(1).expand(-1, src_len, -1)

        energy = torch.tanh(self.W_a(encoder_outputs) + self.U_a(hidden_expanded))

        attention = self.V_a(energy).squeeze(-1)

        if mask is not None:
            attention = attention.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(attention, dim=-1)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class LuongAttention(nn.Module):
    """Luong (Multiplicative) attention mechanism.

    Args:
        hidden_dim: Decoder hidden dimension
        encoder_output_dim: Encoder output dimension
        attention_type: Type of attention ('dot', 'general', 'concat')
    """

    def __init__(
        self,
        hidden_dim: int,
        encoder_output_dim: int,
        attention_type: str = "general",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.attention_type = attention_type.lower()

        if self.attention_type == "general":
            self.W_a = nn.Linear(encoder_output_dim, hidden_dim)
        elif self.attention_type == "concat":
            self.W_a = nn.Linear(encoder_output_dim + hidden_dim, hidden_dim)
            self.v_a = nn.Parameter(torch.randn(hidden_dim))

    def forward(
        self,
        hidden: Tensor,
        encoder_outputs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention weights and context.

        Args:
            hidden: Decoder hidden state [batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_dim]
            mask: Optional mask [batch_size, src_len]

        Returns:
            context: Context vector [batch_size, encoder_output_dim]
            attention_weights: Attention weights [batch_size, src_len]
        """
        src_len = encoder_outputs.size(1)

        if self.attention_type == "dot":
            energy = torch.bmm(
                hidden.unsqueeze(1), encoder_outputs.transpose(1, 2)
            ).squeeze(1)
        elif self.attention_type == "general":
            energy = torch.bmm(
                hidden.unsqueeze(1), self.W_a(encoder_outputs).transpose(1, 2)
            ).squeeze(1)
        else:
            hidden_expanded = hidden.unsqueeze(1).expand(-1, src_len, -1)
            energy = torch.bmm(
                hidden.unsqueeze(1),
                self.W_a(
                    torch.cat([hidden_expanded, encoder_outputs], dim=-1)
                ).transpose(1, 2),
            ).squeeze(1)

        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(energy, dim=-1)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for Transformer.

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute multi-head attention.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask

        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights
        """
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.W_o(context)

        return output, attention_weights


class ConvolutionalAttention(nn.Module):
    """Convolutional attention mechanism using depthwise separable convolutions.

    Args:
        hidden_dim: Hidden dimension
        attention_dim: Attention intermediate dimension
        kernel_size: Convolution kernel size
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int = 512,
        kernel_size: int = 5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.kernel_size = kernel_size

        self.query_proj = nn.Linear(hidden_dim, attention_dim)
        self.key_conv = nn.Conv1d(
            hidden_dim,
            attention_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=attention_dim,
        )
        self.value_conv = nn.Conv1d(
            hidden_dim,
            attention_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=attention_dim,
        )

        self.output_proj = nn.Linear(attention_dim, hidden_dim)

    def forward(
        self,
        hidden: Tensor,
        encoder_outputs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolutional attention.

        Args:
            hidden: Decoder hidden state [batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_dim]
            mask: Optional mask

        Returns:
            context: Context vector
            attention_weights: Attention weights
        """
        batch_size, src_len, _ = encoder_outputs.size()

        query = self.query_proj(hidden).unsqueeze(1)

        encoder_outputs_t = encoder_outputs.transpose(1, 2)
        key = self.key_conv(encoder_outputs_t).transpose(1, 2)
        value = self.value_conv(encoder_outputs_t).transpose(1, 2)

        energy = torch.bmm(query.squeeze(1), key.transpose(1, 2))

        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(energy, dim=-1)

        context = torch.bmm(attention_weights, value).squeeze(1)

        output = self.output_proj(context)

        return output, attention_weights


class LinearizedAttention(nn.Module):
    """Linearized attention for efficient computation.

    Uses feature map approximation for linear complexity.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.elu = nn.ELU()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute linearized attention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional mask

        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        batch_size = query.size(0)

        Q = (
            self.W_q(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = (
            self.W_v(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        Q = self.elu(Q) + 1
        K = self.elu(K) + 1

        Q_normalized = Q / (Q.sum(dim=-1, keepdim=True) + 1e-8)
        K_normalized = K / (K.sum(dim=-2, keepdim=True) + 1e-8)

        context = torch.bmm(
            Q_normalized.view(batch_size * self.num_heads, -1, self.d_k),
            K_normalized.view(batch_size * self.num_heads, -1, self.d_k).transpose(
                1, 2
            ),
        )

        attention_weights = context.view(
            batch_size, self.num_heads, Q.size(2), K.size(2)
        )

        output = torch.bmm(
            attention_weights.view(batch_size * self.num_heads, Q.size(2), K.size(2)),
            V.view(batch_size * self.num_heads, -1, self.d_k),
        )

        output = output.view(batch_size, self.num_heads, Q.size(2), self.d_k)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(output)

        return output, attention_weights


class CausalSelfAttention(nn.Module):
    """Causal self-attention for autoregressive models.

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute causal self-attention.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional additional mask

        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        output = self.W_o(context)

        return output, attention_weights


class LocationAwareAttention(nn.Module):
    """Location-aware attention that incorporates previous attention weights.

    Args:
        hidden_dim: Hidden dimension
        attention_dim: Attention dimension
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int = 512,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        self.W_a = nn.Linear(hidden_dim, attention_dim)
        self.V_a = nn.Linear(attention_dim, 1, bias=False)

        self.location_conv = nn.Conv1d(1, 32, kernel_size=31, padding=15)
        self.location_proj = nn.Linear(32, attention_dim)

    def forward(
        self,
        hidden: Tensor,
        encoder_outputs: Tensor,
        previous_attention: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute location-aware attention.

        Args:
            hidden: Decoder hidden state [batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_dim]
            previous_attention: Previous attention weights [batch_size, src_len]
            mask: Optional mask

        Returns:
            context: Context vector
            attention_weights: Attention weights
        """
        batch_size, src_len, _ = encoder_outputs.size()

        location_features = torch.zeros(batch_size, src_len, device=hidden.device)

        if previous_attention is not None:
            location_input = previous_attention.unsqueeze(1)
            location_conv = self.location_conv(location_input).transpose(1, 2)
            location_features = self.location_proj(location_conv).squeeze(1)

        hidden_expanded = hidden.unsqueeze(1).expand(-1, src_len, -1)

        energy = self.W_a(hidden_expanded) + location_features.unsqueeze(-1)
        energy = torch.tanh(energy).squeeze(-1)

        attention = self.V_a(energy).squeeze(-1)

        if mask is not None:
            attention = attention.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(attention, dim=-1)

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention for encoder-decoder attention.

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute cross-attention between query and key-value pairs.

        Args:
            query: Query tensor [batch_size, tgt_len, d_model]
            key: Key tensor [batch_size, src_len, d_model]
            value: Value tensor [batch_size, src_len, d_model]
            mask: Attention mask

        Returns:
            output: Attention output [batch_size, tgt_len, d_model]
            attention_weights: Attention weights [batch_size, nhead, tgt_len, src_len]
        """
        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = key.size(1)

        Q = (
            self.W_q(query)
            .view(batch_size, tgt_len, self.nhead, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(key)
            .view(batch_size, src_len, self.nhead, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(value)
            .view(batch_size, src_len, self.nhead, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        context = (
            context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        )

        output = self.W_o(context)

        return output, attention_weights

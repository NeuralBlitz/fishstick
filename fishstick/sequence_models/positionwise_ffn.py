"""
Position-wise Feed-Forward Networks

Position-wise feedforward networks for transformer architectures,
including gated linear units, mixture of experts, and variants.
"""

from typing import Optional
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """Position-wise feedforward network (from "Attention Is All You Need").

    Two-layer MLP applied at each position independently:
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) from "Language Modeling with Gated Convolutional Networks".

    Applies a gating mechanism to control information flow:
    GLU(x) = (xW + b) * sigmoid(xV + c)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_model, d_ff)
        self.v = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_1(x)) * torch.sigmoid(self.v(x) + self.w_2(x))


class GatedResidualNetwork(nn.Module):
    """Gated residual network from "Neural Architecture Search with Reinforcement Learning".

    Combines gating with residual connections for better gradient flow.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_model, d_ff)
        self.w_3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.activation(self.w_1(x))
        x = self.dropout(self.w_2(x) * torch.sigmoid(x))
        x = self.w_3(x)
        return self.layer_norm(residual + x)


class SwitchTransformerFFN(nn.Module):
    """Switch Transformer-style Feed-Forward Network with Mixture of Experts.

    Routes each token to one of multiple expert networks,
    from "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity".
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        num_experts: int = 8,
        dropout: float = 0.0,
        capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.routing = nn.Linear(d_model, num_experts)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(num_experts)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.shape

        router_logits = self.routing(x)
        routing_weights = F.softmax(router_logits, dim=-1)

        expert_weights, expert_indices = torch.max(routing_weights, dim=-1)

        expert_weights = expert_weights.unsqueeze(-1)

        output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            expert_mask = (expert_indices == i).unsqueeze(-1)
            expert_input = x[expert_mask.expand_as(x)].view(-1, d_model)

            if expert_input.size(0) > 0:
                expert_output = expert(expert_input)
                output[expert_mask] = expert_output

        return self.dropout(output * expert_weights)


class Conv1dFFN(nn.Module):
    """Conv1d-based Feed-Forward Network.

    Uses 1D convolution for local information mixing,
    from "ConvBERT: Improving BERT with Span-based Dynamic Convolution".
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model,
            d_ff,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.activation(self.conv(x))
        x = x.transpose(1, 2)
        return self.proj(self.dropout(x))


class PositionwiseFFNWithConv(nn.Module):
    """Position-wise FFN with convolutions for local context.

    Combines standard FFN with depthwise convolution.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.ffn(x)

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        return self.layer_norm(residual + x)


class FeedForwardChunk(nn.Module):
    """Chunked feed-forward network.

    Splits the feed-forward computation into chunks for efficiency.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        chunk_size: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)

        if seq_len <= self.chunk_size:
            return self.ffn(x)

        chunks = []
        for i in range(0, seq_len, self.chunk_size):
            chunk = x[:, i : i + self.chunk_size]
            chunks.append(self.ffn(chunk))

        return torch.cat(chunks, dim=1)


class MLPMixerFFN(nn.Module):
    """MLP-Mixer style feed-forward network.

    Applies channel mixing and token mixing operations.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.channel_mixing = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.token_mixing(x)
        x = x.transpose(1, 2)
        x = self.channel_mixing(x + residual)
        return x


class FastformerFFN(nn.Module):
    """Fastformer-style attention-based FFN.

    From "Fastformer: Additive Attention Can Be Fast".
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.shape

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        attention_weights = F.softmax(
            torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d_model), dim=-1
        )

        global_context = torch.bmm(attention_weights, value)

        query = query * global_context

        output = self.ffn(query)

        return self.layer_norm(x + self.dropout(output))

"""Brain-inspired attention mechanisms for neural networks."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NeuralAttention(nn.Module):
    """Neural attention inspired by cortical microcircuits.

    Implements attention that mimics the dynamics of cortical pyramidal
    neurons with feedforward and feedback connections.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in qkv projection
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply neural attention.

        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Optional attention mask

        Returns:
            Attended output tensor
        """
        B, N, C = x.shape

        q = (
            self.q_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return out


class WinnerTakeAllAttention(nn.Module):
    """Winner-Take-All (WTA) attention mechanism.

    Implements competitive attention where only the most active neurons
    are selected, mimicking lateral inhibition in the brain.

    Args:
        dim: Input dimension
        k: Number of winners to select
        temperature: Softmax temperature
    """

    def __init__(
        self,
        dim: int,
        k: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.k = k
        self.temperature = temperature

        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply WTA attention.

        Args:
            x: Input tensor (batch, num_elements, dim)

        Returns:
            Tuple of (attended_output, selection_mask)
        """
        B, N, C = x.shape

        proj = self.proj(x)
        scores = proj.mean(dim=-1)

        if self.training:
            noise = torch.rand_like(scores) * 0.1
            scores = scores + noise

        topk_scores, topk_idx = torch.topk(scores, min(self.k, N), dim=-1)

        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_idx, 1.0)

        attn_weights = F.softmax(scores / self.temperature, dim=-1)

        attn_weights = attn_weights * mask

        out = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

        return out, mask


class DivisiveNormalization(nn.Module):
    """Divisive normalization inspired by retinal and cortical circuits.

    Implements normalization that divides by a pooled estimate of activity:
        y_i = x_i / (σ + ∑_j w_ij * x_j)

    Args:
        dim: Feature dimension
        num_groups: Number of normalization groups
    """

    def __init__(
        self,
        dim: int,
        num_groups: int = 32,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.epsilon = epsilon

        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply divisive normalization.

        Args:
            x: Input tensor (batch, dim)

        Returns:
            Normalized output
        """
        B, C = x.shape

        group_size = C // self.num_groups

        x_reshaped = x.view(B, self.num_groups, group_size)

        pool = x_reshaped.abs().mean(dim=-1, keepdim=True)

        normalized = x_reshaped / (self.epsilon + pool)

        out = normalized.view(B, C) * self.weight + self.bias

        return out


class NormalizedAttention(nn.Module):
    """Attention with divisive normalization.

    Combines attention weights with normalization inspired by cortical
    gain control mechanisms.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.norm = divisiveNormalization(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply normalized attention."""
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        attn = self.norm(attn.transpose(1, 2)).transpose(1, 2)

        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out


class FeedbackAttention(nn.Module):
    """Feedback attention mimicking top-down cortical pathways.

    Implements attention with recurrent feedback connections that allow
    predictions to modulate bottom-up processing.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        feedback_dim: Feedback pathway dimension
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        feedback_dim: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.feedback_proj = nn.Linear(feedback_dim, dim)

        self.out_proj = nn.Linear(dim, dim)

        self.feedback_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: Tensor,
        feedback: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Apply feedback attention.

        Args:
            x: Input tensor (batch, seq_len, dim)
            feedback: Optional feedback from higher levels

        Returns:
            Tuple of (output, feedback)
        """
        B, N, C = x.shape

        q = (
            self.q_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if feedback is not None:
            feedback_mod = self.feedback_proj(feedback)
            out = out * (1 + self.feedback_gate * torch.tanh(feedback_mod))

        out = self.out_proj(out)

        pooled = out.mean(dim=1)

        return out, pooled


class PredictiveAttention(nn.Module):
    """Predictive attention for hierarchical inference.

    Implements attention that predicts future states and uses predictions
    to modulate current processing, inspired by predictive coding.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        prediction_horizon: Steps to predict ahead
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        prediction_horizon: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.prediction_horizon = prediction_horizon

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.predictor = nn.LSTM(dim, dim, num_layers=2, batch_first=True)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply predictive attention.

        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Tuple of (output, prediction)
        """
        B, N, C = x.shape

        q = (
            self.q_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = F.softmax(attn, dim=-1)

        context = (attn @ v).transpose(1, 2).reshape(B, N, C)

        prediction, _ = self.predictor(context[:, :-1])
        prediction = F.pad(prediction, (0, 0, 0, 1))

        out = context + 0.1 * prediction
        out = self.out_proj(out)

        return out, prediction

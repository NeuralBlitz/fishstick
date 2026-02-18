"""
Autoregressive Transformers for Generative Modeling.

Implements autoregressive generation using transformer architectures:
- GPT-style language model generation
- Autoregressive image generation with transformers
- Positional encoding variants
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.

    Encodes position information using sine and cosine functions.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add learned positional encoding."""
        batch_size, seq_len = x.shape[:2]
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        x = x + self.position_embeddings(positions)
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Applies rotation to queries and keys for relative position encoding.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply rotary positional embedding.

        Args:
            x: Input tensor [..., seq_len, dim]

        Returns:
            Tensor with rotary embeddings applied
        """
        seq_len = x.size(-2)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        return x * cos + self._rotate_half(x) * sin

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half of the dimensions."""
        x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
        return torch.cat([-x2, x1], dim=-1)


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention for autoregressive generation.

    Uses causal masking to prevent attending to future positions.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with causal masking.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(causal_mask == 0, float("-inf"))

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.reshape(batch_size, seq_len, self.d_model)
        out = self.proj(out)

        return out


class TransformerBlock(nn.Module):
    """Single transformer block with causal attention and feedforward."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through transformer block."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AutoregressiveTransformer(nn.Module):
    """
    Autoregressive transformer for generative modeling.

    Generic transformer architecture for autoregressive sequence generation.

    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for training.

        Args:
            x: Input tokens [batch, seq_len]
            mask: Optional attention mask

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        start_tokens: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        Autoregressive generation.

        Args:
            start_tokens: Starting token sequence [batch, seq_len]
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Generated sequence [batch, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            x = start_tokens

            logits = self.forward(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            start_tokens = torch.cat([start_tokens, next_token], dim=1)

        return start_tokens

    def compute_loss(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute language modeling loss.

        Args:
            input_ids: Input token IDs
            labels: Target token IDs (if None, shifted version of input_ids)

        Returns:
            Cross-entropy loss
        """
        if labels is None:
            labels = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]

        logits = self.forward(input_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return loss


class AutoregressiveImageTransformer(nn.Module):
    """
    Autoregressive transformer for image generation.

    Treats image pixels as a sequence for autoregressive generation.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        channels: int = 3,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_embedding = nn.Linear(patch_dim, d_model)
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)

        num_tokens = 256**channels
        self.vocab_size = num_tokens

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Image tensor [batch, channels, height, width]

        Returns:
            Predicted pixel values
        """
        batch_size = x.shape[0]

        x = x.reshape(
            batch_size,
            self.channels,
            self.image_size // self.patch_size,
            self.patch_size,
            self.image_size // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(
            batch_size, -1, self.channels * self.patch_size * self.patch_size
        )

        x = self.patch_embedding(x)
        x = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.head(x)

        return x

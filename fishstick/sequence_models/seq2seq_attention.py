"""
Sequence-to-Sequence Models with Attention

Encoder-decoder architectures with various attention mechanisms
for neural machine translation and sequence generation.
"""

from typing import Optional, Tuple
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Seq2SeqEncoder(nn.Module):
    """Base encoder for sequence-to-sequence models."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        embedded = self.embedding(x)

        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        output, hidden = self.lstm(embedded)

        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention.

    Implements the attention mechanism from "Neural Machine Translation
    by Jointly Learning to Align and Translate" (Bahdanau et al., 2014).
    """

    def __init__(
        self,
        query_size: int,
        key_size: int,
        value_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.hidden_size = hidden_size

        self.W_q = nn.Linear(query_size, hidden_size)
        self.W_k = nn.Linear(key_size, hidden_size)
        self.W_v = nn.Linear(value_size, value_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        query = query.unsqueeze(1)

        scores = self.v(torch.tanh(self.W_q(query) + self.W_k(keys))).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), values)

        return context.squeeze(1), attention_weights


class LuongAttention(nn.Module):
    """Luong (Multiplicative) Attention.

    Implements the attention mechanism from "Effective Approaches
    to Attention-based Neural Machine Translation" (Luong et al., 2015).
    """

    def __init__(
        self,
        query_size: int,
        key_size: int,
        value_size: int,
    ):
        super().__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size

        self.W = nn.Linear(key_size, query_size)

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        query = query.unsqueeze(1)

        scores = torch.bmm(query, self.W(keys).transpose(1, 2)).squeeze(1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), values)

        return context.squeeze(1), attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention for sequence-to-sequence models.

    Implements scaled dot-product attention with multiple heads
    from "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

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

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        output = self.W_o(context)

        return output, attention_weights


class Seq2SeqDecoder(nn.Module):
    """Base decoder for sequence-to-sequence models."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: Tensor,
        hidden: Tuple[Tensor, Tensor],
        encoder_output: Optional[Tensor] = None,
        attention_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        raise NotImplementedError


class AttentionSeq2SeqDecoder(nn.Module):
    """Sequence-to-sequence decoder with attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        encoder_hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        attention_type: str = "bahdanau",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.attention = (
            BahdanauAttention(
                hidden_size,
                encoder_hidden_size,
                encoder_hidden_size,
                hidden_size,
            )
            if attention_type == "bahdanau"
            else LuongAttention(
                hidden_size,
                encoder_hidden_size,
                encoder_hidden_size,
            )
        )

        self.lstm = nn.LSTM(
            embed_dim + encoder_hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: Tensor,
        hidden: Tuple[Tensor, Tensor],
        encoder_output: Tensor,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        last_hidden = hidden[0][-1]

        context, attention_weights = self.attention(
            last_hidden, encoder_output, encoder_output, encoder_mask
        )

        embedded = self.embedding(x)
        embedded = torch.cat([embedded, context.unsqueeze(1)], dim=-1)

        output, hidden = self.lstm(embedded, hidden)

        logits = self.output_proj(output)

        return logits, hidden


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff_out = self.feed_forward(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with multi-head self-attention and cross-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        attn_out, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attn_out)
        x = self.norm2(x)

        ff_out = self.feed_forward(x)
        x = x + self.dropout3(ff_out)
        x = self.norm3(x)

        return x


class Seq2SeqModel(nn.Module):
    """Complete sequence-to-sequence model with attention."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_size: int = 512,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dropout: float = 0.1,
        attention_type: str = "bahdanau",
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.encoder = Seq2SeqEncoder(
            src_vocab_size,
            embed_dim,
            hidden_size,
            num_encoder_layers,
            dropout,
            bidirectional=True,
        )

        encoder_output_size = hidden_size * 2

        self.decoder = AttentionSeq2SeqDecoder(
            tgt_vocab_size,
            embed_dim,
            hidden_size,
            encoder_output_size,
            num_decoder_layers,
            dropout,
            attention_type,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_lengths: Optional[Tensor] = None,
    ) -> Tensor:
        encoder_output, encoder_hidden = self.encoder(src, src_lengths)

        decoder_hidden = (
            encoder_hidden[0][-2:],
            encoder_hidden[1][-2:],
        )

        outputs = []
        for t in range(tgt.size(1)):
            output, decoder_hidden = self.decoder(
                tgt[:, t : t + 1],
                decoder_hidden,
                encoder_output,
            )
            outputs.append(output)

        return torch.cat(outputs, dim=1)

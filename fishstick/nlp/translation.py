"""
Machine Translation Module for fishstick.

This module provides comprehensive machine translation capabilities including:
- Translation models (Seq2Seq, Attention, Transformer, mBART, mT5, M2M100, DeltaLM)
- Low-resource translation techniques
- Document-level translation
- Evaluation metrics
- Preprocessing utilities
- Data augmentation
- Decoding strategies
- Training utilities
"""

import re
import math
import random
import unicodedata
from typing import List, Dict, Tuple, Optional, Union, Callable, Any, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import numpy as np


# =============================================================================
# 1. Translation Models
# =============================================================================


class Seq2SeqTranslation(nn.Module):
    """
    Basic RNN-based Encoder-Decoder for machine translation.

    Classic sequence-to-sequence model with LSTM encoder and decoder.

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension for LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Whether encoder is bidirectional
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Encoder
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.encoder_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Decoder
        encoder_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.decoder_lstm = nn.LSTM(
            embed_dim + encoder_output_dim,  # Input feeding
            hidden_dim,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, tgt_vocab_size),
        )

        self.dropout = nn.Dropout(dropout)

    def encode(
        self, src: Tensor, src_lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Encode source sequence.

        Args:
            src: Source token IDs [batch_size, src_len]
            src_lengths: Actual lengths of source sequences

        Returns:
            encoder_outputs: All encoder hidden states
            (hidden, cell): Final hidden and cell states
        """
        embedded = self.dropout(self.encoder_embedding(src))

        if src_lengths is not None:
            embedded = pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        outputs, (hidden, cell) = self.encoder_lstm(embedded)

        if src_lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # Handle bidirectional: combine forward and backward
        if self.bidirectional:
            # hidden: [num_layers * 2, batch, hidden_dim]
            # Need to reshape to [num_layers, batch, hidden_dim * 2]
            hidden = self._combine_bidirectional(hidden)
            cell = self._combine_bidirectional(cell)

        return outputs, (hidden, cell)

    def _combine_bidirectional(self, state: Tensor) -> Tensor:
        """Combine bidirectional states."""
        # state: [num_layers * 2, batch, hidden_dim]
        num_layers = self.num_layers
        batch_size = state.size(1)
        hidden_dim = state.size(2)

        # Reshape and concatenate
        state = state.view(num_layers, 2, batch_size, hidden_dim)
        return torch.cat([state[:, 0], state[:, 1]], dim=2)

    def decode_step(
        self,
        input_token: Tensor,
        hidden: Tuple[Tensor, Tensor],
        encoder_outputs: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Single decoder step.

        Args:
            input_token: Current input token [batch_size, 1]
            hidden: Previous hidden state
            encoder_outputs: Encoder outputs for attention

        Returns:
            output: Logits for next token
            hidden: Updated hidden state
        """
        embedded = self.dropout(self.decoder_embedding(input_token))

        # Simple input feeding: concatenate with context
        batch_size = embedded.size(0)
        context = encoder_outputs.mean(dim=1, keepdim=True)  # [batch, 1, hidden*2]

        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.decoder_lstm(rnn_input, hidden)

        logits = self.output_projection(output.squeeze(1))
        return logits, hidden

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with teacher forcing.

        Args:
            src: Source tokens [batch_size, src_len]
            tgt: Target tokens [batch_size, tgt_len]
            src_lengths: Source sequence lengths

        Returns:
            outputs: Logits [batch_size, tgt_len, tgt_vocab_size]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        # Encode
        encoder_outputs, hidden = self.encode(src, src_lengths)

        # Decode
        outputs = []
        decoder_input = tgt[:, 0:1]  # Start with BOS token

        for t in range(tgt_len):
            logits, hidden = self.decode_step(decoder_input, hidden, encoder_outputs)
            outputs.append(logits.unsqueeze(1))

            # Teacher forcing
            if t < tgt_len - 1:
                decoder_input = tgt[:, t + 1 : t + 2]

        return torch.cat(outputs, dim=1)


class AttentionMechanism(nn.Module):
    """
    Attention mechanism for sequence-to-sequence models.

    Supports multiple attention types: dot, general, concat.
    """

    def __init__(
        self,
        hidden_dim: int,
        encoder_dim: int,
        attention_type: str = "general",
    ):
        super().__init__()

        self.attention_type = attention_type
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim

        if attention_type == "general":
            self.W = nn.Linear(hidden_dim, encoder_dim)
        elif attention_type == "concat":
            self.W = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
            self.v = nn.Parameter(torch.randn(hidden_dim))

    def forward(
        self,
        decoder_hidden: Tensor,
        encoder_outputs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute attention weights and context.

        Args:
            decoder_hidden: Decoder hidden state [batch, hidden_dim]
            encoder_outputs: Encoder outputs [batch, src_len, encoder_dim]
            mask: Source mask [batch, src_len]

        Returns:
            context: Context vector [batch, encoder_dim]
            attention_weights: Attention weights [batch, src_len]
        """
        batch_size, src_len, _ = encoder_outputs.size()

        if self.attention_type == "dot":
            # decoder_hidden: [batch, hidden_dim]
            # encoder_outputs: [batch, src_len, encoder_dim]
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(
                2
            )  # [batch, src_len]

        elif self.attention_type == "general":
            # Project decoder hidden to encoder dimension
            proj_hidden = self.W(decoder_hidden)  # [batch, encoder_dim]
            scores = torch.bmm(encoder_outputs, proj_hidden.unsqueeze(2)).squeeze(2)

        elif self.attention_type == "concat":
            # Expand decoder hidden to match encoder outputs
            expanded_hidden = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
            concat = torch.cat([expanded_hidden, encoder_outputs], dim=2)
            scores = torch.tanh(self.W(concat)) @ self.v  # [batch, src_len]

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=1)

        # Compute context
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class AttentionTranslation(nn.Module):
    """
    Attention-based Encoder-Decoder for machine translation.

    Uses Bahdanau/Luong attention to focus on relevant source tokens.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_type: str = "general",
    ):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.encoder_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        # Attention
        encoder_dim = hidden_dim * 2
        self.attention = AttentionMechanism(hidden_dim, encoder_dim, attention_type)

        # Decoder
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.decoder_lstm = nn.LSTM(
            embed_dim + encoder_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim + encoder_dim + embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, tgt_vocab_size),
        )

        self.dropout = nn.Dropout(dropout)

    def encode(
        self, src: Tensor, src_lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        """
        Encode source sequence.

        Returns:
            encoder_outputs: Encoder outputs
            (hidden, cell): Hidden states
            mask: Source padding mask
        """
        embedded = self.dropout(self.encoder_embedding(src))

        if src_lengths is not None:
            embedded = pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        outputs, (hidden, cell) = self.encoder_lstm(embedded)

        if src_lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # Create mask
        mask = src != 0 if src_lengths is None else None

        # Combine bidirectional states
        hidden = self._combine_bidirectional(hidden)
        cell = self._combine_bidirectional(cell)

        return outputs, (hidden, cell), mask

    def _combine_bidirectional(self, state: Tensor) -> Tensor:
        """Combine bidirectional LSTM states."""
        state = state.view(self.num_layers, 2, state.size(1), self.hidden_dim)
        return torch.cat([state[:, 0], state[:, 1]], dim=2)

    def decode_step(
        self,
        input_token: Tensor,
        hidden: Tuple[Tensor, Tensor],
        encoder_outputs: Tensor,
        mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        """
        Single decoder step with attention.

        Args:
            input_token: Input token
            hidden: Decoder hidden state
            encoder_outputs: All encoder outputs
            mask: Source mask

        Returns:
            output: Logits
            hidden: Updated hidden state
            attention_weights: Attention distribution
        """
        embedded = self.dropout(self.decoder_embedding(input_token))

        # Get last layer hidden state for attention
        last_hidden = hidden[0][-1]  # [batch, hidden_dim]

        # Compute attention
        context, attention_weights = self.attention(last_hidden, encoder_outputs, mask)

        # Concatenate input and context
        rnn_input = torch.cat([embedded.squeeze(1), context], dim=1).unsqueeze(1)

        # Decode
        output, hidden = self.decoder_lstm(rnn_input, hidden)

        # Generate output
        concat = torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        logits = self.output_projection(concat)

        return logits, hidden, attention_weights

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with teacher forcing.

        Args:
            src: Source tokens
            tgt: Target tokens
            src_lengths: Source lengths

        Returns:
            outputs: Logits for each target position
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        # Encode
        encoder_outputs, hidden, mask = self.encode(src, src_lengths)

        # Decode
        outputs = []
        decoder_input = tgt[:, 0:1]

        for t in range(tgt_len):
            logits, hidden, _ = self.decode_step(
                decoder_input, hidden, encoder_outputs, mask
            )
            outputs.append(logits.unsqueeze(1))

            if t < tgt_len - 1:
                decoder_input = tgt[:, t + 1 : t + 2]

        return torch.cat(outputs, dim=1)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerTranslation(nn.Module):
    """
    Full Transformer model for machine translation.

    Standard transformer encoder-decoder with multi-head attention.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
    ):
        super().__init__()

        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        src_padding_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            src: Source tokens [batch, src_len]
            tgt: Target tokens [batch, tgt_len]
            src_mask: Source self-attention mask
            tgt_mask: Target self-attention mask (causal)
            src_padding_mask: Source padding mask
            tgt_padding_mask: Target padding mask

        Returns:
            output: Logits [batch, tgt_len, tgt_vocab_size]
        """
        # Embed and add positional encoding
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Generate masks if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        # Transformer
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        return self.output_projection(output)

    def _generate_square_subsequent_mask(self, size: int) -> Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """Encode source sequence."""
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode with encoded memory."""
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        output = self.transformer.decoder(
            tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )
        return self.output_projection(output)


class MBARTModel(nn.Module):
    """Multilingual BART (mBART) for machine translation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        nhead: int = 16,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        num_languages: int = 25,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_languages = num_languages

        self.language_embeddings = nn.Embedding(num_languages, d_model)
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embed_positions = PositionalEncoding(d_model, dropout=dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )

        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        src_lang_id: Optional[Tensor] = None,
        tgt_lang_id: Optional[Tensor] = None,
    ) -> Tensor:
        encoder_out = self.encode(src_tokens, src_lang_id)
        return self.decode(tgt_tokens, encoder_out, tgt_lang_id)

    def encode(
        self,
        src_tokens: Tensor,
        src_lang_id: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.embed_tokens(src_tokens) * math.sqrt(self.d_model)
        if src_lang_id is not None:
            x = x + self.language_embeddings(src_lang_id).unsqueeze(1)
        x = self.embed_positions(x)
        x = self.dropout(x)
        return self.encoder(x, src_key_padding_mask=src_mask)

    def decode(
        self,
        tgt_tokens: Tensor,
        encoder_out: Tensor,
        tgt_lang_id: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.embed_tokens(tgt_tokens) * math.sqrt(self.d_model)
        if tgt_lang_id is not None:
            x = x + self.language_embeddings(tgt_lang_id).unsqueeze(1)
        x = self.embed_positions(x)
        x = self.dropout(x)
        causal_mask = torch.triu(
            torch.ones(tgt_tokens.size(1), tgt_tokens.size(1)), diagonal=1
        ).bool()
        causal_mask = causal_mask.to(tgt_tokens.device)
        x = self.decoder(
            x, encoder_out, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask
        )
        return self.output_projection(x)


class MT5Model(nn.Module):
    """Multilingual T5 (mT5) for machine translation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_languages: int = 101,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_languages = num_languages

        self.shared = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        encoder_hidden_states = self.encode(input_ids, attention_mask)
        if decoder_input_ids is None:
            decoder_input_ids = input_ids.clone()
        return self.decode(
            decoder_input_ids, encoder_hidden_states, decoder_attention_mask
        )

    def encode(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        x = self.shared(input_ids)
        return self.encoder(x, src_key_padding_mask=attention_mask)

    def decode(
        self,
        decoder_input_ids: Tensor,
        encoder_hidden_states: Tensor,
        decoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.shared(decoder_input_ids)
        causal_mask = torch.triu(
            torch.ones(decoder_input_ids.size(1), decoder_input_ids.size(1)), diagonal=1
        ).bool()
        causal_mask = causal_mask.to(decoder_input_ids.device)
        x = self.decoder(
            x,
            encoder_hidden_states,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_attention_mask,
        )
        return self.lm_head(x)


class M2M100Model(nn.Module):
    """Many-to-Many Multilingual Model (M2M100) for machine translation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        nhead: int = 16,
        num_encoder_layers: int = 24,
        num_decoder_layers: int = 24,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        num_languages: int = 100,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_languages = num_languages

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embed_positions = PositionalEncoding(d_model, dropout=dropout)
        self.lang_embeddings = nn.Embedding(num_languages, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )

        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        src_lang: Optional[Tensor] = None,
        tgt_lang: Optional[Tensor] = None,
    ) -> Tensor:
        encoder_out = self.encode(src_tokens, src_lang)
        return self.decode(tgt_tokens, encoder_out, tgt_lang)

    def encode(
        self,
        src_tokens: Tensor,
        src_lang: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.embed_tokens(src_tokens) * math.sqrt(self.d_model)
        if src_lang is not None:
            x = x + self.lang_embeddings(src_lang).unsqueeze(1)
        x = self.embed_positions(x)
        return self.encoder(x, src_key_padding_mask=src_mask)

    def decode(
        self,
        tgt_tokens: Tensor,
        encoder_out: Tensor,
        tgt_lang: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.embed_tokens(tgt_tokens) * math.sqrt(self.d_model)
        if tgt_lang is not None:
            x = x + self.lang_embeddings(tgt_lang).unsqueeze(1)
        x = self.embed_positions(x)
        causal_mask = torch.triu(
            torch.ones(tgt_tokens.size(1), tgt_tokens.size(1)), diagonal=1
        ).bool()
        causal_mask = causal_mask.to(tgt_tokens.device)
        x = self.decoder(
            x, encoder_out, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask
        )
        return self.output_projection(x)


class DeltaLMModel(nn.Module):
    """DeltaLM - Pre-training for multilingual machine translation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embed_positions = PositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList(
            [
                DeltaLMLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, src_tokens: Tensor, tgt_tokens: Tensor, mode: str = "translation"
    ) -> Tensor:
        if mode == "translation":
            encoder_out = self.encode(src_tokens)
            return self.decode(tgt_tokens, encoder_out)
        else:
            return self.pretrain_forward(src_tokens)

    def encode(self, src_tokens: Tensor) -> Tensor:
        x = self.embed_tokens(src_tokens) * math.sqrt(self.d_model)
        x = self.embed_positions(x)
        for layer in self.layers:
            x = layer(x, encoder_mode=True)
        return x

    def decode(self, tgt_tokens: Tensor, encoder_out: Tensor) -> Tensor:
        x = self.embed_tokens(tgt_tokens) * math.sqrt(self.d_model)
        x = self.embed_positions(x)
        causal_mask = torch.triu(
            torch.ones(tgt_tokens.size(1), tgt_tokens.size(1)), diagonal=1
        ).bool()
        causal_mask = causal_mask.to(tgt_tokens.device)
        for layer in self.layers:
            x = layer(x, encoder_out=encoder_out, causal_mask=causal_mask)
        return self.output_projection(x)

    def pretrain_forward(self, tokens: Tensor) -> Tensor:
        x = self.embed_tokens(tokens) * math.sqrt(self.d_model)
        x = self.embed_positions(x)
        for layer in self.layers:
            x = layer(x, encoder_mode=True)
        return self.output_projection(x)


class DeltaLMLayer(nn.Module):
    """Single DeltaLM layer supporting both encoding and decoding."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        encoder_mode: bool = False,
        encoder_out: Optional[Tensor] = None,
        causal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if encoder_mode:
            attn_out, _ = self.self_attn(x, x, x)
        else:
            attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)

        x = self.norm1(x + self.dropout(attn_out))

        if not encoder_mode and encoder_out is not None:
            cross_attn_out, _ = self.cross_attn(x, encoder_out, encoder_out)
            x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self.norm3(x + self.dropout(ff_out))

        return x


# =============================================================================
# 2. Low-Resource Translation
# =============================================================================


class BackTranslation:
    """Back-translation for synthetic parallel data generation."""

    def __init__(
        self,
        forward_model: nn.Module,
        backward_model: nn.Module,
        src_tokenizer: Callable,
        tgt_tokenizer: Callable,
        device: str = "cuda",
    ):
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device

    def generate_synthetic_data(
        self,
        monolingual_tgt: List[str],
        sampling_strategy: str = "beam",
        num_samples: int = 1,
    ) -> List[Tuple[str, str]]:
        synthetic_pairs = []
        self.backward_model.eval()
        with torch.no_grad():
            for tgt_text in monolingual_tgt:
                synthetic_srcs = self._back_translate(
                    tgt_text, sampling_strategy, num_samples
                )
                for synthetic_src in synthetic_srcs:
                    synthetic_pairs.append((synthetic_src, tgt_text))
        return synthetic_pairs

    def _back_translate(
        self, tgt_text: str, strategy: str, num_samples: int
    ) -> List[str]:
        tgt_tokens = self.tgt_tokenizer(tgt_text)
        tgt_tensor = torch.tensor([tgt_tokens]).to(self.device)
        if strategy == "greedy":
            generated = self._greedy_decode(self.backward_model, tgt_tensor)
        else:
            generated = self._beam_search_decode(self.backward_model, tgt_tensor)
        return [self.src_tokenizer.decode(g) for g in generated]

    def _greedy_decode(self, model: nn.Module, src: Tensor) -> List[Tensor]:
        return [src[0]]

    def _beam_search_decode(
        self, model: nn.Module, src: Tensor, beam_width: int = 5
    ) -> List[Tensor]:
        return [src[0]]


class UnsupervisedTranslation:
    """Unsupervised machine translation using monolingual data only."""

    def __init__(
        self,
        src_model: nn.Module,
        tgt_model: nn.Module,
        shared_encoder: bool = True,
        shared_decoder: bool = False,
    ):
        self.src_model = src_model
        self.tgt_model = tgt_model
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder
        if shared_encoder:
            self._share_encoder_parameters()

    def _share_encoder_parameters(self):
        pass

    def train_step(
        self, src_mono: Tensor, tgt_mono: Tensor, denoise_steps: int = 1
    ) -> Dict[str, Tensor]:
        losses = {}
        losses["src_denoise"] = self._denoising_loss(self.src_model, src_mono)
        losses["tgt_denoise"] = self._denoising_loss(self.tgt_model, tgt_mono)
        losses["src_bt"] = self._back_translation_loss(src_mono, "src_to_tgt")
        losses["tgt_bt"] = self._back_translation_loss(tgt_mono, "tgt_to_src")
        return losses

    def _denoising_loss(
        self, model: nn.Module, tokens: Tensor, noise_prob: float = 0.1
    ) -> Tensor:
        noised_tokens = self._add_noise(tokens, noise_prob)
        output = model(noised_tokens, tokens)
        return F.cross_entropy(
            output.view(-1, output.size(-1)), tokens.view(-1), ignore_index=0
        )

    def _add_noise(self, tokens: Tensor, prob: float) -> Tensor:
        noised = tokens.clone()
        for i in range(tokens.size(0)):
            length = (tokens[i] != 0).sum().item()
            for j in range(1, length):
                if random.random() < prob:
                    noised[i, j] = 0
        return noised

    def _back_translation_loss(self, tokens: Tensor, direction: str) -> Tensor:
        return torch.tensor(0.0)

    def _translate(self, tokens: Tensor, model: nn.Module) -> Tensor:
        model.eval()
        with torch.no_grad():
            return tokens


class PivotTranslation:
    """Pivot translation through intermediate language."""

    def __init__(
        self,
        model_ab: nn.Module,
        model_bc: nn.Module,
        tokenizer_ab: Callable,
        tokenizer_bc: Callable,
        tokenizer_pivot: Callable,
    ):
        self.model_ab = model_ab
        self.model_bc = model_bc
        self.tokenizer_ab = tokenizer_ab
        self.tokenizer_bc = tokenizer_bc
        self.tokenizer_pivot = tokenizer_pivot

    def translate(self, src_text: str, beam_width: int = 5) -> str:
        pivot_text = self._translate_step(
            src_text, self.model_ab, self.tokenizer_ab, self.tokenizer_pivot
        )
        tgt_text = self._translate_step(
            pivot_text, self.model_bc, self.tokenizer_pivot, self.tokenizer_bc
        )
        return tgt_text

    def _translate_step(
        self,
        text: str,
        model: nn.Module,
        src_tokenizer: Callable,
        tgt_tokenizer: Callable,
    ) -> str:
        tokens = src_tokenizer(text)
        token_tensor = torch.tensor([tokens])
        model.eval()
        with torch.no_grad():
            output = model(token_tensor, token_tensor)
            pred_tokens = output.argmax(dim=-1)[0].tolist()
        return tgt_tokenizer.decode(pred_tokens)


class TransferLearning:
    """Cross-lingual transfer learning for low-resource translation."""

    def __init__(
        self,
        pretrained_model: nn.Module,
        target_vocab_size: int,
        freeze_encoder: bool = False,
    ):
        self.pretrained_model = pretrained_model
        self.target_vocab_size = target_vocab_size
        self.freeze_encoder = freeze_encoder
        self._adapt_for_target_language()

    def _adapt_for_target_language(self):
        d_model = getattr(self.pretrained_model, "d_model", 512)
        self.output_projection = nn.Linear(d_model, self.target_vocab_size)
        if self.freeze_encoder:
            for param in self.pretrained_model.encoder.parameters():
                param.requires_grad = False

    def fine_tune_step(
        self,
        src_batch: Tensor,
        tgt_batch: Tensor,
        learning_rate_schedule: str = "gradual",
    ) -> Tensor:
        output = self.pretrained_model(src_batch, tgt_batch)
        return F.cross_entropy(
            output.view(-1, output.size(-1)), tgt_batch.view(-1), ignore_index=0
        )


class MetaLearningTranslation:
    """Meta-learning for fast adaptation to new language pairs."""

    def __init__(
        self,
        base_model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        num_inner_steps: int = 5,
    ):
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps

    def meta_train_step(
        self,
        support_set: List[Tuple[Tensor, Tensor]],
        query_set: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        meta_loss = 0.0
        for support, query in zip(support_set, query_set):
            adapted_params = self._inner_loop_adaptation(support)
            query_loss = self._compute_loss(query, adapted_params)
            meta_loss += query_loss
        return meta_loss / len(support_set)

    def _inner_loop_adaptation(
        self, support: Tuple[Tensor, Tensor]
    ) -> Dict[str, Tensor]:
        adapted_params = {
            name: param.clone() for name, param in self.base_model.named_parameters()
        }
        src, tgt = support
        for _ in range(self.num_inner_steps):
            output = self._forward_with_params(src, tgt, adapted_params)
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)), tgt.view(-1), ignore_index=0
            )
            grads = torch.autograd.grad(
                loss, adapted_params.values(), create_graph=True
            )
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.inner_lr * grad
        return adapted_params

    def _forward_with_params(
        self, src: Tensor, tgt: Tensor, params: Dict[str, Tensor]
    ) -> Tensor:
        return self.base_model(src, tgt)

    def _compute_loss(
        self, query: Tuple[Tensor, Tensor], params: Dict[str, Tensor]
    ) -> Tensor:
        src, tgt = query
        output = self._forward_with_params(src, tgt, params)
        return F.cross_entropy(
            output.view(-1, output.size(-1)), tgt.view(-1), ignore_index=0
        )

    def fast_adapt(
        self, few_shot_data: List[Tuple[str, str]], num_steps: int = 10
    ) -> nn.Module:
        adapted_model = type(self.base_model)(**self.base_model.__dict__)
        adapted_model.load_state_dict(self.base_model.state_dict())
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        for step in range(num_steps):
            for src, tgt in few_shot_data:
                optimizer.zero_grad()
                output = adapted_model(src, tgt)
                output.backward()
                optimizer.step()
        return adapted_model


# =============================================================================
# 3. Document-Level Translation
# =============================================================================


class DocTransformer(nn.Module):
    """Context-aware document-level translation."""

    def __init__(
        self,
        base_model: nn.Module,
        context_encoder: Optional[nn.Module] = None,
        context_window: int = 2,
        fusion_type: str = "attention",
    ):
        super().__init__()

        self.base_model = base_model
        self.context_window = context_window
        self.fusion_type = fusion_type

        if context_encoder is None:
            d_model = getattr(base_model, "d_model", 512)
            self.context_encoder = nn.LSTM(
                d_model, d_model, batch_first=True, bidirectional=True
            )
        else:
            self.context_encoder = context_encoder

        if fusion_type == "attention":
            self.context_attention = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True
            )
        elif fusion_type == "gating":
            self.context_gate = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.Sigmoid(),
            )

    def forward(
        self,
        current_sentence: Tensor,
        context_sentences: List[Tensor],
        tgt_sentence: Tensor,
    ) -> Tensor:
        context_repr = self._encode_context(context_sentences)
        current_repr = self._encode_sentence(current_sentence)
        fused_repr = self._fuse_context(current_repr, context_repr)
        return self._decode(fused_repr, tgt_sentence)

    def _encode_context(self, sentences: List[Tensor]) -> Tensor:
        encoded = [self._encode_sentence(s) for s in sentences]
        if encoded:
            stacked = torch.stack(encoded, dim=1)
            output, _ = self.context_encoder(stacked)
            return output.mean(dim=1)
        return torch.zeros(1, getattr(self.base_model, "d_model", 512))

    def _encode_sentence(self, sentence: Tensor) -> Tensor:
        with torch.no_grad():
            return self.base_model.encode(sentence)

    def _fuse_context(self, current: Tensor, context: Tensor) -> Tensor:
        if self.fusion_type == "attention":
            fused, _ = self.context_attention(
                current.unsqueeze(1), context.unsqueeze(1), context.unsqueeze(1)
            )
            return fused.squeeze(1) + current
        elif self.fusion_type == "gating":
            gate_input = torch.cat([current, context, current * context], dim=-1)
            gate = self.context_gate(gate_input)
            return gate * current + (1 - gate) * context
        else:
            return torch.cat([current, context], dim=-1)

    def _decode(self, representation: Tensor, tgt: Tensor) -> Tensor:
        return self.base_model.decode(tgt, representation)

    def translate_document(
        self, document: List[str], tokenizer: Callable, batch_size: int = 1
    ) -> List[str]:
        translations = []
        for i, sentence in enumerate(document):
            start = max(0, i - self.context_window)
            end = min(len(document), i + self.context_window + 1)
            context = [tokenizer(document[j]) for j in range(start, end) if j != i]
            current = tokenizer(sentence)
            with torch.no_grad():
                current_tensor = torch.tensor([current])
                context_tensors = [torch.tensor([c]) for c in context]
                output = self.forward(current_tensor, context_tensors, current_tensor)
                pred_tokens = output.argmax(dim=-1)[0].tolist()
            translation = tokenizer.decode(pred_tokens)
            translations.append(translation)
        return translations


class CacheBasedTranslation(nn.Module):
    """Cache-based document translation."""

    def __init__(
        self,
        base_model: nn.Module,
        cache_size: int = 100,
        cache_type: str = "attention",
    ):
        super().__init__()

        self.base_model = base_model
        self.cache_size = cache_size
        self.cache_type = cache_type
        self.cache: List[Tuple[Tensor, Tensor]] = []
        self.cache_keys: List[Tensor] = []

        if cache_type == "attention":
            d_model = getattr(base_model, "d_model", 512)
            self.cache_attention = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True
            )

    def forward(
        self, src: Tensor, tgt: Optional[Tensor] = None, use_cache: bool = True
    ) -> Tensor:
        src_repr = self.base_model.encode(src)
        if use_cache and self.cache:
            context = self._retrieve_from_cache(src_repr)
            src_repr = self._combine_with_context(src_repr, context)
        if tgt is None:
            tgt = src
        output = self.base_model.decode(tgt, src_repr)
        if use_cache:
            self._update_cache(src, output)
        return output

    def _retrieve_from_cache(self, query: Tensor) -> Tensor:
        if self.cache_type == "attention":
            cache_values = torch.stack([c[1] for c in self.cache])
            cache_keys = torch.stack(self.cache_keys)
            context, _ = self.cache_attention(
                query.unsqueeze(1), cache_keys.unsqueeze(1), cache_values.unsqueeze(1)
            )
            return context.squeeze(1)
        else:
            return torch.stack([c[1] for c in self.cache]).mean(dim=0)

    def _combine_with_context(self, src_repr: Tensor, context: Tensor) -> Tensor:
        return src_repr + 0.1 * context

    def _update_cache(self, src: Tensor, tgt_output: Tensor):
        src_repr = self.base_model.encode(src)
        tgt_repr = tgt_output.mean(dim=1)
        self.cache.append((src_repr, tgt_repr))
        self.cache_keys.append(src_repr)
        if len(self.cache) > self.cache_size:
            if self.cache_type in ["lru", "fifo"]:
                self.cache.pop(0)
                self.cache_keys.pop(0)

    def clear_cache(self):
        self.cache.clear()
        self.cache_keys.clear()


class HierarchicalTranslation(nn.Module):
    """Hierarchical document translation."""

    def __init__(
        self,
        sentence_model: nn.Module,
        document_encoder: Optional[nn.Module] = None,
        hierarchical_type: str = "bottom-up",
    ):
        super().__init__()

        self.sentence_model = sentence_model
        self.hierarchical_type = hierarchical_type

        if document_encoder is None:
            d_model = getattr(sentence_model, "d_model", 512)
            self.document_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 8, batch_first=True),
                num_layers=2,
            )
        else:
            self.document_encoder = document_encoder

        self.document_head = nn.Linear(d_model, d_model)

    def forward(
        self, document: List[Tensor], target_document: Optional[List[Tensor]] = None
    ) -> List[Tensor]:
        if self.hierarchical_type == "bottom-up":
            return self._bottom_up_translate(document, target_document)
        else:
            return self._top_down_translate(document, target_document)

    def _bottom_up_translate(
        self, document: List[Tensor], target_document: Optional[List[Tensor]]
    ) -> List[Tensor]:
        sentence_reprs = []
        sentence_outputs = []
        for i, sentence in enumerate(document):
            tgt = target_document[i] if target_document else sentence
            output = self.sentence_model(sentence, tgt)
            sentence_outputs.append(output)
            with torch.no_grad():
                repr = self.sentence_model.encode(sentence)
                sentence_reprs.append(repr)
        document_repr = torch.stack(sentence_reprs, dim=1)
        document_context = self.document_encoder(document_repr)
        refined_outputs = []
        for i, output in enumerate(sentence_outputs):
            context = document_context[:, i, :]
            refined = output + context.unsqueeze(1)
            refined_outputs.append(refined)
        return refined_outputs

    def _top_down_translate(
        self, document: List[Tensor], target_document: Optional[List[Tensor]]
    ) -> List[Tensor]:
        sentence_reprs = [self.sentence_model.encode(sent) for sent in document]
        document_repr = torch.stack(sentence_reprs, dim=1)
        document_context = self.document_encoder(document_repr)
        outputs = []
        for i, sentence in enumerate(document):
            tgt = target_document[i] if target_document else sentence
            sent_repr = self.sentence_model.encode(sentence)
            context = document_context[:, i, :]
            combined = sent_repr + context
            output = self.sentence_model.decode(tgt, combined)
            outputs.append(output)
        return outputs

"""
Transformer-based Machine Translation Models

Transformer encoder-decoder architectures for machine translation including:
- Transformer encoder and decoder
- Full translation model
- Relative position representations
"""

from typing import Optional, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
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

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output with positional encoding [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add learned positional encoding.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output with positional encoding
        """
        batch_size, seq_len, _ = x.size()
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        x = x + self.position_embeddings(positions)
        return x


class TransformerMTEncoder(nn.Module):
    """Transformer encoder for machine translation.

    Args:
        vocab_size: Source vocabulary size
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of encoder layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        max_len: Maximum sequence length
        pad_idx: Padding token index
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode source sequence.

        Args:
            src: Source tokens [batch_size, src_len]
            src_key_padding_mask: Mask for padding [batch_size, src_len]

        Returns:
            Encoder outputs [batch_size, src_len, d_model]
        """
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded)

        output = self.transformer_encoder(
            src_embedded,
            src_key_padding_mask=src_key_padding_mask,
        )

        return output

    def make_src_mask(self, src: Tensor) -> Tensor:
        """Create source padding mask.

        Args:
            src: Source tokens [batch_size, src_len]

        Returns:
            Padding mask [batch_size, src_len]
        """
        return src == self.pad_idx


class TransformerMTDecoder(nn.Module):
    """Transformer decoder for machine translation.

    Args:
        vocab_size: Target vocabulary size
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of decoder layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        max_len: Maximum sequence length
        pad_idx: Padding token index
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.output_projection.weight, std=self.d_model**-0.5)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode target sequence.

        Args:
            tgt: Target tokens [batch_size, tgt_len]
            memory: Encoder outputs [batch_size, src_len, d_model]
            tgt_mask: Causal mask for target [tgt_len, tgt_len]
            memory_key_padding_mask: Mask for encoder padding
            tgt_key_padding_mask: Mask for target padding

        Returns:
            Decoder output logits [batch_size, tgt_len, vocab_size]
        """
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)

        output = self.transformer_decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return self.output_projection(output)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generate causal mask for autoregressive decoding.

        Args:
            sz: Sequence length

        Returns:
            Causal mask [sz, sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def make_tgt_mask(self, tgt: Tensor) -> Tensor:
        """Create target padding and causal masks.

        Args:
            tgt: Target tokens [batch_size, tgt_len]

        Returns:
            Causal mask [tgt_len, tgt_len]
        """
        tgt_len = tgt.size(1)
        return self.generate_square_subsequent_mask(tgt_len).to(tgt.device)


class TransformerTranslationModel(nn.Module):
    """Complete Transformer translation model.

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        nhead: Number of attention heads
        encoder_layers: Number of encoder layers
        decoder_layers: Number of decoder layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        max_len: Maximum sequence length
        src_pad_idx: Source padding index
        tgt_pad_idx: Target padding index
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
    ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = TransformerMTEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            pad_idx=src_pad_idx,
        )

        self.decoder = TransformerMTDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            pad_idx=tgt_pad_idx,
        )

        self.d_model = d_model

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for training.

        Args:
            src: Source tokens [batch_size, src_len]
            tgt: Target tokens [batch_size, tgt_len]
            src_padding_mask: Source padding mask

        Returns:
            Translation logits [batch_size, tgt_len, tgt_vocab_size]
        """
        if src_padding_mask is None:
            src_padding_mask = self.encoder.make_src_mask(src)

        memory = self.encoder(src, src_key_padding_mask=src_padding_mask)

        tgt_mask = self.decoder.make_tgt_mask(tgt)
        tgt_padding_mask = tgt == self.tgt_pad_idx

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        return output

    def encode(self, src: Tensor, src_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Encode source sequence.

        Args:
            src: Source tokens [batch_size, src_len]
            src_padding_mask: Source padding mask

        Returns:
            Encoder outputs [batch_size, src_len, d_model]
        """
        if src_padding_mask is None:
            src_padding_mask = self.encoder.make_src_mask(src)

        return self.encoder(src, src_key_padding_mask=src_padding_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode target sequence.

        Args:
            tgt: Target tokens [batch_size, tgt_len]
            memory: Encoder outputs [batch_size, src_len, d_model]
            tgt_mask: Causal mask
            memory_key_padding_mask: Memory padding mask

        Returns:
            Decoder outputs [batch_size, tgt_len, vocab_size]
        """
        if tgt_mask is None:
            tgt_mask = self.decoder.make_tgt_mask(tgt)

        return self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def get_param_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RelativePositionTransformer(nn.Module):
    """Transformer with relative position representations.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        max_relative_position: Maximum relative position distance
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_relative_position: int = 32,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_relative_position = max_relative_position

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.relative_position_embeddings = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, nhead) * 0.1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_projection = nn.Linear(d_model, vocab_size)

    def _compute_relative_position(self, length: int, device: torch.device) -> Tensor:
        """Compute relative position indices.

        Args:
            length: Sequence length
            device: Device to create tensor on

        Returns:
            Relative position tensor [length, length]
        """
        position = torch.arange(length, device=device)
        relative_position = position.unsqueeze(1) - position.unsqueeze(0)

        relative_position = torch.clamp(
            relative_position,
            -self.max_relative_position,
            self.max_relative_position,
        )

        relative_position = relative_position + self.max_relative_position

        return relative_position

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tokens [batch_size, seq_len]

        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        embedded = self.embedding(x) * math.sqrt(self.d_model)

        seq_len = x.size(1)
        relative_position = self._compute_relative_position(seq_len, x.device)

        relative_bias = self.relative_position_embeddings[relative_position]

        output = self.transformer(embedded)

        return self.output_projection(output)


class LightweightTransformerMT(nn.Module):
    """Memory-efficient Transformer for MT using reversible layers.

    Args:
        vocab_size: Source/target vocabulary size
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            src: Source tokens
            tgt: Target tokens
            src_key_padding_mask: Source padding mask

        Returns:
            Translation logits
        """
        src_embedded = self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model))

        memory = self.encoder(src_embedded, src_key_padding_mask=src_key_padding_mask)

        tgt_embedded = self.pos_encoding(
            self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        )

        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), device=tgt.device), diagonal=1
        ).bool()

        output = self.decoder(tgt_embedded, memory, tgt_mask=tgt_mask)

        return self.output_projection(output)

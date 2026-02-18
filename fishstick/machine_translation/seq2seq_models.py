"""
Seq2Seq Translation Models

Advanced sequence-to-sequence models for machine translation including:
- Basic RNN encoder-decoder
- Attention-based encoder-decoder
- Conditional RNN sequences
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class Seq2SeqEncoder(nn.Module):
    """RNN-based encoder for sequence-to-sequence models.

    Args:
        vocab_size: Size of source vocabulary
        embed_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden state
        num_layers: Number of RNN layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional RNN
        rnn_type: Type of RNN cell ('lstm' or 'gru')
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        rnn_type: str = "lstm",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        rnn_cell = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cell(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(
        self, src: Tensor, src_lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Encode source sequence.

        Args:
            src: Source token IDs [batch_size, src_len]
            src_lengths: Actual lengths for packing

        Returns:
            outputs: All hidden states [batch_size, src_len, output_dim]
            hidden: Final hidden state tuple (h, c) for LSTM or h for GRU
        """
        embedded = self.dropout(self.embedding(src))

        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)

        return outputs, hidden

    def get_output_dim(self) -> int:
        return self.output_dim


class Seq2SeqDecoder(nn.Module):
    """RNN-based decoder for sequence-to-sequence models.

    Args:
        vocab_size: Size of target vocabulary
        embed_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden state
        encoder_output_dim: Dimension of encoder outputs
        num_layers: Number of RNN layers
        dropout: Dropout probability
        rnn_type: Type of RNN cell ('lstm' or 'gru')
        attention: Optional attention module
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        encoder_output_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "lstm",
        attention: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention = attention
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        rnn_input_dim = embed_dim + encoder_output_dim if attention else embed_dim
        rnn_cell = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnn = rnn_cell(
            rnn_input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.output_projection = nn.Linear(
            hidden_dim + encoder_output_dim + embed_dim, vocab_size
        )

    def forward(
        self,
        tgt: Tensor,
        hidden: Tuple[Tensor, Tensor],
        encoder_outputs: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Decode one step.

        Args:
            tgt: Target tokens [batch_size, tgt_len]
            hidden: Previous hidden state
            encoder_outputs: Encoder outputs for attention [batch_size, src_len, enc_dim]
            mask: Mask for encoder outputs

        Returns:
            logits: Output logits [batch_size, tgt_len, vocab_size]
            hidden: Updated hidden state
        """
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)

        embedded = self.dropout(self.embedding(tgt))

        outputs = []
        for t in range(tgt_len):
            embedded_t = embedded[:, t : t + 1, :]

            if self.attention and encoder_outputs is not None:
                context, attn_weights = self.attention(
                    hidden[0][-1], encoder_outputs, mask
                )
                rnn_input = torch.cat([embedded_t, context.unsqueeze(1)], dim=-1)
            else:
                rnn_input = embedded_t

            output, hidden = self.rnn(rnn_input, hidden)

            if self.attention and encoder_outputs is not None:
                output = torch.cat([output, context.unsqueeze(1), embedded_t], dim=-1)

            logits = self.output_projection(output)
            outputs.append(logits)

        logits = torch.cat(outputs, dim=1)
        return logits, hidden

    def forward_step(
        self,
        tgt: Tensor,
        hidden: Tuple[Tensor, Tensor],
        encoder_outputs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Optional[Tensor]]:
        """Single decoding step for autoregressive generation.

        Args:
            tgt: Target token [batch_size, 1]
            hidden: Previous hidden state
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_dim]
            mask: Attention mask

        Returns:
            logits: Output logits [batch_size, 1, vocab_size]
            hidden: Updated hidden state
            attn_weights: Attention weights if available
        """
        embedded = self.dropout(self.embedding(tgt))

        attn_weights = None
        if self.attention is not None:
            context, attn_weights = self.attention(hidden[0][-1], encoder_outputs, mask)
            rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        else:
            rnn_input = embedded

        output, hidden = self.rnn(rnn_input, hidden)

        if self.attention is not None:
            output = torch.cat([output, context.unsqueeze(1), embedded], dim=-1)

        logits = self.output_projection(output)
        return logits, hidden, attn_weights


class AttentionSeq2Seq(nn.Module):
    """Complete Seq2Seq model with attention.

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension
        encoder_layers: Number of encoder layers
        decoder_layers: Number of decoder layers
        dropout: Dropout probability
        bidirectional_encoder: Whether encoder is bidirectional
        attention_type: Type of attention ('bahdanau' or 'luong')
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.3,
        bidirectional_encoder: bool = True,
        attention_type: str = "bahdanau",
    ):
        super().__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        encoder_output_dim = hidden_dim * 2 if bidirectional_encoder else hidden_dim

        self.encoder = Seq2SeqEncoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder,
        )

        if attention_type.lower() == "bahdanau":
            attention = BahdanauAttention(hidden_dim, encoder_output_dim)
        else:
            attention = LuongAttention(hidden_dim, encoder_output_dim)

        self.decoder = Seq2SeqDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_output_dim=encoder_output_dim,
            num_layers=decoder_layers,
            dropout=dropout,
            attention=attention,
        )

        self.generator = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for training.

        Args:
            src: Source tokens [batch_size, src_len]
            tgt: Target tokens [batch_size, tgt_len]
            src_lengths: Source sequence lengths

        Returns:
            logits: Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        mask = None
        if encoder_outputs.size(1) != tgt.size(1):
            mask = torch.zeros(
                encoder_outputs.size(0),
                encoder_outputs.size(1),
                dtype=torch.bool,
                device=src.device,
            )

        logits, _ = self.decoder(tgt, hidden, encoder_outputs, mask)

        return self.generator(logits)

    def encode(
        self, src: Tensor, src_lengths: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple]:
        """Encode source sequence."""
        return self.encoder(src, src_lengths)

    def decode_step(
        self,
        tgt: Tensor,
        hidden: Tuple[Tensor, Tensor],
        encoder_outputs: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Optional[Tensor]]:
        """Single decoder step."""
        return self.decoder.forward_step(tgt, hidden, encoder_outputs, mask)

    def generate(
        self,
        src: Tensor,
        max_length: int = 100,
        start_token: int = 2,
        end_token: int = 3,
        src_lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate translation autoregressively.

        Args:
            src: Source tokens [batch_size, src_len]
            max_length: Maximum generation length
            start_token: Start token ID
            end_token: End token ID
            src_lengths: Source lengths

        Returns:
            translations: Generated token IDs [batch_size, generated_len]
        """
        self.eval()

        encoder_outputs, hidden = self.encode(src, src_lengths)

        batch_size = src.size(0)
        generated = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=src.device
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_length):
            logits, hidden, _ = self.decode_step(
                generated[:, -1:], hidden, encoder_outputs
            )

            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(-1) == end_token)
            if finished.all():
                break

        return generated


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) attention mechanism.

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

        self.W_a = nn.Linear(encoder_output_dim, attention_dim)
        self.U_a = nn.Linear(hidden_dim, attention_dim)
        self.V_a = nn.Linear(attention_dim, 1)

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


class ConditionalRNNSequence(nn.Module):
    """Conditional RNN that incorporates context into RNN computation.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        context_dim: Context vector dimension
        num_layers: Number of RNN layers
        dropout: Dropout probability
        rnn_type: Type of RNN ('lstm', 'gru')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "lstm",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        self.context_projection = nn.Linear(context_dim, hidden_dim)

        rnn_cell = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        self.rnns = nn.ModuleList(
            [
                rnn_cell(
                    input_dim if i == 0 else hidden_dim, hidden_dim, batch_first=True
                )
                for i in range(num_layers)
            ]
        )

        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
        context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass with optional context conditioning.

        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            hidden: Initial hidden state
            context: Context vector [batch_size, context_dim]

        Returns:
            outputs: Output sequence [batch_size, seq_len, hidden_dim]
            hidden: Final hidden state
        """
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        if context is not None:
            context_bias = self.context_projection(context)

        outputs = []
        current_input = x

        for i, (rnn, dropout) in enumerate(zip(self.rnns, self.dropouts)):
            rnn_output, hidden[i] = rnn(current_input, hidden[i])
            rnn_output = dropout(rnn_output)
            outputs.append(rnn_output)

            if context is not None:
                rnn_output = rnn_output + context_bias.unsqueeze(1)

            current_input = rnn_output

        return outputs[-1], hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Initialize hidden states."""
        rnn_cell = nn.LSTM if self.rnn_type == "lstm" else nn.GRU

        hidden = []
        for _ in range(self.num_layers):
            h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
            if self.rnn_type == "lstm":
                c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
                hidden.append((h, c))
            else:
                hidden.append(h)

        return tuple(hidden) if self.rnn_type == "lstm" else tuple(hidden)

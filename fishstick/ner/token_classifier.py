"""
Token Classification module for Named Entity Recognition.

Provides BiLSTM and Transformer-based token classifiers for NER tasks.
Supports BIO, BIOES, and other tagging schemes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import math


class BiLSTMTokenClassifier(nn.Module):
    """BiLSTM-based token classifier for NER.

    A bidirectional LSTM model that performs token-level classification
    for named entity recognition. Supports CRF decoding for sequence-level
    constraints.

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden states
        num_tags: Number of entity tags
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        use_crf: Whether to use CRF decoding
        use_char: Whether to use character-level embeddings
        char_embedding_dim: Dimension of character embeddings
        char_hidden_dim: Dimension of character LSTM hidden state
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_tags: int = 9,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_crf: bool = True,
        use_char: bool = False,
        char_embedding_dim: int = 50,
        char_hidden_dim: int = 50,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.num_layers = num_layers
        self.use_crf = use_crf
        self.use_char = use_char

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.use_char = use_char
        if use_char:
            self.char_embedding = nn.Embedding(
                vocab_size, char_embedding_dim, padding_idx=0
            )
            self.char_lstm = nn.LSTM(
                char_embedding_dim,
                char_hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            char_output_dim = char_hidden_dim * 2
            self.char_projection = nn.Linear(char_output_dim, embedding_dim)
            total_embedding_dim = embedding_dim * 2
        else:
            total_embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            total_embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags),
        )

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        char_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training or inference.

        Args:
            input_ids: Word token indices, shape (batch_size, seq_len)
            tags: Entity tags for training, shape (batch_size, seq_len)
            mask: Attention mask for padding, shape (batch_size, seq_len)
            char_ids: Character indices for char-level embeddings

        Returns:
            If training with CRF: negative log likelihood loss
            Otherwise: emission scores, shape (batch_size, seq_len, num_tags)
        """
        if mask is None:
            mask = input_ids != 0

        word_embeds = self.word_embedding(input_ids)

        if self.use_char and char_ids is not None:
            char_embeds = self.char_embedding(char_ids)
            char_lengths = (char_ids != 0).sum(dim=-1).clamp(min=1)
            char_max_len = char_ids.size(-1)

            char_packed = nn.utils.rnn.pack_padded_sequence(
                char_embeds.view(-1, char_max_len, char_embeds.size(-1)),
                char_lengths.view(-1),
                batch_first=True,
                enforce_sorted=False,
            )
            char_out, _ = self.char_lstm(char_packed)
            char_out, _ = nn.utils.rnn.pad_packed_sequence(char_out, batch_first=True)
            char_repr = char_out.mean(dim=1)
            char_repr = self.char_projection(char_repr)
            char_repr = char_repr.view(input_ids.size(0), input_ids.size(1), -1)

            embeddings = torch.cat([word_embeds, char_repr], dim=-1)
        else:
            embeddings = word_embeds

        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        emissions = self.fc(lstm_out)

        if self.training and tags is not None and self.use_crf:
            loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
            return loss

        return emissions

    def predict(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        char_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict entity tags for input tokens.

        Args:
            input_ids: Word token indices
            mask: Attention mask for padding
            char_ids: Character indices

        Returns:
            Predicted tags, shape (batch_size, seq_len)
        """
        if mask is None:
            mask = input_ids != 0

        emissions = self.forward(input_ids, mask=mask, char_ids=char_ids)

        if self.use_crf:
            predictions = self.crf.decode(emissions, mask=mask)
            return torch.tensor(predictions, device=input_ids.device)

        return emissions.argmax(dim=-1)


class TransformerTokenClassifier(nn.Module):
    """Transformer-based token classifier for NER.

    Uses a transformer encoder for contextualized token representation
    followed by token-level classification.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of token embeddings
        hidden_dim: Dimension of transformer hidden states
        num_tags: Number of entity tags
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_seq_len: Maximum sequence length
        use_crf: Whether to use CRF decoding
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_tags: int = 9,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_crf: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.use_crf = use_crf

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.embedding_projection = nn.Linear(embedding_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training or inference.

        Args:
            input_ids: Token indices, shape (batch_size, seq_len)
            tags: Entity tags for training
            mask: Attention mask

        Returns:
            Loss or emission scores
        """
        batch_size, seq_len = input_ids.shape

        if mask is None:
            mask = input_ids != 0

        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(positions)

        embeddings = token_embeds + position_embeds
        embeddings = self.embedding_projection(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        attn_mask = ~mask

        hidden_states = self.transformer(embeddings, src_key_padding_mask=attn_mask)
        hidden_states = self.dropout(hidden_states)

        emissions = self.classifier(hidden_states)

        if self.training and tags is not None and self.use_crf:
            loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
            return loss

        return emissions

    def predict(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict entity tags."""
        if mask is None:
            mask = input_ids != 0

        emissions = self.forward(input_ids, mask=mask)

        if self.use_crf:
            predictions = self.crf.decode(emissions, mask=mask)
            return torch.tensor(predictions, device=input_ids.device)

        return emissions.argmax(dim=-1)


class NERTokenClassifier(nn.Module):
    """Unified NER token classifier supporting multiple architectures.

    A flexible NER model that can use either BiLSTM or Transformer
    as the encoder, with optional CRF decoding.

    Args:
        vocab_size: Size of vocabulary
        encoder_type: Either 'bilstm' or 'transformer'
        embedding_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden states
        num_tags: Number of entity tags
        num_layers: Number of encoder layers
        num_heads: Number of attention heads (transformer only)
        dropout: Dropout probability
        use_crf: Whether to use CRF decoding
        max_seq_len: Maximum sequence length (transformer only)
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_type: str = "bilstm",
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_tags: int = 9,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_crf: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.encoder_type = encoder_type
        self.num_tags = num_tags

        if encoder_type == "bilstm":
            self.encoder = BiLSTMTokenClassifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_tags=num_tags,
                num_layers=num_layers,
                dropout=dropout,
                use_crf=use_crf,
            )
        elif encoder_type == "transformer":
            self.encoder = TransformerTokenClassifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_tags=num_tags,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_crf=use_crf,
                max_seq_len=max_seq_len,
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        return self.encoder(input_ids, tags=tags, mask=mask, **kwargs)

    def predict(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Predict entity tags."""
        return self.encoder.predict(input_ids, mask=mask, **kwargs)


class CRF(nn.Module):
    """Conditional Random Field for sequence labeling.

    Implements forward algorithm and Viterbi decoding for
    sequence labeling with transition constraints.
    """

    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        self._init_transitions()

    def _init_transitions(self) -> None:
        """Initialize transition parameters with reasonable defaults."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.zeros_(self.start_transitions)
        nn.init.zeros_(self.end_transitions)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute negative log likelihood.

        Args:
            emissions: Emission scores (batch_size, seq_len, num_tags)
            tags: Target tags (batch_size, seq_len)
            mask: Valid position mask (batch_size, seq_len)
            reduction: Reduction method ('mean', 'sum', 'none')

        Returns:
            Negative log likelihood
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == "mean":
            return -llh.mean()
        elif reduction == "sum":
            return -llh.sum()
        return -llh

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unnormalized score for a tag sequence."""
        seq_length, batch_size = tags.shape

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i].float()
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i].float()

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log partition function using forward algorithm."""
        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Decode the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions: Emission scores
            mask: Valid position mask

        Returns:
            List of tag sequences, one per batch element
        """
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """Viterbi decoding implementation."""
        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


def build_token_classifier(
    vocab_size: int,
    num_tags: int,
    encoder_type: str = "bilstm",
    **kwargs,
) -> NERTokenClassifier:
    """Build a token classifier for NER.

    Args:
        vocab_size: Size of vocabulary
        num_tags: Number of entity tags
        encoder_type: Type of encoder ('bilstm' or 'transformer')
        **kwargs: Additional arguments passed to NERTokenClassifier

    Returns:
        Configured NER token classifier
    """
    return NERTokenClassifier(
        vocab_size=vocab_size,
        encoder_type=encoder_type,
        num_tags=num_tags,
        **kwargs,
    )

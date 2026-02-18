"""
Sequence Labeling module for Named Entity Recognition.

Provides sequence labeling models with various tagging schemes:
- BIO (Beginning, Inside, Outside)
- BIOES (Begin, Inside, Outside, End, Single)
- BIOLUE (Begin, Inside, Outside, Last, Unit, End)
- BIEOS (Begin, Inside, End, Outside, Single)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Set
from abc import ABC, abstractmethod
import math


class TaggingScheme(ABC):
    """Abstract base class for NER tagging schemes.

    Defines interface for converting between entity spans and tag sequences.
    """

    @abstractmethod
    def get_num_tags(self, num_entity_types: int) -> int:
        """Get total number of tags given number of entity types."""
        pass

    @abstractmethod
    def encode_tag(self, entity_type: str, tag_type: str) -> int:
        """Encode entity type and tag type to tag index."""
        pass

    @abstractmethod
    def decode_tag(self, tag_idx: int) -> Tuple[str, str]:
        """Decode tag index to entity type and tag type."""
        pass

    @abstractmethod
    def spans_to_tags(
        self,
        text: List[str],
        spans: List[Tuple[int, int, str]],
    ) -> List[str]:
        """Convert entity spans to tag sequence."""
        pass

    @abstractmethod
    def tags_to_spans(
        self,
        text: List[str],
        tags: List[str],
    ) -> List[Tuple[int, int, str]]:
        """Convert tag sequence to entity spans."""
        pass


class BIOESTaggingScheme(TaggingScheme):
    """BIOES tagging scheme.

    Tags:
    - O: Outside any entity
    - B-TYPE: Beginning of entity
    - I-TYPE: Inside entity (continuation)
    - E-TYPE: End of entity
    - S-TYPE: Single token entity

    Example: [B-PER, I-PER, E-PER] for "John Smith"
             [S-LOC] for "Paris"
    """

    def __init__(self, entity_types: Optional[List[str]] = None):
        self.entity_types = entity_types or []
        self._tag_to_idx: Dict[str, int] = {}
        self._idx_to_tag: Dict[int, str] = {}
        self._build_mapping()

    def _build_mapping(self) -> None:
        """Build tag to index mapping."""
        tags = ["O"]
        for entity_type in self.entity_types:
            tags.extend(
                [
                    f"B-{entity_type}",
                    f"I-{entity_type}",
                    f"E-{entity_type}",
                    f"S-{entity_type}",
                ]
            )

        for idx, tag in enumerate(tags):
            self._tag_to_idx[tag] = idx
            self._idx_to_tag[idx] = tag

    def get_num_tags(self, num_entity_types: int) -> int:
        """Get total number of tags."""
        return 1 + num_entity_types * 4

    def encode_tag(self, entity_type: str, tag_type: str) -> int:
        """Encode entity type and tag type to tag index."""
        tag = f"{tag_type}-{entity_type}"
        return self._tag_to_idx.get(tag, 0)

    def decode_tag(self, tag_idx: int) -> Tuple[str, str]:
        """Decode tag index to entity type and tag type."""
        tag = self._idx_to_tag.get(tag_idx, "O")
        if tag == "O":
            return "O", "O"

        tag_type, entity_type = tag.split("-", 1)
        return entity_type, tag_type

    def spans_to_tags(
        self,
        text: List[str],
        spans: List[Tuple[int, int, str]],
    ) -> List[str]:
        """Convert entity spans to BIOES tags."""
        tags = ["O"] * len(text)

        for start, end, entity_type in spans:
            length = end - start + 1

            if length == 1:
                tags[start] = f"S-{entity_type}"
            else:
                tags[start] = f"B-{entity_type}"
                for i in range(start + 1, end):
                    tags[i] = f"I-{entity_type}"
                tags[end] = f"E-{entity_type}"

        return tags

    def tags_to_spans(
        self,
        text: List[str],
        tags: List[str],
    ) -> List[Tuple[int, int, str]]:
        """Convert BIOES tags to entity spans."""
        spans: List[Tuple[int, int, str]] = []
        current_entity: Optional[Tuple[int, str]] = None

        for i, tag in enumerate(tags):
            if tag == "O":
                if current_entity is not None:
                    start, entity_type = current_entity
                    spans.append((start, i - 1, entity_type))
                    current_entity = None
            elif tag.startswith("B-"):
                if current_entity is not None:
                    start, entity_type = current_entity
                    spans.append((start, i - 1, entity_type))
                entity_type = tag[2:]
                current_entity = (i, entity_type)
            elif tag.startswith("I-"):
                if current_entity is None:
                    entity_type = tag[2:]
                    current_entity = (i, entity_type)
                else:
                    _, curr_type = current_entity
                    if curr_type != tag[2:]:
                        start, _ = current_entity
                        spans.append((start, i - 1, curr_type))
                        entity_type = tag[2:]
                        current_entity = (i, entity_type)
            elif tag.startswith("E-"):
                if current_entity is not None:
                    start, entity_type = current_entity
                    if entity_type == tag[2:]:
                        spans.append((start, i, entity_type))
                    else:
                        spans.append((start, i - 1, entity_type))
                        spans.append((i, i, tag[2:]))
                    current_entity = None
            elif tag.startswith("S-"):
                if current_entity is not None:
                    start, entity_type = current_entity
                    spans.append((start, i - 1, entity_type))
                spans.append((i, i, tag[2:]))
                current_entity = None

        if current_entity is not None:
            start, entity_type = current_entity
            spans.append((start, len(text) - 1, entity_type))

        return spans


class BIOLUETaggingScheme(TaggingScheme):
    """BIOLUE tagging scheme.

    Tags:
    - O: Outside any entity
    - B-TYPE: Beginning
    - I-TYPE: Inside
    - L-TYPE: Last
    - U-TYPE: Unit (single token)
    - E-TYPE: End
    """

    def __init__(self, entity_types: Optional[List[str]] = None):
        self.entity_types = entity_types or []
        self._tag_to_idx: Dict[str, int] = {}
        self._idx_to_tag: Dict[int, str] = {}
        self._build_mapping()

    def _build_mapping(self) -> None:
        tags = ["O"]
        for entity_type in self.entity_types:
            tags.extend(
                [
                    f"B-{entity_type}",
                    f"I-{entity_type}",
                    f"L-{entity_type}",
                    f"U-{entity_type}",
                    f"E-{entity_type}",
                ]
            )

        for idx, tag in enumerate(tags):
            self._tag_to_idx[tag] = idx
            self._idx_to_tag[idx] = tag

    def get_num_tags(self, num_entity_types: int) -> int:
        return 1 + num_entity_types * 5

    def encode_tag(self, entity_type: str, tag_type: str) -> int:
        tag = f"{tag_type}-{entity_type}"
        return self._tag_to_idx.get(tag, 0)

    def decode_tag(self, tag_idx: int) -> Tuple[str, str]:
        tag = self._idx_to_tag.get(tag_idx, "O")
        if tag == "O":
            return "O", "O"
        tag_type, entity_type = tag.split("-", 1)
        return entity_type, tag_type

    def spans_to_tags(
        self,
        text: List[str],
        spans: List[Tuple[int, int, str]],
    ) -> List[str]:
        tags = ["O"] * len(text)

        for start, end, entity_type in spans:
            length = end - start + 1

            if length == 1:
                tags[start] = f"U-{entity_type}"
            else:
                tags[start] = f"B-{entity_type}"
                for i in range(start + 1, end):
                    tags[i] = f"I-{entity_type}"
                tags[end] = f"L-{entity_type}"

        return tags

    def tags_to_spans(
        self,
        text: List[str],
        tags: List[str],
    ) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []

        i = 0
        while i < len(tags):
            tag = tags[i]

            if tag == "O":
                i += 1
            elif tag.startswith("U-"):
                entity_type = tag[2:]
                spans.append((i, i, entity_type))
                i += 1
            elif tag.startswith("B-"):
                entity_type = tag[2:]
                start = i
                i += 1

                while i < len(tags) and tags[i] == f"I-{entity_type}":
                    i += 1

                if i < len(tags) and tags[i] == f"L-{entity_type}":
                    spans.append((start, i, entity_type))
                    i += 1
                else:
                    spans.append((start, i - 1, entity_type))
            elif tag.startswith("I-"):
                entity_type = tag[2:]
                start = i
                while i < len(tags) and tags[i] == f"I-{entity_type}":
                    i += 1
                spans.append((start, i - 1, entity_type))
            else:
                i += 1

        return spans


class BIEOS_TAGGING_SCHEME(TaggingScheme):
    """BIEOS tagging scheme (similar to BIOES but with different naming).

    Tags:
    - O: Outside
    - B-TYPE: Beginning
    - I-TYPE: Inside
    - E-TYPE: End
    - S-TYPE: Single
    """

    def __init__(self, entity_types: Optional[List[str]] = None):
        self.entity_types = entity_types or []
        self._tag_to_idx: Dict[str, int] = {}
        self._idx_to_tag: Dict[int, str] = {}
        self._build_mapping()

    def _build_mapping(self) -> None:
        tags = ["O"]
        for entity_type in self.entity_types:
            tags.extend(
                [
                    f"B-{entity_type}",
                    f"I-{entity_type}",
                    f"E-{entity_type}",
                    f"S-{entity_type}",
                ]
            )

        for idx, tag in enumerate(tags):
            self._tag_to_idx[tag] = idx
            self._idx_to_tag[idx] = tag

    def get_num_tags(self, num_entity_types: int) -> int:
        return 1 + num_entity_types * 4

    def encode_tag(self, entity_type: str, tag_type: str) -> int:
        tag = f"{tag_type}-{entity_type}"
        return self._tag_to_idx.get(tag, 0)

    def decode_tag(self, tag_idx: int) -> Tuple[str, str]:
        tag = self._idx_to_tag.get(tag_idx, "O")
        if tag == "O":
            return "O", "O"
        tag_type, entity_type = tag.split("-", 1)
        return entity_type, tag_type

    def spans_to_tags(
        self,
        text: List[str],
        spans: List[Tuple[int, int, str]],
    ) -> List[str]:
        return BIOESTaggingScheme(self.entity_types).spans_to_tags(text, spans)

    def tags_to_spans(
        self,
        text: List[str],
        tags: List[str],
    ) -> List[Tuple[int, int, str]]:
        return BIOESTaggingScheme(self.entity_types).tags_to_spans(text, tags)


class SequenceLabeler(nn.Module):
    """Base class for sequence labeling models.

    Provides common functionality for sequence labeling with
    different encoder architectures.
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_crf: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_crf = use_crf

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.dropout = nn.Dropout(dropout)

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def predict(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class LSTMSequenceLabeler(SequenceLabeler):
    """LSTM-based sequence labeler.

    Bidirectional LSTM encoder with optional CRF decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_crf: bool = True,
        use_char: bool = False,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_tags=num_tags,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_crf=use_crf,
        )

        self.use_char = use_char

        if use_char:
            self.char_embedding = nn.Embedding(vocab_size, 50, padding_idx=0)
            self.char_lstm = nn.LSTM(50, 50, batch_first=True, bidirectional=True)
            char_out_dim = 100
            self.char_projection = nn.Linear(char_out_dim, embedding_dim)
            total_dim = embedding_dim * 2
        else:
            total_dim = embedding_dim

        self.lstm = nn.LSTM(
            total_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        char_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        word_embeds = self.embedding(input_ids)

        if self.use_char and char_ids is not None:
            char_embeds = self.char_embedding(char_ids)
            char_lengths = (char_ids != 0).sum(dim=-1).clamp(min=1)
            char_packed = nn.utils.rnn.pack_padded_sequence(
                char_embeds.view(-1, char_ids.size(-1), char_embeds.size(-1)),
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

        emissions = self.classifier(lstm_out)

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
        if mask is None:
            mask = input_ids != 0

        emissions = self.forward(input_ids, mask=mask, char_ids=char_ids)

        if self.use_crf:
            predictions = self.crf.decode(emissions, mask=mask)
            return torch.tensor(predictions, device=input_ids.device)

        return emissions.argmax(dim=-1)


class TransformerSequenceLabeler(SequenceLabeler):
    """Transformer-based sequence labeler.

    Transformer encoder with optional CRF decoding.
    """

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_crf: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_tags=num_tags,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_crf=use_crf,
        )

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.projection = nn.Linear(embedding_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        embeddings = self.projection(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        hidden_states = self.transformer(embeddings, src_key_padding_mask=~mask)
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
        if mask is None:
            mask = input_ids != 0

        emissions = self.forward(input_ids, mask=mask)

        if self.use_crf:
            predictions = self.crf.decode(emissions, mask=mask)
            return torch.tensor(predictions, device=input_ids.device)

        return emissions.argmax(dim=-1)


class CRF(nn.Module):
    """Conditional Random Field for sequence labeling."""

    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
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

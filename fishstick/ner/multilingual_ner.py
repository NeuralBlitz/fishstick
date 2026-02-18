"""
Multi-lingual NER module for Named Entity Recognition.

Provides tools for cross-lingual and multi-lingual NER:
- Cross-lingual NER with shared representations
- Multi-lingual encoder with language adapters
- Zero-shot NER capabilities
- Translation-based NER
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Set
import math


class MultiLingualEncoder(nn.Module):
    """Multi-lingual encoder for NER across multiple languages.

    Uses a shared encoder with language-specific embeddings
    to enable cross-lingual transfer.

    Args:
        vocab_size: Size of vocabulary (per language)
        embedding_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden states
        num_languages: Number of supported languages
        num_layers: Number of encoder layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_languages: int = 10,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_languages = num_languages

        self.language_embeddings = nn.Embedding(num_languages, embedding_dim)

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projection = nn.Linear(embedding_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        language_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input with language awareness.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            language_ids: Language IDs (batch_size,)
            mask: Attention mask

        Returns:
            Encoded representations (batch_size, seq_len, hidden_dim)
        """
        token_embeds = self.token_embedding(input_ids)

        lang_embeds = self.language_embeddings(language_ids)
        lang_embeds = lang_embeds.unsqueeze(1).expand(-1, input_ids.size(1), -1)

        embeddings = token_embeds + lang_embeds
        embeddings = self.dropout(embeddings)

        encoded = self.encoder(embeddings, src_key_padding_mask=~mask)

        hidden_states = self.projection(encoded)

        return hidden_states


class LanguageAdapter(nn.Module):
    """Language adapter for multi-lingual NER.

    Adapts a shared encoder to language-specific features
    using residual connections and layer normalization.

    Args:
        hidden_dim: Dimension of hidden states
        num_languages: Number of languages
        adapter_dim: Dimension of adapter hidden layer
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_languages: int = 10,
        adapter_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_languages = num_languages

        self.language_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, adapter_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(adapter_dim, hidden_dim),
                )
                for _ in range(num_languages)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_languages)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        language_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply language-specific adaptation.

        Args:
            hidden_states: Shared encoder representations
            language_ids: Language identifiers

        Returns:
            Adapted representations
        """
        adapted_states = []

        for b in range(hidden_states.size(0)):
            lang_id = language_ids[b].item()

            adapted = self.language_adapters[lang_id](hidden_states[b])
            adapted = self.layer_norms[lang_id](hidden_states[b] + adapted)
            adapted_states.append(adapted)

        return torch.stack(adapted_states)


class CrossLingualNER(nn.Module):
    """Cross-lingual NER model.

    Enables NER in target languages using training data
    from source languages through shared representations.

    Args:
        vocab_size: Size of vocabulary
        num_entity_types: Number of entity types
        num_languages: Number of languages to support
        embedding_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden states
        encoder_layers: Number of encoder layers
        dropout: Dropout probability
        use_adapter: Whether to use language adapters
    """

    def __init__(
        self,
        vocab_size: int,
        num_entity_types: int = 9,
        num_languages: int = 10,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        encoder_layers: int = 4,
        dropout: float = 0.1,
        use_adapter: bool = True,
    ):
        super().__init__()

        self.num_entity_types = num_entity_types
        self.num_languages = num_languages
        self.use_adapter = use_adapter

        self.encoder = MultiLingualEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_languages=num_languages,
            num_layers=encoder_layers,
            dropout=dropout,
        )

        if use_adapter:
            self.adapter = LanguageAdapter(
                hidden_dim=hidden_dim,
                num_languages=num_languages,
                dropout=dropout,
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_entity_types),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        language_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for cross-lingual NER.

        Args:
            input_ids: Token IDs
            language_ids: Language identifiers
            tags: Entity tags for training
            mask: Attention mask

        Returns:
            Loss or predictions
        """
        if mask is None:
            mask = input_ids != 0

        hidden_states = self.encoder(input_ids, language_ids, mask)

        if self.use_adapter:
            hidden_states = self.adapter(hidden_states, language_ids)

        hidden_states = self.dropout(hidden_states)

        logits = self.classifier(hidden_states)

        if self.training and tags is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_entity_types),
                tags.view(-1),
                ignore_index=0,
            )
            return loss

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        language_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict entity tags."""
        logits = self.forward(input_ids, language_ids, mask=mask)
        return logits.argmax(dim=-1)


class ZeroShotNER(nn.Module):
    """Zero-shot NER using entity type embeddings.

    Enables recognition of novel entity types not seen during training
    by using semantic representations of entity type names.

    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Dimension of hidden states
        entity_type_dim: Dimension of entity type embeddings
        num_known_types: Number of entity types seen during training
        encoder_type: Type of encoder ('lstm' or 'transformer')
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        entity_type_dim: int = 128,
        num_known_types: int = 9,
        encoder_type: str = "transformer",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.entity_type_dim = entity_type_dim
        self.num_known_types = num_known_types

        self.type_embedding = nn.Embedding(num_known_types + 1, entity_type_dim)

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        if encoder_type == "lstm":
            self.encoder = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            encoder_out_dim = hidden_dim * 2
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
            encoder_out_dim = hidden_dim

        self.type_scorer = nn.Linear(encoder_out_dim + entity_type_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for zero-shot NER.

        Args:
            input_ids: Token IDs
            type_ids: Entity type IDs to score against
            labels: Labels for training
            mask: Attention mask

        Returns:
            Scores for each token-type pair
        """
        if mask is None:
            mask = input_ids != 0

        embeddings = self.token_embedding(input_ids)
        embeddings = self.dropout(embeddings)

        if isinstance(self.encoder, nn.LSTM):
            encoded, _ = self.encoder(embeddings)
        else:
            encoded = self.encoder(embeddings, src_key_padding_mask=~mask)

        encoded = self.dropout(encoded)

        type_embeds = self.type_embedding(type_ids)

        batch_size, seq_len = input_ids.shape
        num_types = type_ids.size(1)

        encoded_expanded = encoded.unsqueeze(2).expand(-1, -1, num_types, -1)
        type_expanded = type_embeds.unsqueeze(1).expand(-1, seq_len, -1, -1)

        combined = torch.cat([encoded_expanded, type_expanded], dim=-1)

        scores = self.type_scorer(combined).squeeze(-1)

        if self.training and labels is not None:
            loss = F.binary_cross_entropy_with_logits(scores, labels.float())
            return loss

        return scores

    def predict(
        self,
        input_ids: torch.Tensor,
        type_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Predict entity types for tokens."""
        scores = self.forward(input_ids, type_ids, mask=mask)
        return (torch.sigmoid(scores) > threshold).long()


class TranslationBasedNER(nn.Module):
    """Translation-based NER for low-resource languages.

    Leverages machine translation to project entity annotations
    from high-resource to low-resource languages.

    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Dimension of hidden states
        num_entity_types: Number of entity types
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_entity_types: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_entity_types = num_entity_types

        self.source_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        self.target_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        self.encoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.alignment_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_entity_types),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        tags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with translation alignment.

        Args:
            source_ids: Source language token IDs
            target_ids: Target language token IDs
            source_mask: Source attention mask
            target_mask: Target attention mask
            tags: Entity tags

        Returns:
            Loss or predictions
        """
        source_embeds = self.source_embedding(source_ids)
        target_embeds = self.target_embedding(target_ids)

        source_embeds = self.dropout(source_embeds)
        target_embeds = self.dropout(target_embeds)

        source_encoded, _ = self.encoder(source_embeds)
        target_encoded, _ = self.encoder(target_embeds)

        source_encoded = self.dropout(source_encoded)
        target_encoded = self.dropout(target_encoded)

        alignment_weights = torch.bmm(source_encoded, target_encoded.transpose(1, 2))

        if source_mask is not None:
            alignment_weights = alignment_weights.masked_fill(
                ~source_mask.unsqueeze(-1), float("-inf")
            )

        alignment_weights = F.softmax(alignment_weights, dim=1)

        aligned_source = torch.bmm(alignment_weights, source_encoded)

        aligned_source = self.alignment_projection(aligned_source)

        combined = torch.cat([target_encoded, aligned_source], dim=-1)

        logits = self.classifier(combined)

        if self.training and tags is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_entity_types),
                tags.view(-1),
                ignore_index=0,
            )
            return loss

        return logits

    def predict(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict entity tags."""
        logits = self.forward(source_ids, target_ids, source_mask, target_mask)
        return logits.argmax(dim=-1)


class PseudoLabelNER(nn.Module):
    """Pseudo-labeling based NER for cross-lingual transfer.

    Uses high-confidence predictions on target language
    as supervision for training.

    Args:
        vocab_size: Size of vocabulary
        hidden_dim: Dimension of hidden states
        num_entity_types: Number of entity types
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_entity_types: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_entity_types = num_entity_types

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_entity_types),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        encoded = self.encoder(embeddings, src_key_padding_mask=~mask)
        encoded = self.dropout(encoded)

        logits = self.classifier(encoded)

        if self.training and tags is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_entity_types),
                tags.view(-1),
                ignore_index=0,
            )
            return loss

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict with confidence scores."""
        logits = self.forward(input_ids, mask=mask)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        confidences = probs.max(dim=-1)[0]

        return preds, confidences

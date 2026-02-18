"""
Aspect-based sentiment analysis module.

Provides models for identifying sentiment towards specific aspects within text,
including aspect extraction, aspect-level sentiment classification, and opinion mining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from collections import defaultdict


class AspectExtractor(nn.Module):
    """Extracts aspect terms from text using sequence labeling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_tags: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tags = num_tags

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.tag_classifier = nn.Linear(hidden_dim * 2, num_tags)

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        logits = self.tag_classifier(lstm_out)

        if tags is not None and self.training:
            loss = F.cross_entropy(
                logits.view(-1, self.num_tags),
                tags.view(-1),
                reduction="none",
            )
            masked_loss = loss * mask.view(-1)
            return masked_loss.mean()

        return logits

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.forward(input_ids, mask=mask)
        predictions = logits.argmax(dim=-1)
        return predictions


class AspectSentimentClassifier(nn.Module):
    """Classifies sentiment for specific aspect terms in context."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_sentiment_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_sentiment_classes = num_sentiment_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.aspect_marker = nn.Embedding(2, embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_sentiment_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        aspect_positions: torch.Tensor,
        sentiment_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        embeddings = self.embedding(input_ids)

        aspect_markers = torch.zeros_like(input_ids)
        for b in range(batch_size):
            for start, end in aspect_positions[b]:
                if start < seq_len and end <= seq_len:
                    aspect_markers[b, start:end] = 1

        aspect_embeds = self.aspect_marker(aspect_markers)
        combined_embeds = embeddings + aspect_embeds

        encoded = self.encoder(combined_embeds)

        context_repr = encoded.mean(dim=1)

        aspect_reprs = []
        for b in range(batch_size):
            start, end = aspect_positions[b][0]
            if end > start:
                aspect_vec = encoded[b, start:end].mean(dim=0)
            else:
                aspect_vec = torch.zeros(self.embedding_dim, device=input_ids.device)
            aspect_reprs.append(aspect_vec)

        aspect_repr = torch.stack(aspect_reprs)

        left_context = encoded[:, : seq_len // 2, :].mean(dim=1)
        right_context = encoded[:, seq_len // 2 :, :].mean(dim=1)

        combined = torch.cat(
            [context_repr, aspect_repr, left_context, right_context], dim=-1
        )

        logits = self.classifier(combined)

        if sentiment_labels is not None and self.training:
            loss = F.cross_entropy(logits, sentiment_labels)
            return loss

        return logits

    def predict(
        self, input_ids: torch.Tensor, aspect_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, aspect_positions)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class RelationAwareAspectExtractor(nn.Module):
    """Extracts aspects using relation-aware graph neural network."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_aspect_tags: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.word_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dependency_edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.aspect_classifier = nn.Linear(hidden_dim * 2, num_aspect_tags)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        dep_edges: Optional[torch.Tensor] = None,
        aspect_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.word_lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        if dep_edges is not None:
            batch_size, seq_len, _ = lstm_out.shape

            edge_weights = []
            for b in range(batch_size):
                src_nodes = lstm_out[b]
                for src_idx, tgt_idx in dep_edges[b]:
                    if src_idx < seq_len and tgt_idx < seq_len:
                        edge_score = self.dependency_edge_mlp(
                            torch.cat([src_nodes[src_idx], src_nodes[tgt_idx]], dim=-1)
                        )
                        edge_weights.append(edge_score)

            if edge_weights:
                edge_repr = torch.stack(edge_weights).view(batch_size, seq_len, -1)
                lstm_out = lstm_out + edge_repr

        logits = self.aspect_classifier(lstm_out)

        if aspect_labels is not None and self.training:
            loss = F.cross_entropy(
                logits.view(-1, self.num_aspect_tags),
                aspect_labels.view(-1),
                reduction="none",
            )
            masked_loss = loss * mask.view(-1)
            return masked_loss.mean()

        return logits

    def predict(
        self, input_ids: torch.Tensor, dep_edges: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.forward(input_ids, dep_edges)
        predictions = logits.argmax(dim=-1)
        return predictions


class EndToEndAspectSentiment(nn.Module):
    """End-to-end model for joint aspect extraction and sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_encoder_layers: int = 3,
        num_aspect_tags: int = 3,
        num_sentiment_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.aspect_tagger = nn.Linear(embedding_dim, num_aspect_tags)

        self.aspect_attention = nn.Linear(embedding_dim, 1)

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sentiment_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        aspect_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        encoded = self.encoder(embeddings)
        encoded = self.dropout(encoded)

        aspect_logits = self.aspect_tagger(encoded)

        aspect_probs = F.softmax(aspect_logits, dim=-1)

        aspect_weights = self.aspect_attention(encoded)
        aspect_weights = aspect_weights.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        aspect_weights = F.softmax(aspect_weights, dim=1)

        aspect_context = torch.bmm(aspect_weights.transpose(1, 2), encoded).squeeze(1)

        global_context = encoded.mean(dim=1)

        sentiment_input = torch.cat(
            [global_context, aspect_context, encoded[:, 0, :]], dim=-1
        )

        sentiment_logits = self.sentiment_classifier(sentiment_input)

        total_loss = None
        if self.training:
            if aspect_labels is not None:
                aspect_loss = F.cross_entropy(
                    aspect_logits.view(-1, self.aspect_tagger.out_features),
                    aspect_labels.view(-1),
                    reduction="none",
                )
                aspect_loss = (aspect_loss * mask.view(-1)).mean()

            if sentiment_labels is not None:
                sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_labels)

            if aspect_labels is not None and sentiment_labels is not None:
                total_loss = aspect_loss + sentiment_loss
            elif sentiment_labels is not None:
                total_loss = sentiment_loss
            elif aspect_labels is not None:
                total_loss = aspect_loss

        return aspect_logits, sentiment_logits, total_loss

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aspect_logits, sentiment_logits, _ = self.forward(input_ids, mask=mask)
        aspect_preds = aspect_logits.argmax(dim=-1)
        sentiment_preds = sentiment_logits.argmax(dim=-1)
        sentiment_probs = F.softmax(sentiment_logits, dim=-1)
        return aspect_preds, sentiment_preds, sentiment_probs


class AspectOpinionPairExtractor(nn.Module):
    """Extracts aspect-opinion pairs from text."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_aspect_tags: int = 3,
        num_opinion_tags: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_aspect_tags = num_aspect_tags
        self.num_opinion_tags = num_opinion_tags

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.aspect_classifier = nn.Linear(hidden_dim * 2, num_aspect_tags)
        self.opinion_classifier = nn.Linear(hidden_dim * 2, num_opinion_tags)

    def forward(
        self,
        input_ids: torch.Tensor,
        aspect_labels: Optional[torch.Tensor] = None,
        opinion_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        aspect_logits = self.aspect_classifier(lstm_out)
        opinion_logits = self.opinion_classifier(lstm_out)

        total_loss = None
        if self.training:
            if aspect_labels is not None:
                aspect_loss = F.cross_entropy(
                    aspect_logits.view(-1, self.num_aspect_tags),
                    aspect_labels.view(-1),
                    reduction="none",
                )
                aspect_loss = (aspect_loss * mask.view(-1)).mean()

            if opinion_labels is not None:
                opinion_loss = F.cross_entropy(
                    opinion_logits.view(-1, self.num_opinion_tags),
                    opinion_labels.view(-1),
                    reduction="none",
                )
                opinion_loss = (opinion_loss * mask.view(-1)).mean()

            if aspect_labels is not None and opinion_labels is not None:
                total_loss = aspect_loss + opinion_loss
            elif aspect_labels is not None:
                total_loss = aspect_loss
            elif opinion_labels is not None:
                total_loss = opinion_loss

        return aspect_logits, opinion_logits, total_loss

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aspect_logits, opinion_logits, _ = self.forward(input_ids, mask=mask)
        aspect_preds = aspect_logits.argmax(dim=-1)
        opinion_preds = opinion_logits.argmax(dim=-1)
        return aspect_preds, opinion_preds


class CategoryAspectSentiment(nn.Module):
    """Aspect-based sentiment with predefined aspect categories."""

    def __init__(
        self,
        vocab_size: int,
        num_categories: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_sentiment_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_categories = num_categories
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.category_embedding = nn.Embedding(num_categories, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim * 2,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sentiment_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        category_ids: torch.Tensor,
        sentiment_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        word_embeds = self.embedding(input_ids)
        word_embeds = self.dropout(word_embeds)

        cat_embeds = self.category_embedding(category_ids)
        cat_embeds = cat_embeds.unsqueeze(1).expand_as(word_embeds)

        combined = torch.cat([word_embeds, cat_embeds], dim=-1)
        combined = self.dropout(combined)

        encoded = self.encoder(combined, src_key_padding_mask=~mask)

        pooled = encoded.mean(dim=1)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        if sentiment_labels is not None and self.training:
            loss = F.cross_entropy(logits, sentiment_labels)
            return loss

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        category_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, category_ids, mask=mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


def build_aspect_model(
    vocab_size: int,
    model_type: str = "extractor",
    num_aspect_tags: int = 3,
    num_sentiment_classes: int = 3,
    **kwargs,
) -> nn.Module:
    """Build an aspect-based sentiment model by type."""
    model_registry = {
        "extractor": AspectExtractor,
        "classifier": AspectSentimentClassifier,
        "relation": RelationAwareAspectExtractor,
        "endtoend": EndToEndAspectSentiment,
        "pair": AspectOpinionPairExtractor,
        "category": CategoryAspectSentiment,
    }

    if model_type not in model_registry:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(model_registry.keys())}"
        )

    if model_type == "extractor":
        return model_registry[model_type](
            vocab_size=vocab_size, num_tags=num_aspect_tags, **kwargs
        )
    elif model_type == "classifier":
        return model_registry[model_type](
            vocab_size=vocab_size, num_sentiment_classes=num_sentiment_classes, **kwargs
        )
    elif model_type == "category":
        return model_registry[model_type](
            vocab_size=vocab_size, num_sentiment_classes=num_sentiment_classes, **kwargs
        )
    else:
        return model_registry[model_type](
            vocab_size=vocab_size,
            num_aspect_tags=num_aspect_tags,
            num_sentiment_classes=num_sentiment_classes,
            **kwargs,
        )

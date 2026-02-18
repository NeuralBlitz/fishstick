"""
Opinion mining module.

Provides models for extracting opinions, sentiment tuples, and subjective information from text,
including target-opinion-sentiment extraction and opinion pair classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass


@dataclass
class OpinionTriple:
    """Represents an opinion triple (target, opinion, sentiment)."""

    target: str
    opinion: str
    sentiment: str
    polarity: int
    confidence: float = 1.0


class OpinionExtractor(nn.Module):
    """Extracts opinion expressions from text using sequence labeling."""

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

        self.classifier = nn.Linear(hidden_dim * 2, num_tags)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        logits = self.classifier(lstm_out)

        if labels is not None and self.training:
            loss = F.cross_entropy(
                logits.view(-1, self.num_tags),
                labels.view(-1),
                reduction="none",
            )
            masked_loss = loss * mask.view(-1)
            return masked_loss.mean()

        return logits

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, mask=mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class OpinionRelationClassifier(nn.Module):
    """Classifies relations between opinion elements."""

    def __init__(
        self,
        vocab_size: int,
        num_relations: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_relations = num_relations

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        span1_positions: torch.Tensor,
        span2_positions: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)

        span1_reprs = []
        span2_reprs = []

        batch_size = input_ids.size(0)
        for b in range(batch_size):
            start1, end1 = span1_positions[b]
            start2, end2 = span2_positions[b]

            if end1 > start1:
                span1_repr = lstm_out[b, start1:end1].mean(dim=0)
            else:
                span1_repr = torch.zeros(self.hidden_dim * 2, device=input_ids.device)

            if end2 > start2:
                span2_repr = lstm_out[b, start2:end2].mean(dim=0)
            else:
                span2_repr = torch.zeros(self.hidden_dim * 2, device=input_ids.device)

            span1_reprs.append(span1_repr)
            span2_reprs.append(span2_repr)

        span1_repr = torch.stack(span1_reprs)
        span2_repr = torch.stack(span2_reprs)

        global_repr = lstm_out.mean(dim=1)

        combined = torch.cat([span1_repr, span2_repr, global_repr], dim=-1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        span1_positions: torch.Tensor,
        span2_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, span1_positions, span2_positions)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class SubjectivityClassifier(nn.Module):
    """Classifies text as subjective or objective."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        pooled = lstm_out.mean(dim=1)

        logits = self.classifier(pooled)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, mask=mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class SentimentTupleExtractor(nn.Module):
    """Extracts (aspect, opinion, sentiment) tuples from text."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_aspect_tags: int = 3,
        num_opinion_tags: int = 3,
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.aspect_tagger = nn.Linear(embedding_dim, num_aspect_tags)
        self.opinion_tagger = nn.Linear(embedding_dim, num_opinion_tags)

        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=embedding_dim * 3,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.sentiment_encoder = nn.TransformerEncoder(encoder_layer2, num_layers=1)

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sentiment_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        aspect_labels: Optional[torch.Tensor] = None,
        opinion_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        encoded = self.encoder(embeddings, src_key_padding_mask=~mask)

        aspect_logits = self.aspect_tagger(encoded)
        opinion_logits = self.opinion_tagger(encoded)

        aspect_repr = torch.bmm(
            F.softmax(aspect_logits, dim=-1).transpose(1, 2),
            encoded,
        ).squeeze(1)

        opinion_repr = torch.bmm(
            F.softmax(opinion_logits, dim=-1).transpose(1, 2),
            encoded,
        ).squeeze(1)

        sentiment_input = torch.cat(
            [
                encoded.mean(dim=1, keepdim=True).expand(-1, encoded.size(1), -1),
                aspect_repr.unsqueeze(1).expand(-1, encoded.size(1), -1),
                opinion_repr.unsqueeze(1).expand(-1, encoded.size(1), -1),
            ],
            dim=-1,
        )

        sentiment_encoded = self.sentiment_encoder(
            sentiment_input, src_key_padding_mask=~mask
        )

        sentiment_logits = self.sentiment_classifier(sentiment_encoded.mean(dim=1))

        total_loss = None
        if self.training:
            losses = []
            if aspect_labels is not None:
                aspect_loss = F.cross_entropy(
                    aspect_logits.view(-1, self.aspect_tagger.out_features),
                    aspect_labels.view(-1),
                    reduction="none",
                )
                losses.append((aspect_loss * mask.view(-1)).mean())

            if opinion_labels is not None:
                opinion_loss = F.cross_entropy(
                    opinion_logits.view(-1, self.opinion_tagger.out_features),
                    opinion_labels.view(-1),
                    reduction="none",
                )
                losses.append((opinion_loss * mask.view(-1)).mean())

            if sentiment_labels is not None:
                sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_labels)
                losses.append(sentiment_loss)

            if losses:
                total_loss = sum(losses)

        return aspect_logits, opinion_logits, sentiment_logits, total_loss

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aspect_logits, opinion_logits, sentiment_logits, _ = self.forward(
            input_ids, mask=mask
        )

        aspect_preds = aspect_logits.argmax(dim=-1)
        opinion_preds = opinion_logits.argmax(dim=-1)
        sentiment_preds = sentiment_logits.argmax(dim=-1)
        sentiment_probs = F.softmax(sentiment_logits, dim=-1)

        return aspect_preds, opinion_preds, sentiment_preds


class AspectOpinionSentimentClassifier(nn.Module):
    """Joint classifier for aspect, opinion, and sentiment."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_aspect_tags: int = 3,
        num_opinion_tags: int = 3,
        num_sentiment_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.aspect_classifier = nn.Linear(hidden_dim * 2, num_aspect_tags)
        self.opinion_classifier = nn.Linear(hidden_dim * 2, num_opinion_tags)
        self.sentiment_classifier = nn.Linear(hidden_dim * 2, num_sentiment_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        aspect_labels: Optional[torch.Tensor] = None,
        opinion_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        aspect_logits = self.aspect_classifier(lstm_out)
        opinion_logits = self.opinion_classifier(lstm_out)
        sentiment_logits = self.sentiment_classifier(lstm_out.mean(dim=1))

        total_loss = None
        if self.training:
            losses = []
            if aspect_labels is not None:
                aspect_loss = F.cross_entropy(
                    aspect_logits.view(-1, self.aspect_classifier.out_features),
                    aspect_labels.view(-1),
                    reduction="none",
                )
                losses.append((aspect_loss * mask.view(-1)).mean())

            if opinion_labels is not None:
                opinion_loss = F.cross_entropy(
                    opinion_logits.view(-1, self.opinion_classifier.out_features),
                    opinion_labels.view(-1),
                    reduction="none",
                )
                losses.append((opinion_loss * mask.view(-1)).mean())

            if sentiment_labels is not None:
                sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_labels)
                losses.append(sentiment_loss)

            if losses:
                total_loss = sum(losses)

        return aspect_logits, opinion_logits, sentiment_logits, total_loss

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aspect_logits, opinion_logits, sentiment_logits, _ = self.forward(
            input_ids, mask=mask
        )

        aspect_preds = aspect_logits.argmax(dim=-1)
        opinion_preds = opinion_logits.argmax(dim=-1)
        sentiment_preds = sentiment_logits.argmax(dim=-1)

        return aspect_preds, opinion_preds, sentiment_preds


class ComparativeOpinionExtractor(nn.Module):
    """Extracts comparative opinions (better, worse, equal)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_comparison_types: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_comparison_types),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        pooled = lstm_out.mean(dim=1)

        logits = self.classifier(pooled)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, mask=mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class IntensityClassifier(nn.Module):
    """Classifies sentiment intensity on a continuous scale."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)

        pooled = lstm_out.mean(dim=1)

        intensity = self.regressor(pooled).squeeze(-1)

        if labels is not None and self.training:
            loss = F.mse_loss(intensity, labels)
            return loss

        return intensity

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.forward(input_ids, mask=mask)


class OpinionSentimentScorer(nn.Module):
    """Scores sentiment strength for opinion terms."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.scorer = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        scores = self.scorer(lstm_out).squeeze(-1)

        if labels is not None and self.training:
            loss = F.mse_loss(scores, labels)
            return loss

        return scores


class SentimentLexiconLearner(nn.Module):
    """Learns sentiment scores for words from data."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.dropout = nn.Dropout(dropout)

        self.sentiment_score = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        scores = self.sentiment_score(embeddings).squeeze(-1)

        if labels is not None and self.training:
            loss = F.mse_loss(scores, labels)
            return loss

        return scores

    def get_word_sentiment(self, word_id: torch.Tensor) -> float:
        with torch.no_grad():
            score = self.forward(word_id.unsqueeze(0))
            return score.item()


def build_opinion_model(
    vocab_size: int,
    model_type: str = "extractor",
    num_aspect_tags: int = 3,
    num_opinion_tags: int = 3,
    num_sentiment_classes: int = 3,
    **kwargs,
) -> nn.Module:
    """Build an opinion mining model by type."""
    model_registry = {
        "extractor": OpinionExtractor,
        "relation": OpinionRelationClassifier,
        "subjectivity": SubjectivityClassifier,
        "tuple": SentimentTupleExtractor,
        "joint": AspectOpinionSentimentClassifier,
        "comparative": ComparativeOpinionExtractor,
        "intensity": IntensityClassifier,
        "scorer": OpinionSentimentScorer,
        "lexicon": SentimentLexiconLearner,
    }

    if model_type not in model_registry:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(model_registry.keys())}"
        )

    if model_type == "extractor":
        return model_registry[model_type](
            vocab_size=vocab_size, num_tags=num_opinion_tags, **kwargs
        )
    elif model_type in ("tuple", "joint"):
        return model_registry[model_type](
            vocab_size=vocab_size,
            num_aspect_tags=num_aspect_tags,
            num_opinion_tags=num_opinion_tags,
            num_sentiment_classes=num_sentiment_classes,
            **kwargs,
        )
    else:
        return model_registry[model_type](vocab_size=vocab_size, **kwargs)

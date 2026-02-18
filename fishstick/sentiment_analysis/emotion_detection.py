"""
Emotion detection module.

Provides neural network architectures for detecting emotions in text,
including Ekman-based, Plutchik-based, and dimensional emotion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


EMOTION_LABELS_EKMAN = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral",
]
EMOTION_LABELS_PLUTCHIK = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
    "neutral",
]
EMOTION_LABELS_BASIC = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgusted",
    "neutral",
]


class EmotionClassifier(nn.Module):
    """Base emotion classifier with configurable emotion set."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
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

        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        logits = self.classifier(context)

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


class TransformerEmotionClassifier(nn.Module):
    """Transformer-based emotion classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_classes: int = 7,
        dropout: float = 0.2,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        embeddings = self.embedding(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)

        hidden = embeddings + position_embeds
        hidden = self.dropout(hidden)

        if mask is not None:
            mask = ~mask

        encoded = self.encoder(hidden, src_key_padding_mask=mask)

        pooled = encoded[:, 0, :]

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


class MultiLabelEmotionClassifier(nn.Module):
    """Multi-label emotion classifier for detecting multiple emotions simultaneously."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
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

        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        logits = self.classifier(context)

        if labels is not None and self.training:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            return loss

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, mask=mask)
        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).long()
        return predictions, probs


class DimensionalEmotionModel(nn.Module):
    """Dimensional emotion model (valence, arousal, dominance)."""

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
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.valence = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.arousal = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.dominance = nn.Sequential(
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
        pooled = self.dropout(pooled)

        valence_pred = self.valence(pooled).squeeze(-1)
        arousal_pred = self.arousal(pooled).squeeze(-1)
        dominance_pred = self.dominance(pooled).squeeze(-1)

        if labels is not None and self.training:
            loss = F.mse_loss(valence_pred, labels[:, 0])
            loss += F.mse_loss(arousal_pred, labels[:, 1])
            loss += F.mse_loss(dominance_pred, labels[:, 2])
            return loss

        return torch.stack([valence_pred, arousal_pred, dominance_pred], dim=-1)

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        outputs = self.forward(input_ids, mask=mask)
        return outputs


class EmotionGraphNetwork(nn.Module):
    """Graph neural network for emotion detection using emotion relations."""

    def __init__(
        self,
        vocab_size: int,
        num_emotions: int = 7,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_emotions = num_emotions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.emotion_embedding = nn.Embedding(num_emotions, hidden_dim * 2)

        self.graph_layer = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=4, dropout=dropout, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        emotion_hints: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        text_repr = lstm_out.mean(dim=1).unsqueeze(1)

        if emotion_hints is not None:
            emotion_embeds = self.emotion_embedding(emotion_hints)
            emotion_context, _ = self.graph_layer(
                text_repr, emotion_embeds, emotion_embeds
            )
        else:
            emotion_embeds = self.emotion_embedding(
                torch.arange(self.num_emotions, device=input_ids.device)
            ).unsqueeze(0)
            emotion_context, _ = self.graph_layer(
                text_repr, emotion_embeds, emotion_embeds
            )

        combined = torch.cat([text_repr, emotion_context], dim=-1).squeeze(1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        emotion_hints: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, emotion_hints, mask=mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class ContextAwareEmotionClassifier(nn.Module):
    """Emotion classifier that considers conversational context."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 7,
        max_history: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_history = max_history

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.context_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.utterance_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        context_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_embeds = self.embedding(input_ids)
        current_embeds = self.dropout(current_embeds)

        current_out, _ = self.utterance_lstm(current_embeds)
        current_repr = current_out.mean(dim=1)

        if context_ids is not None:
            batch_size, num_utts, seq_len = context_ids.shape

            context_flat = context_ids.view(-1, seq_len)
            context_embeds = self.embedding(context_flat)
            context_embeds = self.dropout(context_embeds)

            context_out, _ = self.context_lstm(context_embeds)
            context_repr = context_out.view(batch_size, num_utts, -1).mean(dim=1)
        else:
            context_repr = torch.zeros(
                current_repr.shape[0], self.hidden_dim * 2, device=current_repr.device
            )

        combined = torch.cat([current_repr, context_repr], dim=-1)
        combined = self.fusion(combined)
        combined = self.dropout(combined)

        logits = self.classifier(combined)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        context_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, context_ids)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class MultiTaskEmotionModel(nn.Module):
    """Multi-task model for emotion classification and sentiment."""

    def __init__(
        self,
        vocab_size: int,
        num_emotion_classes: int = 7,
        num_sentiment_classes: int = 3,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_emotion_classes = num_emotion_classes
        self.num_sentiment_classes = num_sentiment_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        shared_dim = hidden_dim * 2

        self.emotion_classifier = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotion_classes),
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sentiment_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        emotion_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)

        pooled = lstm_out.mean(dim=1)
        pooled = self.dropout(pooled)

        emotion_logits = self.emotion_classifier(pooled)
        sentiment_logits = self.sentiment_classifier(pooled)

        total_loss = None
        if self.training:
            loss = 0.0
            if emotion_labels is not None:
                loss += F.cross_entropy(emotion_logits, emotion_labels)
            if sentiment_labels is not None:
                loss += F.cross_entropy(sentiment_logits, sentiment_labels)
            total_loss = loss

        return emotion_logits, sentiment_logits, total_loss

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emotion_logits, sentiment_logits, _ = self.forward(input_ids, mask=mask)

        emotion_probs = F.softmax(emotion_logits, dim=-1)
        sentiment_probs = F.softmax(sentiment_logits, dim=-1)

        emotion_preds = emotion_logits.argmax(dim=-1)
        sentiment_preds = sentiment_logits.argmax(dim=-1)

        return emotion_preds, emotion_probs, sentiment_preds, sentiment_probs


def build_emotion_model(
    vocab_size: int,
    model_type: str = "lstm",
    emotion_scheme: str = "ekman",
    num_classes: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Build an emotion detection model by type."""
    if num_classes is None:
        if emotion_scheme == "ekman":
            num_classes = 7
        elif emotion_scheme == "plutchik":
            num_classes = 9
        elif emotion_scheme == "basic":
            num_classes = 7
        else:
            num_classes = 7

    model_registry = {
        "lstm": EmotionClassifier,
        "transformer": TransformerEmotionClassifier,
        "multilabel": MultiLabelEmotionClassifier,
        "dimensional": DimensionalEmotionModel,
        "graph": EmotionGraphNetwork,
        "context": ContextAwareEmotionClassifier,
        "multitask": MultiTaskEmotionModel,
    }

    if model_type not in model_registry:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(model_registry.keys())}"
        )

    if model_type in ("dimensional",):
        return model_registry[model_type](vocab_size=vocab_size, **kwargs)
    elif model_type == "multitask":
        return model_registry[model_type](
            vocab_size=vocab_size,
            num_emotion_classes=num_classes,
            **kwargs,
        )
    else:
        return model_registry[model_type](
            vocab_size=vocab_size, num_classes=num_classes, **kwargs
        )

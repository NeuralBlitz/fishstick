"""
Multi-class sentiment analysis module.

Provides models for fine-grained sentiment classification with multiple sentiment classes,
including rating prediction, star classification, and ordinal sentiment models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class FineGrainedSentimentClassifier(nn.Module):
    """Multi-class sentiment classifier with ordinal regression support."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 5,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_ordinal: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_ordinal = use_ordinal

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

        if use_ordinal:
            self.cumulative_classifier = nn.Linear(hidden_dim * 2, num_classes - 1)
        else:
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)

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

        if self.use_ordinal:
            cumulative_logits = self.cumulative_classifier(pooled)

            if labels is not None and self.training:
                loss = self._ordinal_loss(cumulative_logits, labels)
                return loss

            return cumulative_logits
        else:
            logits = self.classifier(pooled)

            if labels is not None and self.training:
                loss = F.cross_entropy(logits, labels)
                return loss

            return logits

    def _ordinal_loss(
        self, cumulative_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        batch_size = labels.size(0)
        device = labels.device

        labels_oh = torch.zeros(batch_size, self.num_classes, device=device)
        labels_oh.scatter_(1, labels.unsqueeze(1), 1)

        cumulative_probs = torch.sigmoid(cumulative_logits)
        cumulative_probs = torch.cat(
            [
                torch.ones(batch_size, 1, device=device),
                cumulative_probs,
                torch.zeros(batch_size, 1, device=device),
            ],
            dim=1,
        )

        class_probs = cumulative_probs[:, 1:] - cumulative_probs[:, :-1]

        class_probs = class_probs + 1e-8

        loss = -torch.sum(labels_oh * torch.log(class_probs), dim=1).mean()

        return loss

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.forward(input_ids, mask=mask)

        if self.use_ordinal:
            probs = torch.sigmoid(output)
            predictions = (probs > 0.5).sum(dim=1)
            predictions = torch.clamp(predictions, 0, self.num_classes - 1)
        else:
            probs = F.softmax(output, dim=-1)
            predictions = output.argmax(dim=-1)

        return predictions, probs


class StarRatingClassifier(nn.Module):
    """Star rating classifier for reviews (1-5 stars)."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        if mask is not None:
            mask = ~mask

        encoded = self.encoder(embeddings, src_key_padding_mask=mask)

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


class RegressionSentimentRegressor(nn.Module):
    """Sentiment regressor returning continuous sentiment score."""

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
            nn.Tanh(),
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

        score = self.regressor(pooled).squeeze(-1)

        if labels is not None and self.training:
            loss = F.mse_loss(score, labels)
            return loss

        return score

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.forward(input_ids, mask=mask)


class MultiLabelSentimentClassifier(nn.Module):
    """Multi-label sentiment classifier for multiple sentiment dimensions."""

    def __init__(
        self,
        vocab_size: int,
        num_sentiments: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_sentiments = num_sentiments

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.sentiment_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 3),
                )
                for _ in range(num_sentiments)
            ]
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

        logits_list = []
        for classifier in self.sentiment_classifiers:
            logits_list.append(classifier(pooled))

        logits = torch.stack(logits_list, dim=1)

        logits = logits.view(-1, 3)

        if labels is not None and self.training:
            labels_flat = labels.view(-1)
            loss = F.cross_entropy(logits, labels_flat)
            return loss

        return logits.view(-1, self.num_sentiments, 3)

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, mask=mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class EnsembleSentimentClassifier(nn.Module):
    """Ensemble of multiple sentiment classifiers."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 3,
        num_models: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_models = num_models

        self.models = nn.ModuleList(
            [
                BiLSTMSentimentModel(
                    vocab_size, embedding_dim, hidden_dim, num_classes, dropout
                )
                for _ in range(num_models)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        all_logits = []
        for model in self.models:
            logits = model(input_ids, mask=mask)
            all_logits.append(logits)

        ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)

        if labels is not None and self.training:
            loss = F.cross_entropy(ensemble_logits, labels)
            return loss

        return ensemble_logits

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, mask=mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class BiLSTMSentimentModel(nn.Module):
    """BiLSTM model for ensemble."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None):
        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        pooled = lstm_out.mean(dim=1)
        return self.classifier(self.dropout(pooled))


class GaussianProcessSentimentClassifier(nn.Module):
    """Bayesian sentiment classifier with dropout for uncertainty estimation."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.5,
        n_dropout_samples: int = 10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_dropout_samples = n_dropout_samples

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
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
        return_uncertainty: bool = False,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)

        pooled = lstm_out.mean(dim=1)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        if return_uncertainty and not self.training:
            return self._compute_uncertainty(input_ids, mask)

        return logits

    def _compute_uncertainty(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()

        all_probs = []
        with torch.no_grad():
            for _ in range(self.n_dropout_samples):
                logits = self.forward(input_ids, mask=mask)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs)

        mean_probs = all_probs.mean(dim=0)
        uncertainty = all_probs.std(dim=0).mean(dim=-1)

        predictions = mean_probs.argmax(dim=-1)

        return predictions, uncertainty


class ContrastiveSentimentClassifier(nn.Module):
    """Contrastive learning based sentiment classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 3,
        temperature: float = 0.1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.temperature = temperature

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.encoder(embeddings)

        pooled = lstm_out.mean(dim=1)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.encoder(embeddings)
        pooled = lstm_out.mean(dim=1)
        return self.projector(pooled)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for sentiment classification."""

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(predictions, dim=-1)

        targets_one_hot = torch.zeros_like(predictions).scatter_(
            1, targets.unsqueeze(1), 1
        )

        smoothed_targets = targets_one_hot * self.confidence + (
            1 - targets_one_hot
        ) * self.smoothing / (self.num_classes - 1)

        loss = (-smoothed_targets * log_probs).sum(dim=-1).mean()

        return loss


def build_multiclass_sentiment_model(
    vocab_size: int,
    model_type: str = "lstm",
    num_classes: int = 5,
    **kwargs,
) -> nn.Module:
    """Build a multi-class sentiment model by type."""
    model_registry = {
        "lstm": FineGrainedSentimentClassifier,
        "transformer": StarRatingClassifier,
        "ordinal": FineGrainedSentimentClassifier,
        "regression": RegressionSentimentRegressor,
        "multilabel": MultiLabelSentimentClassifier,
        "ensemble": EnsembleSentimentClassifier,
        "bayesian": GaussianProcessSentimentClassifier,
        "contrastive": ContrastiveSentimentClassifier,
    }

    if model_type not in model_registry:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(model_registry.keys())}"
        )

    return model_registry[model_type](
        vocab_size=vocab_size, num_classes=num_classes, **kwargs
    )

"""
Document-level sentiment analysis module.

Provides neural network architectures for classifying sentiment at the document level,
including LSTM, Transformer, and multi-modal approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math


class AttentionPooling(nn.Module):
    """Attention-based pooling layer for document representations."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention_scores = self.attention(hidden_states).squeeze(-1)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=1)
        pooled = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)

        return pooled


class BiLSTMDocumentSentiment(nn.Module):
    """BiLSTM-based document sentiment classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = AttentionPooling(hidden_dim * 2)
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
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = input_ids != 0

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        lstm_out, _ = self.lstm(embeddings)

        document_repr = self.attention(lstm_out, mask)
        document_repr = self.dropout(document_repr)

        logits = self.classifier(document_repr)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class TransformerDocumentSentiment(nn.Module):
    """Transformer-based document sentiment classifier."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.2,
        max_seq_length: int = 512,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

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

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        token_embeds = self.embedding(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)

        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        if mask is not None:
            mask = ~mask

        encoded = self.encoder(hidden_states, src_key_padding_mask=mask)

        pooled = encoded.mean(dim=1)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class CNNDocumentSentiment(nn.Module):
    """CNN-based document sentiment classifier with multiple filter sizes."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        num_filters: int = 100,
        filter_sizes: Tuple[int, ...] = (2, 3, 4, 5),
        num_classes: int = 3,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, fs) for fs in filter_sizes]
        )

        self.dropout = nn.Dropout(dropout)

        total_filters = num_filters * len(filter_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = self.dropout(embeddings)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embeddings))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        concat = torch.cat(conv_outputs, dim=1)
        concat = self.dropout(concat)

        logits = self.classifier(concat)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def predict(
        self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class HierarchicalDocumentSentiment(nn.Module):
    """Hierarchical attention network for long document sentiment."""

    def __init__(
        self,
        vocab_size: int,
        word_embedding_dim: int = 128,
        sentence_embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
        max_sentences: int = 50,
        max_words: int = 30,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.sentence_embedding_dim = sentence_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_sentences = max_sentences
        self.max_words = max_words

        self.word_embedding = nn.Embedding(
            vocab_size, word_embedding_dim, padding_idx=0
        )

        self.word_lstm = nn.LSTM(
            word_embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.word_attention = AttentionPooling(hidden_dim * 2)

        self.sentence_lstm = nn.LSTM(
            hidden_dim * 2,
            sentence_embedding_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.sentence_attention = AttentionPooling(sentence_embedding_dim * 2)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(sentence_embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_sentences, num_words = input_ids.shape

        input_ids_flat = input_ids.view(-1, num_words)
        word_embeds = self.word_embedding(input_ids_flat)
        word_embeds = self.dropout(word_embeds)

        word_mask = input_ids_flat != 0
        word_lstm_out, _ = self.word_lstm(word_embeds)

        sentence_reprs = []
        for i in range(batch_size * num_sentences):
            sent_repr = self.word_attention(
                word_lstm_out[i : i + 1], word_mask[i : i + 1]
            )
            sentence_reprs.append(sent_repr)

        sentence_reprs = torch.stack(sentence_reprs).view(batch_size, num_sentences, -1)

        sentence_mask = (input_ids != 0).any(dim=2)
        sentence_lstm_out, _ = self.sentence_lstm(sentence_reprs)

        document_repr = self.sentence_attention(sentence_lstm_out, sentence_mask)
        document_repr = self.dropout(document_repr)

        logits = self.classifier(document_repr)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def predict(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


class SentimentBERTEncoder(nn.Module):
    """BERT-style encoder for sentiment analysis with CLS token pooling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.token_type_embedding = nn.Embedding(2, embedding_dim)

        self.embedding_norm = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        token_embeds = self.token_embedding(input_ids)

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embedding(token_type_ids)

        hidden_states = token_embeds + position_embeds + token_type_embeds
        hidden_states = self.embedding_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if attention_mask is not None:
            attention_mask = ~attention_mask

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)

        hidden_states = self.final_norm(hidden_states)

        return hidden_states


class MultiModalSentiment(nn.Module):
    """Multi-modal sentiment classifier combining text and optional auxiliary features."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        text_encoder_layers: int = 4,
        num_classes: int = 3,
        dropout: float = 0.3,
        aux_dim: int = 0,
    ):
        super().__init__()
        self.aux_dim = aux_dim
        self.num_classes = num_classes

        self.text_encoder = TransformerDocumentSentiment(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=text_encoder_layers,
            num_classes=embedding_dim,
            dropout=dropout,
        )

        text_repr_dim = embedding_dim

        if aux_dim > 0:
            self.aux_fc = nn.Sequential(
                nn.Linear(aux_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fusion_dim = text_repr_dim + hidden_dim
        else:
            fusion_dim = text_repr_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_logits = self.text_encoder(input_ids, mask)

        if self.aux_dim > 0 and aux_features is not None:
            aux_repr = self.aux_fc(aux_features)
            combined = torch.cat([text_logits, aux_repr], dim=-1)
        else:
            combined = text_logits

        logits = self.classifier(combined)

        if labels is not None and self.training:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        aux_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, aux_features, mask)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return predictions, probs


def build_document_sentiment_model(
    vocab_size: int,
    model_type: str = "lstm",
    num_classes: int = 3,
    **kwargs,
) -> nn.Module:
    """Build a document sentiment model by type."""
    model_registry = {
        "lstm": BiLSTMDocumentSentiment,
        "transformer": TransformerDocumentSentiment,
        "cnn": CNNDocumentSentiment,
        "hierarchical": HierarchicalDocumentSentiment,
    }

    if model_type not in model_registry:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(model_registry.keys())}"
        )

    return model_registry[model_type](
        vocab_size=vocab_size, num_classes=num_classes, **kwargs
    )

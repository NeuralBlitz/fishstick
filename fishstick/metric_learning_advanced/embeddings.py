from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LearnableMetricConfig:
    input_dim: int = 512
    embedding_dim: int = 128
    hidden_dim: int = 256


@dataclass
class MahalanobisConfig:
    input_dim: int = 512
    embedding_dim: int = 128
    num_classes: int = 1000


@dataclass
class LSTMEmbeddingConfig:
    input_dim: int = 512
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    bidirectional: bool = True


class LearnableMetric(nn.Module):
    def __init__(
        self, input_dim: int = 512, embedding_dim: int = 128, hidden_dim: int = 256
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.scale = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings * self.scale

    def get_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)
        return F.pairwise_distance(emb1, emb2)


class MahalanobisMetric(nn.Module):
    def __init__(
        self, input_dim: int = 512, embedding_dim: int = 128, num_classes: int = 1000
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        self.class_means = nn.Parameter(torch.zeros(num_classes, embedding_dim))
        self.register_buffer("precision", torch.eye(embedding_dim))

        self.initialized = torch.zeros(num_classes, dtype=torch.bool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        return F.normalize(embeddings, p=2, dim=1)

    def update_class_mean(self, embeddings: torch.Tensor, labels: torch.Tensor):
        unique_labels = labels.unique()

        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]

            mean = class_embeddings.mean(dim=0)
            self.class_means[label] = mean
            self.initialized[label] = True

    def compute_precision(self, embeddings: torch.Tensor, labels: torch.Tensor):
        unique_labels = labels.unique()
        all_embeddings = []

        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]
            centered = class_embeddings - self.class_means[label]
            all_embeddings.append(centered)

        combined = torch.cat(all_embeddings, dim=0)
        cov = torch.matmul(combined.t(), combined) / (combined.size(0) - 1)

        self.precision = torch.inverse(
            cov + 1e-6 * torch.eye(cov.size(0), device=cov.device)
        )

    def get_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)

        diff = emb1 - emb2
        mahal = torch.sqrt(torch.sum(diff @ self.precision * diff, dim=1))
        return mahal


class LSTMEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3 if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_projection = nn.Linear(lstm_output_dim, embedding_dim)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = self.input_projection(x)

        if lengths is not None:
            lengths = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        embeddings = self.output_projection(lstm_out)

        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(
                0
            ) < lengths.unsqueeze(1)
            embeddings = embeddings * mask.unsqueeze(2)
            embeddings = embeddings.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            embeddings = embeddings.mean(dim=1)

        return F.normalize(embeddings, p=2, dim=1)


class BiLSTMAttentionEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0,
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.output_projection = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        x = self.input_projection(x)

        if lengths is not None:
            lengths = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)

        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        if lengths is not None:
            mask = (
                (
                    torch.arange(seq_len, device=x.device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                .float()
                .unsqueeze(2)
            )
            attention_weights = attention_weights * mask

        context = (lstm_out * attention_weights).sum(dim=1)

        embeddings = self.output_projection(context)
        return F.normalize(embeddings, p=2, dim=1)


class TemporalPooling(nn.Module):
    def __init__(
        self, input_dim: int = 512, output_dim: int = 128, pool_type: str = "mean"
    ):
        super().__init__()
        self.pool_type = pool_type
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(
                0
            ) < lengths.unsqueeze(1)
            x = x * mask.unsqueeze(2)

            if self.pool_type == "mean":
                pooled = x.sum(dim=1) / lengths.unsqueeze(1).float()
            elif self.pool_type == "max":
                x = x.masked_fill(~mask.unsqueeze(2), float("-inf"))
                pooled = x.max(dim=1)[0]
            else:
                pooled = x.mean(dim=1)
        else:
            if self.pool_type == "mean":
                pooled = x.mean(dim=1)
            elif self.pool_type == "max":
                pooled = x.max(dim=1)[0]
            else:
                pooled = x.mean(dim=1)

        return F.normalize(self.projection(pooled), p=2, dim=1)

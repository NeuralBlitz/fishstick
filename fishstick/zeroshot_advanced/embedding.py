from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttributeEmbeddingConfig:
    image_dim: int = 2048
    attribute_dim: int = 312
    embedding_dim: int = 1024
    num_attributes: int = 312
    dropout: float = 0.3


@dataclass
class ClassEmbeddingConfig:
    attribute_dim: int = 312
    embedding_dim: int = 1024
    num_seen_classes: int = 100
    num_all_classes: int = 1000
    dropout: float = 0.3


@dataclass
class CLIPLikeConfig:
    image_dim: int = 2048
    text_dim: int = 512
    embedding_dim: int = 512
    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.2


class AttributeEmbedding(nn.Module):
    def __init__(self, config: AttributeEmbeddingConfig):
        super().__init__()
        self.config = config

        self.image_encoder = nn.Sequential(
            nn.Linear(config.image_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.attribute_predictor = nn.Linear(
            config.embedding_dim, config.num_attributes
        )

        self.attribute_embeddings = nn.Parameter(
            torch.randn(config.num_attributes, config.attribute_dim)
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        encoded = self.image_encoder(image_features)
        predicted_attributes = self.attribute_predictor(encoded)
        return encoded, predicted_attributes

    def get_attribute_embedding(
        self, predicted_attributes: torch.Tensor
    ) -> torch.Tensor:
        weights = F.softmax(predicted_attributes, dim=-1)
        attribute_embedding = torch.matmul(weights, self.attribute_embeddings)
        return attribute_embedding


class ClassEmbedding(nn.Module):
    def __init__(self, config: ClassEmbeddingConfig):
        super().__init__()
        self.config = config

        self.attribute_encoder = nn.Sequential(
            nn.Linear(config.attribute_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        self.seen_class_embeddings = nn.Parameter(
            torch.randn(config.num_seen_classes, config.embedding_dim)
        )

        self.unseen_projection = nn.Linear(config.attribute_dim, config.embedding_dim)

    def forward(
        self,
        class_attributes: torch.Tensor,
        use_seen: bool = True,
        class_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_seen and class_idx is not None:
            return F.embedding(class_idx, self.seen_class_embeddings)
        else:
            return self.attribute_encoder(class_attributes)

    def compute_class_embedding(self, class_attributes: torch.Tensor) -> torch.Tensor:
        return self.attribute_encoder(class_attributes)


class CLIPLikeEncoder(nn.Module):
    def __init__(self, config: CLIPLikeConfig):
        super().__init__()
        self.config = config

        self.image_projection = nn.Sequential(
            nn.Linear(config.image_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(config.text_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )

        self.image_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
                batch_first=True,
            ),
            num_layers=config.num_layers,
        )

        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim,
                dropout=config.dropout,
                batch_first=True,
            ),
            num_layers=config.num_layers,
        )

        self.temperature = nn.Parameter(torch.ones(1) * 2.6592)

    def encode_image(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)

        projected = self.image_projection(image_features)
        transformed = self.image_transformer(projected)
        embeddings = F.normalize(transformed, p=2, dim=-1)

        if embeddings.size(1) == 1:
            embeddings = embeddings.squeeze(1)
        return embeddings

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)

        projected = self.text_projection(text_features)
        transformed = self.text_transformer(projected)
        embeddings = F.normalize(transformed, p=2, dim=-1)

        if embeddings.size(1) == 1:
            embeddings = embeddings.squeeze(1)
        return embeddings

    def forward(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.encode_image(image_features)
        text_embeddings = self.encode_text(text_features)
        return image_embeddings, text_embeddings

    def compute_similarity(
        self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(image_embeddings, text_embeddings.T) / self.temperature

    def contrastive_loss(
        self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        logits = self.compute_similarity(image_embeddings, text_embeddings)

        batch_size = image_embeddings.size(0)
        labels = torch.arange(batch_size, device=image_embeddings.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)

        return (loss_i + loss_t) / 2

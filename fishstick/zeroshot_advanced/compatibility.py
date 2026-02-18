from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    image_dim: int = 2048
    attribute_dim: int = 312
    embedding_dim: int = 1024
    latent_dim: int = 512
    dropout: float = 0.3


@dataclass
class SJEConfig:
    image_dim: int = 2048
    attribute_dim: int = 312
    embedding_dim: int = 1024
    num_classes: int = 1000
    dropout: float = 0.3


@dataclass
class ALEConfig:
    image_dim: int = 2048
    attribute_dim: int = 312
    embedding_dim: int = 1024
    num_classes: int = 1000
    dropout: float = 0.3


@dataclass
class CalibratorConfig:
    embedding_dim: int = 1024
    hidden_dim: int = 512
    num_calibration_levels: int = 3
    dropout: float = 0.2


class SAE(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Linear(config.image_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.image_dim),
        )

        self.attribute_projection = nn.Linear(config.latent_dim, config.attribute_dim)

        self.W = nn.Parameter(torch.randn(config.latent_dim, config.attribute_dim))
        self.b = nn.Parameter(torch.zeros(config.attribute_dim))

    def forward(
        self, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(image_features)
        reconstructed = self.decoder(latent)
        attribute_pred = torch.matmul(latent, self.W) + self.b
        return latent, reconstructed, attribute_pred

    def get_compatibility_score(
        self, image_features: torch.Tensor, class_attributes: torch.Tensor
    ) -> torch.Tensor:
        latent = self.encoder(image_features)
        compatibility = torch.matmul(latent, class_attributes.T)
        return compatibility

    def reconstruct_attribute(self, latent: torch.Tensor) -> torch.Tensor:
        return self.attribute_projection(latent)

    def semantic_loss(
        self, latent: torch.Tensor, target_attributes: torch.Tensor
    ) -> torch.Tensor:
        pred_attributes = torch.matmul(latent, self.W) + self.b
        return F.mse_loss(pred_attributes, target_attributes)

    def reconstruction_loss(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(original, reconstructed)


class SJE(nn.Module):
    def __init__(self, config: SJEConfig):
        super().__init__()
        self.config = config

        self.image_encoder = nn.Sequential(
            nn.Linear(config.image_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        self.attribute_encoder = nn.Sequential(
            nn.Linear(config.attribute_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        self.class_embeddings = nn.Parameter(
            torch.randn(config.num_classes, config.embedding_dim)
        )

    def forward(
        self,
        image_features: torch.Tensor,
        class_attributes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image_embedding = self.image_encoder(image_features)

        if class_attributes is not None:
            class_embedding = self.attribute_encoder(class_attributes)
        else:
            class_embedding = F.normalize(self.class_embeddings, p=2, dim=1)

        return image_embedding, class_embedding

    def get_compatibility_score(
        self, image_embedding: torch.Tensor, class_embedding: torch.Tensor
    ) -> torch.Tensor:
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
        class_embedding = F.normalize(class_embedding, p=2, dim=1)
        return torch.matmul(image_embedding, class_embedding.T)

    def predict(
        self, image_features: torch.Tensor, class_attributes: torch.Tensor
    ) -> torch.Tensor:
        image_embedding = self.image_encoder(image_features)
        class_embedding = self.attribute_encoder(class_attributes)

        scores = self.get_compatibility_score(image_embedding, class_embedding)
        predictions = torch.argmax(scores, dim=-1)

        return predictions


class ALE(nn.Module):
    def __init__(self, config: ALEConfig):
        super().__init__()
        self.config = config

        self.image_encoder = nn.Sequential(
            nn.Linear(config.image_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        self.attribute_weights = nn.Parameter(
            torch.randn(config.embedding_dim, config.attribute_dim)
        )
        self.attribute_bias = nn.Parameter(torch.zeros(config.attribute_dim))

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_embedding = self.image_encoder(image_features)
        return image_embedding

    def get_compatibility_score(
        self, image_features: torch.Tensor, class_attributes: torch.Tensor
    ) -> torch.Tensor:
        image_embedding = self.image_encoder(image_features)

        image_embedding_norm = F.normalize(image_embedding, p=2, dim=1)
        class_attributes_norm = F.normalize(class_attributes, p=2, dim=1)

        scores = torch.matmul(
            torch.matmul(image_embedding_norm, self.attribute_weights),
            class_attributes_norm.T,
        )

        return scores

    def bilinear_compatibility(
        self, image_embedding: torch.Tensor, class_attribute: torch.Tensor
    ) -> torch.Tensor:
        intermediate = torch.matmul(image_embedding, self.attribute_weights)
        score = torch.sum(intermediate * class_attribute, dim=-1)
        return score


class EmbeddingCalibrator(nn.Module):
    def __init__(self, config: CalibratorConfig):
        super().__init__()
        self.config = config

        self.calibration_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.embedding_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.embedding_dim),
                )
                for _ in range(config.num_calibration_levels)
            ]
        )

        self.gate = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim), nn.Sigmoid()
        )

    def forward(self, embeddings: torch.Tensor, level: int = 0) -> torch.Tensor:
        if level >= len(self.calibration_layers):
            level = len(self.calibration_layers) - 1

        calibrated = self.calibration_layers[level](embeddings)
        gate_values = self.gate(embeddings)

        output = gate_values * calibrated + (1 - gate_values) * embeddings

        return output

    def calibrate(
        self,
        embeddings: torch.Tensor,
        target_distribution: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        calibrated = embeddings

        for layer in self.calibration_layers:
            residual = layer(calibrated)
            gate_values = self.gate(calibrated)
            calibrated = gate_values * residual + (1 - gate_values) * calibrated

        if target_distribution is not None:
            calibrated = self.match_distribution(calibrated, target_distribution)

        return calibrated

    def match_distribution(
        self, embeddings: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        source_mean = embeddings.mean(dim=0, keepdim=True)
        source_std = embeddings.std(dim=0, keepdim=True) + 1e-8

        target_mean = target.mean(dim=0, keepdim=True)
        target_std = target.std(dim=0, keepdim=True) + 1e-8

        normalized = (embeddings - source_mean) / source_std
        calibrated = normalized * target_std + target_mean

        return calibrated

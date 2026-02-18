from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CVAEConfig:
    image_dim: int = 2048
    attribute_dim: int = 312
    embedding_dim: int = 256
    latent_dim: int = 128
    hidden_dim: int = 512
    num_classes: int = 100
    dropout: float = 0.3


@dataclass
class GenerativeZSLConfig:
    image_dim: int = 2048
    attribute_dim: int = 312
    embedding_dim: int = 256
    latent_dim: int = 128
    hidden_dim: int = 512
    num_seen_classes: int = 50
    num_all_classes: int = 1000
    dropout: float = 0.3


class CVAEEncoder(nn.Module):
    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.config = config

        self.image_encoder = nn.Sequential(
            nn.Linear(config.image_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.attribute_encoder = nn.Sequential(
            nn.Linear(config.attribute_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        combined_dim = config.hidden_dim + config.hidden_dim

        self.mu_layer = nn.Linear(combined_dim, config.latent_dim)
        self.logvar_layer = nn.Linear(combined_dim, config.latent_dim)

    def forward(
        self, image_features: torch.Tensor, class_attributes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_encoded = self.image_encoder(image_features)
        attr_encoded = self.attribute_encoder(class_attributes)

        combined = torch.cat([image_encoded, attr_encoded], dim=-1)

        mu = self.mu_layer(combined)
        logvar = self.logvar_layer(combined)

        return mu, logvar


class CVAEDecoder(nn.Module):
    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.config = config

        self.latent_to_hidden = nn.Sequential(
            nn.Linear(config.latent_dim + config.attribute_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.image_dim),
        )

    def forward(
        self, latent: torch.Tensor, class_attributes: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([latent, class_attributes], dim=-1)
        hidden = self.latent_to_hidden(combined)
        reconstructed = self.output_layer(hidden)
        return reconstructed


class CVAEZeroShot(nn.Module):
    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.config = config

        self.encoder = CVAEEncoder(config)
        self.decoder = CVAEDecoder(config)

        self.prior_mu = nn.Parameter(torch.zeros(config.latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(config.latent_dim))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self, image_features: torch.Tensor, class_attributes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(image_features, class_attributes)
        latent = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(latent, class_attributes)

        return reconstructed, mu, logvar, latent

    def generate(
        self, class_attributes: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        prior_std = torch.exp(0.5 * self.prior_logvar)

        batch_size = class_attributes.size(0)
        latent = self.prior_mu.unsqueeze(0).expand(num_samples * batch_size, -1)
        latent_std = prior_std.unsqueeze(0).expand(num_samples * batch_size, -1)
        eps = torch.randn_like(latent_std)
        latent = latent + eps * latent_std

        class_attrs_repeated = class_attributes.unsqueeze(0).expand(num_samples, -1, -1)
        class_attrs_repeated = class_attrs_repeated.contiguous().view(
            num_samples * batch_size, -1
        )

        generated = self.decoder(latent, class_attrs_repeated)

        return generated.view(num_samples, batch_size, -1)

    def compute_loss(
        self,
        image_features: torch.Tensor,
        class_attributes: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        reconstructed, mu, logvar, latent = self.forward(
            image_features, class_attributes
        )

        recon_loss = F.mse_loss(reconstructed, image_features, reduction="sum")

        prior_mu = self.prior_mu.expand_as(mu)
        prior_logvar = self.prior_logvar.expand_as(logvar)

        kl_loss = -0.5 * torch.sum(
            1
            + logvar
            - prior_logvar
            - (mu - prior_mu).pow(2) / prior_logvar.exp()
            - logvar.exp()
        )

        total_loss = recon_loss + beta * kl_loss

        losses = {
            "total": total_loss,
            "reconstruction": recon_loss,
            "kl_divergence": kl_loss,
        }

        return total_loss, losses


class GenerativeZeroShotClassifier(nn.Module):
    def __init__(self, config: GenerativeZSLConfig):
        super().__init__()
        self.config = config

        self.cvae = CVAEZeroShot(
            CVAEConfig(
                image_dim=config.image_dim,
                attribute_dim=config.attribute_dim,
                embedding_dim=config.embedding_dim,
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
                num_classes=config.num_seen_classes,
                dropout=config.dropout,
            )
        )

        self.seen_class_attributes = nn.Parameter(
            torch.randn(config.num_seen_classes, config.attribute_dim)
        )

        self.unseen_class_attributes = nn.Parameter(
            torch.randn(
                config.num_all_classes - config.num_seen_classes, config.attribute_dim
            )
        )

        self.image_encoder = nn.Sequential(
            nn.Linear(config.image_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        generate_unseen: bool = True,
        num_generated: int = 5,
    ) -> torch.Tensor:
        seen_attributes = F.normalize(self.seen_class_attributes, p=2, dim=1)

        _, _, _, latent = self.cvae(image_features, seen_attributes[:1])
        image_embedding = self.image_encoder(image_features)

        seen_compatibility = torch.matmul(image_embedding, seen_attributes.T)

        if generate_unseen and self.unseen_class_attributes.size(0) > 0:
            generated_features = self.cvae.generate(
                self.unseen_class_attributes, num_samples=num_generated
            )

            _, _, _, generated_latents = self.cvae(
                generated_features.view(-1, self.config.image_dim),
                self.unseen_class_attributes.unsqueeze(1)
                .expand(num_generated, self.unseen_class_attributes.size(0), -1)
                .contiguous()
                .view(-1, self.config.attribute_dim),
            )

            generated_embeddings = self.image_encoder(
                generated_features.view(-1, self.config.image_dim)
            )
            generated_embeddings = generated_embeddings.view(
                num_generated, self.unseen_class_attributes.size(0), -1
            ).mean(dim=0)

            generated_embeddings = F.normalize(generated_embeddings, p=2, dim=1)
            unseen_compatibility = torch.matmul(image_embedding, generated_embeddings.T)

            scores = torch.cat([seen_compatibility, unseen_compatibility], dim=-1)
        else:
            scores = seen_compatibility

        return scores

    def predict(
        self, image_features: torch.Tensor, generate_unseen: bool = True
    ) -> torch.Tensor:
        scores = self.forward(image_features, generate_unseen)
        predictions = torch.argmax(scores, dim=-1)
        return predictions

    def train_step(
        self,
        image_features: torch.Tensor,
        class_labels: torch.Tensor,
        beta: float = 1.0,
    ) -> dict:
        class_attrs = F.embedding(class_labels, self.seen_class_attributes)

        loss, losses = self.cvae.compute_loss(image_features, class_attrs, beta)

        return {"loss": loss, **losses}

    def generate_synthetic_features(
        self, class_id: int, num_samples: int = 10
    ) -> torch.Tensor:
        if class_id < self.config.num_seen_classes:
            class_attrs = self.seen_class_attributes[class_id : class_id + 1]
        else:
            idx = class_id - self.config.num_seen_classes
            if idx >= self.unseen_class_attributes.size(0):
                raise ValueError(f"Class ID {class_id} out of range")
            class_attrs = self.unseen_class_attributes[idx : idx + 1]

        generated = self.cvae.generate(class_attrs, num_samples=num_samples)
        return generated.squeeze(0)

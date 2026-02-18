"""
CLIP: Contrastive Language-Image Pre-Training

Dual encoder architecture for image-text matching with contrastive loss.
Supports zero-shot classification and image/text embedding generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class CLIPConfig:
    image_dim: int = 768
    text_dim: int = 512
    embed_dim: int = 512
    vision_patch_size: int = 32
    vision_width: int = 768
    vision_layers: int = 12
    vision_heads: int = 12
    text_context_length: int = 77
    text_heads: int = 8
    text_layers: int = 12
    temperature_init: float = 0.07


class ImageEncoder(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        self.patch_embed = nn.Conv2d(
            3,
            config.vision_width,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size,
        )

        self.position_embedding = nn.Parameter(
            torch.randn(1, 197, config.vision_width) * 0.02
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.vision_width) * 0.02)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.vision_width,
                nhead=config.vision_heads,
                dim_feedforward=config.vision_width * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=config.vision_layers,
        )

        self.ln_final = nn.LayerNorm(config.vision_width)
        self.projection = nn.Linear(config.vision_width, config.embed_dim)

    def forward(self, images: Tensor) -> Tensor:
        batch_size = images.shape[0]

        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.position_embedding

        x = self.transformer(x)
        x = self.ln_final(x)

        cls_output = x[:, 0]

        embeddings = self.projection(cls_output)
        return F.normalize(embeddings, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, config: CLIPConfig, vocab_size: int = 49408):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(vocab_size, config.text_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.text_context_length, config.text_dim) * 0.02
        )

        self.projection_up = nn.Linear(config.text_dim, config.vision_width)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.vision_width,
                nhead=config.text_heads,
                dim_feedforward=config.vision_width * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=config.text_layers,
        )

        self.ln_final = nn.LayerNorm(config.vision_width)
        self.projection = nn.Linear(config.vision_width, config.embed_dim)

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = tokens.shape[0]

        x = self.token_embedding(tokens)

        seq_len = min(x.shape[1], self.config.text_context_length)
        x = x[:, :seq_len]

        x = x + self.position_embedding[:, :seq_len]

        x = self.projection_up(x)

        if mask is not None:
            mask = mask[:, :seq_len]

        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.ln_final(x)

        pooled = x.mean(dim=1)

        embeddings = self.projection(pooled)
        return F.normalize(embeddings, dim=-1)


class CLIPModel(nn.Module):
    def __init__(self, config: CLIPConfig, vocab_size: int = 49408):
        super().__init__()
        self.config = config

        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config, vocab_size)

        self.temperature = nn.Parameter(torch.tensor(config.temperature_init))

    def encode_image(self, images: Tensor) -> Tensor:
        return self.image_encoder(images)

    def encode_text(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.text_encoder(tokens, mask)

    def forward(
        self, images: Tensor, tokens: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(tokens, mask)
        return image_embeds, text_embeds

    def contrastive_loss(
        self, image_embeds: Tensor, text_embeds: Tensor, labels: Optional[Tensor] = None
    ) -> Tensor:
        logits = torch.matmul(image_embeds, text_embeds.T) * torch.exp(self.temperature)

        batch_size = image_embeds.shape[0]
        if labels is None:
            labels = torch.arange(batch_size, device=image_embeds.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

    def zero_shot_classify(
        self, images: Tensor, class_descriptions: List[str], tokenizer
    ) -> Tensor:
        self.eval()
        with torch.no_grad():
            image_embeds = self.encode_image(images)

            text_tokens = []
            text_masks = []
            for desc in class_descriptions:
                encoded = tokenizer(
                    desc,
                    padding=True,
                    truncation=True,
                    max_length=self.config.text_context_length,
                    return_tensors="pt",
                )
                text_tokens.append(encoded["input_ids"])
                text_masks.append(encoded["attention_mask"])

            text_tokens = torch.cat(text_tokens, dim=0).to(images.device)
            text_masks = torch.cat(text_masks, dim=0).to(images.device)

            text_embeds = self.encode_text(text_tokens, text_masks)

            logits = torch.matmul(image_embeds, text_embeds.T) * torch.exp(
                self.temperature
            )
            probs = F.softmax(logits, dim=-1)

        return probs


def clip_loss(
    image_embeds: Tensor, text_embeds: Tensor, temperature: float = 0.07
) -> Tensor:
    logits = torch.matmul(image_embeds, text_embeds.T) * temperature

    batch_size = image_embeds.shape[0]
    labels = torch.arange(batch_size, device=image_embeds.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2


class CLIPWithProjection(nn.Module):
    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 512,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self, image_features: Tensor, text_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        image_embeds = F.normalize(self.image_projection(image_features), dim=-1)
        text_embeds = F.normalize(self.text_projection(text_features), dim=-1)
        return image_embeds, text_embeds


DEFAULT_CLIP_CONFIG = CLIPConfig()


__all__ = [
    "CLIPConfig",
    "ImageEncoder",
    "TextEncoder",
    "CLIPModel",
    "clip_loss",
    "CLIPWithProjection",
    "DEFAULT_CLIP_CONFIG",
]

"""
Vision-Language Models for fishstick

This module provides vision-language models including:
- CLIP-style dual encoders
- BLIP-style image captioning
- Vision-language transformers
"""

from typing import Optional, Tuple, List
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CLIPVisionEncoder(nn.Module):
    """CLIP-style vision encoder with patch embeddings and transformer."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.ln_pre = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)

        return x


class CLIPTextEncoder(nn.Module):
    """CLIP-style text encoder with transformer."""

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_length: int = 77,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, L = x.shape

        x = self.token_embedding(x)
        x = x + self.pos_embedding[:, :L, :]
        x = self.drop(x)

        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.ln(x)

        return x


class CLIPModel(nn.Module):
    """CLIP model for vision-language tasks."""

    def __init__(
        self,
        embed_dim: int = 512,
        image_size: int = 224,
        patch_size: int = 16,
        vision_layers: int = 12,
        text_layers: int = 12,
        vocab_size: int = 49408,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=vision_layers,
            dropout=dropout,
        )
        self.text_encoder = CLIPTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=text_layers,
            dropout=dropout,
        )

        self.image_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / 0.07))
        )

    def encode_image(self, image: Tensor) -> Tensor:
        features = self.vision_encoder(image)
        image_features = features[:, 0, :]
        image_features = self.image_projection(image_features)
        return F.normalize(image_features, dim=-1)

    def encode_text(self, text: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        features = self.text_encoder(text, mask)
        text_features = features[:, 0, :]
        text_features = self.text_projection(text_features)
        return F.normalize(text_features, dim=-1)

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        text_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, text_mask)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class BLIPImageEncoder(nn.Module):
    """BLIP-style image encoder with ViT."""

    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        x = self.transformer(x)
        return self.ln(x)


class BLIPTextEncoder(nn.Module):
    """BLIP-style text encoder."""

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_length: int = 77,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, L = x.shape
        x = self.embedding(x) + self.pos_embedding[:, :L, :]
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.ln(x)


class BLIPModel(nn.Module):
    """BLIP model for image captioning and VQA."""

    def __init__(
        self,
        embed_dim: int = 768,
        image_size: int = 384,
        vocab_size: int = 30522,
    ):
        super().__init__()
        self.image_encoder = BLIPImageEncoder(
            image_size=image_size, embed_dim=embed_dim
        )
        self.text_encoder = BLIPTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)

        self.image_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)

    def encode_image(self, image: Tensor) -> Tensor:
        features = self.image_encoder(image)
        image_emb = self.image_projection(features[:, 0, :])
        return F.normalize(image_emb, dim=-1)

    def encode_text(self, text: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        features = self.text_encoder(text, mask)
        text_emb = self.text_projection(features[:, 0, :])
        return F.normalize(text_emb, dim=-1)

    def forward(
        self, image: Tensor, text: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, mask)
        return image_features, text_features


class VisionLanguageTransformer(nn.Module):
    """Unified vision-language transformer for multi-modal understanding."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_embed = nn.Linear(embed_dim, embed_dim)
        self.text_embed = nn.Linear(embed_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        image_mask: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        image_emb = self.image_embed(image_features)
        text_emb = self.text_embed(text_features)

        combined = torch.cat([image_emb, text_emb], dim=1)
        combined = self.transformer(combined)
        return self.ln(combined)


def create_clip_model(
    embed_dim: int = 512,
    image_size: int = 224,
    **kwargs,
) -> CLIPModel:
    """Factory function to create a CLIP model."""
    return CLIPModel(embed_dim=embed_dim, image_size=image_size, **kwargs)


def create_blip_model(
    embed_dim: int = 768,
    image_size: int = 384,
    **kwargs,
) -> BLIPModel:
    """Factory function to create a BLIP model."""
    return BLIPModel(embed_dim=embed_dim, image_size=image_size, **kwargs)

"""
Multi-Modal Encoders
"""

from typing import Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Image encoder for multi-modal learning."""

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256),
            num_layers=num_layers,
        )

        self.projection = nn.Linear(64, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.projection(x)


class TextEncoder(nn.Module):
    """Text encoder for multi-modal learning."""

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return x.mean(dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class AudioEncoder(nn.Module):
    """Audio encoder for multi-modal learning."""

    def __init__(
        self,
        input_dim: int = 80,
        embed_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=num_layers,
        )

        self.projection = nn.Linear(256, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        return self.projection(x.mean(dim=1))


class MultiModalEncoder(nn.Module):
    """Multi-modal encoder combining different modalities."""

    def __init__(
        self,
        image_channels: int = 3,
        vocab_size: int = 30000,
        audio_dim: int = 80,
        embed_dim: int = 256,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            input_channels=image_channels, embed_dim=embed_dim
        )
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.audio_encoder = AudioEncoder(input_dim=audio_dim, embed_dim=embed_dim)

        self.fusion = nn.Linear(embed_dim * 3, embed_dim)

    def forward(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        audio: Optional[Tensor] = None,
    ) -> Tensor:
        features = []

        if image is not None:
            features.append(self.image_encoder(image))

        if text is not None:
            features.append(self.text_encoder(text))

        if audio is not None:
            features.append(self.audio_encoder(audio))

        if not features:
            raise ValueError("At least one modality must be provided")

        concatenated = torch.cat(features, dim=-1)
        return self.fusion(concatenated)


class CLIPEncoder(nn.Module):
    """CLIP-like encoder for vision-language tasks."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.image_proj = nn.Linear(512, embed_dim)
        self.text_proj = nn.Linear(512, embed_dim)

    def encode_image(self, image_features: Tensor) -> Tensor:
        return F.normalize(self.image_proj(image_features), dim=-1)

    def encode_text(self, text_features: Tensor) -> Tensor:
        return F.normalize(self.text_proj(text_features), dim=-1)

    def forward(self, image: Tensor, text: Tensor) -> Tensor:
        image_emb = self.encode_image(image)
        text_emb = self.encode_text(text)
        return image_emb, text_emb

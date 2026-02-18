from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VideoCLIPConfig:
    embed_dim: int = 512
    video_layers: int = 12
    video_heads: int = 8
    video_hidden_dim: int = 512
    text_layers: int = 12
    text_heads: int = 8
    text_hidden_dim: int = 512
    num_frames: int = 8
    image_size: int = 224
    vocab_size: int = 49408
    dropout: float = 0.0
    attn_dropout: float = 0.0


class VideoPatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VideoCLIPAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = query.shape

        q = (
            self.q_proj(query)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(key)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(value)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dim)
        x = self.out_proj(x)
        return x


class VideoCLIPTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = VideoCLIPAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class VideoCLIPEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                VideoCLIPTransformerBlock(
                    embed_dim, num_heads, mlp_ratio, dropout, attn_dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_seq_len: int = 77,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.encoder = VideoCLIPEncoder(
            embed_dim, num_layers, num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = input_ids.shape

        token_embeds = self.token_embedding(input_ids)

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        pos_embeds = self.position_embedding(positions)

        x = token_embeds + pos_embeds

        for layer in self.encoder.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x


class VideoCLIPProjection(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class VideoCLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        video_layers: int = 12,
        video_heads: int = 8,
        video_hidden_dim: int = 512,
        text_layers: int = 12,
        text_heads: int = 8,
        text_hidden_dim: int = 512,
        num_frames: int = 8,
        image_size: int = 224,
        vocab_size: int = 49408,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.video_patch_embed = VideoPatchEmbed(image_size, 16, 3, video_hidden_dim)
        self.video_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames * ((image_size // 16) ** 2) + 1, video_hidden_dim)
        )
        self.video_cls_token = nn.Parameter(torch.zeros(1, 1, video_hidden_dim))

        self.video_encoder = VideoCLIPEncoder(
            video_hidden_dim,
            video_layers,
            video_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.video_projection = VideoCLIPProjection(video_hidden_dim, embed_dim)

        self.text_encoder = TextEncoder(
            vocab_size, text_hidden_dim, text_layers, text_heads, dropout=dropout
        )
        self.text_projection = VideoCLIPProjection(text_hidden_dim, embed_dim)

        self.temperature = nn.Parameter(torch.ones([]) * 2.6592)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.video_pos_embed, std=0.02)
        nn.init.normal_(self.video_cls_token, std=0.02)

    def encode_video(
        self,
        video: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = video.shape[0]

        video_tokens = self.video_patch_embed(video)

        cls_tokens = self.video_cls_token.expand(B, -1, -1)
        video_tokens = torch.cat([cls_tokens, video_tokens], dim=1)

        video_tokens = video_tokens + self.video_pos_embed

        video_features = self.video_encoder(video_tokens, mask)
        video_features = video_features[:, 0]

        video_features = self.video_projection(video_features)
        video_features = F.normalize(video_features, dim=-1)

        return video_features

    def encode_text(
        self,
        text: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_features = self.text_encoder(text, mask)
        text_features = text_features[:, 0]

        text_features = self.text_projection(text_features)
        text_features = F.normalize(text_features, dim=-1)

        return text_features

    def forward(
        self,
        video: torch.Tensor,
        text: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_features = self.encode_video(video, video_mask)
        text_features = self.encode_text(text, text_mask)

        return video_features, text_features

    def contrastive_loss(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        logits = (video_features @ text_features.t()) * torch.exp(self.temperature)

        labels = torch.arange(len(logits), device=logits.device)

        loss_video = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.t(), labels)

        return (loss_video + loss_text) / 2


class VideoTextRetrieval(nn.Module):
    def __init__(
        self,
        video_embed_dim: int = 512,
        text_embed_dim: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 1,
    ):
        super().__init__()
        self.video_proj = nn.Linear(video_embed_dim, hidden_dim)
        self.text_proj = nn.Linear(text_embed_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        video_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        video_proj = self.video_proj(video_features)
        text_proj = self.text_proj(text_features)

        combined = torch.cat([video_proj, text_proj], dim=-1)
        output = self.classifier(combined)

        return output


class ActionLocalizer(nn.Module):
    def __init__(
        self,
        video_embed_dim: int = 512,
        num_classes: int = 1,
        num_segments: int = 8,
    ):
        super().__init__()
        self.num_segments = num_segments

        self.temporal_conv = nn.Conv1d(
            video_embed_dim, video_embed_dim, kernel_size=3, padding=1
        )

        self.classifier = nn.Sequential(
            nn.Linear(video_embed_dim, video_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(video_embed_dim // 2, num_classes),
        )

    def forward(self, video_features: torch.Tensor) -> torch.Tensor:
        B, T, D = video_features.shape

        x = video_features.transpose(1, 2)
        x = self.temporal_conv(x)
        x = x.transpose(1, 2)
        x = x + video_features

        segment_scores = self.classifier(x)

        return segment_scores


def build_videoclip(config: Optional[VideoCLIPConfig] = None, **kwargs) -> VideoCLIP:
    if config is None:
        config = VideoCLIPConfig(**kwargs)
    return VideoCLIP(
        embed_dim=config.embed_dim,
        video_layers=config.video_layers,
        video_heads=config.video_heads,
        video_hidden_dim=config.video_hidden_dim,
        text_layers=config.text_layers,
        text_heads=config.text_heads,
        text_hidden_dim=config.text_hidden_dim,
        num_frames=config.num_frames,
        image_size=config.image_size,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
    )

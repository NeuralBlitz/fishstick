"""
Audio-Visual Learning for fishstick

This module provides audio-visual learning models:
- Audio-visual correspondence learning
- Audio-visual event detection
- Sound localization
"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AudioEncoder2D(nn.Module):
    """Audio encoder using 2D convolutions on spectrograms."""

    def __init__(
        self,
        input_channels: int = 1,
        embed_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.conv_layers(x)
        return self.projection(features)


class VideoEncoder(nn.Module):
    """Video encoder using 3D convolutions."""

    def __init__(
        self,
        input_channels: int = 3,
        embed_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.conv3d_layers(x)
        return self.projection(features)


class AudioVisualCorrespondence(nn.Module):
    """Audio-visual correspondence learning model."""

    def __init__(
        self,
        audio_embed_dim: int = 512,
        video_embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.audio_encoder = AudioEncoder2D(embed_dim=audio_embed_dim, dropout=dropout)
        self.video_encoder = VideoEncoder(embed_dim=video_embed_dim, dropout=dropout)

        self.correspondence_head = nn.Sequential(
            nn.Linear(audio_embed_dim + video_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        audio_features = self.audio_encoder(audio)
        video_features = self.video_encoder(video)

        combined = torch.cat([audio_features, video_features], dim=-1)
        correspondence_score = self.correspondence_head(combined)

        return audio_features, video_features, correspondence_score.squeeze(-1)


class AudioVisualEventDetector(nn.Module):
    """Audio-visual event detection model."""

    def __init__(
        self,
        num_classes: int = 10,
        audio_embed_dim: int = 512,
        video_embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.audio_encoder = AudioEncoder2D(embed_dim=audio_embed_dim, dropout=dropout)
        self.video_encoder = VideoEncoder(embed_dim=video_embed_dim, dropout=dropout)

        self.fusion = nn.Sequential(
            nn.Linear(audio_embed_dim + video_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
    ) -> Tensor:
        audio_features = self.audio_encoder(audio)
        video_features = self.video_encoder(video)

        combined = torch.cat([audio_features, video_features], dim=-1)
        fused = self.fusion(combined)

        return self.classifier(fused)


class SoundLocalization(nn.Module):
    """Sound localization model for audio-visual correspondence."""

    def __init__(
        self,
        video_embed_dim: int = 512,
        audio_embed_dim: int = 512,
        hidden_dim: int = 256,
        num_attention_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.video_encoder = VideoEncoder(embed_dim=video_embed_dim, dropout=dropout)
        self.audio_encoder = AudioEncoder2D(embed_dim=audio_embed_dim, dropout=dropout)

        self.video_projection = nn.Linear(video_embed_dim, hidden_dim)
        self.audio_projection = nn.Linear(audio_embed_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        self.localization_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        audio_features = self.audio_encoder(audio)
        video_features = self.video_encoder(video)

        audio_emb = self.audio_projection(audio_features).unsqueeze(1)
        video_emb = self.video_projection(video_features).unsqueeze(1)

        attended, attention_weights = self.cross_attention(
            audio_emb, video_emb, video_emb
        )

        localization = self.localization_head(attended.squeeze(1))

        return localization, attention_weights, audio_features


class AudioVisualSync(nn.Module):
    """Audio-visual synchronization model."""

    def __init__(
        self,
        audio_embed_dim: int = 512,
        video_embed_dim: int = 512,
        hidden_dim: int = 256,
        num_shifts: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_shifts = num_shifts

        self.audio_encoder = AudioEncoder2D(embed_dim=audio_embed_dim, dropout=dropout)
        self.video_encoder = VideoEncoder(embed_dim=video_embed_dim, dropout=dropout)

        self.sync_head = nn.Sequential(
            nn.Linear(audio_embed_dim + video_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_shifts * 2 + 1),
        )

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        audio_features = self.audio_encoder(audio)
        video_features = self.video_encoder(video)

        combined = torch.cat([audio_features, video_features], dim=-1)
        sync_logits = self.sync_head(combined)

        shift_predictions = sync_logits.view(-1, self.num_shifts * 2 + 1)
        sync_confidence = F.softmax(shift_predictions, dim=-1)

        return shift_predictions, sync_confidence, audio_features


class ContrastiveAudioVisualLoss(nn.Module):
    """Contrastive loss for audio-visual learning."""

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        audio_features: Tensor,
        video_features: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        audio_emb = F.normalize(audio_features, dim=-1)
        video_emb = F.normalize(video_features, dim=-1)

        similarity = (audio_emb @ video_emb.t()) / self.temperature

        if labels is not None:
            positives = similarity[labels.bool()]
            negatives = similarity[~labels.bool()]

            pos_sim = positives.mean()
            neg_sim = negatives.mean()

            loss = F.relu(neg_sim - pos_sim + self.margin)
        else:
            labels = torch.arange(len(audio_features), device=audio_features.device)
            loss = F.cross_entropy(similarity, labels)

        return loss


def create_audiovisual_model(
    task: str = "correspondence",
    **kwargs,
) -> nn.Module:
    """Factory function to create audio-visual models."""
    if task == "correspondence":
        return AudioVisualCorrespondence(**kwargs)
    elif task == "event_detection":
        return AudioVisualEventDetector(**kwargs)
    elif task == "localization":
        return SoundLocalization(**kwargs)
    elif task == "sync":
        return AudioVisualSync(**kwargs)
    else:
        raise ValueError(f"Unknown audio-visual task: {task}")

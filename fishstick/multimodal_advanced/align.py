"""
Audio-Visual: Cross-modal attention and audio-visual correspondence

Includes SyncNet for audio-visual synchronization and cross-modal attention
mechanisms for learning audio-visual representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class AudioVisualConfig:
    audio_dim: int = 256
    video_dim: int = 512
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    syncnet_hidden: int = 512
    syncnet_layers: int = 3


class CrossModalAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_mask: Optional[Tensor] = None,
        key_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = query.shape[0]

        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if query_mask is not None:
            scores = scores.masked_fill(
                query_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf")
            )

        if key_mask is not None:
            scores = scores.masked_fill(
                key_mask.unsqueeze(1).unsqueeze(3) == 0, float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )

        output = self.out_proj(context)

        return output


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioVisualConfig):
        super().__init__()
        self.config = config

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=7, stride=3, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, config.audio_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, audio: Tensor) -> Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        features = self.conv_layers(audio)
        features = features.mean(dim=-1)

        return features


class VideoEncoder(nn.Module):
    def __init__(self, config: AudioVisualConfig):
        super().__init__()
        self.config = config

        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(
                3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(
                32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(
                64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
        )

        self.fc = nn.Linear(128, config.video_dim)

    def forward(self, video: Tensor) -> Tensor:
        features = self.conv3d_layers(video)
        features = features.mean(dim=(-2, -1))
        features = features.view(features.size(0), -1)
        features = self.fc(features)

        return features


class AudioVisualEncoder(nn.Module):
    def __init__(self, config: AudioVisualConfig):
        super().__init__()
        self.config = config

        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config)

        self.audio_to_hidden = nn.Linear(config.audio_dim, config.hidden_dim)
        self.video_to_hidden = nn.Linear(config.video_dim, config.hidden_dim)

        self.audio_cross_attn = CrossModalAttention(
            query_dim=config.hidden_dim,
            key_dim=config.hidden_dim,
            value_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.video_cross_attn = CrossModalAttention(
            query_dim=config.hidden_dim,
            key_dim=config.hidden_dim,
            value_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.audio_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.video_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, audio: Tensor, video: Tensor, return_modalities: bool = False
    ) -> Tuple[Tensor, Tensor]:
        audio_features = self.audio_encoder(audio)
        video_features = self.video_encoder(video)

        audio_hidden = self.audio_to_hidden(audio_features)
        video_hidden = self.video_to_hidden(video_features)

        audio_context = self.audio_cross_attn(audio_hidden, video_hidden, video_hidden)
        video_context = self.video_cross_attn(video_hidden, audio_hidden, audio_hidden)

        audio_output = self.audio_fusion(audio_context + audio_hidden)
        video_output = self.video_fusion(video_context + video_hidden)

        if return_modalities:
            return audio_output, video_output, audio_features, video_features

        return audio_output, video_output


class SyncNet(nn.Module):
    def __init__(self, config: AudioVisualConfig):
        super().__init__()
        self.config = config

        self.audio_mlp = nn.Sequential(
            nn.Linear(config.audio_dim, config.syncnet_hidden),
            nn.ReLU(),
            nn.Linear(config.syncnet_hidden, config.syncnet_hidden),
            nn.ReLU(),
        )

        self.video_mlp = nn.Sequential(
            nn.Linear(config.video_dim, config.syncnet_hidden),
            nn.ReLU(),
            nn.Linear(config.syncnet_hidden, config.syncnet_hidden),
            nn.ReLU(),
        )

        self.sync_head = nn.Sequential(
            nn.Linear(config.syncnet_hidden * 2, config.syncnet_hidden),
            nn.ReLU(),
            nn.Linear(config.syncnet_hidden, 1),
        )

    def forward(
        self,
        audio_features: Tensor,
        video_features: Tensor,
        audio_offsets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        audio_hidden = self.audio_mlp(audio_features)
        video_hidden = self.video_mlp(video_features)

        combined = torch.cat([audio_hidden, video_hidden], dim=-1)
        sync_score = self.sync_head(combined)

        if audio_offsets is not None:
            offsets_expanded = audio_offsets.unsqueeze(-1).expand_as(sync_score)
            sync_score = sync_score + offsets_expanded * 0.1

        return sync_score.squeeze(-1), torch.cat([audio_hidden, video_hidden], dim=-1)


class AudioVisualCorrespondence(nn.Module):
    def __init__(self, config: AudioVisualConfig):
        super().__init__()
        self.config = config

        self.encoder = AudioVisualEncoder(config)
        self.syncnet = SyncNet(config)

        self.correspondence_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
        compute_sync: bool = True,
        compute_correspondence: bool = True,
    ) -> dict:
        audio_output, video_output, audio_features, video_features = self.encoder(
            audio, video, return_modalities=True
        )

        outputs = {}

        if compute_sync:
            sync_scores, sync_features = self.syncnet(audio_features, video_features)
            outputs["sync_scores"] = sync_scores
            outputs["sync_features"] = sync_features

        if compute_correspondence:
            combined = torch.cat([audio_output, video_output], dim=-1)
            correspondence_logits = self.correspondence_head(combined)
            outputs["correspondence"] = torch.sigmoid(correspondence_logits)

        outputs["audio_features"] = audio_output
        outputs["video_features"] = video_output

        return outputs


def contrastive_alignment_loss(
    audio_features: Tensor, video_features: Tensor, temperature: float = 0.1
) -> Tensor:
    audio_features = F.normalize(audio_features, dim=-1)
    video_features = F.normalize(video_features, dim=-1)

    similarity = torch.matmul(audio_features, video_features.T) / temperature

    batch_size = audio_features.shape[0]
    labels = torch.arange(batch_size, device=audio_features.device)

    loss_a2v = F.cross_entropy(similarity, labels)
    loss_v2a = F.cross_entropy(similarity.T, labels)

    return (loss_a2v + loss_v2a) / 2


def syncnet_contrastive_loss(
    audio_features: List[Tensor], video_features: List[Tensor], margin: float = 0.5
) -> Tensor:
    total_loss = 0

    for audio_feat, video_feat in zip(audio_features, video_features):
        batch_size = audio_feat.shape[0]

        similarity = F.cosine_similarity(
            audio_feat.unsqueeze(1), video_feat.unsqueeze(0), dim=-1
        )

        pos_sim = torch.diag(similarity)

        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity.device)
        neg_sims = similarity.masked_fill(~neg_mask, float("-inf"))

        neg_sim_max, _ = neg_sims.max(dim=1)

        loss = F.relu(neg_sim_max - pos_sim + margin).mean()
        total_loss += loss

    return total_loss / len(audio_features)


DEFAULT_AUDIO_VISUAL_CONFIG = AudioVisualConfig()


__all__ = [
    "AudioVisualConfig",
    "CrossModalAttention",
    "AudioEncoder",
    "VideoEncoder",
    "AudioVisualEncoder",
    "SyncNet",
    "AudioVisualCorrespondence",
    "contrastive_alignment_loss",
    "syncnet_contrastive_loss",
    "DEFAULT_AUDIO_VISUAL_CONFIG",
]

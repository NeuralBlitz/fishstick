"""
Video Summarization Module for fishstick

Comprehensive video summarization tools including:
- Supervised summarization models
- Unsupervised/weakly-supervised approaches
- Key frame selection
- Summary generation

Author: Fishstick Team
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class SummarizationConfig:
    """Configuration for video summarization."""

    feature_dim: int = 2048
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    num_keyframes: int = 10
    lambda_rank: float = 0.5
    lambda_recon: float = 0.5


class KeyFrameSelector(nn.Module):
    """
    Key Frame Selection Network.

    Learns to select representative key frames from videos.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Score each frame for selection.

        Args:
            features: Video features (B, T, D)
            mask: Optional mask for valid frames

        Returns:
            Tuple of (scores, selected_features)
        """
        B, T, D = features.shape

        encoded = self.encoder(features)

        scores = self.selector(encoded).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        probs = torch.sigmoid(scores)

        return scores, probs

    def select_k_frames(
        self,
        features: Tensor,
        k: int,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Select top-k frames.

        Args:
            features: Video features (B, T, D)
            k: Number of frames to select
            mask: Optional mask for valid frames

        Returns:
            Tuple of (selected_indices, selected_features)
        """
        scores, _ = self.forward(features, mask)

        B, T = scores.shape

        _, indices = torch.topk(scores, min(k, T), dim=-1)

        batch_indices = (
            torch.arange(B, device=features.device)
            .unsqueeze(-1)
            .expand(-1, indices.size(-1))
        )

        selected_features = features[batch_indices, indices]

        return indices, selected_features


class SupervisedSummarizer(nn.Module):
    """
    Supervised Video Summarization Model.

    Uses ground truth importance scores for training.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        use_attention: Whether to use attention mechanism
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()

        self.use_attention = use_attention

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute importance scores for each frame.

        Args:
            features: Video features (B, T, D)
            mask: Optional mask for valid frames

        Returns:
            Importance scores (B, T)
        """
        encoded, _ = self.encoder(features)

        if self.use_attention:
            attn_weights = self.attention(encoded)
            attn_weights = attn_weights.squeeze(-1)

            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1)

            context = torch.bmm(attn_weights.unsqueeze(1), encoded)
            context = context.squeeze(1)

            decoded = self.decoder(context)
        else:
            decoded = self.decoder(encoded)

        scores = decoded.squeeze(-1)

        return scores

    def get_summary(
        self,
        features: Tensor,
        threshold: float = 0.5,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate video summary based on importance scores.

        Args:
            features: Video features
            threshold: Importance threshold
            mask: Optional mask

        Returns:
            Tuple of (summary_indices, summary_features)
        """
        scores = self.forward(features, mask)

        selected_indices = (scores > threshold).nonzero(as_tuple=True)

        summary_features = features[selected_indices]

        return selected_indices, summary_features


class UnsupervisedSummarizer(nn.Module):
    """
    Unsupervised Video Summarization Model.

    Uses autoencoder reconstruction and diversity regularizer.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        latent_dim: Latent dimension for autoencoder
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.latent_proj = nn.Linear(hidden_dim * 2, latent_dim)

        self.decoder = nn.LSTM(
            latent_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.reconstructor = nn.Linear(hidden_dim * 2, input_dim)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with reconstruction.

        Args:
            features: Video features (B, T, D)
            mask: Optional mask

        Returns:
            Tuple of (scores, reconstructed_features, latent)
        """
        encoded, (h, c) = self.encoder(features)

        h = torch.cat([h[-2], h[-1]], dim=-1)

        latent = self.latent_proj(h)

        latent_expanded = latent.unsqueeze(1).expand(-1, features.size(1), -1)

        decoded, _ = self.decoder(latent_expanded)

        reconstructed = self.reconstructor(decoded)

        scores = self.scorer(encoded).squeeze(-1)

        return scores, reconstructed, latent

    def compute_loss(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
        lambda_diversity: float = 0.5,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute unsupervised loss.

        Args:
            features: Original video features
            mask: Optional mask
            lambda_diversity: Weight for diversity regularizer

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        scores, reconstructed, latent = self.forward(features, mask)

        recon_loss = F.mse_loss(reconstructed, features)

        scores_normalized = torch.sigmoid(scores)

        importance = scores_normalized.mean(dim=-1, keepdim=True)

        diversity_loss = torch.abs(importance - scores_normalized).mean()

        total_loss = recon_loss + lambda_diversity * diversity_loss

        loss_dict = {
            "reconstruction": recon_loss,
            "diversity": diversity_loss,
            "total": total_loss,
        }

        return total_loss, loss_dict


class SummaryGenerator(nn.Module):
    """
    Summary Generation Network.

    Generates natural language summaries from selected keyframes.

    Args:
        image_dim: Image feature dimension
        hidden_dim: Hidden dimension
        vocab_size: Vocabulary size
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        image_dim: int = 2048,
        hidden_dim: int = 512,
        vocab_size: int = 50000,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.decoder = nn.LSTM(
            hidden_dim + hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        image_features: Tensor,
        captions: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """
        Generate captions for summary.

        Args:
            image_features: Keyframe features (B, K, D)
            captions: Target captions (B, L)
            lengths: Caption lengths

        Returns:
            Word predictions
        """
        B, K, D = image_features.shape

        projected = self.image_proj(image_features)

        context = projected.mean(dim=1, keepdim=True)
        context = context.expand(-1, captions.size(1), -1)

        embeddings = self.embedding(captions)

        decoder_input = torch.cat([embeddings, context], dim=-1)

        output, _ = self.decoder(decoder_input)

        predictions = self.output(output)

        return predictions


class SummarizationLoss(nn.Module):
    """
    Loss functions for video summarization.

    Combines ranking loss, reconstruction loss, and diversity loss.
    """

    def __init__(
        self,
        lambda_rank: float = 0.5,
        lambda_recon: float = 0.3,
        lambda_diversity: float = 0.2,
    ):
        super().__init__()

        self.lambda_rank = lambda_rank
        self.lambda_recon = lambda_recon
        self.lambda_diversity = lambda_diversity

    def ranking_loss(
        self,
        scores: Tensor,
        labels: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Pairwise ranking loss for importance scores.

        Args:
            scores: Predicted scores (B, T)
            labels: Ground truth importance (B, T)
            mask: Valid frames mask

        Returns:
            Ranking loss
        """
        B, T = scores.shape

        scores_i = scores.unsqueeze(2).expand(-1, -1, T)
        scores_j = scores.unsqueeze(1).expand(-1, T, -1)

        label_i = labels.unsqueeze(2).expand(-1, -1, T)
        label_j = labels.unsqueeze(1).expand(-1, T, -1)

        diff = (label_i - label_j).clamp(min=0)

        margin = (scores_i - scores_j).sigmoid()

        loss = (diff * margin).sum() / (B * T * T + 1e-8)

        if mask is not None:
            mask_expanded = mask.unsqueeze(2) & mask.unsqueeze(1)
            loss = (loss * mask_expanded.float()).sum() / (
                mask_expanded.float().sum() + 1e-8
            )

        return loss

    def reconstruction_loss(
        self,
        original: Tensor,
        reconstructed: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Reconstruction loss for autoencoder."""
        loss = F.mse_loss(original, reconstructed)

        if mask is not None:
            loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)

        return loss

    def diversity_loss(
        self,
        scores: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Diversity loss to encourage uniform selection.

        Args:
            scores: Predicted scores (B, T)
            mask: Valid frames mask

        Returns:
            Diversity loss
        """
        scores_norm = torch.sigmoid(scores)

        importance = scores_norm.mean(dim=-1, keepdim=True)

        loss = torch.abs(importance - scores_norm).mean()

        if mask is not None:
            loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-8)

        return loss

    def forward(
        self,
        scores: Tensor,
        labels: Optional[Tensor] = None,
        original: Optional[Tensor] = None,
        reconstructed: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute combined loss.

        Args:
            scores: Predicted scores
            labels: Ground truth importance scores
            original: Original features
            reconstructed: Reconstructed features
            mask: Valid frames mask

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}

        if labels is not None and self.lambda_rank > 0:
            rank_loss = self.ranking_loss(scores, labels, mask)
            loss_dict["ranking"] = rank_loss
        else:
            rank_loss = 0

        if original is not None and reconstructed is not None and self.lambda_recon > 0:
            recon_loss = self.reconstruction_loss(original, reconstructed, mask)
            loss_dict["reconstruction"] = recon_loss
        else:
            recon_loss = 0

        if self.lambda_diversity > 0:
            div_loss = self.diversity_loss(scores, mask)
            loss_dict["diversity"] = div_loss
        else:
            div_loss = 0

        total = (
            self.lambda_rank * rank_loss
            + self.lambda_recon * recon_loss
            + self.lambda_diversity * div_loss
        )

        loss_dict["total"] = total

        return total, loss_dict


class AttentionBasedSelector(nn.Module):
    """
    Attention-based Key Frame Selector.

    Uses self-attention to capture temporal dependencies.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute importance scores.

        Args:
            features: Video features (B, T, D)
            mask: Optional mask

        Returns:
            Importance scores (B, T)
        """
        projected = self.input_proj(features)

        transformed = self.transformer(projected, src_key_padding_mask=mask)

        scores = self.scorer(transformed).squeeze(-1)

        return scores


class DiversityRegularizer(nn.Module):
    """
    Diversity Regularizer for Summary Selection.

    Encourages selection of diverse frames.

    Args:
        feature_dim: Feature dimension
        num_selected: Number of frames to select
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        num_selected: int = 10,
    ):
        super().__init__()

        self.num_selected = num_selected

        self.selector = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_selected),
        )

    def forward(
        self,
        features: Tensor,
        scores: Tensor,
    ) -> Tensor:
        """
        Compute diversity loss.

        Args:
            features: Video features (B, T, D)
            scores: Selection scores (B, T)

        Returns:
            Diversity loss
        """
        B, T, D = features.shape

        _, top_indices = torch.topk(scores, self.num_selected, dim=-1)

        batch_indices = (
            torch.arange(B, device=features.device)
            .unsqueeze(-1)
            .expand(-1, self.num_selected)
        )

        selected_features = features[batch_indices, top_indices]

        selected_features = selected_features.reshape(B, self.num_selected, D)

        similarity = torch.bmm(selected_features, selected_features.transpose(1, 2))

        identity = (
            torch.eye(self.num_selected, device=features.device)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        diversity = (similarity - identity).abs().mean()

        return diversity


def create_supervised_summarizer(
    input_dim: int = 2048,
    hidden_dim: int = 256,
    num_layers: int = 2,
    **kwargs,
) -> SupervisedSummarizer:
    """
    Create supervised video summarizer.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        **kwargs: Additional arguments

    Returns:
        Supervised summarizer model
    """
    dropout = kwargs.get("dropout", 0.3)
    use_attention = kwargs.get("use_attention", True)

    return SupervisedSummarizer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=use_attention,
    )


def create_unsupervised_summarizer(
    input_dim: int = 2048,
    hidden_dim: int = 256,
    latent_dim: int = 128,
    num_layers: int = 2,
    **kwargs,
) -> UnsupervisedSummarizer:
    """
    Create unsupervised video summarizer.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        latent_dim: Latent dimension
        num_layers: Number of LSTM layers
        **kwargs: Additional arguments

    Returns:
        Unsupervised summarizer model
    """
    dropout = kwargs.get("dropout", 0.3)

    return UnsupervisedSummarizer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

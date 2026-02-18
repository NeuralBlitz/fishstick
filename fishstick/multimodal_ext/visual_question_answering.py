"""
Visual Question Answering for fishstick

This module provides VQA models:
- Attention-based VQA
- Bottom-up attention VQA
- Language-guided attention
"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class QuestionEncoder(nn.Module):
    """Question encoder using LSTM or GRU."""

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.projection = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim
        )

    def forward(self, question: Tensor, length: Tensor) -> Tuple[Tensor, Tensor]:
        embedded = self.dropout(self.embedding(question))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, length.cpu(), batch_first=True, enforce_sorted=False
        )
        output, (hidden, cell) = self.rnn(packed)

        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        question_features = self.projection(hidden)

        return question_features, output


class BottomUpAttention(nn.Module):
    """Bottom-up attention for VQA using object proposals."""

    def __init__(
        self,
        image_dim: int = 2048,
        hidden_dim: int = 512,
        num_objects: int = 36,
    ):
        super().__init__()
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.question_projection = nn.Linear(hidden_dim, hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        image_features: Tensor,
        question_features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        image_emb = self.image_projection(image_features)
        question_emb = self.question_projection(question_features)

        question_expanded = question_emb.unsqueeze(1).expand_as(image_emb)
        combined = torch.cat([image_emb, question_expanded], dim=-1)

        attention_weights = self.attention(combined)
        attention_weights = F.softmax(attention_weights, dim=1)

        attended_features = (image_features * attention_weights).sum(dim=1)

        return attended_features, attention_weights


class SAN(nn.Module):
    """Stacked Attention Network for VQA."""

    def __init__(
        self,
        image_dim: int = 512,
        question_dim: int = 512,
        hidden_dim: int = 512,
        num_attention_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.question_projection = nn.Linear(question_dim, hidden_dim)

        self.attention_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_attention_layers)
            ]
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        image_features: Tensor,
        question_features: Tensor,
    ) -> Tensor:
        image_emb = self.image_projection(image_features)
        question_emb = self.question_projection(question_features)

        attention_input = torch.cat(
            [image_emb, question_emb.unsqueeze(1).expand_as(image_emb)], dim=-1
        )

        for attention_layer in self.attention_layers:
            attention_weights = attention_layer(attention_input)
            attention_weights = F.softmax(attention_weights, dim=1)

            attended = (image_features * attention_weights).sum(dim=1, keepdim=True)
            attention_input = torch.cat(
                [image_emb, attended.expand_as(image_emb)], dim=-1
            )

        fused = self.fusion(
            torch.cat([image_emb.mean(dim=1, keepdim=True), attended], dim=-1)
        )
        return fused.squeeze(1)


class VQAModel(nn.Module):
    """Complete VQA model."""

    def __init__(
        self,
        vocab_size: int = 30000,
        answer_dim: int = 1000,
        image_dim: int = 2048,
        hidden_dim: int = 512,
        num_attention_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size,
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, answer_dim)

    def forward(
        self,
        image: Tensor,
        question: Tensor,
        question_length: Tensor,
    ) -> Tensor:
        question_features, _ = self.question_encoder(question, question_length)
        image_features = self.image_encoder(image)

        fused = self.fusion(torch.cat([question_features, image_features], dim=-1))
        logits = self.classifier(fused)

        return logits


class VQAWithAttention(nn.Module):
    """VQA model with attention mechanism."""

    def __init__(
        self,
        vocab_size: int = 30000,
        answer_dim: int = 1000,
        image_dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.question_encoder = QuestionEncoder(
            vocab_size=vocab_size,
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.bottom_up = BottomUpAttention(
            image_dim=image_dim,
            hidden_dim=hidden_dim,
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, answer_dim)

    def forward(
        self,
        image_features: Tensor,
        question: Tensor,
        question_length: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        question_features, _ = self.question_encoder(question, question_length)
        attended_features, attention_weights = self.bottom_up(
            image_features, question_features
        )

        fused = self.fusion(torch.cat([question_features, attended_features], dim=-1))
        logits = self.classifier(fused)

        return logits, attention_weights


class LXMERTStyleEncoder(nn.Module):
    """LXMERT-style encoder for VQA with cross-modal attention."""

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.image_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.text_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        image_mask: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        image_emb = self.image_projection(image_features)
        text_emb = self.text_projection(text_features)

        image_cross, _ = self.image_attention(
            image_emb, text_emb, text_emb, key_padding_mask=text_mask
        )
        text_cross, _ = self.text_attention(
            text_emb, image_emb, image_emb, key_padding_mask=image_mask
        )

        return image_cross, text_cross


def create_vqa_model(
    model_type: str = "baseline",
    **kwargs,
) -> nn.Module:
    """Factory function to create VQA models."""
    if model_type == "baseline":
        return VQAModel(**kwargs)
    elif model_type == "attention":
        return VQAWithAttention(**kwargs)
    elif model_type == "lxmert":
        return LXMERTStyleEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown VQA model type: {model_type}")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Wav2Vec2FeatureEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 512,
        num_layers: int = 7,
        kernel_sizes: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2),
        strides: Tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList()
        in_ch = in_channels
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_ch,
                    hidden_channels if i < num_layers - 1 else hidden_channels,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=(kernel_sizes[i] - 1) // 2,
                )
            )
            in_ch = hidden_channels

        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1) if x.dim() == 2 else x

        for conv in self.conv_layers:
            x = conv(x)
            x = self.activation(x)

        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x


class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self, hidden_channels: int = 512, projection_dim: int = 768):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.projection = nn.Linear(hidden_channels, projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


class Wav2Vec2ContextNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_projection(x)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        x = self.layer_norm(x)
        return x


class Wav2Vec2Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        feature_encoder_hidden: int = 512,
        feature_projection_dim: int = 768,
        context_hidden_dim: int = 1024,
        num_context_layers: int = 24,
        num_attention_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_encoder = Wav2Vec2FeatureEncoder(
            in_channels=in_channels,
            hidden_channels=feature_encoder_hidden,
        )

        self.feature_projection = Wav2Vec2FeatureProjection(
            hidden_channels=feature_encoder_hidden,
            projection_dim=feature_projection_dim,
        )

        self.context_network = Wav2Vec2ContextNetwork(
            input_dim=feature_projection_dim,
            hidden_dim=context_hidden_dim,
            num_layers=num_context_layers,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        self.num_quantization_groups = 32
        self.num_vars_per_group = 2
        self.quantizer = Wav2Vec2GumbelVectorQuantizer(
            dim=feature_projection_dim,
            num_groups=self.num_quantization_groups,
            num_vars=self.num_vars_per_group,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features = self.feature_encoder(waveform)
        features = self.feature_projection(features)

        if mask is not None:
            features = apply_mask(features, mask)

        context = self.context_network(features)

        quantized, prob_perplexity, codevector_indices = self.quantizer(features)

        return context, quantized, prob_perplexity, codevector_indices


class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_groups: int = 32,
        num_vars: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.num_vars = num_vars
        self.num_codevectors = num_groups * num_vars
        self.temperature = temperature

        self.embedding = nn.Parameter(torch.randn(num_groups, num_vars, dim))
        self.projection = nn.Linear(dim, num_groups * num_vars)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.projection(z)
        logits = logits.view(z.size(0), z.size(1), self.num_groups, self.num_vars)

        probs = F.softmax(logits / self.temperature, dim=-1)

        indices = probs.argmax(dim=-1)

        quantized = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        quantized = quantized.sum(dim=-2)
        quantized = quantized @ self.embedding.view(
            self.num_groups * self.num_vars, self.dim
        )

        prob_perplexity = self._compute_prob_perplexity(probs)

        return quantized, prob_perplexity, indices

    def _compute_prob_perplexity(self, probs: torch.Tensor) -> torch.Tensor:
        dist = probs.mean(dim=0)
        perplexity = 1 / (dist.pow(2).sum(dim=-1) + 1e-10)
        return perplexity.mean()


def apply_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    padding_value: float = 0.0,
) -> torch.Tensor:
    if mask is None:
        return x

    mask_expanded = mask.unsqueeze(-1).expand_as(x)
    x = x.masked_fill(mask_expanded, padding_value)
    return x


def contrastive_loss(
    context: torch.Tensor,
    target: torch.Tensor,
    negative_samples: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    context = F.normalize(context, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    negative_samples = F.normalize(negative_samples, p=2, dim=-1)

    pos_logits = (context * target).sum(dim=-1, keepdim=True) / temperature

    neg_logits = torch.matmul(context, negative_samples.transpose(-2, -1)) / temperature

    logits = torch.cat([pos_logits, neg_logits], dim=-1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=context.device)

    loss = F.cross_entropy(logits, labels)
    return loss


class Wav2Vec2Loss(nn.Module):
    def __init__(
        self,
        contrastive_temperature: float = 0.1,
        diversity_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.contrastive_temperature = contrastive_temperature
        self.diversity_loss_weight = diversity_loss_weight

    def forward(
        self,
        context: torch.Tensor,
        quantized: torch.Tensor,
        perplexity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        neg_samples = quantized

        contrastive = contrastive_loss(
            context,
            quantized,
            neg_samples,
            temperature=self.contrastive_temperature,
        )

        diversity = 1.0 - (perplexity / (context.size(-1) ** 0.5))

        total_loss = contrastive + self.diversity_loss_weight * diversity

        return total_loss, contrastive

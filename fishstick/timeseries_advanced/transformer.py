import math
import torch
import torch.nn as nn


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(1, seq_len, self.d_model, device=x.device)
        pe[0, :, 0::2] = torch.sin(positions * div_term)
        pe[0, :, 1::2] = torch.cos(positions * div_term)
        return self.dropout(x + pe)


class InvertedBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        expansion_ratio: float = 4.0,
    ):
        super().__init__()
        self.expansion_ratio = expansion_ratio
        hidden = int(in_channels * expansion_ratio)
        self.expand = (
            nn.Linear(in_channels, hidden)
            if hidden_dim is None
            else nn.Linear(in_channels, hidden_dim)
        )
        self.reduce = (
            nn.Linear(hidden, out_channels)
            if hidden_dim is None
            else nn.Linear(hidden_dim, out_channels)
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.reduce(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        patch_size: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * input_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch, channels, seq_len = x.shape
        num_patches = (seq_len - self.patch_size) // self.patch_size + 1
        patches = x.unfold(2, self.patch_size, self.patch_size).permute(0, 1, 2, 3)
        patches = patches.contiguous().view(
            batch, num_patches, self.patch_size * channels
        )
        patches = self.proj(patches)
        return self.dropout(patches)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        patch_size: int = 16,
        output_len: int = 1,
        use_inverted_bottleneck: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_len = output_len
        self.use_inverted_bottleneck = use_inverted_bottleneck

        self.patch_embed = PatchEmbedding(input_dim, patch_size, d_model, dropout)

        if use_inverted_bottleneck:
            self.input_proj = InvertedBottleneck(input_dim, d_model, d_model)
        else:
            self.input_proj = nn.Linear(input_dim, d_model)

        self.temporal_embedding = TemporalEmbedding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch, seq_len, feat_dim = x.shape
        x = self.input_proj(x)
        x = self.temporal_embedding(x)
        patches = self.patch_embed(x.permute(0, 2, 1))
        encoded = self.transformer(patches)
        pooled = encoded.mean(dim=1)
        output = self.output_head(pooled)
        return output

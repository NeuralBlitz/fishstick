import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConformerConvolution(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        self.layer_norm = nn.LayerNorm(channels)
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)
        return x


class ConformerMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.layer_norm(x)

        attn_out, _ = self.attention(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = self.dropout(attn_out)
        return attn_out


class ConformerFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)

        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.dropout2(x)

        return residual + 0.5 * x


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.self_attn = ConformerMultiHeadAttention(d_model, num_heads, dropout)
        self.conv = ConformerConvolution(d_model, conv_kernel_size, dropout)
        self.ff2 = ConformerFeedForward(d_model, dim_feedforward, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.ff1(x)
        x = self.self_attn(x, key_padding_mask)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.layer_norm(x)
        return x


class Conformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers

        if input_dim != d_model:
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            self.input_projection = None

        self.positional_encoding = ConformerPositionalEncoding(d_model, dropout)

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(d_model)

        if num_classes is not None:
            self.classifier = nn.Linear(d_model, num_classes)
        else:
            self.classifier = None

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.input_projection is not None:
            x = self.input_projection(x)

        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, key_padding_mask)

        x = self.layer_norm(x)

        if self.classifier is not None:
            x = self.classifier(x)

        return x


class ConformerPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        inv_freq = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).float()
        pos_enc = torch.outer(positions, self.inv_freq)
        pos_enc = torch.cat([pos_enc.sin(), pos_enc.cos()], dim=-1)
        pos_enc = pos_enc.unsqueeze(0)
        x = x + pos_enc
        return self.dropout(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conformer = Conformer(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.conformer(x, key_padding_mask)

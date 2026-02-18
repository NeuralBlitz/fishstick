"""
Vision-Language Module for Fishstick

Comprehensive vision-language module implementing state-of-the-art encoders,
fusion methods, pretraining approaches, tasks, datasets, and evaluation metrics.

Author: Fishstick AI Framework
"""

from __future__ import annotations

import math
import copy
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union,
    Tuple,
    Callable,
    Sequence,
    Iterator,
    NamedTuple,
    Protocol,
    runtime_checkable,
)
from enum import Enum, auto
from pathlib import Path
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from PIL import Image


# =============================================================================
# Vision Encoders
# =============================================================================


class PatchEmbedding(nn.Module):
    """Patch embedding layer for vision transformers."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class CLIPVisionEncoder(nn.Module):
    """CLIP-style vision encoder.

    Vision Transformer encoder used in CLIP for learning visual representations.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_classes: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


class ImageTransformer(nn.Module):
    """Image Transformer encoder with flexible architecture."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_type: str = "cls",  # "cls", "global_pool", "all_tokens"
    ):
        super().__init__()
        self.output_type = output_type

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if self.output_type == "cls":
            return x[:, 0]
        elif self.output_type == "global_pool":
            return x[:, 1:].mean(dim=1)
        else:
            return x


class ConvNextEncoder(nn.Module):
    """ConvNeXt encoder for vision tasks.

    Pure convolutional architecture with modern design patterns.
    """

    def __init__(
        self,
        in_channels: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        num_classes: int = 0,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        if num_classes > 0:
            self.head = nn.Linear(dims[-1], num_classes)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input_x + self.drop_path(x)
        return x


class ViTVisualEncoder(nn.Module):
    """Vision Transformer (ViT) encoder."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        representation_size: Optional[int] = None,
        num_classes: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        dpr = [x.item() for x in torch.linspace(0, dropout, num_layers)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
                for i in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        if representation_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh(),
            )
            self.num_features = representation_size
        else:
            self.pre_logits = nn.Identity()

        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.pre_logits(x)
        x = self.head(x)
        return x


# =============================================================================
# Language Encoders
# =============================================================================


class TokenEmbedding(nn.Module):
    """Token embedding with positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class CLIPTextEncoder(nn.Module):
    """CLIP-style text encoder.

    Transformer-based text encoder used in CLIP.
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_length: int = 77,
        dropout: float = 0.0,
        eos_token_id: int = 49407,
    ):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(max_length, embed_dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )

        self.ln_final = nn.LayerNorm(embed_dim)
        self.text_projection = nn.Parameter(torch.randn(embed_dim, embed_dim))

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, text: Tensor) -> Tensor:
        B, seq_len = text.shape

        x = self.token_embedding(text)
        x = x + self.positional_embedding[:seq_len]

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = self.transformer(x, mask=mask, is_causal=True)

        x = self.ln_final(x)

        # Take features from the EOT (End Of Text) embedding
        eot_mask = text == self.eos_token_id
        eot_indices = eot_mask.float().argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot_indices]

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x


class BERTLanguageEncoder(nn.Module):
    """BERT-style language encoder.

    Bidirectional transformer encoder for language understanding.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_length: int = 512,
        dropout: float = 0.1,
        type_vocab_size: int = 2,
        num_classes: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_length, embed_dim)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embed_dim)

        self.LayerNorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.pooler = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )

        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        B, seq_len = input_ids.shape

        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # Transformer
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask  # Invert for PyTorch convention

        encoder_outputs = self.encoder(embeddings, src_key_padding_mask=attention_mask)

        # Pooler
        pooled_output = self.pooler(encoder_outputs[:, 0])

        outputs = {
            "last_hidden_state": encoder_outputs,
            "pooler_output": pooled_output,
        }

        if self.num_classes > 0:
            outputs["logits"] = self.classifier(pooled_output)

        return outputs


class GPTLanguageEncoder(nn.Module):
    """GPT-style language encoder.

    Decoder-only transformer for autoregressive language modeling.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [GPTBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        B, seq_len = input_ids.shape

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)

        x = self.dropout(token_embeds + position_embeds)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
        ).bool()

        # Transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        outputs = {"logits": logits, "hidden_states": x}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            outputs["loss"] = loss

        return outputs


class GPTBlock(nn.Module):
    """GPT transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class T5LanguageEncoder(nn.Module):
    """T5-style language encoder.

    Text-to-text transfer transformer encoder.
    """

    def __init__(
        self,
        vocab_size: int = 32128,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 512,
        dropout: float = 0.1,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                T5Block(embed_dim, num_heads, dropout, is_decoder)
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # Create padding mask
        if attention_mask is not None:
            attention_mask = attention_mask.bool()

        # Causal mask for decoder
        if self.is_decoder:
            seq_len = x.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1) | causal_mask.unsqueeze(0)
            else:
                attention_mask = causal_mask.unsqueeze(0)

        # Transformer blocks
        for block in self.blocks:
            x = block(
                x,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )

        x = self.final_layer_norm(x)

        return {"last_hidden_state": x}


class T5Block(nn.Module):
    """T5 transformer block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.is_decoder = is_decoder

        self.self_attn = T5Attention(embed_dim, num_heads, dropout)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        if is_decoder:
            self.encoder_attn = T5Attention(embed_dim, num_heads, dropout)
            self.layer_norm_enc = nn.LayerNorm(embed_dim)

        self.feed_forward = T5FeedForward(embed_dim, dropout)
        self.layer_norm_ff = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Self-attention
        residual = x
        x = self.layer_norm_1(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        # Cross-attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            residual = x
            x = self.layer_norm_enc(x)
            x = self.encoder_attn(
                x,
                encoder_attention_mask,
                encoder_hidden_states,
                encoder_hidden_states,
            )
            x = residual + x

        # Feed-forward
        residual = x
        x = self.layer_norm_ff(x)
        x = self.feed_forward(x)
        x = residual + x

        return x


class T5Attention(nn.Module):
    """T5 attention mechanism."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_value_states: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len = hidden_states.shape[:2]

        query_states = self.q(hidden_states)

        if key_value_states is not None:
            key_states = self.k(key_value_states)
            value_states = self.v(key_value_states)
        else:
            key_states = self.k(hidden_states)
            value_states = self.v(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query_states, key_states.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(1), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        attn_output = self.o(attn_output)

        return attn_output


class T5FeedForward(nn.Module):
    """T5 feed-forward network."""

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.dense_1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dense_2 = nn.Linear(embed_dim * 4, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


# =============================================================================
# Fusion Methods
# =============================================================================


class ConcatFusion(nn.Module):
    """Concatenation-based fusion.

    Simple concatenation of vision and language features.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        output_dim: int,
        use_projection: bool = True,
    ):
        super().__init__()
        self.use_projection = use_projection

        if use_projection:
            self.vision_proj = nn.Linear(vision_dim, output_dim // 2)
            self.text_proj = nn.Linear(text_dim, output_dim // 2)
            self.output_dim = output_dim
        else:
            self.output_dim = vision_dim + text_dim

        self.fusion = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        vision_features: Tensor,
        text_features: Tensor,
    ) -> Tensor:
        if self.use_projection:
            vision_features = self.vision_proj(vision_features)
            text_features = self.text_proj(text_features)

        fused = torch.cat([vision_features, text_features], dim=-1)
        fused = self.fusion(fused)
        return fused


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion.

    Language attends to vision features and vice versa.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.vision_to_text = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.text_to_vision = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        vision_features: Tensor,
        text_features: Tensor,
        vision_mask: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)

        # Vision attends to text
        vision_attended = vision_proj
        for layer in self.vision_to_text:
            vision_attended = layer(vision_attended, text_proj, text_mask)

        # Text attends to vision
        text_attended = text_proj
        for layer in self.text_to_vision:
            text_attended = layer(text_attended, vision_proj, vision_mask)

        fused = torch.cat(
            [vision_attended.mean(dim=1), text_attended.mean(dim=1)], dim=-1
        )
        fused = self.fusion(fused)
        return fused


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attn_output, _ = self.cross_attn(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
        )
        query = self.norm1(query + attn_output)
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        return query


class BilinearFusion(nn.Module):
    """Bilinear pooling fusion.

    Uses bilinear pooling for second-order feature interactions.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        output_dim: int,
        rank: int = 16,
    ):
        super().__init__()
        self.rank = rank

        # Low-rank bilinear pooling
        self.vision_low_rank = nn.Linear(vision_dim, output_dim * rank)
        self.text_low_rank = nn.Linear(text_dim, output_dim * rank)

        self.output_dim = output_dim

        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        vision_features: Tensor,
        text_features: Tensor,
    ) -> Tensor:
        vision_low = self.vision_low_rank(vision_features)
        text_low = self.text_low_rank(text_features)

        B = vision_features.size(0)
        vision_low = vision_low.view(B, self.output_dim, self.rank)
        text_low = text_low.view(B, self.output_dim, self.rank)

        # Bilinear pooling
        fused = torch.bmm(vision_low, text_low.transpose(1, 2))
        fused = fused.sum(dim=-1)  # Sum over rank dimension

        # Signed square root and L2 normalization
        fused = torch.sign(fused) * torch.sqrt(torch.abs(fused) + 1e-8)
        fused = F.normalize(fused, p=2, dim=-1)

        fused = self.fusion(fused)
        return fused


class MultimodalTransformer(nn.Module):
    """Unified multimodal transformer.

    Single transformer processing both vision and language tokens.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_vision_tokens: int = 197,  # 14x14 patches + 1 cls
        max_text_tokens: int = 77,
    ):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Modality type embeddings
        self.vision_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Position embeddings
        self.vision_pos_embed = nn.Parameter(
            torch.zeros(1, max_vision_tokens, hidden_dim)
        )
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_text_tokens, hidden_dim))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        nn.init.normal_(self.vision_type_embed, std=0.02)
        nn.init.normal_(self.text_type_embed, std=0.02)
        nn.init.normal_(self.vision_pos_embed, std=0.02)
        nn.init.normal_(self.text_pos_embed, std=0.02)

    def forward(
        self,
        vision_features: Tensor,
        text_features: Tensor,
        vision_mask: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Project to common space
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)

        # Add type and position embeddings
        B, V, _ = vision_proj.shape
        _, T, _ = text_proj.shape

        vision_proj = (
            vision_proj + self.vision_type_embed + self.vision_pos_embed[:, :V]
        )
        text_proj = text_proj + self.text_type_embed + self.text_pos_embed[:, :T]

        # Concatenate
        multimodal_tokens = torch.cat([vision_proj, text_proj], dim=1)

        # Create attention mask
        if vision_mask is not None or text_mask is not None:
            if vision_mask is None:
                vision_mask = torch.ones(
                    B, V, dtype=torch.bool, device=vision_features.device
                )
            if text_mask is None:
                text_mask = torch.ones(
                    B, T, dtype=torch.bool, device=text_features.device
                )
            attention_mask = torch.cat([vision_mask, text_mask], dim=1)
            attention_mask = ~attention_mask  # PyTorch convention
        else:
            attention_mask = None

        # Process through transformer
        output = self.transformer(
            multimodal_tokens, src_key_padding_mask=attention_mask
        )
        output = self.norm(output)

        return output


class CoAttentionFusion(nn.Module):
    """Co-attention fusion.

    Deep co-attention between vision and language features.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.co_attention_layers = nn.ModuleList(
            [
                CoAttentionLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        vision_features: Tensor,
        text_features: Tensor,
        vision_mask: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)

        for layer in self.co_attention_layers:
            vision_proj, text_proj = layer(
                vision_proj, text_proj, vision_mask, text_mask
            )

        # Global average pooling
        if vision_mask is not None:
            vision_mask_expanded = vision_mask.unsqueeze(-1).float()
            vision_pooled = (vision_proj * vision_mask_expanded).sum(
                dim=1
            ) / vision_mask_expanded.sum(dim=1)
        else:
            vision_pooled = vision_proj.mean(dim=1)

        if text_mask is not None:
            text_mask_expanded = text_mask.unsqueeze(-1).float()
            text_pooled = (text_proj * text_mask_expanded).sum(
                dim=1
            ) / text_mask_expanded.sum(dim=1)
        else:
            text_pooled = text_proj.mean(dim=1)

        fused = torch.cat([vision_pooled, text_pooled], dim=-1)
        fused = self.fusion(fused)
        return fused


class CoAttentionLayer(nn.Module):
    """Single co-attention layer."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        # Vision-guided text attention
        self.v2t_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        self.v2t_norm1 = nn.LayerNorm(hidden_dim)
        self.v2t_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.v2t_norm2 = nn.LayerNorm(hidden_dim)

        # Text-guided vision attention
        self.t2v_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        self.t2v_norm1 = nn.LayerNorm(hidden_dim)
        self.t2v_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.t2v_norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        vision_features: Tensor,
        text_features: Tensor,
        vision_mask: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # Vision -> Text attention
        attn_out, _ = self.v2t_attn(
            text_features,
            vision_features,
            vision_features,
            key_padding_mask=~vision_mask if vision_mask is not None else None,
        )
        text_features = self.v2t_norm1(text_features + attn_out)
        ffn_out = self.v2t_ffn(text_features)
        text_features = self.v2t_norm2(text_features + ffn_out)

        # Text -> Vision attention
        attn_out, _ = self.t2v_attn(
            vision_features,
            text_features,
            text_features,
            key_padding_mask=~text_mask if text_mask is not None else None,
        )
        vision_features = self.t2v_norm1(vision_features + attn_out)
        ffn_out = self.t2v_ffn(vision_features)
        vision_features = self.t2v_norm2(vision_features + ffn_out)

        return vision_features, text_features


# =============================================================================
# Pretraining Methods
# =============================================================================


class CLIPPretraining(nn.Module):
    """CLIP pretraining with contrastive learning.

    Learns joint vision-language representations through contrastive learning.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int = 512,
        temperature: float = 0.07,
        vision_dim: int = 768,
        text_dim: int = 512,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

        self.vision_projection = nn.Linear(vision_dim, embed_dim)
        self.text_projection = nn.Linear(text_dim, embed_dim)

    def encode_image(self, images: Tensor) -> Tensor:
        features = self.vision_encoder(images)
        projected = self.vision_projection(features)
        return F.normalize(projected, dim=-1)

    def encode_text(self, text: Tensor) -> Tensor:
        features = self.text_encoder(text)
        if isinstance(features, dict):
            features = features["last_hidden_state"][:, 0]
        projected = self.text_projection(features)
        return F.normalize(projected, dim=-1)

    def forward(
        self,
        images: Tensor,
        text: Tensor,
    ) -> Dict[str, Tensor]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)

        # Compute similarity matrix
        logits_per_image = image_features @ text_features.T / self.temperature
        logits_per_text = text_features @ image_features.T / self.temperature

        # Contrastive loss
        batch_size = images.size(0)
        labels = torch.arange(batch_size, device=images.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2

        return {
            "loss": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "image_features": image_features,
            "text_features": text_features,
        }


class ALIGNPretraining(nn.Module):
    """ALIGN pretraining with noise-robust contrastive learning.

    Large-scale alignment of noisy image-text pairs.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int = 640,
        temperature: float = 0.07,
        vision_dim: int = 2048,
        text_dim: int = 768,
        momentum: float = 0.995,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        self.momentum = momentum

        self.vision_projection = nn.Linear(vision_dim, embed_dim)
        self.text_projection = nn.Linear(text_dim, embed_dim)

        # Momentum encoders for stable training
        self.vision_encoder_m = copy.deepcopy(vision_encoder)
        self.text_encoder_m = copy.deepcopy(text_encoder)
        self.vision_projection_m = copy.deepcopy(self.vision_projection)
        self.text_projection_m = copy.deepcopy(self.text_projection)

        for param in self.vision_encoder_m.parameters():
            param.requires_grad = False
        for param in self.text_encoder_m.parameters():
            param.requires_grad = False
        for param in self.vision_projection_m.parameters():
            param.requires_grad = False
        for param in self.text_projection_m.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoders."""
        for param, param_m in zip(
            self.vision_encoder.parameters(), self.vision_encoder_m.parameters()
        ):
            param_m.data = param_m.data * self.momentum + param.data * (
                1.0 - self.momentum
            )

        for param, param_m in zip(
            self.text_encoder.parameters(), self.text_encoder_m.parameters()
        ):
            param_m.data = param_m.data * self.momentum + param.data * (
                1.0 - self.momentum
            )

        for param, param_m in zip(
            self.vision_projection.parameters(), self.vision_projection_m.parameters()
        ):
            param_m.data = param_m.data * self.momentum + param.data * (
                1.0 - self.momentum
            )

        for param, param_m in zip(
            self.text_projection.parameters(), self.text_projection_m.parameters()
        ):
            param_m.data = param_m.data * self.momentum + param.data * (
                1.0 - self.momentum
            )

    def forward(
        self,
        images: Tensor,
        text: Tensor,
    ) -> Dict[str, Tensor]:
        # Online encoders
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(text)
        if isinstance(text_features, dict):
            text_features = text_features["pooler_output"]

        image_embeds = F.normalize(self.vision_projection(image_features), dim=-1)
        text_embeds = F.normalize(self.text_projection(text_features), dim=-1)

        # Momentum encoders
        with torch.no_grad():
            self._momentum_update()

            image_features_m = self.vision_encoder_m(images)
            text_features_m = self.text_encoder_m(text)
            if isinstance(text_features_m, dict):
                text_features_m = text_features_m["pooler_output"]

            image_embeds_m = F.normalize(
                self.vision_projection_m(image_features_m), dim=-1
            )
            text_embeds_m = F.normalize(self.text_projection_m(text_features_m), dim=-1)

        # Compute similarity matrices
        sim_i2t = image_embeds @ text_embeds_m.T / self.temperature
        sim_t2i = text_embeds @ image_embeds_m.T / self.temperature

        # Contrastive loss
        batch_size = images.size(0)
        labels = torch.arange(batch_size, device=images.device)

        loss_i2t = F.cross_entropy(sim_i2t, labels)
        loss_t2i = F.cross_entropy(sim_t2i, labels)
        loss = (loss_i2t + loss_t2i) / 2

        return {
            "loss": loss,
            "image_features": image_embeds,
            "text_features": text_embeds,
        }


class FlorencePretraining(nn.Module):
    """Florence unified pretraining.

    One foundation model for various vision-language tasks.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        unified_decoder: nn.Module,
        embed_dim: int = 768,
        vision_dim: int = 768,
        text_dim: int = 768,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.unified_decoder = unified_decoder

        self.vision_adapter = nn.Linear(vision_dim, embed_dim)
        self.text_adapter = nn.Linear(text_dim, embed_dim)

    def forward(
        self,
        images: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        task_type: str = "contrastive",
    ) -> Dict[str, Tensor]:
        if task_type == "contrastive":
            return self._contrastive_forward(images, text)
        elif task_type == "captioning":
            return self._captioning_forward(images, text)
        elif task_type == "grounding":
            return self._grounding_forward(images, text)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _contrastive_forward(
        self,
        images: Tensor,
        text: Tensor,
    ) -> Dict[str, Tensor]:
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(text)
        if isinstance(text_features, dict):
            text_features = text_features["pooler_output"]

        image_embeds = F.normalize(self.vision_adapter(image_features), dim=-1)
        text_embeds = F.normalize(self.text_adapter(text_features), dim=-1)

        # InfoNCE loss
        logits = image_embeds @ text_embeds.T / 0.07
        labels = torch.arange(len(images), device=images.device)
        loss = F.cross_entropy(logits, labels)

        return {"loss": loss}

    def _captioning_forward(
        self,
        images: Tensor,
        text: Tensor,
    ) -> Dict[str, Tensor]:
        # Image-conditioned captioning
        image_features = self.vision_encoder(images)
        image_embeds = self.vision_adapter(image_features)

        # Decode captions
        outputs = self.unified_decoder(text, context=image_embeds)
        return outputs

    def _grounding_forward(
        self,
        images: Tensor,
        text: Tensor,
    ) -> Dict[str, Tensor]:
        # Visual grounding
        image_features = self.vision_encoder(images)
        image_embeds = self.vision_adapter(image_features)

        text_features = self.text_encoder(text)
        if isinstance(text_features, dict):
            text_features = text_features["last_hidden_state"]
        text_embeds = self.text_adapter(text_features)

        # Compute cross-modal attention for grounding
        similarity = torch.bmm(text_embeds, image_embeds.transpose(1, 2))
        return {"similarity": similarity}


class BLIPPretraining(nn.Module):
    """BLIP: Bootstrapping Language-Image Pre-training.

    Multi-modal mixture of encoder-decoder with caption filtering.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        text_decoder: nn.Module,
        embed_dim: int = 256,
        vision_dim: int = 768,
        text_dim: int = 768,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.text_decoder = text_decoder

        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.itm_head = nn.Linear(text_dim, 2)  # Image-text matching

        self.temp = nn.Parameter(0.07 * torch.ones([]))

    def forward(
        self,
        images: Tensor,
        text: Tensor,
        text_atts: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        mode: str = "multimodal",
    ) -> Dict[str, Tensor]:
        if mode == "multimodal":
            return self._multimodal_forward(images, text, text_atts, labels)
        elif mode == "contrastive":
            return self._contrastive_forward(images, text)
        elif mode == "generation":
            return self._generation_forward(images, text, labels)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _contrastive_forward(
        self,
        images: Tensor,
        text: Tensor,
    ) -> Dict[str, Tensor]:
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(text)
        if isinstance(text_features, dict):
            text_features = text_features["last_hidden_state"][:, 0]

        image_feat = F.normalize(self.vision_proj(image_features), dim=-1)
        text_feat = F.normalize(self.text_proj(text_features), dim=-1)

        sim_i2t = image_feat @ text_feat.T / self.temp
        sim_t2i = text_feat @ image_feat.T / self.temp

        batch_size = images.size(0)
        labels = torch.arange(batch_size, device=images.device)

        loss = (F.cross_entropy(sim_i2t, labels) + F.cross_entropy(sim_t2i, labels)) / 2

        return {"loss": loss, "image_feat": image_feat, "text_feat": text_feat}

    def _multimodal_forward(
        self,
        images: Tensor,
        text: Tensor,
        text_atts: Tensor,
        labels: Tensor,
    ) -> Dict[str, Tensor]:
        # Image-text contrastive loss
        image_features = self.vision_encoder(images)
        image_embeds = self.vision_proj(image_features)

        text_features = self.text_encoder(text, attention_mask=text_atts)
        if isinstance(text_features, dict):
            text_features = text_features["last_hidden_state"][:, 0]
        text_embeds = self.text_proj(text_features)

        # Image-text matching
        output_pos = self.text_encoder(
            text,
            attention_mask=text_atts,
            # Add image embeddings as additional input
        )

        # ITM loss
        itm_logits = self.itm_head(text_features)
        itm_loss = F.cross_entropy(itm_logits, labels)

        return {"loss": itm_loss}

    def _generation_forward(
        self,
        images: Tensor,
        text: Tensor,
        labels: Tensor,
    ) -> Dict[str, Tensor]:
        # Caption generation
        image_features = self.vision_encoder(images)

        # Decode with image context
        outputs = self.text_decoder(text, encoder_hidden_states=image_features)

        # LM loss
        lm_loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        return {"loss": lm_loss}


class ALBEFPretraining(nn.Module):
    """ALBEF: Align before Fuse pretraining.

    Contrastive learning before fusion with momentum distillation.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        multimodal_encoder: nn.Module,
        embed_dim: int = 256,
        vision_dim: int = 768,
        text_dim: int = 768,
        momentum: float = 0.995,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.multimodal_encoder = multimodal_encoder

        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.momentum = momentum

        # Momentum models
        self.vision_encoder_m = copy.deepcopy(vision_encoder)
        self.text_encoder_m = copy.deepcopy(text_encoder)
        self.vision_proj_m = copy.deepcopy(self.vision_proj)
        self.text_proj_m = copy.deepcopy(self.text_proj)
        self.multimodal_encoder_m = copy.deepcopy(multimodal_encoder)

        for param in self.vision_encoder_m.parameters():
            param.requires_grad = False
        for param in self.text_encoder_m.parameters():
            param.requires_grad = False
        for param in self.vision_proj_m.parameters():
            param.requires_grad = False
        for param in self.text_proj_m.parameters():
            param.requires_grad = False
        for param in self.multimodal_encoder_m.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum models."""
        for model, model_m in [
            (self.vision_encoder, self.vision_encoder_m),
            (self.text_encoder, self.text_encoder_m),
            (self.vision_proj, self.vision_proj_m),
            (self.text_proj, self.text_proj_m),
            (self.multimodal_encoder, self.multimodal_encoder_m),
        ]:
            for param, param_m in zip(model.parameters(), model_m.parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

    def forward(
        self,
        images: Tensor,
        text: Tensor,
        text_atts: Tensor,
        alpha: float = 0.4,
    ) -> Dict[str, Tensor]:
        # Image and text features
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(text, attention_mask=text_atts)
        if isinstance(text_features, dict):
            text_features = text_features["last_hidden_state"]

        # Projection for contrastive learning
        image_feat = F.normalize(self.vision_proj(image_features), dim=-1)
        text_feat = F.normalize(self.text_proj(text_features[:, 0]), dim=-1)

        # Contrastive loss (ITC)
        sim_i2t = image_feat @ text_feat.T / self.temp
        sim_t2i = text_feat @ image_feat.T / self.temp

        batch_size = images.size(0)
        labels = torch.arange(batch_size, device=images.device)

        loss_itc = (
            F.cross_entropy(sim_i2t, labels) + F.cross_entropy(sim_t2i, labels)
        ) / 2

        # Image-text matching (ITM)
        multimodal_features = self.multimodal_encoder(
            image_features.unsqueeze(1),
            text_features,
        )

        # Momentum distillation
        with torch.no_grad():
            self._momentum_update()

            image_features_m = self.vision_encoder_m(images)
            text_features_m = self.text_encoder_m(text, attention_mask=text_atts)
            if isinstance(text_features_m, dict):
                text_features_m = text_features_m["last_hidden_state"]

            image_feat_m = F.normalize(self.vision_proj_m(image_features_m), dim=-1)
            text_feat_m = F.normalize(self.text_proj_m(text_features_m[:, 0]), dim=-1)

            sim_i2t_m = image_feat_m @ text_feat_m.T / self.temp
            sim_t2i_m = text_feat_m @ image_feat_m.T / self.temp

            # Soft labels from momentum model
            soft_labels_i2t = F.softmax(sim_i2t_m, dim=1)
            soft_labels_t2i = F.softmax(sim_t2i_m, dim=1)

        # Distillation loss
        loss_distill = (
            F.kl_div(
                F.log_softmax(sim_i2t, dim=1), soft_labels_i2t, reduction="batchmean"
            )
            + F.kl_div(
                F.log_softmax(sim_t2i, dim=1), soft_labels_t2i, reduction="batchmean"
            )
        ) / 2

        loss = loss_itc + alpha * loss_distill

        return {
            "loss": loss,
            "loss_itc": loss_itc,
            "loss_distill": loss_distill,
        }


# =============================================================================
# Tasks
# =============================================================================


class ImageCaptioning(nn.Module):
    """Image captioning task.

    Generate natural language descriptions for images.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_decoder: nn.Module,
        vocab_size: int = 30522,
        hidden_dim: int = 768,
        max_length: int = 50,
        vision_dim: int = 768,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.max_length = max_length
        self.vocab_size = vocab_size

        self.vision_adapter = nn.Linear(vision_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        images: Tensor,
        captions: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # Encode image
        image_features = self.vision_encoder(images)
        image_embeds = self.vision_adapter(image_features)

        if captions is not None:
            # Training mode
            outputs = self.text_decoder(
                captions,
                encoder_hidden_states=image_embeds.unsqueeze(1),
            )

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = self.output_proj(outputs)

            # Shift for teacher forcing
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = captions[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            return {"loss": loss, "logits": logits}
        else:
            # Inference mode - generate captions
            generated_ids = self.generate(image_embeds)
            return {"generated_ids": generated_ids}

    @torch.no_grad()
    def generate(
        self,
        image_embeds: Tensor,
        max_length: Optional[int] = None,
        beam_size: int = 5,
    ) -> Tensor:
        max_length = max_length or self.max_length
        batch_size = image_embeds.size(0)

        # Start with BOS token (assume 101)
        input_ids = torch.full(
            (batch_size, 1),
            101,
            dtype=torch.long,
            device=image_embeds.device,
        )

        for _ in range(max_length):
            outputs = self.text_decoder(
                input_ids,
                encoder_hidden_states=image_embeds.unsqueeze(1),
            )

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = self.output_proj(outputs)

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS (assume 102)
            if (next_token == 102).all():
                break

        return input_ids


class VisualQuestionAnswering(nn.Module):
    """Visual Question Answering task.

    Answer natural language questions about images.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        fusion_module: nn.Module,
        num_answers: int = 3129,
        hidden_dim: int = 768,
        vision_dim: int = 768,
        text_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.fusion_module = fusion_module

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
        )

    def forward(
        self,
        images: Tensor,
        questions: Tensor,
        question_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # Encode image
        image_features = self.vision_encoder(images)
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)

        # Encode question
        text_outputs = self.text_encoder(questions, attention_mask=question_mask)
        if isinstance(text_outputs, dict):
            text_features = text_outputs["last_hidden_state"]
        else:
            text_features = text_outputs

        # Fuse modalities
        fused_features = self.fusion_module(
            image_features,
            text_features,
            vision_mask=None,
            text_mask=question_mask,
        )

        # Classify
        logits = self.classifier(fused_features)

        outputs = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss

        return outputs


class ImageTextRetrieval(nn.Module):
    """Image-text retrieval task.

    Cross-modal retrieval: find images given text or text given images.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int = 256,
        vision_dim: int = 768,
        text_dim: int = 768,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.temperature = temperature

    def encode_image(self, images: Tensor) -> Tensor:
        features = self.vision_encoder(images)
        projected = self.vision_proj(features)
        return F.normalize(projected, dim=-1)

    def encode_text(
        self, text: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        outputs = self.text_encoder(text, attention_mask=attention_mask)
        if isinstance(outputs, dict):
            features = outputs["pooler_output"]
        else:
            features = outputs.mean(dim=1)
        projected = self.text_proj(features)
        return F.normalize(projected, dim=-1)

    def forward(
        self,
        images: Tensor,
        text: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(text, attention_mask)

        # Compute similarity
        similarity = image_embeds @ text_embeds.T / self.temperature

        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "similarity": similarity,
        }

    def retrieve_images(
        self,
        text: Tensor,
        image_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        top_k: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        """Retrieve top-k images for given text."""
        text_embeds = self.encode_text(text, attention_mask)
        similarity = text_embeds @ image_embeds.T

        top_k_values, top_k_indices = torch.topk(similarity, k=top_k, dim=-1)
        return top_k_values, top_k_indices

    def retrieve_text(
        self,
        images: Tensor,
        text_embeds: Tensor,
        top_k: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        """Retrieve top-k texts for given images."""
        image_embeds = self.encode_image(images)
        similarity = image_embeds @ text_embeds.T

        top_k_values, top_k_indices = torch.topk(similarity, k=top_k, dim=-1)
        return top_k_values, top_k_indices


class VisualGrounding(nn.Module):
    """Visual grounding task.

    Ground natural language expressions in images (bounding boxes).
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        hidden_dim: int = 256,
        vision_dim: int = 768,
        text_dim: int = 768,
        num_queries: int = 100,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.num_queries = num_queries

        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # DETR-style transformer decoder
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Prediction heads
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.class_embed = nn.Linear(hidden_dim, 2)  # object vs. no-object

    def forward(
        self,
        images: Tensor,
        text: Tensor,
        text_mask: Optional[Tensor] = None,
        targets: Optional[List[Dict]] = None,
    ) -> Dict[str, Tensor]:
        # Encode image (patch features)
        image_features = self.vision_encoder(images)
        if image_features.dim() == 2:
            # Add dummy sequence dimension
            image_features = image_features.unsqueeze(1)
        image_embeds = self.vision_proj(image_features)

        # Encode text
        text_outputs = self.text_encoder(text, attention_mask=text_mask)
        if isinstance(text_outputs, dict):
            text_features = text_outputs["last_hidden_state"]
        else:
            text_features = text_outputs
        text_embeds = self.text_proj(text_features)

        # Cross-modal attention in decoder
        B = images.size(0)
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Concatenate image and text as memory
        memory = torch.cat([image_embeds, text_embeds], dim=1)

        # Decode
        hs = self.decoder(query_embeds, memory)

        # Predictions
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        outputs = {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
        }

        if targets is not None:
            loss = self.compute_loss(outputs, targets)
            outputs["loss"] = loss

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict],
    ) -> Tensor:
        """Compute DETR-style loss."""
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        # Simplified loss - in practice use Hungarian matching
        target_classes = torch.cat([t["labels"] for t in targets])
        target_boxes = torch.cat([t["boxes"] for t in targets])

        # Classification loss
        loss_ce = F.cross_entropy(
            pred_logits.view(-1, pred_logits.shape[-1]),
            target_classes,
        )

        # Box regression loss
        loss_bbox = F.l1_loss(pred_boxes.view(-1, 4), target_boxes)

        loss = loss_ce + 5 * loss_bbox
        return loss


class MultimodalClassification(nn.Module):
    """CLIP-style multimodal classification.

    Zero-shot classification using vision-language alignment.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        embed_dim: int = 512,
        vision_dim: int = 768,
        text_dim: int = 512,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.temperature = temperature

        self.class_embeddings: Optional[Tensor] = None
        self.class_names: Optional[List[str]] = None

    def encode_image(self, images: Tensor) -> Tensor:
        features = self.vision_encoder(images)
        projected = self.vision_proj(features)
        return F.normalize(projected, dim=-1)

    def encode_text(
        self, text: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        outputs = self.text_encoder(text, attention_mask=attention_mask)
        if isinstance(outputs, dict):
            # Get CLS token or mean pool
            features = outputs.get("pooler_output", outputs["last_hidden_state"][:, 0])
        else:
            features = outputs.mean(dim=1) if outputs.dim() == 3 else outputs
        projected = self.text_proj(features)
        return F.normalize(projected, dim=-1)

    def set_classes(self, class_names: List[str], text_tokenizer: Callable):
        """Set class names for zero-shot classification."""
        self.class_names = class_names

        # Create prompts for each class
        prompts = [f"a photo of a {name}" for name in class_names]

        # Tokenize
        tokens = text_tokenizer(prompts)
        if isinstance(tokens, dict):
            input_ids = tokens["input_ids"]
            attention_mask = tokens.get("attention_mask")
        else:
            input_ids = tokens
            attention_mask = None

        # Encode class embeddings
        with torch.no_grad():
            class_embeds = self.encode_text(input_ids, attention_mask)

        self.class_embeddings = class_embeds

    def forward(
        self,
        images: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if self.class_embeddings is None:
            raise ValueError("Classes not set. Call set_classes() first.")

        image_embeds = self.encode_image(images)

        # Compute similarity with class embeddings
        logits = image_embeds @ self.class_embeddings.T / self.temperature

        outputs = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss

        return outputs


# =============================================================================
# Datasets
# =============================================================================


@dataclass
class VisionLanguageExample:
    """Single example for vision-language tasks."""

    image: Union[Tensor, Image.Image, str]
    text: str
    image_id: Optional[str] = None
    example_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "text": self.text,
            "image_id": self.image_id,
            "example_id": self.example_id,
            "metadata": self.metadata,
        }


class COCOCaptions(Dataset):
    """COCO Captions dataset.

    Microsoft COCO dataset for image captioning.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_length: int = 50,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.images_dir = self.root_dir / "images" / split
        annotations_file = self.root_dir / "annotations" / f"captions_{split}2017.json"

        self.examples = []
        self._load_annotations(annotations_file)

    def _load_annotations(self, annotations_file: Path) -> None:
        """Load COCO annotations."""
        if not annotations_file.exists():
            warnings.warn(f"Annotations file not found: {annotations_file}")
            return

        with open(annotations_file, "r") as f:
            data = json.load(f)

        # Build image id to file name mapping
        id_to_filename = {}
        for img in data["images"]:
            id_to_filename[img["id"]] = img["file_name"]

        # Build examples
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            if image_id in id_to_filename:
                self.examples.append(
                    {
                        "image_file": id_to_filename[image_id],
                        "caption": ann["caption"],
                        "image_id": image_id,
                        "annotation_id": ann["id"],
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Load image
        image_path = self.images_dir / example["image_file"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Return a blank image on error
            image = Image.new("RGB", (224, 224), color="white")

        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        caption = example["caption"]
        if self.tokenizer:
            tokens = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            tokens = {"input_ids": caption}

        return {
            "image": image,
            "caption": caption,
            "tokens": tokens,
            "image_id": example["image_id"],
        }


class Flickr30k(Dataset):
    """Flickr30k dataset for image-text retrieval.

    31,000 images with 5 captions each.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_length: int = 50,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.images_dir = self.root_dir / "flickr30k-images"
        self.examples = []
        self._load_data()

    def _load_data(self) -> None:
        """Load Flickr30k data."""
        # Look for annotations file
        annotations_file = self.root_dir / f"flickr30k_{self.split}.txt"
        if not annotations_file.exists():
            annotations_file = self.root_dir / "results.csv"

        if not annotations_file.exists():
            warnings.warn(f"Annotations not found: {annotations_file}")
            return

        # Parse annotations
        # Format: image_name,caption_number,caption
        import csv

        with open(annotations_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.examples.append(
                    {
                        "image_file": row.get("image_name", row.get("filename", "")),
                        "caption": row.get("caption", row.get(" comment", "")),
                        "caption_id": row.get(
                            "caption_number", row.get("comment_number", "")
                        ),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Load image
        image_path = self.images_dir / example["image_file"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (224, 224), color="white")

        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        caption = example["caption"]
        if self.tokenizer:
            tokens = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            tokens = {"input_ids": caption}

        return {
            "image": image,
            "caption": caption,
            "tokens": tokens,
            "image_file": example["image_file"],
        }


class ConceptualCaptions(Dataset):
    """Conceptual Captions dataset.

    Large-scale web-scale image-caption pairs.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_length: int = 50,
        max_examples: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.images_dir = self.root_dir / split
        self.examples = []
        self._load_data(max_examples)

    def _load_data(self, max_examples: Optional[int] = None) -> None:
        """Load Conceptual Captions data."""
        # Load from TSV file
        tsv_file = self.root_dir / f"ConceptualCaptions_{self.split}.tsv"
        if not tsv_file.exists():
            tsv_file = self.root_dir / f"{self.split}.tsv"

        if not tsv_file.exists():
            warnings.warn(f"TSV file not found: {tsv_file}")
            return

        with open(tsv_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break

                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.examples.append(
                        {
                            "image_url": parts[0],
                            "caption": parts[1],
                        }
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # In practice, download or load image from URL
        # For now, return a placeholder
        image = Image.new("RGB", (224, 224), color="white")

        if self.transform:
            image = self.transform(image)

        caption = example["caption"]
        if self.tokenizer:
            tokens = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            tokens = {"input_ids": caption}

        return {
            "image": image,
            "caption": caption,
            "tokens": tokens,
            "image_url": example["image_url"],
        }


class VisualGenome(Dataset):
    """Visual Genome dataset.

    Dense annotations for visual understanding including:
    - Region descriptions
    - Object annotations
    - Relationships
    - Scene graphs
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        task: str = "regions",  # 'regions', 'vqa', 'grounding'
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_length: int = 50,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.task = task
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.images_dir = self.root_dir / "images"
        self.examples = []
        self._load_data()

    def _load_data(self) -> None:
        """Load Visual Genome annotations."""
        # Load region descriptions
        regions_file = self.root_dir / "region_descriptions.json"

        if not regions_file.exists():
            warnings.warn(f"Annotations not found: {regions_file}")
            return

        with open(regions_file, "r") as f:
            data = json.load(f)

        for item in data:
            image_id = item["id"]
            regions = item.get("regions", [])

            for region in regions:
                self.examples.append(
                    {
                        "image_id": image_id,
                        "region_id": region.get("region_id"),
                        "caption": region.get("phrase", ""),
                        "bbox": [  # x, y, width, height
                            region.get("x", 0),
                            region.get("y", 0),
                            region.get("width", 0),
                            region.get("height", 0),
                        ],
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Load image
        image_file = f"{example['image_id']}.jpg"
        image_path = self.images_dir / image_file

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (224, 224), color="white")

        if self.transform:
            image = self.transform(image)

        caption = example["caption"]
        if self.tokenizer:
            tokens = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            tokens = {"input_ids": caption}

        return {
            "image": image,
            "caption": caption,
            "tokens": tokens,
            "bbox": example["bbox"],
            "image_id": example["image_id"],
        }


# =============================================================================
# Evaluation Metrics
# =============================================================================


class CaptionMetrics:
    """Collection of caption evaluation metrics."""

    @staticmethod
    def bleu(
        references: List[List[str]], hypotheses: List[str], max_n: int = 4
    ) -> Dict[str, float]:
        """Compute BLEU scores.

        Args:
            references: List of reference captions (list of lists of tokens)
            hypotheses: List of hypothesis captions (list of strings)
            max_n: Maximum n-gram size

        Returns:
            Dictionary of BLEU scores
        """
        from collections import Counter
        import math

        scores = {}

        for n in range(1, max_n + 1):
            bleu_n = CaptionMetrics._bleu_n(references, hypotheses, n)
            scores[f"bleu{n}"] = bleu_n

        # Compute cumulative BLEU
        weights = [0.25] * 4  # Equal weights for BLEU-4
        scores["bleu"] = CaptionMetrics._bleu_cumulative(
            references, hypotheses, weights
        )

        return scores

    @staticmethod
    def _bleu_n(
        references: List[List[str]],
        hypotheses: List[str],
        n: int,
    ) -> float:
        """Compute BLEU-n score."""
        clipped_count = 0
        count = 0

        for refs, hyp in zip(references, hypotheses):
            hyp_tokens = hyp.lower().split()
            hyp_ngrams = CaptionMetrics._get_ngrams(hyp_tokens, n)

            max_ref_ngrams = Counter()
            for ref in refs:
                ref_tokens = ref.lower().split()
                ref_ngrams = Counter(CaptionMetrics._get_ngrams(ref_tokens, n))
                max_ref_ngrams |= ref_ngrams

            clipped_count += sum((hyp_ngrams & max_ref_ngrams).values())
            count += sum(hyp_ngrams.values())

        if count == 0:
            return 0.0

        precision = clipped_count / count

        # Brevity penalty
        hyp_len = sum(len(h.split()) for h in hypotheses)
        ref_len = sum(min(len(r.split()) for r in refs) for refs in references)

        if hyp_len > ref_len:
            bp = 1
        else:
            bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0

        return bp * precision

    @staticmethod
    def _bleu_cumulative(
        references: List[List[str]],
        hypotheses: List[str],
        weights: List[float],
    ) -> float:
        """Compute cumulative BLEU score."""
        scores = []
        for n, weight in enumerate(weights, 1):
            score = CaptionMetrics._bleu_n(references, hypotheses, n)
            if score > 0:
                scores.append(weight * math.log(score))

        if not scores:
            return 0.0

        # Brevity penalty
        hyp_len = sum(len(h.split()) for h in hypotheses)
        ref_len = sum(min(len(r.split()) for r in refs) for refs in references)

        if hyp_len > ref_len:
            bp = 1
        else:
            bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0

        return bp * math.exp(sum(scores))

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from tokens."""
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def meteor(references: List[List[str]], hypotheses: List[str]) -> float:
        """Compute METEOR score.

        Metric for Evaluation of Translation with Explicit ORdering.
        """
        scores = []

        for refs, hyp in zip(references, hypotheses):
            best_score = 0
            for ref in refs:
                score = CaptionMetrics._meteor_single(ref, hyp)
                best_score = max(best_score, score)
            scores.append(best_score)

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _meteor_single(reference: str, hypothesis: str) -> float:
        """Compute METEOR for a single reference-hypothesis pair."""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        # Count unigram matches
        ref_counts = Counter(ref_tokens)
        hyp_counts = Counter(hyp_tokens)

        matches = sum((ref_counts & hyp_counts).values())

        if matches == 0:
            return 0.0

        precision = matches / len(hyp_tokens)
        recall = matches / len(ref_tokens)

        # F-mean
        f_mean = (10 * precision * recall) / (9 * precision + recall + 1e-8)

        # Fragmentation penalty
        chunks = CaptionMetrics._count_chunks(ref_tokens, hyp_tokens)
        frag_penalty = 0.5 * (chunks / matches) ** 3 if matches > 0 else 0

        return f_mean * (1 - frag_penalty)

    @staticmethod
    def _count_chunks(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
        """Count number of matching chunks."""
        # Simplified chunk counting
        ref_set = set(ref_tokens)
        hyp_set = set(hyp_tokens)

        matches = ref_set & hyp_set

        if not matches:
            return 0

        # Count as single chunk for simplicity
        return 1

    @staticmethod
    def rouge(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores.

        Recall-Oriented Understudy for Gisting Evaluation.
        """
        rouge_1 = CaptionMetrics._rouge_n(references, hypotheses, 1)
        rouge_2 = CaptionMetrics._rouge_n(references, hypotheses, 2)
        rouge_l = CaptionMetrics._rouge_l(references, hypotheses)

        return {
            "rouge1": rouge_1,
            "rouge2": rouge_2,
            "rougeL": rouge_l,
        }

    @staticmethod
    def _rouge_n(
        references: List[List[str]],
        hypotheses: List[str],
        n: int,
    ) -> float:
        """Compute ROUGE-n score."""
        scores = []

        for refs, hyp in zip(references, hypotheses):
            hyp_tokens = hyp.lower().split()
            hyp_ngrams = set(CaptionMetrics._get_ngrams(hyp_tokens, n))

            best_score = 0
            for ref in refs:
                ref_tokens = ref.lower().split()
                ref_ngrams = set(CaptionMetrics._get_ngrams(ref_tokens, n))

                matches = len(hyp_ngrams & ref_ngrams)
                if len(ref_ngrams) > 0:
                    score = matches / len(ref_ngrams)
                    best_score = max(best_score, score)

            scores.append(best_score)

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _rouge_l(references: List[List[str]], hypotheses: List[str]) -> float:
        """Compute ROUGE-L (Longest Common Subsequence)."""
        scores = []

        for refs, hyp in zip(references, hypotheses):
            hyp_tokens = hyp.lower().split()

            best_score = 0
            for ref in refs:
                ref_tokens = ref.lower().split()
                lcs_length = CaptionMetrics._lcs_length(ref_tokens, hyp_tokens)

                if len(ref_tokens) > 0 and len(hyp_tokens) > 0:
                    precision = lcs_length / len(hyp_tokens)
                    recall = lcs_length / len(ref_tokens)

                    if precision + recall > 0:
                        f1 = (2 * precision * recall) / (precision + recall)
                        best_score = max(best_score, f1)

            scores.append(best_score)

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    @staticmethod
    def cider(references: List[List[str]], hypotheses: List[str]) -> float:
        """Compute CIDEr score.

        Consensus-based Image Description Evaluation.
        """
        # TF-IDF weighted n-gram matching
        from collections import defaultdict

        # Document frequency
        document_frequency = defaultdict(int)
        n_docs = len(references)

        for refs in references:
            unique_ngrams = set()
            for ref in refs:
                tokens = ref.lower().split()
                for n in range(1, 5):
                    ngrams = CaptionMetrics._get_ngrams(tokens, n)
                    unique_ngrams.update(ngrams)
            for ngram in unique_ngrams:
                document_frequency[ngram] += 1

        # Compute CIDEr
        scores = []

        for refs, hyp in zip(references, hypotheses):
            hyp_tokens = hyp.lower().split()
            hyp_counts = defaultdict(int)

            for n in range(1, 5):
                ngrams = CaptionMetrics._get_ngrams(hyp_tokens, n)
                for ngram in ngrams:
                    hyp_counts[ngram] += 1

            # TF-IDF vector for hypothesis
            hyp_vector = {}
            for ngram, count in hyp_counts.items():
                tf = count / len(hyp_tokens)
                idf = math.log(n_docs / (document_frequency[ngram] + 1))
                hyp_vector[ngram] = tf * idf

            # Compare with references
            ref_scores = []
            for ref in refs:
                ref_tokens = ref.lower().split()
                ref_counts = defaultdict(int)

                for n in range(1, 5):
                    ngrams = CaptionMetrics._get_ngrams(ref_tokens, n)
                    for ngram in ngrams:
                        ref_counts[ngram] += 1

                ref_vector = {}
                for ngram, count in ref_counts.items():
                    tf = count / len(ref_tokens)
                    idf = math.log(n_docs / (document_frequency[ngram] + 1))
                    ref_vector[ngram] = tf * idf

                # Cosine similarity
                dot_product = sum(
                    hyp_vector.get(ngram, 0) * ref_vector.get(ngram, 0)
                    for ngram in set(hyp_vector.keys()) | set(ref_vector.keys())
                )

                hyp_norm = math.sqrt(sum(v**2 for v in hyp_vector.values()))
                ref_norm = math.sqrt(sum(v**2 for v in ref_vector.values()))

                if hyp_norm > 0 and ref_norm > 0:
                    similarity = dot_product / (hyp_norm * ref_norm)
                    ref_scores.append(similarity)

            if ref_scores:
                scores.append(max(ref_scores))

        return (
            sum(scores) / len(scores) * 10 if scores else 0.0
        )  # CIDEr is multiplied by 10

    @staticmethod
    def spice(references: List[List[str]], hypotheses: List[str]) -> float:
        """Compute SPICE score.

        Semantic Propositional Image Caption Evaluation.

        Note: This is a simplified approximation. Full SPICE requires
        scene graph parsing which is complex.
        """
        # Simplified SPICE using semantic overlap
        scores = []

        for refs, hyp in zip(references, hypotheses):
            hyp_tokens = set(hyp.lower().split())

            best_score = 0
            for ref in refs:
                ref_tokens = set(ref.lower().split())

                # F1 score as approximation
                intersection = hyp_tokens & ref_tokens
                if len(intersection) > 0:
                    precision = len(intersection) / len(hyp_tokens) if hyp_tokens else 0
                    recall = len(intersection) / len(ref_tokens) if ref_tokens else 0

                    if precision + recall > 0:
                        f1 = (2 * precision * recall) / (precision + recall)
                        best_score = max(best_score, f1)

            scores.append(best_score)

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def spider(references: List[List[str]], hypotheses: List[str]) -> float:
        """Compute SPIDEr score.

        SPIDEr = (CIDEr + SPICE) / 2
        """
        cider_score = CaptionMetrics.cider(references, hypotheses)
        spice_score = CaptionMetrics.spice(references, hypotheses)

        return (cider_score + spice_score) / 2

    @classmethod
    def compute_all(
        cls,
        references: List[List[str]],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """Compute all metrics."""
        results = {}

        # BLEU scores
        bleu_scores = cls.bleu(references, hypotheses)
        results.update(bleu_scores)

        # METEOR
        results["meteor"] = cls.meteor(references, hypotheses)

        # ROUGE
        rouge_scores = cls.rouge(references, hypotheses)
        results.update(rouge_scores)

        # CIDEr
        results["cider"] = cls.cider(references, hypotheses)

        # SPICE
        results["spice"] = cls.spice(references, hypotheses)

        # SPIDEr
        results["spider"] = cls.spider(references, hypotheses)

        return results


# =============================================================================
# Utility Functions
# =============================================================================


def collate_vision_language_batch(
    batch: List[Dict[str, Any]],
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """Collate function for vision-language batches."""
    images = torch.stack([item["image"] for item in batch])

    # Stack or pad token sequences
    if "tokens" in batch[0] and isinstance(batch[0]["tokens"], dict):
        input_ids = torch.stack(
            [item["tokens"]["input_ids"].squeeze(0) for item in batch]
        )
        attention_mask = (
            torch.stack([item["tokens"]["attention_mask"].squeeze(0) for item in batch])
            if "attention_mask" in batch[0]["tokens"]
            else None
        )

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "captions": [item["caption"] for item in batch],
        }
    else:
        return {
            "images": images,
            "captions": [item["caption"] for item in batch],
        }


def create_vision_language_model(
    vision_encoder_type: str = "clip",
    text_encoder_type: str = "clip",
    fusion_type: str = "concat",
    task_type: str = "retrieval",
    **kwargs,
) -> nn.Module:
    """Factory function to create vision-language models.

    Args:
        vision_encoder_type: Type of vision encoder
        text_encoder_type: Type of text encoder
        fusion_type: Type of fusion module
        task_type: Type of task
        **kwargs: Additional arguments

    Returns:
        Vision-language model
    """
    # Create vision encoder
    if vision_encoder_type == "clip":
        vision_encoder = CLIPVisionEncoder(**kwargs.get("vision_kwargs", {}))
    elif vision_encoder_type == "vit":
        vision_encoder = ViTVisualEncoder(**kwargs.get("vision_kwargs", {}))
    elif vision_encoder_type == "convnext":
        vision_encoder = ConvNextEncoder(**kwargs.get("vision_kwargs", {}))
    elif vision_encoder_type == "transformer":
        vision_encoder = ImageTransformer(**kwargs.get("vision_kwargs", {}))
    else:
        raise ValueError(f"Unknown vision encoder: {vision_encoder_type}")

    # Create text encoder
    if text_encoder_type == "clip":
        text_encoder = CLIPTextEncoder(**kwargs.get("text_kwargs", {}))
    elif text_encoder_type == "bert":
        text_encoder = BERTLanguageEncoder(**kwargs.get("text_kwargs", {}))
    elif text_encoder_type == "gpt":
        text_encoder = GPTLanguageEncoder(**kwargs.get("text_kwargs", {}))
    elif text_encoder_type == "t5":
        text_encoder = T5LanguageEncoder(**kwargs.get("text_kwargs", {}))
    else:
        raise ValueError(f"Unknown text encoder: {text_encoder_type}")

    # Create fusion module
    if fusion_type == "concat":
        fusion = ConcatFusion(**kwargs.get("fusion_kwargs", {}))
    elif fusion_type == "cross_attention":
        fusion = CrossAttentionFusion(**kwargs.get("fusion_kwargs", {}))
    elif fusion_type == "bilinear":
        fusion = BilinearFusion(**kwargs.get("fusion_kwargs", {}))
    elif fusion_type == "multimodal_transformer":
        fusion = MultimodalTransformer(**kwargs.get("fusion_kwargs", {}))
    elif fusion_type == "co_attention":
        fusion = CoAttentionFusion(**kwargs.get("fusion_kwargs", {}))
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    # Create task model
    if task_type == "retrieval":
        model = ImageTextRetrieval(
            vision_encoder=vision_encoder,
            text_encoder=text_encoder,
            **kwargs.get("task_kwargs", {}),
        )
    elif task_type == "captioning":
        model = ImageCaptioning(
            vision_encoder=vision_encoder,
            text_decoder=text_encoder,  # Use encoder as decoder for simplicity
            **kwargs.get("task_kwargs", {}),
        )
    elif task_type == "vqa":
        model = VisualQuestionAnswering(
            vision_encoder=vision_encoder,
            text_encoder=text_encoder,
            fusion_module=fusion,
            **kwargs.get("task_kwargs", {}),
        )
    elif task_type == "classification":
        model = MultimodalClassification(
            vision_encoder=vision_encoder,
            text_encoder=text_encoder,
            **kwargs.get("task_kwargs", {}),
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return model


__all__ = [
    # Vision Encoders
    "CLIPVisionEncoder",
    "ImageTransformer",
    "ConvNextEncoder",
    "ViTVisualEncoder",
    # Language Encoders
    "CLIPTextEncoder",
    "BERTLanguageEncoder",
    "GPTLanguageEncoder",
    "T5LanguageEncoder",
    # Fusion Methods
    "ConcatFusion",
    "CrossAttentionFusion",
    "BilinearFusion",
    "MultimodalTransformer",
    "CoAttentionFusion",
    # Pretraining
    "CLIPPretraining",
    "ALIGNPretraining",
    "FlorencePretraining",
    "BLIPPretraining",
    "ALBEFPretraining",
    # Tasks
    "ImageCaptioning",
    "VisualQuestionAnswering",
    "ImageTextRetrieval",
    "VisualGrounding",
    "MultimodalClassification",
    # Datasets
    "COCOCaptions",
    "Flickr30k",
    "ConceptualCaptions",
    "VisualGenome",
    "VisionLanguageExample",
    # Evaluation
    "CaptionMetrics",
    # Utilities
    "collate_vision_language_batch",
    "create_vision_language_model",
]

from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer


@dataclass
class MAEConfig:
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.75
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    lr: float = 0.001
    weight_decay: float = 0.05
    epochs: int = 800


@dataclass
class BEiTConfig:
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.4
    n_queries: int = 8192
    lr: float = 0.001
    weight_decay: float = 0.05
    epochs: int = 800


@dataclass
class MaskedImageConfig:
    patch_size: int = 16
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.5
    lr: float = 0.001
    epochs: int = 100


@dataclass
class MaskedLanguageConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.15
    lr: float = 0.001
    epochs: int = 100


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x


class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        n_patches: int = 196,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, decoder_embed_dim)
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 16 * 16 * 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.size(0), mask.size(1), 1)
        x = torch.where(mask.unsqueeze(-1), mask_tokens, x)

        x = x + self.decoder_pos_embed
        x = self.pos_drop(x)

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x


class MAE(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        config: Optional[MAEConfig] = None,
    ):
        super().__init__()
        self.config = config or MAEConfig()
        self.mask_ratio = self.config.mask_ratio

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
        )

        n_patches = (img_size // self.config.patch_size) ** 2
        self.decoder = MAEDecoder(
            embed_dim=self.config.embed_dim,
            decoder_embed_dim=self.config.decoder_embed_dim,
            decoder_depth=self.config.decoder_depth,
            decoder_num_heads=self.config.decoder_num_heads,
            n_patches=n_patches,
            mlp_ratio=self.config.mlp_ratio,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        n_patches = (x.size(2) // self.config.patch_size) ** 2

        visible_indices, mask = self._generate_mask(B, n_patches, x.device)

        x = self.encoder(x)
        x = x[:, 1:]

        visible_tokens = torch.gather(
            x, 1, visible_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )

        pred = self.decoder(visible_tokens, mask)
        return pred, mask

    def _generate_mask(
        self, B: int, n_patches: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_visible = int(n_patches * (1 - self.mask_ratio))
        noise = torch.rand(B, n_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_visible = ids_shuffle[:, :n_visible]
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones(B, n_patches, device=device)
        mask[:, ids_visible] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return ids_visible, mask


class BEiTViTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class BEiT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        config: Optional[BEiTConfig] = None,
    ):
        super().__init__()
        self.config = config or BEiTConfig()
        self.mask_ratio = self.config.mask_ratio

        self.encoder = BEiTViTEncoder(
            img_size=img_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
        )

        self.num_queries = self.config.n_queries
        self.queries = nn.Parameter(
            torch.zeros(1, self.num_queries, self.config.embed_dim)
        )
        self.query_norm = nn.LayerNorm(self.config.embed_dim)
        self.decoder_norm = nn.LayerNorm(self.config.embed_dim)

        self.vq_embed = nn.Embedding(self.num_queries, self.config.embed_dim)
        self.predictor = nn.Linear(self.config.embed_dim, self.config.patch_size**2 * 3)

    def forward(
        self, x: torch.Tensor, raw_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = x.size(0)
        n_patches = (x.size(2) // self.config.patch_size) ** 2

        mask = self._generate_mask(B, n_patches, x.device)
        x = self._apply_mask(x, mask)

        x = self.encoder(x)

        queries = self.queries.repeat(B, 1, 1)
        queries = self.query_norm(queries)

        cls_token = x[:, 0:1]
        patch_tokens = x[:, 1:]

        q = queries + patch_tokens.mean(dim=1, keepdim=True)
        for block in self.encoder.blocks:
            q = block(q)

        q = self.decoder_norm(q)
        pred = self.predictor(q)

        return pred

    def _generate_mask(
        self, B: int, n_patches: int, device: torch.device
    ) -> torch.Tensor:
        mask = torch.rand(B, n_patches, device=device) < self.mask_ratio
        return mask

    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        patch_h = H // self.config.patch_size
        patch_w = W // self.config.patch_size

        x = x.unfold(2, self.config.patch_size, self.config.patch_size).unfold(
            3, self.config.patch_size, self.config.patch_size
        )
        x = x.contiguous().view(
            B, C, patch_h * patch_w, self.config.patch_size, self.config.patch_size
        )
        x = x.permute(0, 2, 1, 3, 4)

        mask_expanded = (
            mask.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, C, self.config.patch_size, self.config.patch_size)
        )
        x = x.masked_fill(mask_expanded, 0)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, C, H, W)
        return x


class MaskedImageModeler(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        config: Optional[MaskedImageConfig] = None,
    ):
        super().__init__()
        self.config = config or MaskedImageConfig()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = self.config.mask_ratio

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        n_patches = (x.size(2) // self.config.patch_size) ** 2

        mask = torch.rand(B, n_patches, device=x.device) < self.mask_ratio
        masked_x = self._apply_mask(x, mask)

        features = self.encoder(masked_x)
        pred = self.decoder(features, mask)

        return pred, mask

    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x


class BERTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attention(x)
        return self.norm(x + attn)


class BERTFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, int(hidden_size * mlp_ratio))
        self.activation = nn.GELU()
        self.dense_out = nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.dense(x)
        hidden = self.activation(hidden)
        hidden = self.dense_out(hidden)
        return self.norm(x + self.dropout(hidden))


class BERTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = BERTAttention(hidden_size, num_heads, dropout)
        self.ff = BERTFeedForward(hidden_size, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.ff(x)
        return x


class MaskedLanguageModeler(nn.Module):
    def __init__(self, config: Optional[MaskedLanguageConfig] = None):
        super().__init__()
        self.config = config or MaskedLanguageConfig()
        self.mask_ratio = self.config.mask_ratio

        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.position_embedding = nn.Embedding(512, self.config.hidden_size)

        self.blocks = nn.ModuleList(
            [
                BERTBlock(
                    self.config.hidden_size,
                    self.config.num_heads,
                    self.config.mlp_ratio,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.config.hidden_size)

        self.mlm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.mlm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape

        token_embeds = self.token_embedding(input_ids)
        position_ids = (
            torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        )
        position_embeds = self.position_embedding(position_ids)

        x = token_embeds + position_embeds

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.mlm_head(x)

        mask = torch.rand(B, L, device=input_ids.device) < self.mask_ratio
        return logits, mask

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        masked_lm_loss = loss_fct(
            logits.view(-1, self.config.vocab_size), labels.view(-1)
        )
        masked_lm_loss = masked_lm_loss * mask.view(-1).float()
        return masked_lm_loss.sum() / mask.sum()


def train_mae(
    model: MAE,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        pred, mask = model(images)

        loss = F.mse_loss(pred, torch.zeros_like(pred))
        loss = loss * mask.float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_beit(
    model: BEiT,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        pred = model(images)

        loss = F.mse_loss(pred, torch.zeros_like(pred))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_masked_image(
    model: MaskedImageModeler,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        pred, mask = model(images)

        loss = F.mse_loss(pred, torch.zeros_like(pred))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_masked_language(
    model: MaskedLanguageModeler,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits, mask = model(input_ids)

        loss = model.compute_loss(logits, labels, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

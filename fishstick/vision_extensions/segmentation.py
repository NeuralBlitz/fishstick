"""
Image Segmentation Models

Advanced segmentation architectures including U-Net, DeepLabV3+,
Segmenter (ViT-based), and MaskFormer-style models.

References:
    - U-Net: https://arxiv.org/abs/1505.04597
    - DeepLabV3+: https://arxiv.org/abs/1802.02611
    - Segmenter: https://arxiv.org/abs/2105.05633
    - MaskFormer: https://arxiv.org/abs/2107.06278
"""

from typing import List, Tuple, Optional, Dict, Any
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """
    U-Net style decoder block with skip connection support.

    Args:
        in_channels: Input channel dimension
        skip_channels: Skip connection channel dimension
        out_channels: Output channel dimension
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input tensor
            skip: [B, C_skip, H_skip, W_skip] skip connection
        Returns:
            [B, out_channels, H*2, W*2] upsampled tensor
        """
        x = self.upconv(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Semantic Segmentation.

    Encoder-decoder architecture with skip connections for
    precise localization in medical and general segmentation.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        base_channels: Base number of channels for first layer
        depth: Number of downsampling/upsampling levels
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        ch = base_channels
        for i in range(depth):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else ch // 2, ch, 3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, ch, 3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
            )

            if i < depth - 1:
                self.encoder.append(nn.MaxPool2d(2))

            ch *= 2

        for i in range(depth - 1):
            self.decoder.append(DecoderBlock(ch, ch // 2, ch // 2))
            ch //= 2

        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input image
        Returns:
            [B, num_classes, H, W] segmentation logits
        """
        skips = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                continue
            skips.append(x)

        skips = skips[:-1][::-1]

        for i, decoder in enumerate(self.decoder):
            skip = skips[i] if i < len(skips) else None
            x = decoder(x, skip)

        return self.final(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Multi-scale feature extraction using atrous convolutions
    with different dilation rates.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        atrous_rates: List of dilation rates
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: List[int] = [6, 12, 18],
    ):
        super().__init__()

        modules = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.modules = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input feature
        Returns:
            [B, out_channels, H, W] multi-scale features
        """
        h, w = x.size()[2:]

        res = []
        for module in self.modules:
            res.append(module(x))

        res[-1] = F.interpolate(
            res[-1], size=(h, w), mode="bilinear", align_corners=False
        )

        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ Architecture for Semantic Segmentation.

    Combines ASPP multi-scale features with low-level features
    via decoder for improved boundary precision.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        backbone_channels: List of backbone feature channel dims
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        backbone_channels: List[int] = [256, 512, 1024, 2048],
    ):
        super().__init__()

        self.aspp = ASPP(backbone_channels[-1], 256)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, features: List[Tensor], input_size: Tuple[int, int]) -> Tensor:
        """
        Args:
            features: List of backbone feature maps [C1, C2, C3, C4]
            input_size: (H, W) of original input
        Returns:
            [B, num_classes, H, W] segmentation logits
        """
        x = features[-1]

        x = self.aspp(x)

        low_level = features[0]

        x = F.interpolate(
            x, size=low_level.shape[2:], mode="bilinear", align_corners=False
        )

        x = torch.cat([x, low_level], dim=1)

        x = self.decoder(x)

        x = self.classifier(x)

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x


class SegmenterMaskDecoder(nn.Module):
    """
    Segmenter Mask Decoder for ViT-based segmentation.

    Args:
        embed_dim: Embedding dimension
        num_classes: Number of segmentation classes
        decoder_depth: Number of decoder layers
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 21,
        decoder_depth: int = 3,
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(embed_dim, 256)

        self.decoder_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(256)
        self.mask_embed = nn.Linear(256, num_classes)

    def forward(self, x: Tensor, patch_embeddings: Tensor) -> Tensor:
        """
        Args:
            x: [B, 1+num_patches, embed_dim] encoded features
            patch_embeddings: [B, num_patches, embed_dim] patch tokens
        Returns:
            [B, num_classes, H, W] segmentation masks
        """
        B, N, C = patch_embeddings.shape

        x = self.decoder_embed(x)

        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)

        mask_tokens = self.mask_embed(x[:, 1:])

        h = w = int(math.sqrt(N - 1))
        masks = mask_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)

        return masks


class Segmenter(nn.Module):
    """
    Segmenter: Transformer-based Semantic Segmentation.

    Uses ViT encoder with mask transformer decoder for
    semantic segmentation.

    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        embed_dim: Transformer embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        patch_size: Patch size for tokenization
    """

    import math

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (224 // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=depth,
        )

        self.decoder = SegmenterMaskDecoder(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W] input image
        Returns:
            [B, num_classes, H, W] segmentation logits
        """
        B, C, H, W = x.shape

        patch_embeddings = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls_tokens, patch_embeddings], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)

        masks = self.decoder(x, patch_embeddings)

        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)

        return masks


class MaskFormerHead(nn.Module):
    """
    MaskFormer-style Segmentation Head.

    Generates per-pixel embeddings and classifies them into
    semantic/instance masks.

    Args:
        in_channels: Input channel dimension
        num_classes: Number of segmentation classes
        hidden_dim: Hidden dimension for projections
        num_queries: Number of mask queries
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 100,
    ):
        super().__init__()

        self.num_queries = num_queries

        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=6,
        )

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mask_output = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            features: [B, C, H, W] feature tensor
        Returns:
            Dict with 'pred_logits' and 'pred_masks'
        """
        B, C, H, W = features.shape

        x = self.input_proj(features)

        x_flat = x.flatten(2).permute(0, 2, 1)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        hs = self.transformer(queries + x_flat.mean(1, keepdim=True))

        pred_logits = self.class_embed(hs)

        mask_embeds = self.mask_embed(hs)

        mask_embeds = mask_embeds.unsqueeze(-1).unsqueeze(-1)

        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeds, x)

        return {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
        }


def create_segmentation_model(
    model_type: str,
    num_classes: int = 21,
    in_channels: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create segmentation models.

    Args:
        model_type: Type of segmentation model ('unet', 'deeplabv3+', 'segmenter')
        num_classes: Number of segmentation classes
        in_channels: Number of input channels
        **kwargs: Additional arguments for specific models

    Returns:
        Segmentation model

    Raises:
        ValueError: If model_type is not recognized
    """
    models = {
        "unet": UNet,
        "deeplabv3+": DeepLabV3Plus,
        "segmenter": Segmenter,
    }

    if model_type.lower() not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(models.keys())}"
        )

    return models[model_type.lower()](
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs,
    )

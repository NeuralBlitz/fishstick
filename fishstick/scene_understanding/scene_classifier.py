"""
Scene Classification Module

Provides scene classification models for understanding image context.
"""

from typing import Tuple, List, Optional, Union, Dict, Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SceneClassifier(nn.Module):
    """
    Base class for scene classification models.
    """

    def __init__(
        self,
        num_classes: int = 365,
        feature_dim: int = 2048,
    ):
        """
        Args:
            num_classes: Number of scene categories
            feature_dim: Dimension of extracted features
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Tuple of (logits, features)
        """
        raise NotImplementedError


class ResNetSceneClassifier(SceneClassifier):
    """
    ResNet-based scene classifier with multi-scale features.
    """

    def __init__(
        self,
        num_classes: int = 365,
        backbone: str = "resnet50",
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_classes: Number of scene categories
            backbone: Backbone architecture (resnet18, resnet34, resnet50, resnet101)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__(num_classes=num_classes)

        import torchvision.models as models

        if backbone == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.avgpool = resnet.avgpool

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Tuple of (logits [B, num_classes], features [B, feature_dim])
        """
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.flatten(1)

        logits = self.classifier(features)

        return logits, features


class MultiScaleSceneFeatures(nn.Module):
    """
    Extract multi-scale features for scene understanding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        scales: List[int] = [1, 2, 4, 8],
    ):
        """
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels
            scales: List of downsampling scales
        """
        super().__init__()
        self.scales = scales

        self.encoders = nn.ModuleDict()

        for scale in scales:
            self.encoders[str(scale)] = self._make_encoder(
                in_channels, base_channels * scale, scale
            )

        self.fusion = nn.ModuleList(
            [nn.Conv2d(base_channels * s, base_channels * 4, 1) for s in scales]
        )

    def _make_encoder(
        self,
        in_channels: int,
        channels: int,
        scale: int,
    ) -> nn.Module:
        """Create encoder for a specific scale."""
        layers = [
            nn.Conv2d(
                in_channels, channels, 3, stride=2 if scale > 1 else 1, padding=1
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        ]

        for _ in range(int(np.log2(scale))):
            layers.extend(
                [
                    nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
                    nn.BatchNorm2d(channels * 2),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = channels * 2

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Dictionary of multi-scale features
        """
        import numpy as np

        outputs = {}

        for scale in self.scales:
            scaled_x = F.interpolate(
                x,
                scale_factor=1 / scale,
                mode="bilinear",
                align_corners=False,
            )
            outputs[str(scale)] = self.encoders[str(scale)](scaled_x)

        return outputs


class VisionTransformerSceneClassifier(SceneClassifier):
    """
    Vision Transformer-based scene classifier.
    """

    def __init__(
        self,
        num_classes: int = 365,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_classes: Number of scene categories
            img_size: Input image size
            patch_size: Size of patches
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super().__init__(num_classes=num_classes, feature_dim=embed_dim)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Tuple of (logits [B, num_classes], features [B, embed_dim])
        """
        B = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        features = x[:, 0]
        logits = self.head(features)

        return logits, features


class TransformerBlock(nn.Module):
    """
    Transformer block for vision transformer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SceneContextEncoder(nn.Module):
    """
    Encode global scene context for improved classification.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        context_dim: int = 512,
    ):
        """
        Args:
            feature_dim: Dimension of input features
            context_dim: Dimension of output context
        """
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim, context_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(context_dim * 2, context_dim),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: Feature tensor [B, C, H, W]

        Returns:
            Context weights [B, context_dim]
        """
        global_feat = self.global_pool(features).flatten(1)
        context = self.context_encoder(global_feat)
        return context


def create_scene_classifier(
    model_type: str = "resnet",
    num_classes: int = 365,
    **kwargs,
) -> SceneClassifier:
    """
    Factory function to create scene classifiers.

    Args:
        model_type: Type of model ('resnet', 'vit')
        num_classes: Number of scene categories
        **kwargs: Additional model-specific arguments

    Returns:
        Scene classifier instance
    """
    if model_type == "resnet":
        return ResNetSceneClassifier(
            num_classes=num_classes,
            backbone=kwargs.get("backbone", "resnet50"),
            pretrained=kwargs.get("pretrained", False),
            freeze_backbone=kwargs.get("freeze_backbone", False),
        )
    elif model_type == "vit":
        return VisionTransformerSceneClassifier(
            num_classes=num_classes,
            img_size=kwargs.get("img_size", 224),
            patch_size=kwargs.get("patch_size", 16),
            embed_dim=kwargs.get("embed_dim", 768),
            depth=kwargs.get("depth", 12),
            num_heads=kwargs.get("num_heads", 12),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

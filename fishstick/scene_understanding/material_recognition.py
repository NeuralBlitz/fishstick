"""
Material & Surface Recognition Module

Provides models for per-pixel material classification, texture
description, and reflectance property estimation. Supports BRDF-aware
encoding and dense material segmentation for scene understanding.
"""

from typing import Tuple, List, Optional, Union, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math

MATERIAL_CLASSES: List[str] = [
    "fabric",
    "leather",
    "metal",
    "wood",
    "plastic",
    "glass",
    "ceramic",
    "stone",
    "paper",
    "rubber",
    "concrete",
    "carpet",
    "foliage",
    "skin",
    "food",
    "water",
    "sky",
    "other",
]


class TextureDescriptor(nn.Module):
    """
    Local texture pattern descriptor network.

    Extracts rotation-invariant texture descriptors from image patches
    using a set of oriented Gabor-like learned filters followed by
    orderless pooling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        descriptor_dim: int = 128,
        num_orientations: int = 8,
        patch_size: int = 7,
    ):
        """
        Args:
            in_channels: Number of input channels.
            descriptor_dim: Output descriptor dimensionality.
            num_orientations: Number of learned filter orientations.
            patch_size: Spatial size of oriented filters.
        """
        super().__init__()
        self.num_orientations = num_orientations

        self.oriented_filters = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    descriptor_dim // num_orientations,
                    patch_size,
                    padding=patch_size // 2,
                )
                for _ in range(num_orientations)
            ]
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        ch = descriptor_dim // num_orientations * num_orientations
        self.proj = nn.Sequential(
            nn.Linear(ch, descriptor_dim),
            nn.ReLU(inplace=True),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image or patch ``[B, C, H, W]``.

        Returns:
            Texture descriptor ``[B, descriptor_dim]``.
        """
        responses = [f(x) for f in self.oriented_filters]
        combined = torch.cat(responses, dim=1)
        combined = F.relu(combined, inplace=True)
        pooled = self.pool(combined)
        return F.normalize(self.proj(pooled), p=2, dim=-1)

    def forward_dense(self, x: Tensor) -> Tensor:
        """
        Compute per-pixel texture descriptors.

        Args:
            x: Input image ``[B, C, H, W]``.

        Returns:
            Dense descriptors ``[B, descriptor_dim, H, W]``.
        """
        responses = [f(x) for f in self.oriented_filters]
        combined = torch.cat(responses, dim=1)
        return F.relu(combined, inplace=True)


class MaterialEncoder(nn.Module):
    """
    BRDF-aware feature encoder for material recognition.

    Encodes an input image into features that capture both appearance
    and reflectance cues by combining a standard feature pyramid with
    a specular / diffuse decomposition branch.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 4,
    ):
        """
        Args:
            in_channels: Input image channels.
            base_channels: Base channel width.
            num_stages: Number of encoder stages.
        """
        super().__init__()
        self.num_stages = num_stages

        stages: List[nn.Module] = []
        ch_in = in_channels
        for i in range(num_stages):
            ch_out = base_channels * (2**i)
            stages.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_out, ch_out, 3, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                )
            )
            ch_in = ch_out
        self.stages = nn.ModuleList(stages)

        self.specular_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        """
        Args:
            x: Input image ``[B, C, H, W]``.

        Returns:
            Tuple of (multi-scale features list, specular features ``[B, 32, H, W]``).
        """
        feats: List[Tensor] = []
        h = x
        for stage in self.stages:
            h = stage(h)
            feats.append(h)

        spec = self.specular_branch(x)
        return feats, spec


class ReflectanceEstimator(nn.Module):
    """
    Estimate per-pixel diffuse and specular reflectance properties.

    Predicts a diffuse albedo map and a specular roughness map from
    the encoded material features.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        specular_dim: int = 32,
    ):
        """
        Args:
            feature_dim: Channel dimension of deepest encoder features.
            specular_dim: Channel dimension of specular branch features.
        """
        super().__init__()
        self.diffuse_head = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1),
            nn.Sigmoid(),
        )

        self.roughness_head = nn.Sequential(
            nn.Conv2d(feature_dim + specular_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        deep_feat: Tensor,
        spec_feat: Tensor,
        target_size: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        """
        Args:
            deep_feat: Deepest encoder features ``[B, C, h, w]``.
            spec_feat: Specular features ``[B, 32, H, W]``.
            target_size: ``(H, W)`` for output resolution.

        Returns:
            Dictionary with ``diffuse_albedo`` ``[B,3,H,W]`` and
            ``roughness`` ``[B,1,H,W]``.
        """
        up = F.interpolate(
            deep_feat, size=target_size, mode="bilinear", align_corners=False
        )
        diffuse = self.diffuse_head(up)

        spec_down = F.interpolate(
            spec_feat, size=up.shape[2:], mode="bilinear", align_corners=False
        )
        roughness = self.roughness_head(torch.cat([up, spec_down], dim=1))
        roughness = F.interpolate(
            roughness, size=target_size, mode="bilinear", align_corners=False
        )

        return {"diffuse_albedo": diffuse, "roughness": roughness}


class MaterialSegmentationHead(nn.Module):
    """
    Dense per-pixel material segmentation head.

    Fuses multi-scale encoder features through a lightweight FPN-style
    decoder and produces per-pixel material class logits.
    """

    def __init__(
        self,
        base_channels: int = 64,
        num_stages: int = 4,
        num_classes: int = 18,
    ):
        """
        Args:
            base_channels: Must match encoder base_channels.
            num_stages: Must match encoder num_stages.
            num_classes: Number of material categories.
        """
        super().__init__()
        self.num_classes = num_classes

        laterals: List[nn.Module] = []
        for i in range(num_stages):
            ch = base_channels * (2**i)
            laterals.append(nn.Conv2d(ch, base_channels, 1))
        self.laterals = nn.ModuleList(laterals)

        self.smooth = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, 1),
        )

    def forward(
        self,
        features: List[Tensor],
        target_size: Tuple[int, int],
    ) -> Tensor:
        """
        Args:
            features: Multi-scale encoder features.
            target_size: ``(H, W)`` output resolution.

        Returns:
            Material logits ``[B, num_classes, H, W]``.
        """
        x = self.laterals[-1](features[-1])
        for i in range(len(features) - 2, -1, -1):
            lat = self.laterals[i](features[i])
            x = F.interpolate(
                x, size=lat.shape[2:], mode="bilinear", align_corners=False
            )
            x = x + lat

        x = self.smooth(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.classifier(x)


class MaterialClassifier(nn.Module):
    """
    End-to-end per-pixel material classification network.

    Combines BRDF-aware encoding, texture description, reflectance
    estimation, and dense material segmentation into a unified model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 4,
        num_classes: int = 18,
        descriptor_dim: int = 128,
    ):
        """
        Args:
            in_channels: Input image channels.
            base_channels: Base encoder channel width.
            num_stages: Encoder depth.
            num_classes: Number of material categories.
            descriptor_dim: Texture descriptor dimensionality.
        """
        super().__init__()
        self.encoder = MaterialEncoder(in_channels, base_channels, num_stages)
        self.texture = TextureDescriptor(in_channels, descriptor_dim)
        self.seg_head = MaterialSegmentationHead(base_channels, num_stages, num_classes)

        feat_dim = base_channels * (2 ** (num_stages - 1))
        self.reflectance = ReflectanceEstimator(feat_dim, 32)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: Input image ``[B, 3, H, W]``.

        Returns:
            Dictionary containing:
                - ``material_logits``: ``[B, num_classes, H, W]``
                - ``texture_desc``: ``[B, descriptor_dim]``
                - ``diffuse_albedo``: ``[B, 3, H, W]``
                - ``roughness``: ``[B, 1, H, W]``
        """
        H, W = x.shape[2], x.shape[3]
        feats, spec = self.encoder(x)

        material_logits = self.seg_head(feats, (H, W))
        texture_desc = self.texture(x)
        reflectance = self.reflectance(feats[-1], spec, (H, W))

        return {
            "material_logits": material_logits,
            "texture_desc": texture_desc,
            **reflectance,
        }


def create_material_model(
    model_type: str = "default",
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to create material recognition models.

    Args:
        model_type: Model variant (``"default"``).
        **kwargs: Forwarded to the model constructor.

    Returns:
        Material classifier instance.
    """
    if model_type == "default":
        return MaterialClassifier(
            in_channels=kwargs.get("in_channels", 3),
            base_channels=kwargs.get("base_channels", 64),
            num_stages=kwargs.get("num_stages", 4),
            num_classes=kwargs.get("num_classes", len(MATERIAL_CLASSES)),
            descriptor_dim=kwargs.get("descriptor_dim", 128),
        )
    raise ValueError(f"Unknown material model type: {model_type}")

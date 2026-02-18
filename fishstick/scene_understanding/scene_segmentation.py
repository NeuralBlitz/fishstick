"""
Scene Segmentation Module

Semantic and panoptic segmentation for scene understanding.
"""

from typing import Tuple, List, Optional, Union, Dict, Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SemanticSegmentationHead(nn.Module):
    """
    Semantic segmentation head with various backbones.
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int = 20,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_channels: List of input channel dimensions from encoder
            num_classes: Number of semantic classes
            dropout: Dropout rate
        """
        super().__init__()

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, 256, 1) for in_ch in in_channels]
        )

        self.smooth_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels
            ]
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(256 * len(in_channels), 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.cls_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Args:
            features: Multi-scale features from encoder

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        target_size = features[0].shape[2:]

        laterals = []
        for i, (feat, lateral_conv, smooth_conv) in enumerate(
            zip(features, self.lateral_convs, self.smooth_convs)
        ):
            lateral = lateral_conv(feat)

            if i > 0:
                lateral = F.interpolate(
                    lateral,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            lateral = smooth_conv(lateral)
            laterals.append(lateral)

        fused = torch.cat(laterals, dim=1)
        fused = self.fuse_conv(fused)

        output = self.cls_conv(fused)

        return output


class PSPModule(nn.Module):
    """
    Pyramid Pooling Module for scene segmentation.
    """

    def __init__(
        self,
        in_channels: int = 2048,
        pool_sizes: List[int] = [1, 2, 3, 6],
        out_channels: int = 512,
    ):
        """
        Args:
            in_channels: Input channel dimension
            pool_sizes: List of pooling window sizes
            out_channels: Output channel dimension
        """
        super().__init__()

        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, out_channels // len(pool_sizes), 1),
                    nn.BatchNorm2d(out_channels // len(pool_sizes)),
                    nn.ReLU(inplace=True),
                )
                for pool_size in pool_sizes
            ]
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input features [B, C, H, W]

        Returns:
            Fused features [B, out_channels, H, W]
        """
        h, w = x.size(2), x.size(3)

        pyramids = []
        for stage in self.stages:
            pyramid = stage(x)
            pyramid = F.interpolate(
                pyramid,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pyramids.append(pyramid)

        output = torch.cat([x] + pyramids, dim=1)
        output = self.bottleneck(output)

        return output


class SceneSegmentationNetwork(nn.Module):
    """
    Complete scene segmentation network.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 20,
        pretrained: bool = False,
        output_stride: int = 16,
    ):
        """
        Args:
            backbone: Encoder backbone architecture
            num_classes: Number of semantic classes
            pretrained: Whether to use pretrained encoder
            output_stride: Output stride for encoder
        """
        super().__init__()

        self.encoder = self._make_encoder(backbone, pretrained, output_stride)

        encoder_channels = self._get_encoder_channels(backbone, output_stride)

        if "resnet" in backbone:
            self.psp = PSPModule(
                in_channels=encoder_channels[-1],
                pool_sizes=[1, 2, 3, 6],
                out_channels=512,
            )
            decoder_in_channels = 512
        else:
            self.psp = None
            decoder_in_channels = encoder_channels[-1]

        self.decoder = SemanticSegmentationHead(
            in_channels=[
                encoder_channels[0],
                encoder_channels[1],
                encoder_channels[2],
                decoder_in_channels,
            ],
            num_classes=num_classes,
        )

    def _make_encoder(
        self,
        backbone: str,
        pretrained: bool,
        output_stride: int,
    ) -> nn.Module:
        """Create encoder backbone."""
        import torchvision.models as models

        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        layers = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool]

        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        layers_dict = {}

        for name in layer_names:
            layers_dict[name] = getattr(resnet, name)

        return nn.ModuleDict({"features": nn.Sequential(*layers), **layers_dict})

    def _get_encoder_channels(
        self,
        backbone: str,
        output_stride: int,
    ) -> List[int]:
        """Get encoder output channels."""
        if backbone in ["resnet50", "resnet101"]:
            return [256, 512, 1024, 2048]
        return [256, 512, 1024, 2048]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        features = []

        x = self.encoder["features"](x)
        features.append(self.encoder["layer1"](x))
        features.append(self.encoder["layer2"](features[-1]))
        features.append(self.encoder["layer3"](features[-1]))
        features.append(self.encoder["layer4"](features[-1]))

        if self.psp is not None:
            features[-1] = self.psp(features[-1])

        output = self.decoder(features)

        return output


class BoundaryAwareSegmentation(nn.Module):
    """
    Segmentation with boundary awareness for improved scene understanding.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 20,
        boundary_channels: int = 64,
    ):
        """
        Args:
            in_channels: Input feature channels
            num_classes: Number of semantic classes
            boundary_channels: Number of boundary prediction channels
        """
        super().__init__()

        self.semantic_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, 1),
        )

        self.boundary_head = nn.Sequential(
            nn.Conv2d(in_channels, boundary_channels, 3, padding=1),
            nn.BatchNorm2d(boundary_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(boundary_channels, 1, 1),
            nn.Sigmoid(),
        )

        self.boundary_refine = BoundaryRefinementModule(num_classes)

    def forward(
        self,
        features: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: Input features [B, C, H, W]

        Returns:
            Tuple of (semantic_logits, boundary_predictions)
        """
        semantic = self.semantic_head(features)
        boundary = self.boundary_head(features)

        semantic = self.boundary_refine(semantic, boundary)

        return semantic, boundary


class BoundaryRefinementModule(nn.Module):
    """
    Refine segmentation using boundary information.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.refine_conv = nn.Sequential(
            nn.Conv2d(num_classes + 1, num_classes, 3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        semantic: Tensor,
        boundary: Tensor,
    ) -> Tensor:
        """
        Args:
            semantic: Semantic segmentation logits [B, num_classes, H, W]
            boundary: Boundary predictions [B, 1, H, W]

        Returns:
            Refined segmentation
        """
        combined = torch.cat([semantic, boundary], dim=1)
        refined = self.refine_conv(combined)
        return refined


class PanopticSegmentationHead(nn.Module):
    """
    Panoptic segmentation head combining semantic and instance predictions.
    """

    def __init__(
        self,
        num_classes: int = 20,
        embedding_dim: int = 128,
        feature_channels: int = 256,
    ):
        """
        Args:
            num_classes: Number of semantic classes
            embedding_dim: Dimension of instance embeddings
            feature_channels: Input feature channels
        """
        super().__init__()

        self.semantic_head = nn.Conv2d(feature_channels, num_classes, 1)

        self.instance_embedding = nn.Sequential(
            nn.Conv2d(feature_channels, embedding_dim, 1),
        )

        self.instance_center = nn.Sequential(
            nn.Conv2d(feature_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(
        self,
        features: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            features: Input features [B, C, H, W]

        Returns:
            Dictionary with semantic, instance embeddings, and center heatmaps
        """
        semantic = self.semantic_head(features)
        embeddings = self.instance_embedding(features)
        centers = self.instance_center(features)

        return {
            "semantic": semantic,
            "instance_embeddings": embeddings,
            "center_heatmaps": torch.sigmoid(centers),
        }


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture for scene segmentation.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 20,
        pretrained: bool = False,
        output_stride: int = 16,
    ):
        """
        Args:
            backbone: Encoder backbone
            num_classes: Number of segmentation classes
            pretrained: Use pretrained weights
            output_stride: Output stride
        """
        super().__init__()

        self.backbone = self._build_backbone(backbone, pretrained, output_stride)

        in_channels = 2048 if "resnet" in backbone else 512

        self.aspp = ASPP(in_channels, [6, 12, 18], 256)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def _build_backbone(
        self,
        backbone: str,
        pretrained: bool,
        output_stride: int,
    ) -> nn.Module:
        """Build backbone network."""
        import torchvision.models as models

        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        else:
            resnet = models.resnet101(pretrained=pretrained)

        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        input_size = x.shape[2:]

        features = self.backbone(x)
        features = self.aspp(features)

        output = self.decoder(features)
        output = F.interpolate(
            output,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return output


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    """

    def __init__(
        self,
        in_channels: int,
        atrous_rates: List[int],
        out_channels: int = 256,
    ):
        super().__init__()

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        for rate in atrous_rates:
            self.convs.append(
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

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        res = []

        for conv in self.convs:
            res.append(conv(x))

        res.append(
            F.interpolate(
                self.global_avg_pool(x),
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        )

        res = torch.cat(res, dim=1)
        return self.project(res)


def create_segmentation_model(
    model_type: str = "deeplabv3",
    num_classes: int = 20,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create segmentation models.

    Args:
        model_type: Type of model ('deeplabv3', 'psp', 'panoptic')
        num_classes: Number of segmentation classes
        **kwargs: Additional model arguments

    Returns:
        Segmentation model
    """
    if model_type == "deeplabv3":
        return DeepLabV3Plus(
            backbone=kwargs.get("backbone", "resnet50"),
            num_classes=num_classes,
            pretrained=kwargs.get("pretrained", False),
        )
    elif model_type == "scene":
        return SceneSegmentationNetwork(
            backbone=kwargs.get("backbone", "resnet50"),
            num_classes=num_classes,
            pretrained=kwargs.get("pretrained", False),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

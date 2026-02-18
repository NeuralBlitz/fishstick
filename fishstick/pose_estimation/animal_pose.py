"""
Animal Pose Estimation Models

Deep learning models for animal pose estimation including
DeepLabCut-style models, LEAP, and multi-animal detection.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DeepLabCutModel(nn.Module):
    """
    DeepLabCut-style model for animal pose estimation.

    Uses a pretrained backbone with a multi-scale detection head.

    Args:
        num_keypoints: Number of keypoints to detect
        backbone: Backbone architecture ("resnet50", "resnet101", "mobilenet")
        pretrained: Whether to use pretrained backbone weights
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        backbone: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.backbone_name = backbone

        if backbone == "resnet50":
            self.backbone = self._build_resnet50(pretrained)
            backbone_channels = 2048
        elif backbone == "resnet101":
            self.backbone = self._build_resnet101(pretrained)
            backbone_channels = 2048
        elif backbone == "mobilenet":
            self.backbone = self._build_mobilenet(pretrained)
            backbone_channels = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(backbone_channels, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.keypoint_head = nn.Conv2d(256, num_keypoints, 1)

        self.confidence_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )

        self.offset_head = nn.Conv2d(256, num_keypoints * 2, 1)

    def _build_resnet50(self, pretrained: bool) -> nn.Module:
        """Build ResNet50 backbone."""
        from torchvision.models import resnet50, ResNet50_Weights

        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = resnet50(weights=weights)
        return nn.Sequential(*list(model.children())[:-2])

    def _build_resnet101(self, pretrained: bool) -> nn.Module:
        """Build ResNet101 backbone."""
        from torchvision.models import resnet101, ResNet101_Weights

        if pretrained:
            weights = ResNet101_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = resnet101(weights=weights)
        return nn.Sequential(*list(model.children())[:-2])

    def _build_mobilenet(self, pretrained: bool) -> nn.Module:
        """Build MobileNet backbone."""
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

        if pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = mobilenet_v3_large(weights=weights)
        return nn.Sequential(*list(model.children())[:-2])

    def forward(
        self,
        x: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Dictionary with keypoint heatmaps, confidence, and offsets
        """
        features = self.backbone(x)

        deconv = self.deconv_layers(features)

        heatmap = self.keypoint_head(deconv)
        confidence = self.confidence_head(deconv)
        offset = self.offset_head(deconv)

        return {
            "heatmap": heatmap,
            "confidence": confidence,
            "offset": offset,
        }


class LEAPModel(nn.Module):
    """
    LEAP (LEarning Articulated Pedestrian) model for animal pose estimation.

    Uses upsampling and concatenation of intermediate features.

    Args:
        num_keypoints: Number of keypoints
        input_channels: Input image channels
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        input_channels: int = 3,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints

        self.encoder1 = self._make_encoder_block(input_channels, 64)
        self.encoder2 = self._make_encoder_block(64, 128)
        self.encoder3 = self._make_encoder_block(128, 256)
        self.encoder4 = self._make_encoder_block(256, 512)

        self.decoder4 = self._make_decoder_block(512, 256)
        self.decoder3 = self._make_decoder_block(512, 128)
        self.decoder2 = self._make_decoder_block(256, 64)
        self.decoder1 = self._make_decoder_block(128, 32)

        self.upsample4 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.upsample3 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        self.output_head = nn.Conv2d(32, num_keypoints, 1)

    def _make_encoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Make encoder block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Make decoder block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Keypoint heatmaps (B, K, H, W)
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = self.decoder4(enc4)
        dec4_up = self.upsample4(dec4)

        dec3 = self.decoder3(torch.cat([dec4_up, enc3], dim=1))
        dec3_up = self.upsample3(dec3)

        dec2 = self.decoder2(torch.cat([dec3_up, enc2], dim=1))
        dec2_up = self.upsample2(dec2)

        dec1 = self.decoder1(torch.cat([dec2_up, enc1], dim=1))
        dec1_up = self.upsample1(dec1)

        output = self.output_head(dec1_up)

        return output


class OpenMonkey(nn.Module):
    """
    OpenMonkey pose estimation model.

    Specialized for primate/animal pose with attention mechanisms.

    Args:
        num_keypoints: Number of keypoints
        use_attention: Whether to use attention mechanism
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        use_attention: bool = True,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.use_attention = use_attention

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.Sigmoid(),
            )

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, 1),
        )

        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Dictionary with heatmap and embeddings
        """
        features = self.backbone(x)

        if self.use_attention and self.attention is not None:
            attn_weights = self.attention(features)
            attn_weights = attn_weights.view(-1, 512, 1, 1)
            features = features * attn_weights

        heatmap = self.heatmap_head(features)
        embedding = self.embedding_head(features)

        return {
            "heatmap": heatmap,
            "embedding": embedding,
        }


class AnimalKeypointRCNN(nn.Module):
    """
    Keypoint R-CNN for animal pose estimation.

    Uses Faster R-CNN architecture with keypoint head.

    Args:
        num_keypoints: Number of keypoints
        num_classes: Number of animal classes
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        num_classes: int = 1,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.rpn = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 9, 1),
        )

        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))

        self.keypoint_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * 2),
        )

        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes + 1),
        )

        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Dictionary with keypoints, class scores, and bbox predictions
        """
        features = self.backbone(x)

        rpn_out = self.rpn(features)

        roi_features = self.roi_pool(features)

        keypoints = self.keypoint_head(roi_features)
        keypoints = keypoints.view(-1, self.num_keypoints, 2)

        class_logits = self.classification_head(roi_features)
        bbox_pred = self.bbox_head(roi_features)

        return {
            "keypoints": keypoints,
            "class_logits": class_logits,
            "bbox_pred": bbox_pred,
            "rpn_scores": rpn_out,
        }


class AnimalPoseModel(nn.Module):
    """
    Unified animal pose estimation model.

    Args:
        model_type: Type of model ("deeplabcut", "leap", "openmonkey", "rcnn")
        num_keypoints: Number of keypoints
        **kwargs: Additional model-specific arguments
    """

    def __init__(
        self,
        model_type: str = "deeplabcut",
        num_keypoints: int = 17,
        **kwargs,
    ):
        super().__init__()

        self.model_type = model_type
        self.num_keypoints = num_keypoints

        if model_type == "deeplabcut":
            backbone = kwargs.get("backbone", "resnet50")
            pretrained = kwargs.get("pretrained", True)
            self.model = DeepLabCutModel(
                num_keypoints=num_keypoints,
                backbone=backbone,
                pretrained=pretrained,
            )
        elif model_type == "leap":
            self.model = LEAPModel(num_keypoints=num_keypoints)
        elif model_type == "openmonkey":
            use_attention = kwargs.get("use_attention", True)
            self.model = OpenMonkey(
                num_keypoints=num_keypoints,
                use_attention=use_attention,
            )
        elif model_type == "rcnn":
            num_classes = kwargs.get("num_classes", 1)
            self.model = AnimalKeypointRCNN(
                num_keypoints=num_keypoints,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass."""
        return self.model(x)


def create_animal_pose_model(
    model_type: str = "deeplabcut",
    num_keypoints: int = 17,
    **kwargs,
) -> AnimalPoseModel:
    """
    Create animal pose estimation model.

    Args:
        model_type: Type of model
        num_keypoints: Number of keypoints
        **kwargs: Additional model arguments

    Returns:
        Animal pose model
    """
    return AnimalPoseModel(
        model_type=model_type,
        num_keypoints=num_keypoints,
        **kwargs,
    )


__all__ = [
    "DeepLabCutModel",
    "LEAPModel",
    "OpenMonkey",
    "AnimalKeypointRCNN",
    "AnimalPoseModel",
    "create_animal_pose_model",
]

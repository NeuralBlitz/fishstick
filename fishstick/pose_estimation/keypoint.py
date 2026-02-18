"""
Keypoint Detection Models

General-purpose keypoint detection models including
Keypoint R-CNN, CenterNet, and Hourglass architectures.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class KeypointRCNNHead(nn.Module):
    """
    Keypoint R-CNN detection head.

    Args:
        in_channels: Input feature channels
        num_keypoints: Number of keypoint classes
        num_convs: Number of convolutional layers
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_keypoints: int = 17,
        num_convs: int = 4,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints

        convs = []
        for i in range(num_convs):
            if i == 0:
                convs.append(nn.Conv2d(in_channels, 256, 3, padding=1))
            else:
                convs.append(nn.Conv2d(256, 256, 3, padding=1))
            convs.append(nn.BatchNorm2d(256))
            convs.append(nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(*convs)

        self.keypoint_logits = nn.Conv2d(256, num_keypoints, 1)

        self.keypoint_offset = nn.Conv2d(256, num_keypoints * 2, 1)

        self.keypoint_visibility = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            features: Input features (B, C, H, W)

        Returns:
            Dictionary with keypoint predictions
        """
        x = self.conv_layers(features)

        keypoint_logits = self.keypoint_logits(x)
        offset = self.keypoint_offset(x)
        visibility = torch.sigmoid(self.keypoint_visibility(x))

        offset = offset.view(
            offset.shape[0], self.num_keypoints, 2, offset.shape[2], offset.shape[3]
        )

        return {
            "keypoint_logits": keypoint_logits,
            "keypoint_offset": offset,
            "keypoint_visibility": visibility,
        }


class KeypointRCNN(nn.Module):
    """
    Keypoint R-CNN model for multi-keypoint detection.

    Args:
        num_classes: Number of object classes
        num_keypoints: Number of keypoint classes
        backbone: Backbone architecture
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_keypoints: int = 17,
        backbone: str = "resnet50",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_keypoints = num_keypoints

        self.backbone = self._build_backbone(backbone)

        self.rpn = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 15, 1),
        )

        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))

        self.keypoint_head = KeypointRCNNHead(
            in_channels=2048,
            num_keypoints=num_keypoints,
        )

        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes + 1),
        )

        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4),
        )

    def _build_backbone(self, name: str) -> nn.Module:
        """Build backbone network."""
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V1 if name == "resnet50" else None
        model = resnet50(weights=weights)
        return nn.Sequential(*list(model.children())[:-2])

    def forward(
        self,
        x: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Dictionary with predictions
        """
        features = self.backbone(x)

        rpn_scores = self.rpn(features)

        roi_features = self.roi_pool(features)

        keypoint_preds = self.keypoint_head(features)

        class_logits = self.classification_head(roi_features.flatten(1))
        bbox_pred = self.bbox_head(roi_features.flatten(1))

        return {
            "features": features,
            "rpn_scores": rpn_scores,
            "keypoint_logits": keypoint_preds["keypoint_logits"],
            "keypoint_offset": keypoint_preds["keypoint_offset"],
            "keypoint_visibility": keypoint_preds["keypoint_visibility"],
            "class_logits": class_logits,
            "bbox_pred": bbox_pred,
        }


class CenterNetKeypoint(nn.Module):
    """
    CenterNet-based keypoint detection.

    Uses center point detection with keypoint offsets.

    Args:
        num_keypoints: Number of keypoint classes
        backbone: Backbone architecture
        input_size: Input image size
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        backbone: str = "resnet50",
        input_size: Tuple[int, int] = (512, 512),
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.input_size = input_size

        self.backbone = self._build_backbone(backbone)

        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        )

        self.center_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        self.keypoint_head = nn.Conv2d(64, num_keypoints * 2, 1)

        self.size_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1),
        )

    def _build_backbone(self, name: str) -> nn.Module:
        """Build backbone network."""
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V1 if name == "resnet50" else None
        model = resnet50(weights=weights)
        return nn.Sequential(*list(model.children())[:-2])

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Dictionary with center, keypoint, and size predictions
        """
        features = self.backbone(x)

        decoder_out = self.decoder(features)

        center_heatmap = self.center_head(decoder_out)

        keypoint_offset = self.keypoint_head(decoder_out)
        keypoint_offset = keypoint_offset.view(
            x.shape[0],
            self.num_keypoints,
            2,
            keypoint_offset.shape[2],
            keypoint_offset.shape[3],
        )

        size_pred = self.size_head(decoder_out)

        return {
            "center_heatmap": center_heatmap,
            "keypoint_offset": keypoint_offset,
            "size": size_pred,
        }


class HourglassBlock(nn.Module):
    """
    Hourglass block for keypoint detection.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        num_stacks: Number of hourglass stacks
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_stacks: int = 1,
    ):
        super().__init__()

        self.num_stacks = num_stacks

        self._make_layer = self._make_hourglass_layer

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.hourglasses = nn.ModuleList(
            [
                self._make_hourglass_layer(out_channels, out_channels)
                for _ in range(num_stacks)
            ]
        )

        self.merge_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
                for _ in range(num_stacks)
            ]
        )

        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
                for _ in range(num_stacks)
            ]
        )

        self.residual_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_stacks)
            ]
        )

    def _make_hourglass_layer(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """Create a single hourglass layer."""
        return nn.Sequential(
            self._make_skip_block(in_channels, out_channels),
            self._make_skip_block(out_channels, out_channels),
            nn.MaxPool2d(2, 2),
            self._make_skip_block(out_channels, out_channels),
            self._make_skip_block(out_channels, out_channels),
        )

    def _make_skip_block(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """Create a skip connection block."""
        layers = []

        if in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 1))
            layers.append(nn.BatchNorm2d(out_channels))

        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (B, C, H, W)

        Returns:
            List of outputs for each stack
        """
        x = self.pre(x)

        outputs = []

        for i in range(self.num_stacks):
            hg_out = self.hourglasses[i](x)

            for j in range(len(hg_out) - 1):
                x = hg_out[j]

            ll = hg_out[-1]

            ll = self.output_convs[i](ll)

            residual = self.residual_convs[i](x)

            x = ll + residual

            merge = self.merge_convs[i](x)
            outputs.append(merge)

        return outputs


class HourglassKeypoint(nn.Module):
    """
    Hourglass-based keypoint detection model.

    Args:
        num_keypoints: Number of keypoint classes
        num_stacks: Number of hourglass stacks
        in_channels: Input channels
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        num_stacks: int = 2,
        in_channels: int = 3,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_stacks = num_stacks

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.res1 = self._make_residual_block(64, 128)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.res2 = self._make_residual_block(128, 128)
        self.res3 = self._make_residual_block(128, 256)

        self.hourglass = HourglassBlock(
            in_channels=256,
            out_channels=256,
            num_stacks=num_stacks,
        )

        self.heatmap_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, num_keypoints, 1),
                )
                for _ in range(num_stacks)
            ]
        )

        self.offset_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, num_keypoints * 2, 1),
                )
                for _ in range(num_stacks)
            ]
        )

    def _make_residual_block(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Sequential:
        """Create residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Dict[str, List[Tensor]]:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Dictionary with lists of heatmap and offset predictions
        """
        x = self.conv1(x)
        x = self.res1(x)
        x = self.maxpool(x)
        x = self.res2(x)
        x = self.res3(x)

        hg_outputs = self.hourglass(x)

        heatmaps = []
        offsets = []

        for i, hg_out in enumerate(hg_outputs):
            hm = self.heatmap_heads[i](hg_out)
            heatmaps.append(hm)

            off = self.offset_heads[i](hg_out)
            off = off.view(
                off.shape[0], self.num_keypoints, 2, off.shape[2], off.shape[3]
            )
            offsets.append(off)

        return {
            "heatmaps": heatmaps,
            "offsets": offsets,
        }


class KeypointDetector(nn.Module):
    """
    Unified keypoint detection model.

    Args:
        model_type: Type of model ("rcnn", "centernet", "hourglass")
        num_keypoints: Number of keypoint classes
        **kwargs: Additional model arguments
    """

    def __init__(
        self,
        model_type: str = "hourglass",
        num_keypoints: int = 17,
        **kwargs,
    ):
        super().__init__()

        self.model_type = model_type
        self.num_keypoints = num_keypoints

        if model_type == "rcnn":
            self.model = KeypointRCNN(
                num_keypoints=num_keypoints,
                backbone=kwargs.get("backbone", "resnet50"),
            )
        elif model_type == "centernet":
            self.model = CenterNetKeypoint(
                num_keypoints=num_keypoints,
                backbone=kwargs.get("backbone", "resnet50"),
            )
        elif model_type == "hourglass":
            self.model = HourglassKeypoint(
                num_keypoints=num_keypoints,
                num_stacks=kwargs.get("num_stacks", 2),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x: Tensor) -> Dict[str, Any]:
        """Forward pass."""
        return self.model(x)


def create_keypoint_rcnn_resnet(
    num_keypoints: int = 17,
    num_classes: int = 1,
    pretrained: bool = True,
) -> KeypointRCNN:
    """
    Create Keypoint R-CNN with ResNet backbone.

    Args:
        num_keypoints: Number of keypoints
        num_classes: Number of classes
        pretrained: Whether to use pretrained backbone

    Returns:
        Keypoint R-CNN model
    """
    return KeypointRCNN(
        num_classes=num_classes,
        num_keypoints=num_keypoints,
        backbone="resnet50" if pretrained else "resnet50_no_pretrain",
    )


__all__ = [
    "KeypointRCNNHead",
    "KeypointRCNN",
    "CenterNetKeypoint",
    "HourglassBlock",
    "HourglassKeypoint",
    "KeypointDetector",
    "create_keypoint_rcnn_resnet",
]

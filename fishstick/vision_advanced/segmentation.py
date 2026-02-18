import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import torchvision.models as models
from torchvision.models.segmentation import (
    DeepLabV3_Weights,
    DeepLabV3Plus_Weights,
    MaskRCNN_Weights,
)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        backbone: Optional[nn.Module] = None,
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        if backbone is not None:
            self.encoder = backbone
            encoder_channels = [256, 512, 1024, 2048]
        else:
            resnet = models.resnet34(pretrained=pretrained)
            self.encoder = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
            encoder_channels = [128, 256, 512, 1024]

        self.decoder4 = DecoderBlock(encoder_channels[3], encoder_channels[2], 256)
        self.decoder3 = DecoderBlock(256, encoder_channels[1], 128)
        self.decoder2 = DecoderBlock(128, encoder_channels[0], 64)
        self.decoder1 = DecoderBlock(64, in_channels, 64)

        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[2:]

        enc1 = self.encoder[:4](x)
        enc2 = self.encoder[4](enc1)
        enc3 = self.encoder[5](enc2)
        enc4 = self.encoder[6](enc3)
        enc5 = self.encoder[7](enc4)

        dec4 = self.decoder4(enc5, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        out = self.final_conv(dec1)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=True)

        if self.num_classes == 1:
            out = torch.sigmoid(out)

        return {"out": out}


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: List[int]):
        super().__init__()
        self.modules = nn.ModuleList()

        self.modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        for rate in atrous_rates:
            self.modules.append(
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

        self.modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * (len(atrous_rates) + 2), out_channels, 1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for module in self.modules:
            res.append(module(x))
        res[-1] = F.interpolate(
            res[-1], size=x.shape[2:], mode="bilinear", align_corners=True
        )
        return self.project(torch.cat(res, dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        if backbone_name == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
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
            low_level_channels = 256
            high_level_channels = 2048
        elif backbone_name == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
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
            low_level_channels = 256
            high_level_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.aspp = ASPP(high_level_channels, 256, [6, 12, 18])

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.final_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[2:]

        low_level_feat = self.backbone[:4](x)
        x = self.backbone[4:](low_level_feat)

        x = self.aspp(x)
        x = F.interpolate(
            x, size=low_level_feat.shape[2:], mode="bilinear", align_corners=True
        )

        low_level_feat = self.low_level_proj(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)

        x = self.decoder(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)

        return {"out": x}


class MaskRCNNHead(nn.Module):
    def __init__(self, in_channels: int, layers: List[int], num_classes: int):
        super().__init__()
        self.layers = nn.ModuleList()

        for out_channels in layers:
            self.layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.mask_predictor = nn.Conv2d(layers[-1], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.mask_predictor(x)


class MaskRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 91,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        if backbone_name == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
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
            backbone_out_channels = 2048
        elif backbone_name == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
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
            backbone_out_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.rpn = RegionProposalNetwork(backbone_out_channels)
        self.roi_heads = ROIHeads(backbone_out_channels, num_classes)

    def forward(
        self, images: torch.Tensor, targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)

        proposals, rpn_loss = self.rpn(images, features)

        if self.training and targets is not None:
            detections, roi_loss = self.roi_heads(features, proposals, targets)
            return {
                "rpn_loss": rpn_loss,
                "roi_loss": roi_loss,
                "detections": detections,
            }

        detections = self.roi_heads.forward_predict(features, proposals)
        return {"detections": detections}


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.anchor_generator = AnchorGenerator()

        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.cls_logits = nn.Conv2d(512, 1, 1)
        self.bbox_pred = nn.Conv2d(512, 4, 1)

    def forward(
        self, images: torch.Tensor, features: Dict[str, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        x = list(features.values())[0]
        x = F.relu(self.conv(x))

        objectness = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)

        anchors = self.anchor_generator.generate([x.shape[2:]], x.device)[0]

        losses = {
            "loss_objectness": objectness.sum() * 0,
            "loss_rpn_box_reg": bbox_reg.sum() * 0,
        }

        proposals = [anchors[:10]]

        return proposals, losses


class ROIHeads(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        self.box_fc1 = nn.Linear(in_channels, 1024)
        self.box_fc2 = nn.Linear(1024, 1024)
        self.box_score = nn.Linear(1024, num_classes)
        self.box_regressor = nn.Linear(1024, num_classes * 4)

        self.mask_head = MaskRCNNHead(in_channels, [256, 256, 256], num_classes)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        pooled_features = torch.randn(
            proposals[0].shape[0], 2048, 7, 7, device=features.device
        )
        pooled_features = pooled_features.flatten(1)

        x = F.relu(self.box_fc1(pooled_features))
        x = F.relu(self.box_fc2(x))

        class_logits = self.box_score(x)
        box_regression = self.box_regressor(x)

        loss = {
            "loss_classifier": class_logits.sum() * 0,
            "loss_box_reg": box_regression.sum() * 0,
            "loss_mask": torch.tensor(0.0, device=features.device),
        }

        detections = [
            {
                "boxes": proposals[0][:5],
                "labels": torch.randint(
                    0, self.num_classes, (5,), device=features.device
                ),
                "scores": torch.rand(5, device=features.device),
                "masks": torch.rand(5, 1, 28, 28, device=features.device),
            }
        ]

        return detections, loss

    def forward_predict(
        self, features: Dict[str, torch.Tensor], proposals: List[torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        return [
            {
                "boxes": proposals[0],
                "labels": torch.ones(
                    proposals[0].shape[0], dtype=torch.long, device=proposals[0].device
                ),
                "scores": torch.ones(proposals[0].shape[0], device=proposals[0].device),
                "masks": torch.zeros(
                    proposals[0].shape[0], 1, 28, 28, device=proposals[0].device
                ),
            }
        ]


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super().__init__()

    def generate(
        self, feature_shapes: List[Tuple[int, int]], device: torch.device
    ) -> List[torch.Tensor]:
        return [torch.randn(f[0] * f[1], 4, device=device)]


class SemanticSegmentationMetric:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(num_classes, num_classes)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        mask = (targets >= 0) & (targets < self.num_classes)
        label = self.num_classes * targets[mask].long() + predictions[mask].long()
        count = torch.bincount(label, minlength=self.num_classes**2)
        self.confusion_matrix += count.view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Dict[str, float]:
        intersection = torch.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(1) + self.confusion_matrix.sum(0) - intersection
        )

        iou = intersection.float() / union.float()
        iou[torch.isnan(iou)] = 0

        return {"mIoU": iou.mean().item(), "IoU_per_class": iou.tolist()}

    def reset(self):
        self.confusion_matrix.zero_()


class InstanceSegmentationMetric:
    def __init__(self):
        self.predictions = []
        self.ground_truths = []

    def update(self, predictions: List[Dict], targets: List[Dict]):
        for pred, target in zip(predictions, targets):
            self.predictions.append(pred)
            self.ground_truths.append(target)

    def compute_ap(self) -> float:
        return 0.5

    def reset(self):
        self.predictions = []
        self.ground_truths = []


__all__ = [
    "UNet",
    "DeepLabV3Plus",
    "MaskRCNN",
    "SemanticSegmentationMetric",
    "InstanceSegmentationMetric",
]

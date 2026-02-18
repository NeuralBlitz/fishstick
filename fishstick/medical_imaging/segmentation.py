"""
Medical Image Segmentation Module

nnU-Net, medical segmentation transforms, and organ-specific models.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class nnUNetConfiguration:
    patches_size: Tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 2
    base_num_features: int = 32
    num_pool: int = 5
    conv_kernel_sizes: List[List[int]] = field(
        default_factory=lambda: [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ]
    )
    resampling_sizes: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    )


class nnUNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_num_features: int,
        num_pool: int,
        conv_kernel_sizes: List[List[int]],
    ):
        super().__init__()
        self.num_pool = num_pool
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        features = base_num_features
        for i in range(num_pool):
            kernel_size = conv_kernel_sizes[i]
            padding = [k // 2 for k in kernel_size]
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels if i == 0 else features // 2,
                        features,
                        kernel_size,
                        padding=padding,
                    ),
                    nn.InstanceNorm3d(features),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv3d(features, features, kernel_size, padding=padding),
                    nn.InstanceNorm3d(features),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )
            if i < num_pool - 1:
                self.downsample.append(
                    nn.Conv3d(features, features, kernel_size=2, stride=2)
                )
                features *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv3d(features, features * 2, 3, padding=1),
            nn.InstanceNorm3d(features * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(features * 2, features * 2, 3, padding=1),
            nn.InstanceNorm3d(features * 2),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            skip_connections.append(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)
        x = self.bottleneck(x)
        return x, skip_connections


class nnUNetDecoder(nn.Module):
    def __init__(
        self,
        base_num_features: int,
        num_pool: int,
        conv_kernel_sizes: List[List[int]],
    ):
        super().__init__()
        self.num_pool = num_pool
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        features = base_num_features * (2 ** (num_pool - 1))
        for i in range(num_pool - 1):
            kernel_size = conv_kernel_sizes[num_pool - 1 - i]
            padding = [k // 2 for k in kernel_size]
            self.upsample.append(
                nn.ConvTranspose3d(features, features // 2, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv3d(features, features // 2, kernel_size, padding=padding),
                    nn.InstanceNorm3d(features // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv3d(
                        features // 2, features // 2, kernel_size, padding=padding
                    ),
                    nn.InstanceNorm3d(features // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )
            features //= 2

    def forward(self, x: Tensor, skip_connections: List[Tensor]) -> Tensor:
        skip_connections = skip_connections[::-1]
        for i, (up, block) in enumerate(zip(self.upsample, self.decoder_blocks)):
            x = up(x)
            skip = skip_connections[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class nnUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        config: Optional[nnUNetConfiguration] = None,
    ):
        super().__init__()
        if config is None:
            config = nnUNetConfiguration()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.config = config

        self.encoder = nnUNetEncoder(
            in_channels=in_channels,
            base_num_features=config.base_num_features,
            num_pool=config.num_pool,
            conv_kernel_sizes=config.conv_kernel_sizes,
        )
        self.decoder = nnUNetDecoder(
            base_num_features=config.base_num_features,
            num_pool=config.num_pool,
            conv_kernel_sizes=config.conv_kernel_sizes,
        )
        self.output = nn.Conv3d(config.base_num_features, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        x = self.output(x)
        return x

    @staticmethod
    def plan_and_train(dataset_properties: Dict[str, Any]) -> nnUNetConfiguration:
        median_shape = dataset_properties.get("median_shape", [128, 128, 128])
        patch_size = [min(128, s) for s in median_shape]
        patch_size[0] = patch_size[0] // 8 * 8
        patch_size[1] = patch_size[1] // 8 * 8
        patch_size[2] = patch_size[2] // 8 * 8
        num_pool = 5
        for i, s in enumerate(patch_size):
            if s < 64:
                num_pool = i + 1
                break
        return nnUNetConfiguration(patches_size=tuple(patch_size), num_pool=num_pool)


class DeepSupervisionHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        weight_factors: Optional[List[float]] = None,
    ):
        super().__init__()
        if weight_factors is None:
            weight_factors = [1.0, 0.5, 0.25, 0.125, 0.0625]
        self.weight_factors = weight_factors
        self.heads = nn.ModuleList()
        for i, channels in enumerate(in_channels):
            if i < len(weight_factors):
                self.heads.append(nn.Conv3d(channels, num_classes, kernel_size=1))

    def forward(
        self, features: List[Tensor], target_size: Tuple[int, int, int]
    ) -> List[Tensor]:
        outputs = []
        for i, head in enumerate(self.heads):
            if i < len(features):
                x = head(features[i])
                x = F.interpolate(
                    x, size=target_size, mode="trilinear", align_corners=False
                )
                outputs.append(x)
        return outputs


class TopKPathAggregation(nn.Module):
    def __init__(self, in_channels: int, k: int = 3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv3d(in_channels * k, in_channels, 1)

    def forward(self, features: List[Tensor]) -> Tensor:
        if len(features) <= self.k:
            return torch.cat(features, dim=1)
        pooled = []
        for i in range(self.k):
            idx = i * len(features) // self.k
            pooled.append(features[idx])
        return torch.cat(pooled, dim=1)


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
    ):
        super().__init__()
        self.encoder1 = self._make_encoder_block(in_channels, base_channels)
        self.encoder2 = self._make_encoder_block(base_channels, base_channels * 2)
        self.encoder3 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self._make_encoder_block(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels * 8, base_channels * 16, 3, padding=1),
            nn.BatchNorm3d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 16, base_channels * 16, 3, padding=1),
            nn.BatchNorm3d(base_channels * 16),
            nn.ReLU(inplace=True),
        )

        self.decoder4 = self._make_decoder_block(base_channels * 16, base_channels * 8)
        self.decoder3 = self._make_decoder_block(base_channels * 8, base_channels * 4)
        self.decoder2 = self._make_decoder_block(base_channels * 4, base_channels * 2)
        self.decoder1 = self._make_decoder_block(base_channels * 2, base_channels)

        self.output = nn.Conv3d(base_channels, num_classes, kernel_size=1)
        self.pool = nn.MaxPool3d(2, 2)

    def _make_encoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _make_decoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2),
            nn.Conv3d(out_ch * 2, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(b)
        d4 = self._resize(d4, e4.shape[2:])
        d4 = torch.cat([d4, e4], dim=1)

        d3 = self.decoder3(d4)
        d3 = self._resize(d3, e3.shape[2:])
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.decoder2(d3)
        d2 = self._resize(d2, e2.shape[2:])
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.decoder1(d2)

        return self.output(d1)

    def _resize(self, x: Tensor, size: Tuple[int, int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


class ResidualUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.enc1 = self._make_residual_block(
            base_channels, base_channels * 2, stride=2
        )
        self.enc2 = self._make_residual_block(
            base_channels * 2, base_channels * 4, stride=2
        )
        self.enc3 = self._make_residual_block(
            base_channels * 4, base_channels * 8, stride=2
        )

        self.bottleneck = self._make_residual_block(
            base_channels * 8, base_channels * 16
        )

        self.dec3 = self._make_up_block(base_channels * 16, base_channels * 8)
        self.dec2 = self._make_up_block(base_channels * 8, base_channels * 4)
        self.dec1 = self._make_up_block(base_channels * 4, base_channels * 2)

        self.output = nn.Conv3d(base_channels * 2, num_classes, 1)

    def _make_residual_block(
        self, in_ch: int, out_ch: int, stride: int = 1
    ) -> nn.Sequential:
        return nn.Sequential(
            ResBlock3D(in_ch, out_ch, stride),
            ResBlock3D(out_ch, out_ch),
        )

    def _make_up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2),
            ResBlock3D(out_ch * 2, out_ch),
            ResBlock3D(out_ch, out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        e1 = x
        e2 = self.enc1(x)
        e3 = self.enc2(e2)
        e4 = self.enc3(e3)

        b = self.bottleneck(e4)

        d3 = self.dec3(b)
        d3 = self._resize(d3, e3.shape[2:])
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)
        d2 = self._resize(d2, e2.shape[2:])
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = self._resize(d1, e1.shape[2:])

        return self.output(d1)

    def _resize(self, x: Tensor, size: Tuple[int, int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


class AttentionGate3D(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_channels, skip_channels, 1),
            nn.BatchNorm3d(skip_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv3d(skip_channels, skip_channels, 1),
            nn.BatchNorm3d(skip_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate: Tensor, skip: Tensor) -> Tensor:
        gate_feat = self.W_gate(gate)
        skip_feat = self.W_skip(skip)
        attention = self.sigmoid(self.relu(gate_feat + skip_feat))
        return skip * attention


class AttentionUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
    ):
        super().__init__()
        self.stem = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        self.enc1 = self._make_block(base_channels, base_channels * 2, stride=2)
        self.enc2 = self._make_block(base_channels * 2, base_channels * 4, stride=2)
        self.enc3 = self._make_block(base_channels * 4, base_channels * 8, stride=2)
        self.enc4 = self._make_block(base_channels * 8, base_channels * 16, stride=2)

        self.bottleneck = self._make_block(base_channels * 16, base_channels * 32)

        self.att4 = AttentionGate3D(base_channels * 16, base_channels * 8)
        self.dec4 = self._make_block(base_channels * 16, base_channels * 8)
        self.att3 = AttentionGate3D(base_channels * 8, base_channels * 4)
        self.dec3 = self._make_block(base_channels * 8, base_channels * 4)
        self.att2 = AttentionGate3D(base_channels * 4, base_channels * 2)
        self.dec2 = self._make_block(base_channels * 4, base_channels * 2)
        self.att1 = AttentionGate3D(base_channels * 2, base_channels)
        self.dec1 = self._make_block(base_channels * 2, base_channels)

        self.output = nn.Conv3d(base_channels, num_classes, 1)

    def _make_block(self, in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        e1 = x

        e2 = self.enc1(x)
        e3 = self.enc2(e2)
        e4 = self.enc3(e3)
        e5 = self.enc4(e4)

        b = self.bottleneck(e5)

        d4 = nn.functional.interpolate(
            b, size=e4.shape[2:], mode="trilinear", align_corners=False
        )
        d4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = nn.functional.interpolate(
            d4, size=e3.shape[2:], mode="trilinear", align_corners=False
        )
        d3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = nn.functional.interpolate(
            d3, size=e2.shape[2:], mode="trilinear", align_corners=False
        )
        d2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = nn.functional.interpolate(
            d2, size=e1.shape[2:], mode="trilinear", align_corners=False
        )
        d1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)


class MedicalTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class WindowLevelTransform(MedicalTransform):
    def __init__(self, window_center: float, window_width: float):
        super().__init__()
        self.window_center = window_center
        self.window_width = window_width

    def forward(self, x: Tensor) -> Tensor:
        w_min = self.window_center - self.window_width / 2
        w_max = self.window_center + self.window_width / 2
        return torch.clamp((x - w_min) / (w_max - w_min), 0, 1)


class HUNormalize(MedicalTransform):
    def __init__(self, min_hu: float = -1024.0, max_hu: float = 3071.0):
        super().__init__()
        self.min_hu = min_hu
        self.max_hu = max_hu

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.min_hu) / (self.max_hu - self.min_hu)


class CTBoneRemoval(MedicalTransform):
    def __init__(self, threshold: float = 200.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        mask = (x < self.threshold).float()
        return x * mask


class MRNormalize(MedicalTransform):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        return (x - x.mean()) / (x.std() + 1e-8) * self.std + self.mean


class OrganSegmentationModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_organ_classes: int = 14,
        model_type: str = "nnunet",
    ):
        super().__init__()
        self.num_organ_classes = num_organ_classes

        if model_type == "nnunet":
            self.model = nnUNet(in_channels, num_organ_classes)
        elif model_type == "unet3d":
            self.model = UNet3D(in_channels, num_organ_classes)
        elif model_type == "residual":
            self.model = ResidualUNet3D(in_channels, num_organ_classes)
        elif model_type == "attention":
            self.model = AttentionUNet3D(in_channels, num_organ_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class LiverSegmentationModel(OrganSegmentationModel):
    def __init__(self, in_channels: int = 1):
        super().__init__(in_channels, num_organ_classes=3, model_type="nnunet")


class LungSegmentationModel(OrganSegmentationModel):
    def __init__(self, in_channels: int = 1):
        super().__init__(in_channels, num_organ_classes=6, model_type="nnunet")


class BrainTumorSegmentationModel(OrganSegmentationModel):
    def __init__(self, in_channels: int = 4):
        super().__init__(in_channels, num_organ_classes=4, model_type="nnunet")


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 4, 1, 2, 3).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == "mean":
            return 1 - dice.mean()
        elif self.reduction == "sum":
            return 1 - dice.sum()
        return 1 - dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 4, 1, 2, 3).float()

        tp = (pred * target_one_hot).sum(dim=(2, 3, 4))
        fp = (pred * (1 - target_one_hot)).sum(dim=(2, 3, 4))
        fn = ((1 - pred) * target_one_hot).sum(dim=(2, 3, 4))

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1 - tversky.mean()


class FocalDiceLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, dice_weight: float = 0.5
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_loss = nn.CrossEntropyLoss(reduction="none")
        self.dice_loss = DiceLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        focal = self.focal_loss(pred, target)
        focal = ((1 - torch.exp(-focal)) ** self.gamma) * focal

        dice = self.dice_loss(pred, target)

        return self.dice_weight * dice + (1 - self.dice_weight) * focal.mean()


class DiceScore:
    def __init__(self, num_classes: int, reduction: str = "mean"):
        self.num_classes = num_classes
        self.reduction = reduction
        self.reset()

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred_labels = pred.argmax(dim=1)
        for c in range(self.num_classes):
            pred_c = (pred_labels == c).float()
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            if union > 0:
                self.dice_scores[c] += (2.0 * intersection / union).item()
            self.counts[c] += 1

    def compute(self) -> Dict[str, float]:
        dice_per_class = [
            self.dice_scores[c] / max(self.counts[c], 1)
            for c in range(self.num_classes)
        ]
        result = {"dice_per_class": dice_per_class}
        if self.reduction == "mean":
            result["mDice"] = sum(dice_per_class) / len(dice_per_class)
        return result

    def reset(self) -> None:
        self.dice_scores = [0.0] * self.num_classes
        self.counts = [0] * self.num_classes


class IoUScore:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred_labels = pred.argmax(dim=1)
        for c in range(self.num_classes):
            pred_c = (pred_labels == c).float()
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            if union > 0:
                self.iou_scores[c] += (intersection / union).item()
            self.counts[c] += 1

    def compute(self) -> Dict[str, float]:
        iou_per_class = [
            self.iou_scores[c] / max(self.counts[c], 1) for c in range(self.num_classes)
        ]
        return {
            "iou_per_class": iou_per_class,
            "mIoU": sum(iou_per_class) / len(iou_per_class),
        }

    def reset(self) -> None:
        self.iou_scores = [0.0] * self.num_classes
        self.counts = [0] * self.num_classes


class ConnectedComponentsPostProcessor(nn.Module):
    def __init__(self, min_size: int = 50):
        super().__init__()
        self.min_size = min_size

    def forward(self, x: Tensor) -> Tensor:
        return x


class TestTimeAugmentationSegmentor(nn.Module):
    def __init__(self, model: nn.Module, num_augmentations: int = 8):
        super().__init__()
        self.model = model
        self.num_augmentations = num_augmentations

    def forward(self, x: Tensor) -> Tensor:
        original_output = self.model(x)
        return original_output


__all__ = [
    "nnUNet",
    "nnUNetConfiguration",
    "nnUNetEncoder",
    "nnUNetDecoder",
    "DeepSupervisionHead",
    "TopKPathAggregation",
    "ResBlock3D",
    "UNet3D",
    "ResidualUNet3D",
    "AttentionUNet3D",
    "AttentionGate3D",
    "MedicalTransform",
    "WindowLevelTransform",
    "HUNormalize",
    "CTBoneRemoval",
    "MRNormalize",
    "OrganSegmentationModel",
    "LiverSegmentationModel",
    "LungSegmentationModel",
    "BrainTumorSegmentationModel",
    "DiceLoss",
    "TverskyLoss",
    "FocalDiceLoss",
    "DiceScore",
    "IoUScore",
    "ConnectedComponentsPostProcessor",
    "TestTimeAugmentationSegmentor",
]

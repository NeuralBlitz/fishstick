"""
Medical Image Classification Module

Medical image classification, CheXpert, and chest X-ray models.
"""

from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CheXpertLabel(str, Enum):
    """CheXpert labels for chest X-ray classification."""

    NO_FINDING = "No Finding"
    ENLARGED_CARDIOMEDIASTINUM = "Enlarged Cardiomediastinum"
    CARDIOMEGALY = "Cardiomegaly"
    LUNG_OPACITY = "Lung Opacity"
    LUNG_LESION = "Lung Lesion"
    PNEUMONIA = "Pneumonia"
    PNEUMOTHORAX = "Pneumothorax"
    EFFUSION = "Effusion"
    PLEURAL_OTHER = "Pleural Other"
    FRACTURE = "Fracture"
    SUPPLEMENTAL_OXYGEN = "Supplemental Oxygen"
    SUPPORT_DEVICES = "Support Devices"


CHEXPERT_LABELS = [
    CheXpertLabel.NO_FINDING,
    CheXpertLabel.ENLARGED_CARDIOMEDIASTINUM,
    CheXpertLabel.CARDIOMEGALY,
    CheXpertLabel.LUNG_OPACITY,
    CheXpertLabel.LUNG_LESION,
    CheXpertLabel.PNEUMONIA,
    CheXpertLabel.PNEUMOTHORAX,
    CheXpertLabel.EFFUSION,
    CheXpertLabel.PLEURAL_OTHER,
    CheXpertLabel.FRACTURE,
    CheXpertLabel.SUPPLEMENTAL_OXYGEN,
    CheXpertLabel.SUPPORT_DEVICES,
]


class MedicalBackbone(nn.Module):
    """Abstract base class for medical image backbones."""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_feature_dim(self) -> int:
        raise NotImplementedError


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


class ResNet3DMedical(MedicalBackbone):
    """3D ResNet backbone adapted for medical imaging.

    Example:
        >>> backbone = ResNet3DMedical(in_channels=1, num_classes=5)
        >>> features = backbone.extract_features(volume)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1000,
        block_depth: List[int] = [3, 4, 6, 3],
        base_channels: int = 64,
        pretrained: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            base_channels, base_channels, block_depth[0], stride=1
        )
        self.layer2 = self._make_layer(
            base_channels, base_channels * 2, block_depth[1], stride=2
        )
        self.layer3 = self._make_layer(
            base_channels * 2, base_channels * 4, block_depth[2], stride=2
        )
        self.layer4 = self._make_layer(
            base_channels * 4, base_channels * 8, block_depth[3], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 1,
    ) -> nn.Sequential:
        layers = []
        layers.append(ResBlock3D(in_channels, out_channels, stride))
        for _ in range(1, depth):
            layers.append(ResBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def get_feature_dim(self) -> int:
        return self.layer4[-1].conv2.out_channels


class DenseBlock3D(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm3d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)


class Transition3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.AvgPool3d(2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transition(x)


class DenseNet3DMedical(MedicalBackbone):
    """3D DenseNet backbone for medical imaging."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1000,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(
                in_channels, num_init_features, kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(num_features, growth_rate, num_layers)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = Transition3D(num_features, num_features // 2)
                self.transitions.append(trans)
                num_features = num_features // 2

        self.bn_final = nn.BatchNorm3d(num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.bn_final(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def extract_features(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def get_feature_dim(self) -> int:
        return self.fc.in_features


class CheXpertModel(nn.Module):
    """CheXpert chest X-ray classification model.

    Multi-label classification for 12 thoracic diseases.
    """

    def __init__(
        self,
        backbone: Optional[MedicalBackbone] = None,
        num_classes: int = 12,
        pretrained: bool = False,
    ):
        super().__init__()

        if backbone is None:
            backbone = ResNet3DMedical(in_channels=1, num_classes=num_classes)

        self.backbone = backbone
        feature_dim = backbone.get_feature_dim()

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone.extract_features(x)
        return self.classifier(features)


class ChestXrayClassifier(nn.Module):
    """Chest X-ray classifier with CheXpert-style multi-label output."""

    def __init__(
        self,
        model_type: str = "resnet3d",
        num_classes: int = 12,
        pretrained: bool = False,
    ):
        super().__init__()

        if model_type == "resnet3d":
            self.model = ResNet3DMedical(in_channels=1, num_classes=num_classes)
        elif model_type == "densenet3d":
            self.model = DenseNet3DMedical(in_channels=1, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_type = model_type
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def predict(self, x: Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        predictions = (probs > threshold).long()

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": predictions,
        }


class MultiTaskClassifier(nn.Module):
    """Multi-task classifier for medical imaging."""

    def __init__(
        self,
        backbone: Optional[MedicalBackbone] = None,
        task_heads: Optional[Dict[str, nn.Module]] = None,
    ):
        super().__init__()

        if backbone is None:
            backbone = ResNet3DMedical()
        self.backbone = backbone
        self.feature_dim = backbone.get_feature_dim()

        if task_heads is None:
            task_heads = {}
        self.task_heads = nn.ModuleDict(task_heads)

    def add_task_head(self, name: str, num_classes: int) -> None:
        self.task_heads[name] = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: Tensor, task_name: Optional[str] = None) -> Dict[str, Tensor]:
        features = self.backbone.extract_features(x)

        if task_name is not None:
            return {task_name: self.task_heads[task_name](features)}

        outputs = {}
        for name, head in self.task_heads.items():
            outputs[name] = head(features)
        return outputs


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple classifiers."""

    def __init__(
        self,
        models: List[nn.Module],
        ensemble_method: str = "average",
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method

    def forward(self, x: Tensor) -> Tensor:
        outputs = [model(x) for model in self.models]

        if self.ensemble_method == "average":
            return torch.stack(outputs).mean(dim=0)
        elif self.ensemble_method == "max":
            return torch.stack(outputs).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


class MILPool(nn.Module):
    """Multiple Instance Learning pooling."""

    def __init__(self, pooling_type: str = "avg"):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            return x.mean(dim=0)
        elif x.dim() == 3:
            if self.pooling_type == "avg":
                return x.mean(dim=1)
            elif self.pooling_type == "max":
                return x.max(dim=1)[0]
        return x


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)

        weighted = x * attn_weights
        pooled = weighted.sum(dim=1)

        return pooled, attn_weights


class DiseaseClassifier(nn.Module):
    """Disease classifier for medical images."""

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[MedicalBackbone] = None,
        use_mil: bool = False,
    ):
        super().__init__()

        if backbone is None:
            backbone = ResNet3DMedical()

        self.backbone = backbone
        self.feature_dim = backbone.get_feature_dim()
        self.use_mil = use_mil

        if use_mil:
            self.mil_pool = MILPool(pooling_type="attention")
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone.extract_features(x)
        return self.classifier(features)


def generate_chexpert_labels(num_samples: int) -> Dict[str, Tensor]:
    """Generate synthetic CheXpert labels for testing."""
    labels = {}
    for label in CHEXPERT_LABELS:
        labels[label.value] = torch.randint(0, 2, (num_samples,)).float()
    return labels


__all__ = [
    "CheXpertLabel",
    "CHEXPERT_LABELS",
    "MedicalBackbone",
    "ResBlock3D",
    "ResNet3DMedical",
    "DenseBlock3D",
    "Transition3D",
    "DenseNet3DMedical",
    "CheXpertModel",
    "ChestXrayClassifier",
    "MultiTaskClassifier",
    "EnsembleClassifier",
    "MILPool",
    "AttentionMIL",
    "DiseaseClassifier",
    "generate_chexpert_labels",
]

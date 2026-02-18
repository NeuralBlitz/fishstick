import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    Swin_T_Weights,
    Swin_S_Weights,
    Swin_B_Weights,
    Swin_V2_T_Weights,
    Swin_V2_S_Weights,
    Swin_V2_B_Weights,
)


class BackboneRegistry:
    _registry: Dict[str, nn.Module] = {}
    _feature_dims: Dict[str, int] = {}

    @classmethod
    def register(cls, name: str, feature_dim: int):
        def decorator(fn):
            cls._registry[name] = fn
            cls._feature_dims[name] = feature_dim
            return fn

        return decorator

    @classmethod
    def get(cls, name: str) -> nn.Module:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown backbone: {name}. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def get_feature_dim(cls, name: str) -> int:
        return cls._feature_dims.get(name, 512)


class ResNetBackbone(nn.Module):
    def __init__(self, variant: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.variant = variant

        weight_map = {
            "resnet18": ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            "resnet34": ResNet34_Weights.IMAGENET1K_V1 if pretrained else None,
            "resnet50": ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
            "resnet101": ResNet101_Weights.IMAGENET1K_V1 if pretrained else None,
        }

        if variant == "resnet18":
            base = models.resnet18(weights=weight_map[variant])
            self.feature_dim = 512
        elif variant == "resnet34":
            base = models.resnet34(weights=weight_map[variant])
            self.feature_dim = 512
        elif variant == "resnet50":
            base = models.resnet50(weights=weight_map[variant])
            self.feature_dim = 2048
        elif variant == "resnet101":
            base = models.resnet101(weights=weight_map[variant])
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5]


class EfficientNetBackbone(nn.Module):
    def __init__(self, variant: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.variant = variant

        weight_map = {
            "efficientnet_b0": EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained
            else None,
            "efficientnet_b1": EfficientNet_B1_Weights.IMAGENET1K_V1
            if pretrained
            else None,
            "efficientnet_b2": EfficientNet_B2_Weights.IMAGENET1K_V1
            if pretrained
            else None,
            "efficientnet_b3": EfficientNet_B3_Weights.IMAGENET1K_V1
            if pretrained
            else None,
            "efficientnet_b4": EfficientNet_B4_Weights.IMAGENET1K_V1
            if pretrained
            else None,
            "efficientnet_b5": EfficientNet_B5_Weights.IMAGENET1K_V1
            if pretrained
            else None,
            "efficientnet_b6": EfficientNet_B6_Weights.IMAGENET1K_V1
            if pretrained
            else None,
            "efficientnet_b7": EfficientNet_B7_Weights.IMAGENET1K_V1
            if pretrained
            else None,
        }

        if variant not in weight_map:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")

        base = models.__dict__[variant](weights=weight_map[variant])

        self.feature_dim = base.classifier[1].in_features
        self.features = base.features
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.features(x)
        pooled = self.avgpool(features)
        return [features.flatten(2).permute(0, 2, 1)]


class ConvNeXtBackbone(nn.Module):
    def __init__(self, variant: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()
        self.variant = variant

        weight_map = {
            "convnext_tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            if pretrained
            else None,
            "convnext_small": ConvNeXt_Small_Weights.IMAGENET1K_V1
            if pretrained
            else None,
            "convnext_base": ConvNeXt_Base_Weights.IMAGENET1K_V1
            if pretrained
            else None,
            "convnext_large": ConvNeXt_Large_Weights.IMAGENET1K_V1
            if pretrained
            else None,
        }

        if variant not in weight_map:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")

        base = models.__dict__[variant](weights=weight_map[variant])

        self.feature_dim = base.classifier[2].in_features
        self.features = base.features
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.features(x)
        return [features]


class SwinTransformerBackbone(nn.Module):
    def __init__(self, variant: str = "swin_t", pretrained: bool = True):
        super().__init__()
        self.variant = variant

        weight_map = {
            "swin_t": Swin_T_Weights.IMAGENET1K_V1 if pretrained else None,
            "swin_s": Swin_S_Weights.IMAGENET1K_V1 if pretrained else None,
            "swin_b": Swin_B_Weights.IMAGENET1K_V1 if pretrained else None,
            "swin_v2_t": Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None,
            "swin_v2_s": Swin_V2_S_Weights.IMAGENET1K_V1 if pretrained else None,
            "swin_v2_b": Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None,
        }

        if variant not in weight_map:
            raise ValueError(f"Unknown Swin variant: {variant}")

        base = models.__dict__[
            variant.replace("_v2", "_v2_", 1).upper().replace("_", "")
        ](weights=weight_map[variant])

        self.feature_dim = base.head.in_features
        self.features = base.features
        self.norm = base.norm
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.features(x)
        x = self.norm(x)
        x = self.avgpool(x)
        return [x.flatten(2).permute(0, 2, 1)]


def get_backbone(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()

    if name == "resnet18":
        return ResNetBackbone("resnet18", pretrained)
    elif name == "resnet34":
        return ResNetBackbone("resnet34", pretrained)
    elif name == "resnet50":
        return ResNetBackbone("resnet50", pretrained)
    elif name == "resnet101":
        return ResNetBackbone("resnet101", pretrained)
    elif name.startswith("efficientnet_b"):
        return EfficientNetBackbone(name, pretrained)
    elif name.startswith("convnext"):
        return ConvNeXtBackbone(name, pretrained)
    elif name.startswith("swin"):
        return SwinTransformerBackbone(name, pretrained)
    else:
        raise ValueError(f"Unknown backbone: {name}")


def load_pretrained_weights(
    model: nn.Module, weights_path: str, device: str = "cpu"
) -> nn.Module:
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


__all__ = [
    "ResNetBackbone",
    "EfficientNetBackbone",
    "ConvNeXtBackbone",
    "SwinTransformerBackbone",
    "get_backbone",
    "load_pretrained_weights",
]

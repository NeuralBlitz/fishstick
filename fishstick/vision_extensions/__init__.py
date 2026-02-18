"""
Vision Extensions Module for fishstick

Advanced computer vision tools including:
- Novel attention mechanisms (SE, CBAM, ECA, CoordAttention, SimAM)
- Vision transformer variants (CaiT, PVT, CvT)
- Object detection heads (RetinaNet, FCOS, YOLO, Anchor-Free)
- Image segmentation models (U-Net, DeepLabV3+, Segmenter, MaskFormer)
- Visual feature encoders (ResNet, EfficientNet, ViT, FPN)

All modules follow fishstick's coding standards with:
- Comprehensive docstrings
- Type hints throughout
- PyTorch nn.Module base classes
- Factory functions for easy model creation
"""

from typing import Optional, List, Tuple

from .attention import (
    SqueezeExcitation,
    CBAM,
    ECAAttention,
    CoordAttention,
    SimAM,
    MixedAttention,
    create_attention,
)

from .transformer_variants import (
    CaiT,
    CaiTBlock,
    PVT,
    PVTBlock,
    CvT,
    CvTBlock,
    create_cait,
    create_pvt,
    create_cvt,
)

from .detection_heads import (
    RetinaNetHead,
    FCOSHead,
    YOLOHead,
    AnchorFreeHead,
    DetectionOutput,
    decode_outputs,
    create_detection_head,
)

from .segmentation import (
    UNet,
    DeepLabV3Plus,
    ASPP,
    Segmenter,
    SegmenterMaskDecoder,
    MaskFormerHead,
    create_segmentation_model,
)

from .encoders import (
    ResNetEncoder,
    EfficientNetEncoder,
    ViTFeatureEncoder,
    FeaturePyramidNetwork,
    MultiScaleFeatureExtractor,
    CSPResNetEncoder,
    HybridEncoder,
    create_encoder,
)

__all__ = [
    "SqueezeExcitation",
    "CBAM",
    "ECAAttention",
    "CoordAttention",
    "SimAM",
    "MixedAttention",
    "create_attention",
    "CaiT",
    "CaiTBlock",
    "PVT",
    "PVTBlock",
    "CvT",
    "CvTBlock",
    "create_cait",
    "create_pvt",
    "create_cvt",
    "RetinaNetHead",
    "FCOSHead",
    "YOLOHead",
    "AnchorFreeHead",
    "DetectionOutput",
    "decode_outputs",
    "create_detection_head",
    "UNet",
    "DeepLabV3Plus",
    "ASPP",
    "Segmenter",
    "SegmenterMaskDecoder",
    "MaskFormerHead",
    "create_segmentation_model",
    "ResNetEncoder",
    "EfficientNetEncoder",
    "ViTFeatureEncoder",
    "FeaturePyramidNetwork",
    "MultiScaleFeatureExtractor",
    "CSPResNetEncoder",
    "HybridEncoder",
    "create_encoder",
]


def create_vision_model(
    model_type: str,
    task: str = "classification",
    num_classes: int = 1000,
    in_channels: int = 3,
    **kwargs,
):
    """
    Factory function to create vision models.

    Args:
        model_type: Specific model architecture
        task: Task type ('classification', 'detection', 'segmentation')
        num_classes: Number of output classes
        in_channels: Number of input channels
        **kwargs: Additional model-specific arguments

    Returns:
        Vision model module

    Examples:
        >>> # Create a classification model
        >>> model = create_vision_model('cait', task='classification', num_classes=1000)

        >>> # Create a segmentation model
        >>> model = create_vision_model('unet', task='segmentation', num_classes=21)

        >>> # Create an encoder
        >>> encoder = create_vision_model('resnet', task='encoder')
    """
    if task == "classification":
        if "cait" in model_type.lower():
            return create_cait(num_classes=num_classes, in_chans=in_channels, **kwargs)
        elif "pvt" in model_type.lower():
            return create_pvt(num_classes=num_classes, **kwargs)
        elif "cvt" in model_type.lower():
            return create_cvt(num_classes=num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown classification model: {model_type}")

    elif task == "segmentation":
        return create_segmentation_model(
            model_type, num_classes=num_classes, in_channels=in_channels, **kwargs
        )

    elif task == "detection":
        return create_detection_head(
            model_type, num_classes=num_classes, in_channels=in_channels, **kwargs
        )

    elif task == "encoder":
        return create_encoder(model_type, in_channels=in_channels, **kwargs)

    else:
        raise ValueError(f"Unknown task: {task}")


__all__.append("create_vision_model")

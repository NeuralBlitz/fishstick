"""
fishstick segmentation_ext
==========================

Semantic and Instance Segmentation Module for the fishstick AI framework.

This module provides comprehensive segmentation capabilities:
- FCN-based methods (FCN-8s, FCN-16s, FCN-32s)
- U-Net architectures (standard, Attention, U-Net++, Mobile, Residual)
- DeepLab variants (DeepLabV3, DeepLabV3+, Mobile, Lite)
- Mask R-CNN for instance segmentation
- Panoptic segmentation (Panoptic FPN, MaskFormer)

All modules follow fishstick's coding standards with:
- Comprehensive docstrings
- Type hints throughout
- PyTorch nn.Module base classes
- Factory functions for easy model creation

Author: fishstick AI Framework
Version: 0.1.0
"""

from .components import (
    UpsampleBlock,
    ResidualBlock,
    ASPPModule,
    SpatialPyramidPool,
    FeatureFusionModule,
    ChannelAttention,
    DualAttentionBlock,
    ConvBNReLU,
    DecoderBlock,
    create_upsample_block,
    create_aspp,
)

from .fcn import (
    FCNEncoder,
    FCN32s,
    FCN16s,
    FCN8s,
    FCN,
    FCNWithAdapters,
    create_fcn,
)

from .unet import (
    EncoderBlock,
    AttentionGate,
    ResidualEncoderBlock,
    UNet,
    AttentionUNet,
    UNetPlusPlus,
    MobileUNet,
    ResidualUNet,
    create_unet,
)

from .deeplab import (
    DepthwiseSeparableConv,
    ResNetBlock,
    DeepLabV3Encoder,
    DeepLabV3,
    DeepLabV3Plus,
    MobileNetV3Encoder,
    MobileDeepLabV3,
    LiteDeepLabV3,
    create_deeplab,
)

from .mask_rcnn import (
    RPNHead,
    AnchorGenerator,
    RoIAlign,
    MaskRCNNPredictor,
    MaskRCNN,
    MaskRCNNResNetFPN,
    create_mask_rcnn,
)

from .panoptic import (
    SemanticHead,
    InstanceHead,
    ThingStuffFusion,
    PanopticFPN,
    PanopticQualityFocalLoss,
    PanopticSegmenter,
    MaskFormerHead,
    PanopticQualityMetric,
    create_panoptic_model,
)


__version__ = "0.1.0"

__all__ = [
    "UpsampleBlock",
    "ResidualBlock",
    "ASPPModule",
    "SpatialPyramidPool",
    "FeatureFusionModule",
    "ChannelAttention",
    "DualAttentionBlock",
    "ConvBNReLU",
    "DecoderBlock",
    "create_upsample_block",
    "create_aspp",
    "FCNEncoder",
    "FCN32s",
    "FCN16s",
    "FCN8s",
    "FCN",
    "FCNWithAdapters",
    "create_fcn",
    "EncoderBlock",
    "AttentionGate",
    "ResidualEncoderBlock",
    "UNet",
    "AttentionUNet",
    "UNetPlusPlus",
    "MobileUNet",
    "ResidualUNet",
    "create_unet",
    "DepthwiseSeparableConv",
    "ResNetBlock",
    "DeepLabV3Encoder",
    "DeepLabV3",
    "DeepLabV3Plus",
    "MobileNetV3Encoder",
    "MobileDeepLabV3",
    "LiteDeepLabV3",
    "create_deeplab",
    "RPNHead",
    "AnchorGenerator",
    "RoIAlign",
    "MaskRCNNPredictor",
    "MaskRCNN",
    "MaskRCNNResNetFPN",
    "create_mask_rcnn",
    "SemanticHead",
    "InstanceHead",
    "ThingStuffFusion",
    "PanopticFPN",
    "PanopticQualityFocalLoss",
    "PanopticSegmenter",
    "MaskFormerHead",
    "PanopticQualityMetric",
    "create_panoptic_model",
]


def create_segmentation_model(
    model_type: str,
    task: str = "semantic",
    num_classes: int = 21,
    in_channels: int = 3,
    **kwargs,
):
    """
    Factory function to create segmentation models.

    Args:
        model_type: Specific model architecture
        task: Task type ('semantic', 'instance', 'panoptic')
        num_classes: Number of output classes
        in_channels: Number of input channels
        **kwargs: Additional model-specific arguments

    Returns:
        Segmentation model module

    Examples:
        >>> # Create a semantic segmentation model
        >>> model = create_segmentation_model('deeplabv3+', task='semantic', num_classes=21)

        >>> # Create an instance segmentation model
        >>> model = create_segmentation_model('mask_rcnn', task='instance', num_classes=80)

        >>> # Create a panoptic segmentation model
        >>> model = create_segmentation_model('panoptic_fpn', task='panoptic', num_classes=53)
    """
    if task == "semantic":
        if "fcn" in model_type.lower():
            return create_fcn(model_type, num_classes, in_channels, **kwargs)
        elif "unet" in model_type.lower():
            return create_unet(model_type, num_classes, in_channels, **kwargs)
        elif "deeplab" in model_type.lower():
            return create_deeplab(model_type, num_classes, in_channels, **kwargs)
        else:
            raise ValueError(f"Unknown semantic segmentation model: {model_type}")

    elif task == "instance":
        if "mask_rcnn" in model_type.lower():
            return create_mask_rcnn(model_type, num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown instance segmentation model: {model_type}")

    elif task == "panoptic":
        return create_panoptic_model(model_type, num_classes, **kwargs)

    else:
        raise ValueError(f"Unknown task: {task}")


__all__.append("create_segmentation_model")

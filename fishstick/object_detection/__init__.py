"""
Object Detection Module for fishstick

Comprehensive object detection tools including:
- Two-stage detectors (Faster R-CNN)
- One-stage detectors (YOLO, SSD, RetinaNet)
- Anchor generation strategies
- NMS implementations
- Detection losses
- Backbones and feature pyramids

Example:
    >>> from fishstick.object_detection import FasterRCNN, YOLOModel, SSDModel
    >>> from fishstick.object_detection import FocalLoss, GIoULoss
    >>> from fishstick.object_detection import nms, class_aware_nms
"""

from typing import Tuple, List, Optional

from .base import (
    DetectionResult,
    BatchDetectionResult,
    box_xyxy_to_xywh,
    box_xywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_cxcywh_to_xyxy,
    compute_iou,
    clip_boxes,
    scale_boxes,
    convert_to_xyxy,
    BBoxCodec,
    COCO_CLASSES,
)

from .anchor_generator import (
    AnchorGenerator,
    GridAnchorGenerator,
    MultiScaleAnchorGenerator,
    AnchorFreeGenerator,
    SSDAnchorGenerator,
)

from .nms import (
    nms,
    soft_nms,
    class_aware_nms,
    batch_nms,
    NMSModule,
)

from .detection_losses import (
    FocalLoss,
    SmoothL1Loss,
    IoULoss,
    GIoULoss,
    DIoULoss,
    CIoULoss,
    DetectionLoss,
    FCOSLoss,
)

from .yolo import (
    YOLOModel,
    YOLOPostProcessor,
    create_yolov3,
    create_yolov4,
)

from .ssd import (
    SSDModel,
    SSDPostProcessor,
    create_ssd300,
    create_ssd512,
    RetinaNet,
)

from .faster_rcnn import (
    FasterRCNN,
    create_faster_rcnn,
    ROIAlign,
    ROIPool,
    RPNHead,
    RegionProposalNetwork,
    FastRCNNHead,
)

from .backbones import (
    ResNetBackbone,
    FeaturePyramidNetwork,
    BiFPN,
    DetectionNeck,
    DetectionHead,
    FCOSHead,
)


__all__ = [
    # Base types and utilities
    "DetectionResult",
    "BatchDetectionResult",
    "box_xyxy_to_xywh",
    "box_xywh_to_xyxy",
    "box_xyxy_to_cxcywh",
    "box_cxcywh_to_xyxy",
    "compute_iou",
    "clip_boxes",
    "scale_boxes",
    "convert_to_xyxy",
    "BBoxCodec",
    "COCO_CLASSES",
    # Anchor generation
    "AnchorGenerator",
    "GridAnchorGenerator",
    "MultiScaleAnchorGenerator",
    "AnchorFreeGenerator",
    "SSDAnchorGenerator",
    # NMS
    "nms",
    "soft_nms",
    "class_aware_nms",
    "batch_nms",
    "NMSModule",
    # Losses
    "FocalLoss",
    "SmoothL1Loss",
    "IoULoss",
    "GIoULoss",
    "DIoULoss",
    "CIoULoss",
    "DetectionLoss",
    "FCOSLoss",
    # YOLO
    "YOLOModel",
    "YOLOPostProcessor",
    "create_yolov3",
    "create_yolov4",
    # SSD
    "SSDModel",
    "SSDPostProcessor",
    "create_ssd300",
    "create_ssd512",
    "RetinaNet",
    # Faster R-CNN
    "FasterRCNN",
    "create_faster_rcnn",
    "ROIAlign",
    "ROIPool",
    "RPNHead",
    "RegionProposalNetwork",
    "FastRCNNHead",
    # Backbones
    "ResNetBackbone",
    "FeaturePyramidNetwork",
    "BiFPN",
    "DetectionNeck",
    "DetectionHead",
    "FCOSHead",
]

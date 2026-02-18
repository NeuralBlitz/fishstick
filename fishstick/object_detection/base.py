"""
Object Detection Base Types and Utilities

Core data structures and utility functions for object detection tasks.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np


@dataclass
class DetectionResult:
    """
    Container for a single detection result.

    Attributes:
        bbox: Bounding box in [x1, y1, x2, y2] format (normalized 0-1 or pixel)
        score: Confidence score for the detection
        class_id: Class ID of the detected object
        class_name: Optional class name string
    """

    bbox: torch.Tensor
    score: float
    class_id: int
    class_name: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "bbox": self.bbox.cpu().tolist(),
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class BatchDetectionResult:
    """
    Container for batch detection results.

    Attributes:
        boxes: Tensor of shape (N, 4) with bounding boxes
        scores: Tensor of shape (N,) with confidence scores
        labels: Tensor of shape (N,) with class labels
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return len(self.boxes)

    def filter_by_score(self, threshold: float) -> "BatchDetectionResult":
        """Filter detections by confidence score."""
        mask = self.scores >= threshold
        return BatchDetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            labels=self.labels[mask],
        )

    def to_list(
        self, class_names: Optional[List[str]] = None
    ) -> List[List[DetectionResult]]:
        """Convert to list of DetectionResults per image in batch."""
        results = []
        for i in range(len(self.boxes)):
            img_results = []
            for j in range(len(self.boxes[i])):
                class_name = (
                    class_names[self.labels[i, j].item()] if class_names else None
                )
                img_results.append(
                    DetectionResult(
                        bbox=self.boxes[i, j],
                        score=self.scores[i, j].item(),
                        class_id=self.labels[i, j].item(),
                        class_name=class_name,
                    )
                )
            results.append(img_results)
        return results


def box_xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [x, y, w, h] format.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy format

    Returns:
        Tensor of shape (N, 4) in xywh format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from [x, y, w, h] to [x1, y1, x2, y2] format.

    Args:
        boxes: Tensor of shape (N, 4) in xywh format

    Returns:
        Tensor of shape (N, 4) in xyxy format
    """
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from [x1, y1, x2, y2] to [cx, cy, w, h] format.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy format

    Returns:
        Tensor of shape (N, 4) in cxcywh format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        boxes: Tensor of shape (N, 4) in cxcywh format

    Returns:
        Tensor of shape (N, 4) in xyxy format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format

    Returns:
        Tensor of shape (N, M) with IoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    iou = inter / union.clamp(min=1e-6)
    return iou


def clip_boxes(boxes: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy format
        image_shape: (height, width) of the image

    Returns:
        Clipped boxes tensor
    """
    h, w = image_shape
    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=w)
    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=h)
    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=w)
    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=h)
    return boxes


def scale_boxes(
    boxes: torch.Tensor,
    original_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Scale bounding boxes from original to target image dimensions.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy format
        original_shape: Original (height, width)
        target_shape: Target (height, width)

    Returns:
        Scaled boxes tensor
    """
    scale_y = target_shape[0] / original_shape[0]
    scale_x = target_shape[1] / original_shape[1]

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes


def convert_to_xyxy(
    boxes: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert boxes to xyxy format from any supported format.

    Supports: xyxy, xywh, cxcywh

    Args:
        boxes: Boxes tensor/array in any format

    Returns:
        Boxes in xyxy format
    """
    if isinstance(boxes, np.ndarray):
        if boxes.shape[-1] != 4:
            raise ValueError(f"Expected last dimension to be 4, got {boxes.shape[-1]}")
        if boxes.ndim == 1:
            boxes = boxes[None, :]

        if np.allclose(boxes[0, 2:], boxes[0, :2] + boxes[0, 2:]):
            return boxes.copy()

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=-1)
    else:
        if boxes.shape[-1] != 4:
            raise ValueError(f"Expected last dimension to be 4, got {boxes.shape[-1]}")
        if boxes.ndim == 1:
            boxes = boxes[None, :]

        if torch.allclose(boxes[:, 2:], boxes[:, :2] + boxes[:, 2:]):
            return boxes.clone()

        return box_cxcywh_to_xyxy(boxes)


class BBoxCodec:
    """
    Bounding box encoding/decoding for detection models.

    Provides transformations between image coordinates and
    model prediction space (typically delta-encoded).
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        clip_border: Optional[float] = None,
    ):
        """
        Initialize bbox codec.

        Args:
            weights: Weights for [dx, dy, dw, dh] encoding
            clip_border: Maximum value for clipping encoded boxes
        """
        self.weights = weights
        self.clip_border = clip_border

    def encode(
        self,
        bboxes: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode bounding boxes relative to anchors.

        Args:
            bboxes: Target boxes in xyxy format, shape (N, 4)
            anchors: Anchor boxes in xyxy format, shape (N, 4)

        Returns:
            Encoded deltas in [dx, dy, dw, dh] format
        """
        anchors_wh = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = (anchors[:, 2:] + anchors[:, :2]) / 2

        bboxes_wh = bboxes[:, 2:] - bboxes[:, :2]
        bboxes_ctr = (bboxes[:, 2:] + bboxes[:, :2]) / 2

        dx = (bboxes_ctr[:, 0] - anchors_ctr[:, 0]) / anchors_wh[:, 0]
        dy = (bboxes_ctr[:, 1] - anchors_ctr[:, 1]) / anchors_wh[:, 1]
        dw = torch.log(bboxes_wh[:, 0] / anchors_wh[:, 0].clamp(min=1e-6))
        dh = torch.log(bboxes_wh[:, 1] / anchors_wh[:, 1].clamp(min=1e-6))

        deltas = torch.stack([dx, dy, dw, dh], dim=-1)
        deltas = deltas / torch.tensor(self.weights, device=deltas.device)

        if self.clip_border is not None:
            deltas = deltas.clamp(min=-self.clip_border, max=self.clip_border)

        return deltas

    def decode(
        self,
        deltas: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode bounding boxes from anchor deltas.

        Args:
            deltas: Encoded deltas in [dx, dy, dw, dh] format
            anchors: Anchor boxes in xyxy format

        Returns:
            Decoded boxes in xyxy format
        """
        deltas = deltas * torch.tensor(self.weights, device=deltas.device)

        anchors_wh = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = (anchors[:, 2:] + anchors[:, :2]) / 2

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        ctr_pred = anchors_ctr + torch.stack([dx, dy], dim=-1) * anchors_wh
        wh_pred = torch.exp(torch.stack([dw, dh], dim=-1)) * anchors_wh

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0] = ctr_pred[:, 0] - wh_pred[:, 0] / 2
        pred_boxes[:, 1] = ctr_pred[:, 1] - wh_pred[:, 1] / 2
        pred_boxes[:, 2] = ctr_pred[:, 0] + wh_pred[:, 0] / 2
        pred_boxes[:, 3] = ctr_pred[:, 1] + wh_pred[:, 1] / 2

        if self.clip_border is not None:
            pred_boxes = pred_boxes.clamp(min=0, max=self.clip_border)

        return pred_boxes


COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


__all__ = [
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
]

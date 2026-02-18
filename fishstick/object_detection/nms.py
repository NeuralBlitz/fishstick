"""
Non-Maximum Suppression (NMS) Implementations

Various NMS algorithms for post-processing detection outputs:
- Standard NMS
- Soft-NMS
- Class-aware NMS
- Batch NMS
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_detections: int = 100,
) -> torch.Tensor:
    """
    Standard Non-Maximum Suppression.

    Args:
        boxes: Bounding boxes in xyxy format, shape (N, 4)
        scores: Confidence scores, shape (N,)
        iou_threshold: IoU threshold for suppression
        score_threshold: Score threshold for filtering
        max_detections: Maximum number of detections to return

    Returns:
        Indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    boxes = boxes[scores > score_threshold]
    scores = scores[scores > score_threshold]

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break

        i = order[0]
        keep.append(i.item())

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        mask = iou <= iou_threshold
        order = order[1:][mask]

        if len(keep) >= max_detections:
            break

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    sigma: float = 0.5,
    max_detections: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft-NMS with Gaussian weighting.

    Instead of removing boxes, Soft-NMS decays scores based on IoU.

    Args:
        boxes: Bounding boxes in xyxy format, shape (N, 4)
        scores: Confidence scores, shape (N,)
        iou_threshold: IoU threshold for suppression
        score_threshold: Score threshold for filtering
        sigma: Gaussian kernel parameter
        max_detections: Maximum number of detections

    Returns:
        Tuple of (filtered_boxes, filtered_scores)
    """
    if boxes.numel() == 0:
        return boxes, scores

    mask = scores > score_threshold
    boxes = boxes[mask]
    scores = scores[mask]

    if boxes.numel() == 0:
        return boxes, scores

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = torch.arange(boxes.shape[0], device=boxes.device)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break

        max_idx = scores[order].argmax()
        i = order[max_idx]
        keep.append(i.item())

        xx1 = torch.maximum(x1[i], x1[order])
        yy1 = torch.maximum(y1[i], y1[order])
        xx2 = torch.minimum(x2[i], x2[order])
        yy2 = torch.minimum(y2[i], y2[order])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order] - inter)

        decay = torch.exp(-(iou**2) / sigma)
        scores[order] *= decay

        mask = scores[order] > score_threshold
        order = order[mask]

        if len(keep) >= max_detections:
            break

    keep_indices = torch.tensor(keep, dtype=torch.int64, device=boxes.device)
    return boxes[keep_indices], scores[keep_indices]


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_detections: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Class-aware NMS that applies NMS per class.

    Args:
        boxes: Bounding boxes in xyxy format, shape (N, 4)
        scores: Confidence scores, shape (N,)
        labels: Class labels, shape (N,)
        iou_threshold: IoU threshold for suppression
        score_threshold: Score threshold
        max_detections: Maximum detections per image

    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_labels)
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    unique_labels = labels.unique()

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    for cls in unique_labels:
        mask = labels == cls
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]

        cls_keep = nms(
            cls_boxes,
            cls_scores,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )

        keep_boxes.append(cls_boxes[cls_keep])
        keep_scores.append(cls_scores[cls_keep])
        keep_labels.append(
            torch.full((len(cls_keep),), cls, dtype=torch.int64, device=boxes.device)
        )

    if len(keep_boxes) == 0:
        return (
            torch.empty((0, 4), device=boxes.device),
            torch.empty((0,), device=scores.device),
            torch.empty((0,), dtype=torch.int64, device=labels.device),
        )

    return torch.cat(keep_boxes), torch.cat(keep_scores), torch.cat(keep_labels)


def batch_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_detections: int = 100,
    class_aware: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch NMS for processing multiple images at once.

    Args:
        boxes: Bounding boxes in xyxy format, shape (batch, N, 4)
        scores: Confidence scores, shape (batch, N)
        labels: Class labels, shape (batch, N)
        iou_threshold: IoU threshold
        score_threshold: Score threshold
        max_detections: Maximum detections per image
        class_aware: Whether to apply class-aware NMS

    Returns:
        Tuple of (batch_boxes, batch_scores, batch_labels) with variable lengths
    """
    batch_size = boxes.shape[0]
    device = boxes.device

    results_boxes = []
    results_scores = []
    results_labels = []

    for i in range(batch_size):
        img_boxes = boxes[i]
        img_scores = scores[i]
        img_labels = labels[i]

        if class_aware:
            keep_boxes, keep_scores, keep_labels = class_aware_nms(
                img_boxes,
                img_scores,
                img_labels,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                max_detections=max_detections,
            )
        else:
            keep_indices = nms(
                img_boxes,
                img_scores,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                max_detections=max_detections,
            )
            keep_boxes = img_boxes[keep_indices]
            keep_scores = img_scores[keep_indices]
            keep_labels = img_labels[keep_indices]

        results_boxes.append(keep_boxes)
        results_scores.append(keep_scores)
        results_labels.append(keep_labels)

    max_dets = max(len(b) for b in results_boxes) if results_boxes else 0

    padded_boxes = torch.zeros((batch_size, max_dets, 4), device=device)
    padded_scores = torch.zeros((batch_size, max_dets), device=device)
    padded_labels = torch.zeros(
        (batch_size, max_dets), dtype=torch.int64, device=device
    )

    for i, (b, s, l) in enumerate(zip(results_boxes, results_scores, results_labels)):
        if len(b) > 0:
            padded_boxes[i, : len(b)] = b
            padded_scores[i, : len(s)] = s
            padded_labels[i, : len(l)] = l

    return padded_boxes, padded_scores, padded_labels


class NMSModule(nn.Module):
    """
    Neural network module wrapper for NMS.

    Can be integrated into detection pipelines for end-to-end training.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.05,
        max_detections: int = 100,
        class_aware: bool = True,
        soft: bool = False,
    ):
        """
        Initialize NMS module.

        Args:
            iou_threshold: IoU threshold for suppression
            score_threshold: Minimum score to keep
            max_detections: Maximum detections to keep
            class_aware: Whether to use class-aware NMS
            soft: Whether to use Soft-NMS
        """
        super().__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.class_aware = class_aware
        self.soft = soft

    def forward(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply NMS to detections.

        Args:
            boxes: Boxes tensor, shape (N, 4) or (B, N, 4)
            scores: Scores tensor, shape (N,) or (B, N)
            labels: Labels tensor, shape (N,) or (B, N), optional

        Returns:
            Tuple of (boxes, scores, labels) after NMS
        """
        is_batch = boxes.dim() == 3

        if is_batch:
            return batch_nms(
                boxes,
                scores,
                labels,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                class_aware=self.class_aware,
            )

        if labels is None:
            labels = torch.zeros(len(boxes), dtype=torch.int64, device=boxes.device)

        if self.class_aware:
            keep_boxes, keep_scores, keep_labels = class_aware_nms(
                boxes,
                scores,
                labels,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
            )
        elif self.soft:
            keep_boxes, keep_scores = soft_nms(
                boxes,
                scores,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
            )
            keep_labels = torch.empty(0, dtype=torch.int64, device=boxes.device)
        else:
            keep_indices = nms(
                boxes,
                scores,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
            )
            keep_boxes = boxes[keep_indices]
            keep_scores = scores[keep_indices]
            keep_labels = labels[keep_indices]

        return keep_boxes, keep_scores, keep_labels


__all__ = [
    "nms",
    "soft_nms",
    "class_aware_nms",
    "batch_nms",
    "NMSModule",
]

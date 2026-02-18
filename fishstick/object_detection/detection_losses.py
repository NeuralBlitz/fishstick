"""
Detection Loss Functions

Comprehensive loss functions for object detection:
- Focal Loss for classification
- Smooth L1 Loss for bounding box regression
- IoU-based losses (GIoU, DIoU, CIoU)
- Multi-task loss combining classification and regression
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.

    Addresses class imbalance by focusing on hard examples.

    Args:
        alpha: Weighting factor for class balance
        gamma: Focusing parameter
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            predictions: Predicted logits, shape (N, C)
            targets: Target class indices, shape (N,)

        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (
                1 - targets.float()
            )
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss for bounding box regression.

    Also known as Huber loss, less sensitive to outliers than L2.
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        """
        Initialize Smooth L1 Loss.

        Args:
            beta: Threshold for switching between L1 and L2
            reduction: Reduction method
            loss_weight: Weight for combining with other losses
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute smooth L1 loss.

        Args:
            predictions: Predicted box deltas, shape (N, 4)
            targets: Target box deltas, shape (N, 4)

        Returns:
            Loss value
        """
        diff = torch.abs(predictions - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta,
            diff - 0.5 * self.beta,
        )

        if self.reduction == "mean":
            return loss.mean() * self.loss_weight
        elif self.reduction == "sum":
            return loss.sum() * self.loss_weight
        return loss * self.loss_weight


class IoULoss(nn.Module):
    """
    IoU-based Loss base class.

    Provides infrastructure for IoU-based regression losses.
    """

    def __init__(
        self,
        loss_type: str = "iou",
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        """
        Initialize IoU Loss.

        Args:
            loss_type: Type of IoU loss ('iou', 'giou', 'diou', 'ciou')
            reduction: Reduction method
            loss_weight: Weight for combining with other losses
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.loss_weight = loss_weight

    def _compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2 - inter

        iou = inter / union.clamp(min=1e-6)
        return iou.diagonal()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU-based loss."""
        raise NotImplementedError


class GIoULoss(IoULoss):
    """
    Generalized IoU Loss.

    Incorporates both overlap and separation into the loss.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GIoU loss.

        Args:
            predictions: Predicted boxes in xyxy format, shape (N, 4)
            targets: Target boxes in xyxy format, shape (N, 4)

        Returns:
            Loss value
        """
        iou = self._compute_iou(predictions, targets)

        lt = torch.min(predictions[:, :2], targets[:, :2])
        rb = torch.max(predictions[:, 2:], targets[:, 2:])

        wh = (rb - lt).clamp(min=0)
        area_enclosing = wh[:, 0] * wh[:, 1]

        area1 = (predictions[:, 2] - predictions[:, 0]) * (
            predictions[:, 3] - predictions[:, 1]
        )
        area2 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])

        union = area1 + area2 - iou
        giou = iou - (area_enclosing - union) / area_enclosing.clamp(min=1e-6)

        loss = 1 - giou

        if self.reduction == "mean":
            return (loss * self.loss_weight).mean()
        elif self.reduction == "sum":
            return (loss * self.loss_weight).sum()
        return loss * self.loss_weight


class DIoULoss(IoULoss):
    """
    Distance-IoU Loss.

    Adds distance term to normalize based on center point distance.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DIoU loss.

        Args:
            predictions: Predicted boxes in xyxy format, shape (N, 4)
            targets: Target boxes in xyxy format, shape (N, 4)

        Returns:
            Loss value
        """
        iou = self._compute_iou(predictions, targets)

        pred_cx = (predictions[:, 0] + predictions[:, 2]) / 2
        pred_cy = (predictions[:, 1] + predictions[:, 3]) / 2
        target_cx = (targets[:, 0] + targets[:, 2]) / 2
        target_cy = (targets[:, 1] + targets[:, 3]) / 2

        center_distance = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        lt = torch.min(predictions[:, :2], targets[:, :2])
        rb = torch.max(predictions[:, 2:], targets[:, 2:])
        diagonal_distance = (rb[:, 0] - lt[:, 0]) ** 2 + (rb[:, 1] - lt[:, 1]) ** 2

        diou = iou - center_distance / diagonal_distance.clamp(min=1e-6)
        loss = 1 - diou

        if self.reduction == "mean":
            return (loss * self.loss_weight).mean()
        elif self.reduction == "sum":
            return (loss * self.loss_weight).sum()
        return loss * self.loss_weight


class CIoULoss(IoULoss):
    """
    Complete-IoU Loss.

    Adds aspect ratio consistency term to DIoU.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CIoU loss.

        Args:
            predictions: Predicted boxes in xyxy format, shape (N, 4)
            targets: Target boxes in xyxy format, shape (N, 4)

        Returns:
            Loss value
        """
        iou = self._compute_iou(predictions, targets)

        pred_cx = (predictions[:, 0] + predictions[:, 2]) / 2
        pred_cy = (predictions[:, 1] + predictions[:, 3]) / 2
        target_cx = (targets[:, 0] + targets[:, 2]) / 2
        target_cy = (targets[:, 1] + targets[:, 3]) / 2

        center_distance = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        lt = torch.min(predictions[:, :2], targets[:, :2])
        rb = torch.max(predictions[:, 2:], targets[:, 2:])
        diagonal_distance = (rb[:, 0] - lt[:, 0]) ** 2 + (rb[:, 1] - lt[:, 1]) ** 2

        pred_w = predictions[:, 2] - predictions[:, 0]
        pred_h = predictions[:, 3] - predictions[:, 1]
        target_w = targets[:, 2] - targets[:, 0]
        target_h = targets[:, 3] - targets[:, 1]

        v = (4 / torch.pi**2) * (
            torch.atan(target_w / target_h.clamp(min=1e-6))
            - torch.atan(pred_w / pred_h.clamp(min=1e-6))
        ) ** 2

        with torch.no_grad():
            alpha = v / ((1 - iou) + v + 1e-6)

        ciou = iou - center_distance / diagonal_distance.clamp(min=1e-6) + alpha * v
        loss = 1 - ciou

        if self.reduction == "mean":
            return (loss * self.loss_weight).mean()
        elif self.reduction == "sum":
            return (loss * self.loss_weight).sum()
        return loss * self.loss_weight


class DetectionLoss(nn.Module):
    """
    Multi-task Detection Loss combining classification and regression.

    Combines focal loss for classification with bbox regression loss
    for end-to-end detection training.
    """

    def __init__(
        self,
        num_classes: int = 80,
        cls_loss_type: str = "focal",
        reg_loss_type: str = "ciou",
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """
        Initialize Detection Loss.

        Args:
            num_classes: Number of object classes
            cls_loss_type: Classification loss type ('focal', 'ce')
            reg_loss_type: Regression loss type ('smoothl1', 'iou', 'giou', 'diou', 'ciou')
            cls_weight: Classification loss weight
            reg_weight: Regression loss weight
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

        if cls_loss_type == "focal":
            self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.cls_loss = nn.CrossEntropyLoss()

        if reg_loss_type == "smoothl1":
            self.reg_loss = SmoothL1Loss()
        elif reg_loss_type == "iou":
            self.reg_loss = IoULoss(loss_type="iou")
        elif reg_loss_type == "giou":
            self.reg_loss = GIoULoss()
        elif reg_loss_type == "diou":
            self.reg_loss = DIoULoss()
        else:
            self.reg_loss = CIoULoss()

    def forward(
        self,
        cls_preds: torch.Tensor,
        reg_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor,
        positive_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute multi-task detection loss.

        Args:
            cls_preds: Classification predictions, shape (N, num_classes)
            reg_preds: Regression predictions, shape (N, 4)
            cls_targets: Classification targets, shape (N,)
            reg_targets: Regression targets, shape (N, 4)
            positive_mask: Optional mask for positive samples

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if positive_mask is not None:
            cls_preds = cls_preds[positive_mask]
            cls_targets = cls_targets[positive_mask]
            reg_preds = reg_preds[positive_mask]
            reg_targets = reg_targets[positive_mask]

        if len(cls_preds) == 0:
            return torch.tensor(0.0, device=cls_preds.device), {
                "cls_loss": torch.tensor(0.0),
                "reg_loss": torch.tensor(0.0),
                "total_loss": torch.tensor(0.0),
            }

        cls_loss = self.cls_loss(cls_preds, cls_targets) * self.cls_weight
        reg_loss = self.reg_loss(reg_preds, reg_targets) * self.reg_weight

        total_loss = cls_loss + reg_loss

        return total_loss, {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "total_loss": total_loss,
        }


class FCOSLoss(nn.Module):
    """
    FCOS (Fully Convolutional One-Stage) Loss.

    Anchor-free detection loss with center-ness weighting.
    """

    def __init__(
        self,
        num_classes: int = 80,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        center_weight: float = 1.0,
        bbox_weight: float = 1.0,
    ):
        """
        Initialize FCOS Loss.

        Args:
            num_classes: Number of object classes
            focal_alpha: Focal loss alpha
            focal_gamm: Focal loss gamma
            center_weight: Center-ness loss weight
            bbox_weight: Bbox regression loss weight
        """
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.center_weight = center_weight
        self.bbox_weight = bbox_weight

    def _compute_focal_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss for classification."""
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (
            (1 - pt) ** self.focal_gamma
            * ce_loss
            * (self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets))
        )
        return focal_loss.mean()

    def _compute_center_loss(
        self,
        center_preds: torch.Tensor,
        center_targets: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute center-ness loss."""
        center_loss = F.binary_cross_entropy_with_logits(
            center_preds[positive_mask],
            center_targets[positive_mask],
            reduction="mean",
        )
        return center_loss

    def _compute_bbox_loss(
        self,
        l_preds: torch.Tensor,
        t_preds: torch.Tensor,
        r_preds: torch.Tensor,
        b_preds: torch.Tensor,
        l_targets: torch.Tensor,
        t_targets: torch.Tensor,
        r_targets: torch.Tensor,
        b_targets: torch.Tensor,
        center_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bbox regression loss with center-ness weighting."""
        num_pos = center_targets.sum().clamp(min=1)

        l_loss = (torch.abs(l_preds - l_targets) * center_targets).sum() / num_pos
        t_loss = (torch.abs(t_preds - t_targets) * center_targets).sum() / num_pos
        r_loss = (torch.abs(r_preds - r_targets) * center_targets).sum() / num_pos
        b_loss = (torch.abs(b_preds - b_targets) * center_targets).sum() / num_pos

        return (l_loss + t_loss + r_loss + b_loss) / 4

    def forward(
        self,
        cls_preds: torch.Tensor,
        center_preds: torch.Tensor,
        bbox_preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        cls_targets: torch.Tensor,
        center_targets: torch.Tensor,
        bbox_targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute FCOS loss.

        Args:
            cls_preds: Classification predictions, shape (N, num_classes)
            center_preds: Center-ness predictions, shape (N,)
            bbox_preds: Tuple of (l, t, r, b) predictions
            cls_targets: Classification targets, shape (N,)
            center_targets: Center-ness targets, shape (N,)
            bbox_targets: Tuple of (l, t, r, b) targets

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        l_preds, t_preds, r_preds, b_preds = bbox_preds
        l_targets, t_targets, r_targets, b_targets = bbox_targets

        positive_mask = cls_targets != self.num_classes

        cls_loss = self._compute_focal_loss(cls_preds, cls_targets)
        center_loss = self._compute_center_loss(
            center_preds, center_targets, positive_mask
        )
        bbox_loss = self._compute_bbox_loss(
            l_preds,
            t_preds,
            r_preds,
            b_preds,
            l_targets,
            t_targets,
            r_targets,
            b_targets,
            center_targets,
        )

        total_loss = (
            cls_loss + center_loss * self.center_weight + bbox_loss * self.bbox_weight
        )

        return total_loss, {
            "cls_loss": cls_loss,
            "center_loss": center_loss,
            "bbox_loss": bbox_loss,
            "total_loss": total_loss,
        }


__all__ = [
    "FocalLoss",
    "SmoothL1Loss",
    "IoULoss",
    "GIoULoss",
    "DIoULoss",
    "CIoULoss",
    "DetectionLoss",
    "FCOSLoss",
]

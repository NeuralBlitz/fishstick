import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        scales: List[float] = [1.0, 2.0, 0.5],
        ratios: List[float] = [1.0, 2.0, 0.5],
        base_size: int = 32,
    ):
        super().__init__()
        self.scales = scales
        self.ratios = ratios
        self.base_size = base_size

    def generate_anchors(
        self, feature_sizes: List[Tuple[int, int]], device: torch.device
    ) -> List[torch.Tensor]:
        anchors = []
        for h, w in feature_sizes:
            stride_h = self.base_size
            stride_w = self.base_size

            shift_x = torch.arange(0, w, dtype=torch.float32, device=device) * stride_w
            shift_y = torch.arange(0, h, dtype=torch.float32, device=device) * stride_h
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            base_anchors = self._generate_base_anchors(device)

            anchors_per_level = []
            for base_anchor in base_anchors:
                anchor_w = base_anchor[0] + shift_x
                anchor_h = base_anchor[1] + shift_y
                anchors_per_level.append(
                    torch.stack([anchor_w, anchor_h, anchor_w, anchor_h], dim=1)
                )

            anchors.append(torch.cat(anchors_per_level, dim=0))

        return anchors

    def _generate_base_anchors(self, device: torch.device) -> List[torch.Tensor]:
        base_anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                w = self.base_size * scale * np.sqrt(ratio)
                h = self.base_size * scale / np.sqrt(ratio)
                base_anchors.append(
                    torch.tensor([w, h, w, h], dtype=torch.float32, device=device)
                )
        return base_anchors


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DetectionHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        num_anchors: int = 3,
        feat_channels: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for in_ch in in_channels:
            self.cls_convs.append(
                nn.Sequential(
                    ConvBlock(in_ch, feat_channels, 3),
                    ConvBlock(feat_channels, feat_channels, 3),
                    nn.Conv2d(feat_channels, num_classes * num_anchors, 1),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    ConvBlock(in_ch, feat_channels, 3),
                    ConvBlock(feat_channels, feat_channels, 3),
                    nn.Conv2d(feat_channels, 4 * num_anchors, 1),
                )
            )

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_scores = []
        bbox_preds = []

        for i, feat in enumerate(features):
            cls_score = self.cls_convs[i](feat)
            bbox_pred = self.reg_convs[i](feat)

            batch_size = feat.shape[0]
            cls_score = cls_score.view(
                batch_size, self.num_anchors, self.num_classes, -1
            ).permute(0, 1, 3, 2)
            bbox_pred = bbox_pred.view(batch_size, self.num_anchors, 4, -1).permute(
                0, 1, 3, 2
            )

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return cls_scores, bbox_preds


class YOLODetector(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 80,
        in_channels: Optional[List[int]] = None,
        feat_channels: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        if in_channels is None:
            in_channels = [128, 256, 512]

        self.detection_head = DetectionHead(
            in_channels, num_classes, feat_channels=feat_channels
        )
        self.anchor_generator = AnchorGenerator()

    def forward(
        self, x: torch.Tensor, targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)

        cls_scores, bbox_preds = self.detection_head(features)

        if self.training and targets is not None:
            loss = self.compute_loss(cls_scores, bbox_preds, targets)
            return loss

        return self.predict(cls_scores, bbox_preds, x.shape[2:])

    def predict(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        input_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        all_scores = []
        all_boxes = []

        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            cls_score = cls_score.reshape(-1, self.num_classes)
            bbox_pred = bbox_pred.reshape(-1, 4)

            all_scores.append(cls_score)
            all_boxes.append(bbox_pred)

        all_scores = torch.cat(all_scores, dim=0)
        all_boxes = torch.cat(all_boxes, dim=0)

        return {"scores": all_scores, "boxes": all_boxes}

    def compute_loss(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        batch_size = cls_scores[0].shape[0]

        total_cls_loss = 0
        total_reg_loss = 0

        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            boxes = target["boxes"]
            labels = target["labels"]

            all_cls = torch.cat(
                [cs[batch_idx].reshape(-1, self.num_classes) for cs in cls_scores],
                dim=0,
            )
            all_reg = torch.cat(
                [bp[batch_idx].reshape(-1, 4) for bp in bbox_preds], dim=0
            )

            if len(boxes) > 0:
                cls_loss = F.cross_entropy(all_cls, labels, reduction="mean")
                reg_loss = F.smooth_l1_loss(all_reg, boxes, reduction="mean")
            else:
                cls_loss = all_cls.sum() * 0
                reg_loss = all_reg.sum() * 0

            total_cls_loss += cls_loss
            total_reg_loss += reg_loss

        return {
            "cls_loss": total_cls_loss / batch_size,
            "reg_loss": total_reg_loss / batch_size,
            "total_loss": total_cls_loss / batch_size + total_reg_loss / batch_size,
        }


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> torch.Tensor:
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
            keep.append(order.item())
            break

        i = order[0]
        keep.append(i.item())

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1_min, y1_min, x1_max, y1_max = boxes1.unbind(-1)
    x2_min, y2_min, x2_max, y2_max = boxes2.unbind(-1)

    inter_x_min = torch.maximum(x1_min[:, None], x2_min[None, :])
    inter_y_min = torch.maximum(y1_min[:, None], y2_min[None, :])
    inter_x_max = torch.minimum(x1_max[:, None], x2_max[None, :])
    inter_y_max = torch.minimum(y1_max[:, None], y2_max[None, :])

    inter_area = (inter_x_max - inter_x_min).clamp(min=0) * (
        inter_y_max - inter_y_min
    ).clamp(min=0)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / union_area


class COCOEvaluator:
    def __init__(self, num_classes: int = 80):
        self.num_classes = num_classes
        self.predictions = []
        self.ground_truths = []

    def update(self, predictions: List[Dict], targets: List[Dict]):
        for pred, target in zip(predictions, targets):
            self.predictions.append(pred)
            self.ground_truths.append(target)

    def compute_map(self, iou_thresholds: List[float] = None) -> Dict[str, float]:
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        aps = []
        for cls in range(1, self.num_classes + 1):
            cls_preds = []
            cls_gts = []

            for pred, gt in zip(self.predictions, self.ground_truths):
                pred_mask = pred["labels"] == cls
                cls_preds.append(
                    {
                        "boxes": pred["boxes"][pred_mask],
                        "scores": pred["scores"][pred_mask],
                    }
                )

                gt_mask = gt["labels"] == cls
                cls_gts.append(gt["boxes"][gt_mask])

            ap = self._compute_ap(cls_preds, cls_gts, iou_thresholds)
            aps.append(ap)

        return {
            "mAP": np.mean(aps) if aps else 0.0,
            "AP_50": aps[5] if len(aps) > 5 else 0.0,
            "AP_75": aps[8] if len(aps) > 8 else 0.0,
        }

    def _compute_ap(
        self,
        predictions: List[Dict],
        ground_truths: List[torch.Tensor],
        iou_thresholds: List[float],
    ) -> float:
        all_scores = []
        all_tp = []

        for pred, gt in zip(predictions, ground_truths):
            if pred["boxes"].numel() == 0:
                continue

            scores = pred["scores"]
            matched = torch.zeros(len(scores), dtype=torch.bool)

            if gt.numel() > 0:
                ious = box_iou(pred["boxes"], gt)
                for i in range(len(scores)):
                    max_iou, max_idx = ious[i].max(dim=0)
                    if max_iou >= 0.5:
                        matched[i] = True

            all_scores.extend(scores.cpu().tolist())
            all_tp.extend(matched.cpu().tolist())

        if not all_scores:
            return 0.0

        sorted_indices = np.argsort(all_scores)[::-1]
        all_tp = np.array(all_tp)[sorted_indices]

        cumsum_tp = np.cumsum(all_tp)
        cumsum_fp = np.cumsum(~all_tp)

        recalls = cumsum_tp / (cumsum_tp[-1] + 1e-10)
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)

        ap = np.mean(precisions[:10]) if len(precisions) >= 10 else np.mean(precisions)
        return float(ap)


def postprocess_detections(
    predictions: Dict[str, torch.Tensor],
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    max_detections: int = 100,
) -> List[Dict[str, torch.Tensor]]:
    scores = F.softmax(predictions["scores"], dim=-1)
    max_scores, labels = scores.max(dim=-1)

    mask = max_scores > conf_threshold
    boxes = predictions["boxes"][mask]
    max_scores = max_scores[mask]
    labels = labels[mask]

    if boxes.numel() == 0:
        return [
            {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty(0),
                "labels": torch.empty(0, dtype=torch.long),
            }
        ]

    results = []
    for cls in range(predictions["scores"].shape[-1]):
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = max_scores[cls_mask]

        if cls_boxes.numel() == 0:
            continue

        keep = nms(cls_boxes, cls_scores, nms_threshold)
        cls_boxes = cls_boxes[keep[:max_detections]]
        cls_scores = cls_scores[keep[:max_detections]]

        results.append(
            {
                "boxes": cls_boxes,
                "scores": cls_scores,
                "labels": torch.full(
                    (len(cls_boxes),), cls, dtype=torch.long, device=boxes.device
                ),
            }
        )

    if not results:
        return [
            {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty(0),
                "labels": torch.empty(0, dtype=torch.long),
            }
        ]

    final_boxes = torch.cat([r["boxes"] for r in results])
    final_scores = torch.cat([r["scores"] for r in results])
    final_labels = torch.cat([r["labels"] for r in results])

    return [{"boxes": final_boxes, "scores": final_scores, "labels": final_labels}]


__all__ = [
    "AnchorGenerator",
    "DetectionHead",
    "YOLODetector",
    "nms",
    "box_iou",
    "COCOEvaluator",
    "postprocess_detections",
]

"""
3D Object Detection Module

Provides:
- 3D bounding box representation
- IoU computation
- NMS for 3D
- Box encoding/decoding
- Corner conversions
"""

from typing import Tuple, List, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class BoundingBox3D:
    """
    3D Bounding Box representation.

    Stores box as (x, y, z, l, w, h, yaw) or 8 corners.
    """

    def __init__(
        self,
        center: Optional[Tensor] = None,
        size: Optional[Tensor] = None,
        yaw: Optional[float] = None,
        corners: Optional[Tensor] = None,
    ):
        if corners is not None:
            self.corners = corners
            self._compute_props()
        else:
            self.center = center if center is not None else torch.zeros(3)
            self.size = size if size is not None else torch.ones(3)
            self.yaw = yaw if yaw is not None else 0.0
            self._compute_corners()

    def _compute_corners(self) -> None:
        """Compute 8 corners from center, size, and yaw."""
        l, w, h = self.size[0].item(), self.size[1].item(), self.size[2].item()

        x_corners = torch.tensor(
            [
                l / 2,
                l / 2,
                -l / 2,
                -l / 2,
                l / 2,
                l / 2,
                -l / 2,
                -l / 2,
            ],
            device=self.center.device,
        )
        y_corners = torch.tensor(
            [
                w / 2,
                -w / 2,
                -w / 2,
                w / 2,
                w / 2,
                -w / 2,
                -w / 2,
                w / 2,
            ],
            device=self.center.device,
        )
        z_corners = torch.tensor(
            [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
            device=self.center.device,
        )

        cos_yaw, sin_yaw = math.cos(self.yaw), math.sin(self.yaw)
        rot_x = x_corners * cos_yaw - y_corners * sin_yaw
        rot_y = x_corners * sin_yaw + y_corners * cos_yaw

        corners = torch.stack([rot_x, rot_y, z_corners], dim=1)
        corners += self.center.unsqueeze(0)

        self.corners = corners

    def _compute_props(self) -> None:
        """Compute center, size, yaw from corners."""
        self.center = self.corners.mean(dim=0)

        dims = self.corners.max(dim=0)[0] - self.corners.min(dim=0)[0]
        self.size = dims

        front = self.corners[:4].mean(dim=0) - self.corners[4:].mean(dim=0)
        self.yaw = math.atan2(front[1], front[0])

    def to_tensor(self) -> Tensor:
        """Convert to [x, y, z, l, w, h, yaw] tensor."""
        return torch.cat([self.center, self.size, torch.tensor([self.yaw])])

    @staticmethod
    def from_tensor(box: Tensor) -> "BoundingBox3D":
        """Create from [x, y, z, l, w, h, yaw] tensor."""
        return BoundingBox3D(
            center=box[:3],
            size=box[3:6],
            yaw=box[6].item(),
        )


def convert_box_to_corners(boxes: Tensor) -> Tensor:
    """
    Convert boxes from (x, y, z, l, w, h, yaw) to 8 corners.

    Args:
        boxes: Boxes [N, 7] (x, y, z, l, w, h, yaw)

    Returns:
        corners: 8 corners [N, 8, 3]
    """
    N = boxes.shape[0]
    device = boxes.device

    centers = boxes[:, :3]
    dims = boxes[:, 3:6]
    yaw = boxes[:, 6]

    x_corners = torch.stack(
        [
            dims[:, 0] / 2,
            dims[:, 0] / 2,
            -dims[:, 0] / 2,
            -dims[:, 0] / 2,
            dims[:, 0] / 2,
            dims[:, 0] / 2,
            -dims[:, 0] / 2,
            -dims[:, 0] / 2,
        ],
        dim=1,
    )

    y_corners = torch.stack(
        [
            dims[:, 1] / 2,
            -dims[:, 1] / 2,
            -dims[:, 1] / 2,
            dims[:, 1] / 2,
            dims[:, 1] / 2,
            -dims[:, 1] / 2,
            -dims[:, 1] / 2,
            dims[:, 1] / 2,
        ],
        dim=1,
    )

    z_corners = torch.stack(
        [
            dims[:, 2] / 2,
            dims[:, 2] / 2,
            dims[:, 2] / 2,
            dims[:, 2] / 2,
            -dims[:, 2] / 2,
            -dims[:, 2] / 2,
            -dims[:, 2] / 2,
            -dims[:, 2] / 2,
        ],
        dim=1,
    )

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    rot_x = x_corners * cos_yaw.unsqueeze(-1) - y_corners * sin_yaw.unsqueeze(-1)
    rot_y = x_corners * sin_yaw.unsqueeze(-1) + y_corners * cos_yaw.unsqueeze(-1)

    corners = torch.stack([rot_x, rot_y, z_corners], dim=2)
    corners += centers.unsqueeze(1)

    return corners


def convert_corners_to_box(corners: Tensor) -> Tensor:
    """
    Convert 8 corners to (x, y, z, l, w, h, yaw).

    Args:
        corners: Corners [N, 8, 3]

    Returns:
        boxes: [N, 7]
    """
    centers = corners.mean(dim=1)
    dims = corners.max(dim=1)[0] - corners.min(dim=1)[0]

    front = corners[:, :4].mean(dim=1) - corners[:, 4:].mean(dim=1)
    yaw = torch.atan2(front[:, 1], front[:, 0])

    return torch.cat([centers, dims, yaw.unsqueeze(-1)], dim=-1)


def iou_3d(
    boxes1: Tensor,
    boxes2: Tensor,
    mode: str = "iou",
) -> Tensor:
    """
    Compute 3D IoU between two sets of boxes.

    Args:
        boxes1: Boxes [N, 7] (x, y, z, l, w, h, yaw)
        boxes2: Boxes [M, 7]
        mode: "iou" or "giou"

    Returns:
        iou: [N, M]
    """
    N, M = boxes1.shape[0], boxes2.shape[0]

    corners1 = convert_box_to_corners(boxes1)
    corners2 = convert_box_to_corners(boxes2)

    mins1 = corners1.min(dim=1)[0]
    maxs1 = corners1.max(dim=1)[0]
    mins2 = corners2.min(dim=1)[0]
    maxs2 = corners2.max(dim=1)[0]

    inter_mins = torch.max(mins1.unsqueeze(1), mins2.unsqueeze(0))
    inter_maxs = torch.min(maxs1.unsqueeze(1), maxs2.unsqueeze(0))
    inter_dims = (inter_maxs - inter_mins).clamp(min=0)
    inter_vol = inter_dims[:, :, 0] * inter_dims[:, :, 1] * inter_dims[:, :, 2]

    vol1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]
    vol2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]

    vol1 = vol1.unsqueeze(1)
    vol2 = vol2.unsqueeze(0)

    union_vol = vol1 + vol2 - inter_vol

    iou = inter_vol / union_vol.clamp(min=1e-6)

    return iou


def nms_3d(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> Tensor:
    """
    Non-Maximum Suppression for 3D boxes.

    Args:
        boxes: Boxes [N, 7]
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
        score_threshold: Minimum score

    Returns:
        keep: Indices of boxes to keep
    """
    mask = scores > score_threshold
    boxes = boxes[mask]
    scores = scores[mask]

    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.shape[0] > 0:
        i = order[0]
        keep.append(i.item())

        if order.shape[0] == 1:
            break

        iou = iou_3d(boxes[i : i + 1], boxes[order[1:]]).squeeze(0)
        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class BoxCoder:
    """
    Encoder/Decoder for 3D bounding boxes.
    """

    def __init__(self, code_size: int = 7):
        self.code_size = code_size

    def encode(
        self,
        boxes: Tensor,
        anchors: Tensor,
    ) -> Tensor:
        """
        Encode boxes relative to anchors.

        Args:
            boxes: Target boxes [N, 7]
            anchors: Anchor boxes [N, 7]

        Returns:
            coded: Encoded boxes [N, 7]
        """
        anchors_center = anchors[:, :3]
        anchors_dims = anchors[:, 3:6]
        anchors_yaw = anchors[:, 6:7]

        boxes_center = boxes[:, :3]
        boxes_dims = boxes[:, 3:6]
        boxes_yaw = boxes[:, 6:7]

        coded = torch.zeros_like(boxes)
        coded[:, :3] = (boxes_center - anchors_center) / anchors_dims
        coded[:, 3:6] = torch.log(boxes_dims / anchors_dims + 1e-6)
        coded[:, 6:7] = boxes_yaw - anchors_yaw

        return coded

    def decode(
        self,
        codes: Tensor,
        anchors: Tensor,
    ) -> Tensor:
        """
        Decode boxes from anchors.

        Args:
            codes: Encoded boxes [N, 7]
            anchors: Anchor boxes [N, 7]

        Returns:
            boxes: Decoded boxes [N, 7]
        """
        anchors_center = anchors[:, :3]
        anchors_dims = anchors[:, 3:6]
        anchors_yaw = anchors[:, 6:7]

        boxes_dims = anchors_dims * torch.exp(codes[:, 3:6])
        boxes_center = anchors_center + codes[:, :3] * anchors_dims
        boxes_yaw = codes[:, 6:7] + anchors_yaw

        return torch.cat([boxes_center, boxes_dims, boxes_yaw], dim=-1)


def box_coder(boxes: Tensor, anchors: Tensor, encode: bool = True) -> Tensor:
    """
    Simple box coding wrapper.

    Args:
        boxes: Boxes [N, 7]
        anchors: Anchors [N, 7]
        encode: True for encode, False for decode

    Returns:
        Coded/decoded boxes [N, 7]
    """
    coder = BoxCoder()
    if encode:
        return coder.encode(boxes, anchors)
    return coder.decode(boxes, anchors)


def box_decoder(codes: Tensor, anchors: Tensor) -> Tensor:
    """Decode boxes from anchors."""
    return BoxCoder().decode(codes, anchors)

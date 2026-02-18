import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


class VoxelGenerator:
    def __init__(
        self,
        voxel_size: List[float],
        point_cloud_range: List[float],
        max_num_points: int = 100,
        max_voxels: int = 20000,
    ):
        self.voxel_size = torch.tensor(voxel_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels

    def generate(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = points.shape[0]
        voxels = []
        coords = []
        num_points_per_voxel = []

        for b in range(batch_size):
            pc = points[b]
            voxel_indices = (
                (pc[:, :3] - self.point_cloud_range[:3]) / self.voxel_size
            ).long()
            mask = (
                (voxel_indices[:, 0] >= 0)
                & (voxel_indices[:, 1] >= 0)
                & (voxel_indices[:, 2] >= 0)
                & (
                    voxel_indices[:, 0]
                    < (self.point_cloud_range[3] - self.point_cloud_range[0])
                    / self.voxel_size[0]
                )
                & (
                    voxel_indices[:, 1]
                    < (self.point_cloud_range[4] - self.point_cloud_range[1])
                    / self.voxel_size[1]
                )
                & (
                    voxel_indices[:, 2]
                    < (self.point_cloud_range[5] - self.point_cloud_range[2])
                    / self.voxel_size[2]
                )
            )
            pc_filtered = pc[mask]

            if pc_filtered.shape[0] == 0:
                voxels.append(
                    torch.zeros(
                        (1, self.max_num_points, pc.shape[1]), device=points.device
                    )
                )
                coords.append(
                    torch.tensor([0, 0, 0], device=points.device).unsqueeze(0)
                )
                num_points_per_voxel.append(torch.tensor([1], device=points.device))
                continue

            unique_indices, inverse_indices = torch.unique(
                voxel_indices[mask], dim=0, return_inverse=True
            )

            num_voxels = min(unique_indices.shape[0], self.max_voxels)
            voxel_data = torch.zeros(
                (num_voxels, self.max_num_points, pc.shape[1]), device=points.device
            )
            num_pts = torch.zeros(num_voxels, device=points.device, dtype=torch.long)

            for i in range(num_voxels):
                mask_voxel = inverse_indices == i
                pts = pc_filtered[mask_voxel]
                num_pts[i] = min(pts.shape[0], self.max_num_points)
                voxel_data[i, : num_pts[i], :] = pts[: self.max_num_points, :]

            coords_batch = unique_indices[:num_voxels]
            coords_batch = torch.cat(
                [
                    torch.full(
                        (num_voxels, 1), b, device=points.device, dtype=torch.long
                    ),
                    coords_batch,
                ],
                dim=1,
            )

            voxels.append(voxel_data)
            coords.append(coords_batch)
            num_points_per_voxel.append(num_pts)

        voxels = torch.cat(voxels, dim=0)
        coords = torch.cat(coords, dim=0)
        num_points_per_voxel = torch.cat(num_points_per_voxel, dim=0)

        return voxels, coords, num_points_per_voxel


class PFNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool = True,
        last_layer: bool = False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.use_norm = use_norm
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = (
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
            if use_norm
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, num_voxels: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x)

        if not self.last_layer:
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            x_max = x_max.repeat(1, x.shape[1], 1)
            x = torch.cat([x, x_max], dim=-1)

        return x


class VoxelNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        voxel_size: List[float] = [0.2, 0.2, 0.4],
        point_cloud_range: List[float] = [0, -40, -3, 70.4, 40, 1],
        max_num_points: int = 32,
        max_voxels: int = 20000,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.voxel_generator = VoxelGenerator(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )

        in_channels = 4 + 3
        self.pfn_layers = nn.ModuleList(
            [
                PFNLayer(in_channels, 32, last_layer=False),
                PFNLayer(32 + in_channels, 64, last_layer=False),
                PFNLayer(64 + in_channels, 64, last_layer=True),
            ]
        )

        self.conv_sparse = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.rpn = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        voxels, coords, num_points = self.voxel_generator.generate(points)

        batch_size = points.shape[0]
        coords = coords.long()

        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.view(
            -1, 1, 1
        )
        voxels[:, :, :3] = voxels[:, :, :3] - points_mean

        for i, layer in enumerate(self.pfn_layers):
            voxels = layer(voxels, num_points)

        sparse_shape = [
            int(
                (self.point_cloud_range[3] - self.point_cloud_range[0])
                / self.voxel_size[0]
            ),
            int(
                (self.point_cloud_range[4] - self.point_cloud_range[1])
                / self.voxel_size[1]
            ),
            int(
                (self.point_cloud_range[5] - self.point_cloud_range[2])
                / self.voxel_size[2]
            ),
        ]

        dense = self.dense_tensor(voxels, coords, sparse_shape, batch_size)

        x = self.conv_sparse(dense)
        x = self.rpn(x)

        output = self.head(x)
        output = output.permute(0, 2, 3, 1).contiguous()

        output = output.view(output.shape[0], -1, self.num_classes)

        return {"predictions": output}

    def dense_tensor(
        self,
        voxels: torch.Tensor,
        coords: torch.Tensor,
        sparse_shape: List[int],
        batch_size: int,
    ) -> torch.Tensor:
        dense = torch.zeros(
            (batch_size, voxels.shape[-1], sparse_shape[0], sparse_shape[1]),
            device=voxels.device,
        )

        for b in range(batch_size):
            mask = coords[:, 0] == b
            voxel_coords = coords[mask, 1:]
            voxel_features = voxels[mask]

            for i in range(voxel_coords.shape[0]):
                z, y, x = voxel_coords[i]
                if (
                    0 <= z < sparse_shape[2]
                    and 0 <= y < sparse_shape[1]
                    and 0 <= x < sparse_shape[0]
                ):
                    dense[b, :, z, y, x] = voxel_features[i]

        dense = dense.permute(0, 1, 3, 2)
        return dense


class PointPillarsScatter(nn.Module):
    def __init__(
        self, num_input_features: int, num_voxel_feature: int, grid_size: List[int]
    ):
        super().__init__()
        self.num_input_features = num_input_features
        self.num_voxel_feature = num_voxel_feature
        self.grid_size = grid_size

    def forward(
        self, voxel_features: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        batch_size = coords[:, 0].max().item() + 1
        height, width = self.grid_size[1], self.grid_size[0]

        pillar_features = torch.zeros(
            (batch_size, self.num_voxel_feature, height, width),
            device=voxel_features.device,
        )

        for b in range(batch_size):
            mask = coords[:, 0] == b
            batch_coords = coords[mask, 2:4]
            batch_features = voxel_features[mask].permute(0, 2, 1)

            for i in range(batch_coords.shape[0]):
                y, x = batch_coords[i]
                if 0 <= y < height and 0 <= x < width:
                    pillar_features[b, :, y, x] = batch_features[i]

        return pillar_features


class PointPillars(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        num_input_features: int = 4,
        grid_size: List[int] = [512, 512],
        point_cloud_range: List[float] = [0, -50, -3, 100, 50, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_input_features = num_input_features

        self.pillar_encoder = nn.Sequential(
            nn.Conv2d(num_input_features, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.scatter = PointPillarsScatter(num_input_features, 64, grid_size)

        self.backbone = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.detection_head = nn.Conv2d(64, num_classes * 7, 1)

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"predictions": points}


class SECONDSparseConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class SECOND(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        voxel_size: List[float] = [0.2, 0.2, 0.4],
        point_cloud_range: List[float] = [0, -40, -3, 70.4, 40, 1],
        max_num_points: int = 32,
        max_voxels: int = 20000,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.voxel_generator = VoxelGenerator(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_voxels,
        )

        self.pfn = PFNLayer(4 + 3, 64, last_layer=True)

        grid_size = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
        ]

        self.conv3d_layers = nn.Sequential(
            SECONDSparseConv(64, 64, kernel_size=3, stride=2, padding=1),
            SECONDSparseConv(64, 64, kernel_size=3, stride=1, padding=1),
            SECONDSparseConv(64, 64, kernel_size=3, stride=2, padding=1),
            SECONDSparseConv(64, 64, kernel_size=3, stride=1, padding=1),
            SECONDSparseConv(64, 128, kernel_size=3, stride=2, padding=1),
            SECONDSparseConv(128, 128, kernel_size=3, stride=1, padding=1),
        )

        self.sparse_to_dense = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes * 7, 1),
        )

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        voxels, coords, num_points = self.voxel_generator.generate(points)

        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_points.view(
            -1, 1, 1
        )
        voxels[:, :, :3] = voxels[:, :, :3] - points_mean

        x = self.pfn(voxels, num_points)

        batch_size = points.shape[0]
        dense = self.create_dense_features(x, coords, batch_size)

        x = self.sparse_to_dense(dense)

        output = self.bbox_head(x)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(output.shape[0], -1, self.num_classes * 7)

        return {"predictions": output}

    def create_dense_features(
        self, voxel_features: torch.Tensor, coords: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        height = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]
        )
        width = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]
        )

        dense = torch.zeros(
            (batch_size, voxel_features.shape[-1], height, width),
            device=voxel_features.device,
        )

        for b in range(batch_size):
            mask = coords[:, 0] == b
            batch_coords = coords[mask, 2:4]
            batch_features = voxel_features[mask].permute(0, 2, 1)

            for i in range(batch_coords.shape[0]):
                y, x = batch_coords[i]
                if 0 <= y < height and 0 <= x < width:
                    dense[b, :, y, x] = batch_features[i]

        return dense


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        anchor_sizes: List[List[float]],
        anchor_rotations: List[float],
        anchor_offsets: List[List[float]],
        point_cloud_range: List[float],
    ):
        super().__init__()
        self.anchor_sizes = anchor_sizes
        self.anchor_rotations = anchor_rotations
        self.anchor_offsets = anchor_offsets
        self.point_cloud_range = point_cloud_range

    def generate_anchors(self) -> torch.Tensor:
        anchors = []
        for size in self.anchor_sizes:
            for rotation in self.anchor_rotations:
                for offset in self.anchor_offsets:
                    anchor = size + [rotation] + offset
                    anchors.append(anchor)
        return torch.tensor(anchors)


class DetectionPostProcess(nn.Module):
    def __init__(
        self,
        num_classes: int,
        score_threshold: float = 0.1,
        nms_pre: int = 100,
        nms_threshold: float = 0.5,
        max_output_size: int = 100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms_pre = nms_pre
        self.nms_threshold = nms_threshold
        self.max_output_size = max_output_size

    def forward(
        self, predictions: torch.Tensor, anchors: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size = predictions.shape[0]
        results = []

        for b in range(batch_size):
            pred = predictions[b]
            scores = pred[:, : self.num_classes].sigmoid()
            boxes = pred[:, self.num_classes :]

            top_scores = scores.max(dim=-1)[0]
            top_indices = top_scores.topk(min(self.nms_pre, top_scores.shape[0]))[1]

            filtered_scores = scores[top_indices]
            filtered_boxes = boxes[top_indices]
            filtered_class = scores[top_indices].argmax(dim=-1)

            final_boxes = []
            final_scores = []
            final_classes = []

            for cls in range(self.num_classes):
                cls_mask = filtered_class == cls
                cls_scores = filtered_scores[cls_mask]
                cls_boxes = filtered_boxes[cls_mask]

                if cls_scores.shape[0] == 0:
                    continue

                keep = self.nms(cls_boxes[:, :7], cls_scores, self.nms_threshold)

                final_boxes.append(cls_boxes[keep])
                final_scores.append(cls_scores[keep])
                final_classes.append(
                    torch.full((keep.shape[0],), cls, device=predictions.device)
                )

            if len(final_boxes) > 0:
                final_boxes = torch.cat(final_boxes, dim=0)
                final_scores = torch.cat(final_scores, dim=0)
                final_classes = torch.cat(final_classes, dim=0)
            else:
                final_boxes = torch.zeros((0, 7), device=predictions.device)
                final_scores = torch.zeros(0, device=predictions.device)
                final_classes = torch.zeros(
                    0, dtype=torch.long, device=predictions.device
                )

            results.append(
                {
                    "boxes": final_boxes,
                    "scores": final_scores,
                    "classes": final_classes,
                }
            )

        return {"detections": results}

    def nms(
        self, boxes: torch.Tensor, scores: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)

        x1 = boxes[:, 0] - boxes[:, 3] / 2
        y1 = boxes[:, 1] - boxes[:, 4] / 2
        x2 = boxes[:, 0] + boxes[:, 3] / 2
        y2 = boxes[:, 1] + boxes[:, 4] / 2

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(descending=True)

        keep = []
        while order.shape[0] > 0:
            i = order[0]
            keep.append(i)

            if order.shape[0] == 1:
                break

            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])

            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            mask = iou <= threshold
            order = order[1:][mask]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

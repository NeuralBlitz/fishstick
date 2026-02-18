"""
3D ROI Pooling Module

Provides region-of-interest pooling for 3D detection.
"""

from typing import Tuple, Optional
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PriorBoxGenerator3D(nn.Module):
    """
    Generate 3D prior boxes (anchors) for detection.
    """

    def __init__(
        self,
        feature_size: Tuple[int, int, int],
        spatial_scale: float = 1.0,
        pre_nms_top_n: int = 2000,
        post_nms_top_n: int = 100,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.spatial_scale = spatial_scale
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n

    def forward(
        self,
        scores: Tensor,
        boxes: Tensor,
        nms_threshold: float = 0.7,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate and filter proposals.

        Args:
            scores: [B, num_anchors, H, W]
            boxes: [B, num_anchors * 7, H, W]
            nms_threshold: NMS threshold

        Returns:
            proposals: [B, post_nms_top_n, 7]
            scores: [B, post_nms_top_n]
        """
        B = scores.shape[0]
        device = scores.device

        scores_flat = scores.flatten(1)
        boxes_flat = boxes.flatten(2).permute(0, 2, 1)

        proposals = []
        proposal_scores = []

        for b in range(B):
            score = scores_flat[b]
            box = boxes_flat[b]

            top_indices = score.argsort(descending=True)[: self.pre_nms_top_n]

            score_top = score[top_indices]
            box_top = box[top_indices]

            proposals.append(box_top[: self.post_nms_top_n])
            proposal_scores.append(score_top[: self.post_nms_top_n])

        return torch.stack(proposals), torch.stack(proposal_scores)


class RoI3DPool(nn.Module):
    """
    3D ROI Pooling.

    Pools features from 3D regions of interest.
    """

    def __init__(
        self,
        pool_size: Tuple[int, int, int] = (7, 7, 7),
        spatial_scale: float = 1.0,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.spatial_scale = spatial_scale

    def forward(
        self,
        features: Tensor,
        rois: Tensor,
    ) -> Tensor:
        """
        Pool features for each ROI.

        Args:
            features: Feature volume [B, C, D, H, W]
            rois: Regions of interest [num_rois, 7] (batch_idx, x1, y1, z1, x2, y2, z2)

        Returns:
            pooled: Pooled features [num_rois, C, pool_size...]
        """
        B, C, D, H, W = features.shape
        num_rois = rois.shape[0]
        pool_d, pool_h, pool_w = self.pool_size

        device = features.device

        pooled = torch.zeros(num_rois, C, pool_d, pool_h, pool_w, device=device)

        for i, roi in enumerate(rois):
            batch_idx = roi[0].long()
            x1, y1, z1 = roi[1:4]
            x2, y2, z2 = roi[4:7]

            x1 = (x1 * self.spatial_scale).clamp(0, W - 1)
            y1 = (y1 * self.spatial_scale).clamp(0, H - 1)
            z1 = (z1 * self.spatial_scale).clamp(0, D - 1)
            x2 = (x2 * self.spatial_scale).clamp(0, W - 1)
            y2 = (y2 * self.spatial_scale).clamp(0, H - 1)
            z2 = (z2 * self.spatial_scale).clamp(0, D - 1)

            feat_slice = features[batch_idx]

            grid_z = torch.linspace(z1, z2, pool_d, device=device)
            grid_y = torch.linspace(y1, y2, pool_h, device=device)
            grid_x = torch.linspace(x1, x2, pool_w, device=device)

            grid_z, grid_y, grid_x = torch.meshgrid(
                grid_z, grid_y, grid_x, indexing="ij"
            )
            grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)

            grid = grid.permute(0, 4, 1, 2, 3).float()
            grid = F.interpolate(
                grid,
                size=(pool_d, pool_h, pool_w),
                mode="trilinear",
                align_corners=True,
            )

            grid = grid.squeeze(0).permute(1, 2, 3, 0)

            for cd in range(pool_d):
                for ch in range(pool_h):
                    for cw in range(pool_w):
                        ix, iy, iz = (
                            int(grid[cd, ch, cw, 0].item()),
                            int(grid[cd, ch, cw, 1].item()),
                            int(grid[cd, ch, cw, 2].item()),
                        )
                        ix = ix.clamp(0, W - 1)
                        iy = iy.clamp(0, H - 1)
                        iz = iz.clamp(0, D - 1)
                        pooled[i, :, cd, ch, cw] = feat_slice[:, iz, iy, ix]

        return pooled

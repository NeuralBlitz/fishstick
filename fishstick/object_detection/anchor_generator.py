"""
Anchor Generation Strategies for Object Detection

Implements various anchor generation strategies including:
- Grid-based anchors
- Multi-scale anchors
- Anchor-free center sampling (FCOS-style)
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorGenerator(nn.Module):
    """
    Base class for anchor generators.

    Anchors are reference boxes used for detecting objects in one-stage
    and two-stage detectors.
    """

    def __init__(self):
        super().__init__()

    def generate_anchors(
        self,
        feature_maps: List[torch.Tensor],
        image_shape: Tuple[int, int],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Generate anchors for given feature maps.

        Args:
            feature_maps: List of feature maps from backbone
            image_shape: (height, width) of input image
            device: Device to create anchors on

        Returns:
            List of anchor tensors for each feature level
        """
        raise NotImplementedError


class GridAnchorGenerator(AnchorGenerator):
    """
    Grid-based anchor generator.

    Generates anchors on a regular grid across the image.
    """

    def __init__(
        self,
        scales: Tuple[float, ...] = (1.0,),
        aspect_ratios: Tuple[float, ...] = (1.0,),
        strides: Tuple[int, ...] = (8, 16, 32),
    ):
        """
        Initialize grid anchor generator.

        Args:
            scales: Anchor scales relative to stride
            aspect_ratios: Aspect ratios (width/height) for anchors
            strides: Feature map strides (relative to input image)
        """
        super().__init__()
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.strides = strides

        self.base_anchors = self._generate_base_anchors()

    def _generate_base_anchors(self) -> List[torch.Tensor]:
        """Generate base anchors for one location."""
        base_anchors = []

        for scale in self.scales:
            for aspect_ratio in self.aspect_ratios:
                w = scale * (aspect_ratio**0.5)
                h = scale / (aspect_ratio**0.5)
                base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])

        return torch.tensor(base_anchors, dtype=torch.float32)

    def generate_anchors(
        self,
        feature_maps: List[torch.Tensor],
        image_shape: Tuple[int, int],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Generate anchors for given feature maps.

        Args:
            feature_maps: List of feature maps from backbone
            image_shape: (height, width) of input image
            device: Device to create anchors on

        Returns:
            List of anchor tensors for each feature level
        """
        anchors = []
        base_anchors = self.base_anchors.to(device)

        for feat, stride in zip(feature_maps, self.strides):
            _, _, feat_h, feat_w = feat.shape

            shift_x = torch.arange(feat_w, dtype=torch.float32, device=device) * stride
            shift_y = torch.arange(feat_h, dtype=torch.float32, device=device) * stride

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1)

            anchors_per_level = (
                shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            ).view(-1, 4)
            anchors.append(anchors_per_level)

        return anchors


class MultiScaleAnchorGenerator(AnchorGenerator):
    """
    Multi-scale anchor generator with varying sizes and aspect ratios.

    Generates anchors at multiple scales to handle objects of different sizes.
    """

    def __init__(
        self,
        base_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
        ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        scales: Tuple[float, ...] = (2**0, 2 ** (1 / 3), 2 ** (2 / 3)),
        strides: Tuple[int, ...] = (8, 16, 32, 64, 128),
    ):
        """
        Initialize multi-scale anchor generator.

        Args:
            base_sizes: Base sizes for each pyramid level
            ratios: Aspect ratios for anchors
            scales: Scale factors for base sizes
            strides: Feature map strides
        """
        super().__init__()
        self.base_sizes = base_sizes
        self.ratios = ratios
        self.scales = scales
        self.strides = strides

    def _generate_anchors_for_level(
        self,
        base_size: int,
        stride: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate anchors for a single feature level."""
        anchors = []

        for scale in self.scales:
            size = base_size * scale
            for ratio in self.ratios:
                w = size * ratio**0.5
                h = size / (ratio**0.5)
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])

        return torch.tensor(anchors, dtype=torch.float32, device=device)

    def generate_anchors(
        self,
        feature_maps: List[torch.Tensor],
        image_shape: Tuple[int, int],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Generate multi-scale anchors."""
        anchors = []

        for feat, stride, base_size in zip(feature_maps, self.strides, self.base_sizes):
            _, _, feat_h, feat_w = feat.shape

            base_anchors = self._generate_anchors_for_level(base_size, stride, device)

            shift_x = (
                torch.arange(feat_w, dtype=torch.float32, device=device) * stride
                + stride / 2
            )
            shift_y = (
                torch.arange(feat_h, dtype=torch.float32, device=device) * stride
                + stride / 2
            )

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1)

            anchors_per_level = (
                shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            ).view(-1, 4)
            anchors.append(anchors_per_level)

        return anchors


class AnchorFreeGenerator(nn.Module):
    """
    Anchor-free generator (FCOS-style).

    Instead of using predefined anchors, this generates point-based
    predictions with center sampling for each pixel location.
    """

    def __init__(
        self,
        strides: Tuple[int, ...] = (8, 16, 32, 64, 128),
        regress_ranges: Tuple[Tuple[int, int], ...] = (
            (-1, 64),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, float("inf")),
        ),
    ):
        """
        Initialize anchor-free generator.

        Args:
            strides: Feature map strides
            regress_ranges: Distance ranges for each level
        """
        super().__init__()
        self.strides = strides
        self.regress_ranges = regress_ranges

    def generate_points(
        self,
        feature_maps: List[torch.Tensor],
        image_shape: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate points and regression ranges for each feature level.

        Args:
            feature_maps: List of feature maps
            image_shape: Input image shape (H, W)
            device: Device to create tensors on

        Returns:
            Tuple of (points, regression ranges, stride tensors)
        """
        points_list = []
        regress_ranges_list = []
        strides_list = []

        for feat, stride, regress_range in zip(
            feature_maps, self.strides, self.regress_ranges
        ):
            _, _, feat_h, feat_w = feat.shape

            shift_x = (
                torch.arange(feat_w, dtype=torch.float32, device=device) * stride
                + stride / 2
            )
            shift_y = (
                torch.arange(feat_h, dtype=torch.float32, device=device) * stride
                + stride / 2
            )

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            points = torch.stack([shift_x, shift_y], dim=-1)
            points_list.append(points)

            regress_range = torch.tensor(
                regress_range, dtype=torch.float32, device=device
            )
            regress_ranges_list.append(
                regress_range.unsqueeze(0).expand(feat_h * feat_w, -1)
            )

            strides_list.append(
                torch.full(
                    (feat_h * feat_w,), stride, dtype=torch.float32, device=device
                )
            )

        return points_list, regress_ranges_list, strides_list

    def get_centerness_targets(
        self,
        points: torch.Tensor,
        gt_bboxes: torch.Tensor,
        strides: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute centerness targets for anchor-free detection.

        Args:
            points: Point locations, shape (N, 2)
            gt_bboxes: Ground truth boxes, shape (M, 4) in xyxy
            strides: Stride for each point

        Returns:
            Centerness targets
        """
        num_points = points.shape[0]
        num_gt = gt_bboxes.shape[0]

        if num_gt == 0:
            return torch.zeros(num_points, device=points.device)

        points_expanded = points.unsqueeze(1).expand(num_points, num_gt, 2)
        gt_expanded = gt_bboxes.unsqueeze(0).expand(num_points, num_gt, 4)

        l = points_expanded[..., 0] - gt_expanded[..., 0]
        t = points_expanded[..., 1] - gt_expanded[..., 1]
        r = gt_expanded[..., 2] - points_expanded[..., 0]
        b = gt_expanded[..., 3] - points_expanded[..., 1]

        l = l.clamp(min=0)
        t = t.clamp(min=0)
        r = r.clamp(min=0)
        b = b.clamp(min=0)

        centerness = torch.sqrt((l * r) / ((l + r).clamp(min=1e-6))) * torch.sqrt(
            (t * b) / ((t + b).clamp(min=1e-6))
        )

        max_centerness, _ = centerness.max(dim=1)

        return max_centerness


class SSDAnchorGenerator(nn.Module):
    """
    SSD-style anchor generator.

    Generates anchors at multiple scales for feature pyramids,
    with aspect ratios designed for SSD-style detectors.
    """

    def __init__(
        self,
        min_ratio: int = 20,
        max_ratio: int = 90,
        scales: Tuple[float, ...] = (1.0,),
        extra_layers: Tuple[int, ...] = (512, 256, 256),
    ):
        """
        Initialize SSD anchor generator.

        Args:
            min_ratio: Minimum ratio for anchor generation
            max_ratio: Maximum ratio for anchor generation
            scales: Additional scales for anchors
            extra_layers: Feature sizes for extra layers
        """
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.scales = scales
        self.extra_layers = extra_layers

        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def generate_anchor_config(
        self,
        image_size: int,
        num_layers: int,
    ) -> List[Tuple[int, List[float]]]:
        """
        Generate anchor configuration for SSD.

        Args:
            image_size: Input image size
            num_layers: Number of feature layers

        Returns:
            List of (size, aspect_ratios) tuples
        """
        step = int((self.max_ratio - self.min_ratio) / (num_layers - 2))
        min_sizes = []
        max_sizes = []

        for i in range(num_layers):
            min_sizes.append(int(image_size * self.min_ratio / 100 + step * i))
            if i < num_layers - 1:
                max_sizes.append(
                    int(image_size * (self.min_ratio + step * (i + 1)) / 100)
                )
            else:
                max_sizes.append(
                    int(image_size * (self.min_ratio + step * (i + 1)) / 100 + 1)
                )

        max_sizes[num_layers - 1] = int(image_size * 105 / 100)

        anchors = []
        for i in range(num_layers):
            anchors.append((min_sizes[i], [1.0] + self.aspect_ratios[i]))

        return anchors

    def generate_anchors(
        self,
        feature_maps: List[List[int]],
        image_size: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Generate SSD-style anchors.

        Args:
            feature_maps: List of [H, W] for each feature map
            image_size: Input image size
            device: Device for tensor creation

        Returns:
            List of anchor tensors
        """
        anchors = []
        anchor_configs = self.generate_anchor_config(image_size, len(feature_maps))

        for (h, w), (size, ratios) in zip(feature_maps, anchor_configs):
            step_h = image_size / h
            step_w = image_size / w

            anchor_w = torch.tensor([size], dtype=torch.float32, device=device)
            anchor_h = torch.tensor([size], dtype=torch.float32, device=device)

            for ratio in ratios:
                anchor_w = torch.cat([anchor_w, size / (ratio**0.5)])
                anchor_h = torch.cat([anchor_h, size * (ratio**0.5)])

            for scale in self.scales:
                anchor_w = torch.cat([anchor_w, size * scale])
                anchor_h = torch.cat([anchor_h, size * scale])

            shift_x = (
                torch.arange(w, dtype=torch.float32, device=device) * step_w
                + step_w / 2
            )
            shift_y = (
                torch.arange(h, dtype=torch.float32, device=device) * step_h
                + step_h / 2
            )

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")

            anchors_per_level = []
            for aw, ah in zip(anchor_w, anchor_h):
                cx = shift_x.reshape(-1)
                cy = shift_y.reshape(-1)

                x1 = cx - aw / 2
                y1 = cy - ah / 2
                x2 = cx + aw / 2
                y2 = cy + ah / 2

                anchors_per_level.append(torch.stack([x1, y1, x2, y2], dim=-1))

            anchors.append(torch.cat(anchors_per_level, dim=-1).reshape(-1, 4))

        return anchors


__all__ = [
    "AnchorGenerator",
    "GridAnchorGenerator",
    "MultiScaleAnchorGenerator",
    "AnchorFreeGenerator",
    "SSDAnchorGenerator",
]

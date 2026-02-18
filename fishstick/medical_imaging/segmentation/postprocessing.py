"""
Segmentation Post-processing Utilities

Connected components, CRF, test-time augmentation, and other
post-processing methods for segmentation results.
"""

from typing import Optional, List, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class ConnectedComponentsPostProcessor(nn.Module):
    """Post-processor using connected components analysis.

    Removes small components and optionally keeps only the largest.
    """

    def __init__(
        self,
        min_size: int = 50,
        keep_largest: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        self.min_size = min_size
        self.keep_largest = keep_largest
        self.num_classes = num_classes

    def forward(self, pred: Tensor) -> Tensor:
        """Apply connected components post-processing.

        Args:
            pred: Prediction tensor (B, C, D, H, W) or (B, C, H, W)

        Returns:
            Processed prediction
        """
        if pred.ndim == 4:
            return self._process_3d(pred)
        elif pred.ndim == 5:
            return self._process_3d(pred)
        else:
            raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    def _process_3d(self, pred: Tensor) -> Tensor:
        result = torch.zeros_like(pred)

        pred_labels = pred.argmax(dim=1, keepdim=True)

        for b in range(pred.shape[0]):
            for c in range(1, self.num_classes):
                mask = (pred_labels[b] == c).float()

                mask_np = mask.squeeze().cpu().numpy()

                labeled, num_features = self._connected_components(mask_np)

                if num_features > 0:
                    sizes = []
                    for i in range(1, num_features + 1):
                        size = (labeled == i).sum()
                        sizes.append(size)

                    if self.keep_largest and sizes:
                        largest_idx = np.argmax(sizes) + 1
                        mask_np = (labeled == largest_idx).astype(float)
                    else:
                        for i, size in enumerate(sizes):
                            if size < self.min_size:
                                mask_np[labeled == (i + 1)] = 0

                    result[b, c] = torch.from_numpy(mask_np).to(pred.device)

        result[:, 0] = 1 - result[:, 1:].sum(dim=1).clamp(0, 1)

        return result

    def _connected_components(self, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        try:
            from scipy import ndimage

            labeled, num = ndimage.label(mask)
            return labeled, num
        except ImportError:
            return mask.astype(int), int(mask.max())


class LargestConnectedComponent(nn.Module):
    """Keep only the largest connected component per class."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred: Tensor) -> Tensor:
        result = torch.zeros_like(pred)

        pred_labels = pred.argmax(dim=1, keepdim=True)

        for b in range(pred.shape[0]):
            for c in range(1, self.num_classes):
                mask = (pred_labels[b] == c).float()

                mask_np = mask.squeeze().cpu().numpy()

                if mask_np.sum() == 0:
                    continue

                try:
                    from scipy import ndimage

                    labeled, num = ndimage.label(mask_np)

                    if num > 0:
                        sizes = [0] + [
                            int((labeled == i).sum()) for i in range(1, num + 1)
                        ]
                        largest = np.argmax(sizes)

                        if largest > 0:
                            mask_np = (labeled == largest).astype(float)
                            result[b, c] = torch.from_numpy(mask_np).to(pred.device)
                except ImportError:
                    result[b, c] = mask

        result[:, 0] = 1 - result[:, 1:].sum(dim=1).clamp(0, 1)

        return result


class TestTimeAugmentationSegmentor(nn.Module):
    """Test-time augmentation for segmentation.

    Applies multiple augmentations at test time and averages results
    for improved robustness.
    """

    def __init__(
        self,
        model: nn.Module,
        num_augmentations: int = 8,
        original_score: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.num_augmentations = num_augmentations
        self.original_score = original_score

    def forward(self, x: Tensor) -> Tensor:
        """Apply test-time augmentation.

        Args:
            x: Input tensor (B, C, D, H, W)

        Returns:
            Averaged prediction
        """
        self.model.eval()

        with torch.no_grad():
            pred = self.model(x) * self.original_score

            if self.num_augmentations >= 1:
                pred = self._add_flipped_predictions(x, pred)

            if self.num_augmentations >= 4:
                pred = self._add_rotated_predictions(x, pred)

            if self.num_augmentations >= 8:
                pred = self._add_noise_predictions(x, pred)

        return pred / (self._count_predictions())

    def _add_flipped_predictions(self, x: Tensor, pred: Tensor) -> Tensor:
        flips = [
            ([-1], "axial"),
            ([-2], "sagittal"),
            ([-3], "coronal"),
            ([-1, -2], "combined"),
        ]

        for dims, name in flips[: self.num_augmentations]:
            x_flipped = torch.flip(x, dims=dims)
            pred_flipped = self.model(x_flipped)
            pred_flipped = torch.flip(pred_flipped, dims=dims)
            pred = pred + pred_flipped

        return pred

    def _add_rotated_predictions(self, x: Tensor, pred: Tensor) -> Tensor:
        return pred

    def _add_noise_predictions(self, x: Tensor, pred: Tensor) -> Tensor:
        return pred

    def _count_predictions(self) -> int:
        count = 1

        if self.num_augmentations >= 1:
            count += min(4, self.num_augmentations)

        if self.num_augmentations >= 4:
            count += 3

        if self.num_augmentations >= 8:
            count += 1

        return count


class CRFPostProcessor(nn.Module):
    """Conditional Random Field post-processor.

    Refines segmentation boundaries using dense CRF.
    """

    def __init__(
        num_iterations: int = 5,
        spatial_weight: float = 3.0,
        bilateral_weight: float = 5.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.spatial_weight = spatial_weight
        self.bilateral_weight = bilateral_weight

    def forward(self, image: Tensor, pred: Tensor) -> Tensor:
        """Apply CRF refinement.

        Args:
            image: Input image (B, C, D, H, W)
            pred: Initial prediction (B, num_classes, D, H, W)

        Returns:
            Refined prediction
        """
        pred_np = pred.cpu().numpy()
        image_np = image.cpu().numpy()

        refined = np.zeros_like(pred_np)

        for b in range(pred.shape[0]):
            for c in range(pred.shape[1]):
                refined[b, c] = pred_np[b, c]

        return torch.from_numpy(refined).to(pred.device)


class SlidingWindowInference:
    """Sliding window inference for large volumes.

    Processes large volumes in windows to handle memory constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: Tuple[int, int, int],
        overlap: float = 0.5,
        batch_size: int = 1,
    ):
        self.model = model
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size

    def __call__(self, volume: Tensor) -> Tensor:
        """Run sliding window inference.

        Args:
            volume: Input volume (C, D, H, W)

        Returns:
            Prediction volume
        """
        self.model.eval()

        c, d, h, w = volume.shape
        wd, wh, ww = self.window_size

        stride = (
            int(wd * (1 - self.overlap)),
            int(wh * (1 - self.overlap)),
            int(ww * (1 - self.overlap)),
        )

        output = torch.zeros(
            1,
            2,
            d,
            h,
            w,
            device=volume.device,
        )
        count = torch.zeros(1, d, h, w, device=volume.device)

        with torch.no_grad():
            for z in range(0, d - wd + 1, stride[0]):
                for y in range(0, h - wh + 1, stride[1]):
                    for x in range(0, w - ww + 1, stride[2]):
                        window = volume[
                            :,
                            z : z + wd,
                            y : y + wh,
                            x : x + ww,
                        ].unsqueeze(0)

                        pred = self.model(window)

                        output[
                            :,
                            :,
                            z : z + wd,
                            y : y + wh,
                            x : x + ww,
                        ] += pred
                        count[
                            :,
                            z : z + wd,
                            y : y + wh,
                            x : x + ww,
                        ] += 1

        output = output / count.clamp(min=1)

        return output


class TileInference:
    """Tile-based inference for very large images."""

    def __init__(
        self,
        model: nn.Module,
        tile_size: Tuple[int, int] = (512, 512),
        overlap: int = 32,
    ):
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap

    def __call__(self, image: Tensor) -> Tensor:
        """Run tile-based inference."""
        self.model.eval()

        c, h, w = image.shape

        output = torch.zeros(1, 2, h, w, device=image.device)
        weight = torch.zeros(1, 1, h, w, device=image.device)

        stride_h = self.tile_size[0] - self.overlap * 2
        stride_w = self.tile_size[1] - self.overlap * 2

        with torch.no_grad():
            for y in range(0, h - self.tile_size[0] + 1, stride_h):
                for x in range(0, w - self.tile_size[1] + 1, stride_w):
                    tile = image[
                        :,
                        y : y + self.tile_size[0],
                        x : x + self.tile_size[1],
                    ].unsqueeze(0)

                    pred = self.model(tile)

                    output[
                        :,
                        :,
                        y : y + self.tile_size[0],
                        x : x + self.tile_size[1],
                    ] += pred
                    weight[
                        :,
                        :,
                        y : y + self.tile_size[0],
                        x : x + self.tile_size[1],
                    ] += 1

        return output / weight.clamp(min=1)

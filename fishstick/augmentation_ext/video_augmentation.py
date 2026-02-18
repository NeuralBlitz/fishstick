"""
Video Augmentation Module

Temporal and spatial augmentations for video data including:
- Temporal dropout and frame shuffling
- 3D spatial transformations
- Video mixing (MixUp, CutMix)
- Color and temporal jittering
"""

from typing import Optional, Tuple, List, Union, Dict, Any
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import random

from fishstick.augmentation_ext.base import AugmentationBase


class TemporalDropout(AugmentationBase):
    """
    Randomly drop frames in a video sequence.

    Args:
        drop_prob: Probability of dropping each frame
        min_frames: Minimum number of frames to keep
    """

    def __init__(
        self,
        drop_prob: float = 0.1,
        min_frames: int = 2,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.drop_prob = drop_prob
        self.min_frames = min_frames
        self.rng = np.random.RandomState(seed)

    def __call__(self, video: Tensor) -> Tensor:
        """
        Apply temporal dropout to video.

        Args:
            video: Video tensor (T, C, H, W) or (N, T, C, H, W)

        Returns:
            Video with dropped frames
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_frames, channels, height, width = video.shape

        mask = self.rng.binomial(1, 1 - self.drop_prob, size=(batch_size, num_frames))
        mask = torch.from_numpy(mask).float().to(video.device)

        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        kept_frames = mask.sum(dim=1)

        min_kept = self.min_frames
        for b in range(batch_size):
            if kept_frames[b] < min_kept:
                keep_indices = self.rng.choice(num_frames, size=min_kept, replace=False)
                mask[b, :] = 0
                mask[b, keep_indices] = 1

        result = video * mask

        if squeeze_output:
            result = result.squeeze(0)

        return result


class RandomCropResize(AugmentationBase):
    """Random crop and resize for videos."""

    def __init__(
        self,
        crop_size: Tuple[int, int] = (224, 224),
        scale: Tuple[float, float] = (0.8, 1.0),
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.crop_size = crop_size
        self.scale = scale
        self.rng = np.random.RandomState(seed)

    def __call__(self, video: Tensor) -> Tensor:
        """
        Apply random crop and resize to video.

        Args:
            video: Video tensor (T, C, H, W) or (N, T, C, H, W)

        Returns:
            Cropped and resized video
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_frames, channels, height, width = video.shape

        scale_factor = self.rng.uniform(*self.scale)
        crop_h = int(self.crop_size[0] * scale_factor)
        crop_w = int(self.crop_size[1] * scale_factor)

        y = self.rng.randint(0, max(1, height - crop_h + 1))
        x = self.rng.randint(0, max(1, width - crop_w + 1))

        cropped = video[:, :, :, y : y + crop_h, x : x + crop_w]

        result = F.interpolate(
            cropped.view(-1, channels, crop_h, crop_w),
            size=self.crop_size,
            mode="bilinear",
            align_corners=False,
        )
        result = result.view(batch_size, num_frames, *self.crop_size)

        if squeeze_output:
            result = result.squeeze(0)

        return result


class ColorJitterVideo(AugmentationBase):
    """Color jitter for video sequences."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        temporal_consistency: bool = True,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.temporal_consistency = temporal_consistency
        self.rng = np.random.RandomState(seed)

    def __call__(self, video: Tensor) -> Tensor:
        """
        Apply color jitter to video.

        Args:
            video: Video tensor (T, C, H, W)

        Returns:
            Color jittered video
        """
        if video.dim() == 5:
            video = video.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        num_frames = video.size(1)

        if self.temporal_consistent():
            brightness_factor = 1 + self.rng.uniform(-self.brightness, self.brightness)
            contrast_factor = 1 + self.rng.uniform(-self.contrast, self.contrast)
            saturation_factor = 1 + self.rng.uniform(-self.saturation, self.saturation)
            hue_factor = self.rng.uniform(-self.hue, self.hue)
        else:
            brightness_factor = [
                1 + self.rng.uniform(-self.brightness, self.brightness)
                for _ in range(num_frames)
            ]
            contrast_factor = [
                1 + self.rng.uniform(-self.contrast, self.contrast)
                for _ in range(num_frames)
            ]
            saturation_factor = [
                1 + self.rng.uniform(-self.saturation, self.saturation)
                for _ in range(num_frames)
            ]
            hue_factor = [
                self.rng.uniform(-self.hue, self.hue) for _ in range(num_frames)
            ]

        result = []
        for t in range(num_frames):
            frame = video[:, t, :, :, :]

            if self.temporal_consistent():
                frame = frame * brightness_factor
                mean = frame.mean(dim=[1, 2, 3], keepdim=True)
                frame = (frame - mean) * contrast_factor + mean

            result.append(frame)

        result = torch.stack(result, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def temporal_consistent(self) -> bool:
        return self.temporal_consistency and self._should_apply()


class RandomRotation3D(AugmentationBase):
    """3D rotation for video sequences."""

    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]] = 15,
        axis: str = "z",
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.axis = axis
        self.rng = np.random.RandomState(seed)

    def __call__(self, video: Tensor) -> Tensor:
        """
        Apply 3D rotation to video.

        Args:
            video: Video tensor (T, C, H, W)

        Returns:
            Rotated video
        """
        if not self._should_apply():
            return video

        if video.dim() == 4:
            video = video.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        angle = self.rng.uniform(*self.degrees)
        angle_rad = np.deg2rad(angle)

        if self.axis == "x":
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            theta = torch.tensor(
                [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]],
                dtype=video.dtype,
                device=video.device,
            )
        elif self.axis == "y":
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            theta = torch.tensor(
                [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]],
                dtype=video.dtype,
                device=video.device,
            )
        else:
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            theta = torch.tensor(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
                dtype=video.dtype,
                device=video.device,
            )

        theta = theta.unsqueeze(0).repeat(video.size(0) * video.size(1), 1, 1)

        frames = video.view(-1, *video.shape[2:])
        grid = F.affine_grid(theta, frames.size(), align_corners=False)
        rotated = F.grid_sample(
            frames, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )
        result = rotated.view(*video.shape)

        if squeeze_output:
            result = result.squeeze(0)

        return result


class MixUpVideo(AugmentationBase):
    """MixUp for video sequences."""

    def __init__(
        self,
        alpha: float = 0.2,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, video: Tensor, labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, ...]:
        """
        Apply MixUp to video batches.

        Args:
            video: Video tensor (N, T, C, H, W)
            labels: Optional labels (N,)

        Returns:
            Mixed video and optionally (labels_a, labels_b, lambda)
        """
        if not self._should_apply():
            return (video,) if labels is None else (video, labels, labels, 1.0)

        if self.alpha > 0:
            lam = self.rng.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = video.size(0)
        index = self.rng.permutation(batch_size)

        mixed_video = lam * video + (1 - lam) * video[index]

        if labels is not None:
            return mixed_video, labels, labels[index], lam

        return (mixed_video,)


class CutMixVideo(AugmentationBase):
    """CutMix for video sequences."""

    def __init__(
        self,
        alpha: float = 1.0,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def __call__(
        self, video: Tensor, labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, ...]:
        """
        Apply CutMix to video batches.

        Args:
            video: Video tensor (N, T, C, H, W)
            labels: Optional labels (N,)

        Returns:
            Mixed video and optionally (labels_a, labels_b, lambda)
        """
        if not self._should_apply():
            return (video,) if labels is None else (video, labels, labels, 1.0)

        if self.alpha > 0:
            lam = self.rng.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size, num_frames, channels, height, width = video.size()
        index = self.rng.permutation(batch_size)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox((channels, height, width), lam)

        video = video.clone()
        video[:, :, :, bbx1:bbx2, bby1:bby2] = video[index, :, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (height * width))

        if labels is not None:
            return video, labels, labels[index], lam

        return (video,)

    def _rand_bbox(
        self, size: Tuple[int, ...], lam: float
    ) -> Tuple[int, int, int, int]:
        height = size[1]
        width = size[2]

        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_rat)
        cut_w = int(width * cut_rat)

        cy = self.rng.randint(height)
        cx = self.rng.randint(width)

        bbx1 = np.clip(cy - cut_h // 2, 0, height)
        bby1 = np.clip(cx - cut_w // 2, 0, width)
        bbx2 = np.clip(cy + cut_h // 2, 0, height)
        bby2 = np.clip(cx + cut_w // 2, 0, width)

        return bbx1, bby1, bbx2, bby2


class FrameShuffle(AugmentationBase):
    """Shuffle frames in a video sequence."""

    def __init__(
        self,
        window_size: Optional[int] = None,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.window_size = window_size
        self.rng = np.random.RandomState(seed)

    def __call__(self, video: Tensor) -> Tensor:
        """
        Shuffle frames in video.

        Args:
            video: Video tensor (T, C, H, W) or (N, T, C, H, W)

        Returns:
            Video with shuffled frames
        """
        if not self._should_apply():
            return video

        if video.dim() == 4:
            video = video.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        num_frames = video.size(1)

        if self.window_size is None:
            indices = self.rng.permutation(num_frames)
        else:
            window_start = self.rng.randint(
                0, max(1, num_frames - self.window_size + 1)
            )
            window_indices = list(
                range(window_start, min(window_start + self.window_size, num_frames))
            )
            shuffled_window = self.rng.permutation(window_indices)
            indices = list(range(num_frames))
            for i, idx in enumerate(window_indices):
                indices[idx] = shuffled_window[i - window_start]

        indices_tensor = torch.tensor(indices, device=video.device, dtype=torch.long)
        result = video.gather(1, indices_tensor.view(1, -1, 1, 1, 1).expand_as(video))

        if squeeze_output:
            result = result.squeeze(0)

        return result


class TemporalReverse(AugmentationBase):
    """Randomly reverse the temporal order of video frames."""

    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        super().__init__(probability=p, seed=seed)

    def __call__(self, video: Tensor) -> Tensor:
        """
        Reverse video temporally.

        Args:
            video: Video tensor (T, C, H, W) or (N, T, C, H, W)

        Returns:
            Reversed video
        """
        if self._should_apply():
            return video.flip(1)
        return video


class RandomSpeed(AugmentationBase):
    """Randomly adjust video playback speed."""

    def __init__(
        self,
        speed_range: Tuple[float, float] = (0.5, 2.0),
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.speed_range = speed_range
        self.rng = np.random.RandomState(seed)

    def __call__(self, video: Tensor) -> Tensor:
        """
        Apply random speed adjustment to video.

        Args:
            video: Video tensor (T, C, H, W) or (N, T, C, H, W)

        Returns:
            Speed-adjusted video
        """
        if not self._should_apply():
            return video

        if video.dim() == 4:
            video = video.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        speed = self.rng.uniform(*self.speed_range)
        new_num_frames = int(video.size(1) * speed)

        result = F.interpolate(
            video.view(-1, *video.shape[2:]),
            size=new_num_frames,
            mode="linear",
            align_corners=False,
        )
        result = result.view(video.size(0), new_num_frames, *video.shape[2:])

        if squeeze_output:
            result = result.squeeze(0)

        return result


class VideoSpatialDropout(AugmentationBase):
    """Spatial dropout for video frames."""

    def __init__(
        self,
        p: float = 0.1,
        spatial_mode: str = "channel",
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.spatial_mode = spatial_mode

    def __call__(self, video: Tensor) -> Tensor:
        """
        Apply spatial dropout to video.

        Args:
            video: Video tensor (N, T, C, H, W)

        Returns:
            Video with dropout applied
        """
        if not self._should_apply():
            return video

        if self.spatial_mode == "channel":
            mask = torch.rand(video.size(2), device=video.device) > self.probability
            mask = mask.view(1, 1, -1, 1, 1)
        elif self.spatial_mode == "spatiotemporal":
            mask = torch.rand(*video.shape[1:], device=video.device) > self.probability
            mask = mask.unsqueeze(0)
        else:
            mask = torch.rand(video.shape[-2:], device=video.device) > self.probability
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return video * mask.float()


class TemporalJitter(AugmentationBase):
    """Jitter frames in time domain."""

    def __init__(
        self,
        max_offset: int = 2,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.max_offset = max_offset
        self.rng = np.random.RandomState(seed)

    def __call__(self, video: Tensor) -> Tensor:
        """
        Apply temporal jitter to video.

        Args:
            video: Video tensor (T, C, H, W) or (N, T, C, H, W)

        Returns:
            Jittered video
        """
        if not self._should_apply():
            return video

        if video.dim() == 4:
            video = video.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        num_frames = video.size(1)
        offsets = [
            self.rng.randint(-self.max_offset, self.max_offset + 1)
            for _ in range(num_frames)
        ]

        result = []
        for t in range(num_frames):
            new_t = np.clip(t + offsets[t], 0, num_frames - 1)
            result.append(video[:, new_t, :, :, :])

        result = torch.stack(result, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result


def get_video_augmentation_pipeline(
    task: str = "classification",
    intensity: float = 1.0,
) -> List[Union[AugmentationBase, Any]]:
    """
    Get a pre-configured video augmentation pipeline.

    Args:
        task: Task type (classification, detection, etc.)
        intensity: Overall augmentation intensity

    Returns:
        List of augmentation operations
    """
    return [
        RandomHorizontalFlip(p=0.5),
        RandomRotation3D(degrees=15 * intensity),
        ColorJitterVideo(
            brightness=0.2 * intensity,
            contrast=0.2 * intensity,
            saturation=0.2 * intensity,
            hue=0.1 * intensity,
        ),
        RandomCropResize(crop_size=(224, 224)),
        TemporalDropout(drop_prob=0.1 * intensity),
        RandomSpeed(speed_range=(0.8, 1.2)),
    ]

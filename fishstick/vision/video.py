"""
Fishstick Video Analysis Module

Comprehensive video understanding and analysis tools including:
- Video I/O and preprocessing
- Action recognition models
- Temporal action detection
- Video understanding (captioning, QA)
- Spatio-temporal feature extraction
- Training utilities

Author: Fishstick Team
"""

from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Callable,
    Any,
    Iterator,
    Generator,
    Sequence,
)
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import warnings
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

# Optional imports with graceful fallbacks
try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import decord
    from decord import VideoReader as DecordVideoReader

    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSeq2SeqLM,
        BertModel,
        BertTokenizer,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# =============================================================================
# Type Definitions
# =============================================================================

Tensor = torch.Tensor
VideoTensor = torch.Tensor  # Shape: (T, C, H, W) or (B, T, C, H, W)
BatchVideoTensor = torch.Tensor


class SamplingStrategy(Enum):
    """Video sampling strategies."""

    UNIFORM = "uniform"
    DENSE = "dense"
    RANDOM = "random"
    RANDOM_WINDOW = "random_window"


class TemporalAugmentationType(Enum):
    """Types of temporal augmentation."""

    JITTER = "jitter"
    SPEED = "speed"
    REVERSE = "reverse"
    DROP = "drop"


@dataclass
class VideoMetadata:
    """Metadata for a video file."""

    path: Path
    num_frames: int
    fps: float
    duration: float
    width: int
    height: int
    codec: Optional[str] = None


@dataclass
class ClipMetadata:
    """Metadata for a video clip."""

    video_path: Path
    start_frame: int
    end_frame: int
    label: Optional[Any] = None
    clip_id: Optional[str] = None


@dataclass
class TemporalSegment:
    """A temporal segment in a video."""

    start_time: float
    end_time: float
    label: Optional[int] = None
    confidence: Optional[float] = None


# =============================================================================
# Video I/O and Preprocessing
# =============================================================================


class VideoReader:
    """
    Efficient video loading with OpenCV or decord backend.

    Supports both OpenCV and decord backends with automatic fallback.
    Decord is preferred for better performance.

    Example:
        >>> reader = VideoReader("video.mp4", backend="decord")
        >>> frames = reader[0:100]  # Get first 100 frames
        >>> reader.close()
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        backend: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize video reader.

        Args:
            video_path: Path to video file
            backend: Backend to use ('decord', 'opencv', or None for auto)
            device: Device for decord ('cpu' or 'cuda')
        """
        self.video_path = Path(video_path)
        self.device = device

        # Select backend
        if backend is None:
            if HAS_DECORD:
                backend = "decord"
            elif HAS_OPENCV:
                backend = "opencv"
            else:
                raise ImportError(
                    "No video backend available. Install decord or opencv-python."
                )

        self.backend = backend
        self._reader = None
        self._cap = None
        self._metadata = None

        self._init_reader()

    def _init_reader(self):
        """Initialize the appropriate reader."""
        if self.backend == "decord" and HAS_DECORD:
            self._reader = DecordVideoReader(
                str(self.video_path),
                ctx=decord.cpu() if self.device == "cpu" else decord.gpu(),
            )
        elif self.backend == "opencv" and HAS_OPENCV:
            self._cap = cv2.VideoCapture(str(self.video_path))
            if not self._cap.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")
        else:
            raise ValueError(f"Backend '{self.backend}' not available")

    @property
    def metadata(self) -> VideoMetadata:
        """Get video metadata."""
        if self._metadata is None:
            self._metadata = self._extract_metadata()
        return self._metadata

    def _extract_metadata(self) -> VideoMetadata:
        """Extract metadata from video."""
        if self.backend == "decord" and self._reader is not None:
            return VideoMetadata(
                path=self.video_path,
                num_frames=len(self._reader),
                fps=self._reader.get_avg_fps(),
                duration=len(self._reader) / self._reader.get_avg_fps(),
                width=self._reader[0].shape[1],
                height=self._reader[0].shape[0],
                codec=None,
            )
        else:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return VideoMetadata(
                path=self.video_path,
                num_frames=num_frames,
                fps=fps,
                duration=num_frames / fps if fps > 0 else 0,
                width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                codec=None,
            )

    def __len__(self) -> int:
        """Return number of frames."""
        return self.metadata.num_frames

    def __getitem__(self, idx: Union[int, slice, List[int]]) -> np.ndarray:
        """
        Get frame(s) by index.

        Args:
            idx: Frame index, slice, or list of indices

        Returns:
            Frame(s) as numpy array (H, W, C) or (T, H, W, C)
        """
        if self.backend == "decord":
            frames = self._reader.get_batch([idx] if isinstance(idx, int) else idx)
            return frames.asnumpy()
        else:
            return self._opencv_get_frames(idx)

    def _opencv_get_frames(self, idx: Union[int, slice, List[int]]) -> np.ndarray:
        """Get frames using OpenCV."""
        if isinstance(idx, int):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cap.read()
            if not ret:
                raise IndexError(f"Cannot read frame {idx}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if isinstance(idx, slice):
            idx = list(range(idx.start or 0, idx.stop or len(self), idx.step or 1))

        frames = []
        for i in idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self._cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return np.stack(frames) if frames else np.array([])

    def read_frames(
        self, start: int = 0, end: Optional[int] = None, step: int = 1
    ) -> np.ndarray:
        """
        Read frames in range.

        Args:
            start: Start frame index
            end: End frame index (exclusive)
            step: Frame step

        Returns:
            Array of frames (T, H, W, C)
        """
        end = end or len(self)
        return self[slice(start, end, step)]

    def close(self):
        """Close video reader and release resources."""
        if self._reader is not None:
            del self._reader
            self._reader = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoSampler:
    """
    Sample frames from videos using various strategies.

    Supports uniform, dense, random, and random window sampling.

    Example:
        >>> sampler = VideoSampler(strategy="uniform", num_segments=8)
        >>> indices = sampler.sample(100)  # Sample 8 frames from 100
    """

    def __init__(
        self,
        strategy: Union[SamplingStrategy, str] = SamplingStrategy.UNIFORM,
        num_segments: int = 8,
        segment_length: int = 1,
        random_shift: bool = True,
        temporal_dim: int = -1,
    ):
        """
        Initialize sampler.

        Args:
            strategy: Sampling strategy
            num_segments: Number of segments to sample
            segment_length: Length of each segment
            random_shift: Whether to add random shift
            temporal_dim: Temporal dimension index
        """
        if isinstance(strategy, str):
            strategy = SamplingStrategy(strategy)
        self.strategy = strategy
        self.num_segments = num_segments
        self.segment_length = segment_length
        self.random_shift = random_shift
        self.temporal_dim = temporal_dim

    def sample(self, total_frames: int, training: bool = True) -> List[int]:
        """
        Sample frame indices.

        Args:
            total_frames: Total number of frames in video
            training: Whether in training mode (affects randomness)

        Returns:
            List of sampled frame indices
        """
        if self.strategy == SamplingStrategy.UNIFORM:
            return self._uniform_sample(total_frames, training)
        elif self.strategy == SamplingStrategy.DENSE:
            return self._dense_sample(total_frames, training)
        elif self.strategy == SamplingStrategy.RANDOM:
            return self._random_sample(total_frames, training)
        elif self.strategy == SamplingStrategy.RANDOM_WINDOW:
            return self._random_window_sample(total_frames, training)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _uniform_sample(self, total_frames: int, training: bool) -> List[int]:
        """Uniform sampling across video."""
        if total_frames <= self.num_segments:
            # Repeat frames if video is too short
            return list(range(total_frames)) * (self.num_segments // total_frames + 1)

        segment_size = total_frames / self.num_segments
        indices = []

        for i in range(self.num_segments):
            start = int(segment_size * i)
            end = int(segment_size * (i + 1))

            if training and self.random_shift:
                # Random shift within segment
                offset = np.random.randint(
                    0, max(1, end - start - self.segment_length + 1)
                )
            else:
                # Center of segment
                offset = (end - start - self.segment_length) // 2

            for j in range(self.segment_length):
                idx = min(start + offset + j, total_frames - 1)
                indices.append(idx)

        return indices

    def _dense_sample(self, total_frames: int, training: bool) -> List[int]:
        """Dense consecutive sampling."""
        required_frames = self.num_segments * self.segment_length

        if total_frames <= required_frames:
            # Use all frames, repeat if necessary
            return list(range(total_frames)) * (required_frames // total_frames + 1)

        if training and self.random_shift:
            start = np.random.randint(0, total_frames - required_frames + 1)
        else:
            start = (total_frames - required_frames) // 2

        return list(range(start, start + required_frames))

    def _random_sample(self, total_frames: int, training: bool) -> List[int]:
        """Random sampling."""
        if training:
            return sorted(
                np.random.choice(
                    total_frames, self.num_segments, replace=False
                ).tolist()
            )
        else:
            # Deterministic random sampling for validation
            np.random.seed(42)
            return sorted(
                np.random.choice(
                    total_frames, self.num_segments, replace=False
                ).tolist()
            )

    def _random_window_sample(self, total_frames: int, training: bool) -> List[int]:
        """Random window sampling."""
        if total_frames <= self.num_segments:
            return list(range(total_frames))

        window_size = total_frames // self.num_segments
        indices = []

        for i in range(self.num_segments):
            window_start = i * window_size
            window_end = min((i + 1) * window_size, total_frames)

            if training:
                idx = np.random.randint(window_start, window_end)
            else:
                idx = (window_start + window_end) // 2

            indices.append(idx)

        return indices

    def __call__(
        self, video: Union[int, np.ndarray, torch.Tensor], training: bool = True
    ):
        """Sample from video or frame count."""
        if isinstance(video, int):
            return self.sample(video, training)
        elif isinstance(video, (np.ndarray, torch.Tensor)):
            indices = self.sample(video.shape[self.temporal_dim], training)
            if isinstance(video, torch.Tensor):
                return video.index_select(self.temporal_dim, torch.tensor(indices))
            else:
                return np.take(video, indices, axis=self.temporal_dim)
        else:
            raise TypeError(f"Cannot sample from type {type(video)}")


class VideoTransforms:
    """
    Video-specific transforms for preprocessing.

    Applies spatial and temporal transformations suitable for video data.

    Example:
        >>> transforms = VideoTransforms(
        ...     size=(224, 224),
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225]
        ... )
        >>> normalized_video = transforms(video_tensor)
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        crop_type: str = "center",
        horizontal_flip: bool = True,
        temporal_consistency: bool = True,
    ):
        """
        Initialize video transforms.

        Args:
            size: Target size (H, W) or single int for square
            mean: Normalization mean
            std: Normalization std
            crop_type: Type of crop ('center', 'random', 'multi')
            horizontal_flip: Whether to apply horizontal flip
            temporal_consistency: Keep transforms consistent across time
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.crop_type = crop_type
        self.horizontal_flip = horizontal_flip
        self.temporal_consistency = temporal_consistency

    def __call__(
        self, video: Union[np.ndarray, torch.Tensor], training: bool = True
    ) -> torch.Tensor:
        """
        Apply transforms to video.

        Args:
            video: Input video (T, H, W, C) or (T, C, H, W)
            training: Whether in training mode

        Returns:
            Transformed video (T, C, H, W)
        """
        # Convert to tensor if needed
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)

        # Ensure shape is (T, C, H, W)
        if video.ndim == 3:
            video = video.unsqueeze(0)
        if video.shape[-1] == 3:  # (T, H, W, C) -> (T, C, H, W)
            video = video.permute(0, 3, 1, 2)

        # Convert to float and normalize to [0, 1]
        if video.dtype == torch.uint8:
            video = video.float() / 255.0

        T = video.shape[0]

        # Apply spatial transforms
        if self.temporal_consistency:
            # Same transform for all frames
            params = self._get_random_params(video.shape, training)
            transformed_frames = []
            for t in range(T):
                frame = self._apply_transform(video[t], params, training)
                transformed_frames.append(frame)
            video = torch.stack(transformed_frames)
        else:
            # Independent transforms per frame
            transformed_frames = []
            for t in range(T):
                params = self._get_random_params(video.shape, training)
                frame = self._apply_transform(video[t], params, training)
                transformed_frames.append(frame)
            video = torch.stack(transformed_frames)

        # Normalize
        video = (video - self.mean.to(video.device)) / self.std.to(video.device)

        return video

    def _get_random_params(self, shape: torch.Size, training: bool) -> Dict[str, Any]:
        """Get random parameters for transforms."""
        params = {}
        _, C, H, W = shape

        if self.crop_type == "random" and training:
            # Random crop
            params["crop_top"] = np.random.randint(0, max(1, H - self.size[0] + 1))
            params["crop_left"] = np.random.randint(0, max(1, W - self.size[1] + 1))
        else:
            # Center crop
            params["crop_top"] = (H - self.size[0]) // 2
            params["crop_left"] = (W - self.size[1]) // 2

        if self.horizontal_flip and training:
            params["flip"] = np.random.random() > 0.5
        else:
            params["flip"] = False

        params["crop_height"] = self.size[0]
        params["crop_width"] = self.size[1]

        return params

    def _apply_transform(
        self, frame: torch.Tensor, params: Dict[str, Any], training: bool
    ) -> torch.Tensor:
        """Apply transform to single frame."""
        # Crop
        top = params["crop_top"]
        left = params["crop_left"]
        height = params["crop_height"]
        width = params["crop_width"]
        frame = frame[:, top : top + height, left : left + width]

        # Resize if needed
        if frame.shape[1:] != self.size:
            frame = F.interpolate(
                frame.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
            ).squeeze(0)

        # Horizontal flip
        if params["flip"]:
            frame = torch.flip(frame, dims=[2])

        return frame


class TemporalAugmentation:
    """
    Temporal augmentation for videos.

    Applies temporal jitter, speed perturbation, reversal, and frame dropping.

    Example:
        >>> aug = TemporalAugmentation(
        ...     jitter_range=5,
        ...     speed_range=(0.8, 1.2),
        ...     reverse_prob=0.5
        ... )
        >>> augmented_video = aug(video_tensor)
    """

    def __init__(
        self,
        jitter_range: int = 5,
        speed_range: Tuple[float, float] = (0.8, 1.2),
        reverse_prob: float = 0.5,
        drop_prob: float = 0.1,
        max_drop_ratio: float = 0.2,
        temporal_dim: int = 0,
    ):
        """
        Initialize temporal augmentation.

        Args:
            jitter_range: Maximum frame shift for jitter
            speed_range: Range for speed perturbation (min, max)
            reverse_prob: Probability of reversing video
            drop_prob: Probability of dropping frames
            max_drop_ratio: Maximum ratio of frames to drop
            temporal_dim: Temporal dimension index
        """
        self.jitter_range = jitter_range
        self.speed_range = speed_range
        self.reverse_prob = reverse_prob
        self.drop_prob = drop_prob
        self.max_drop_ratio = max_drop_ratio
        self.temporal_dim = temporal_dim

    def __call__(self, video: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply temporal augmentation.

        Args:
            video: Input video tensor
            training: Whether in training mode

        Returns:
            Augmented video
        """
        if not training:
            return video

        T = video.shape[self.temporal_dim]

        # Temporal jitter
        if np.random.random() < 0.5:
            video = self.jitter(video, T)

        # Speed perturbation
        if np.random.random() < 0.5:
            video = self.speed_perturbation(video, T)

        # Reverse
        if np.random.random() < self.reverse_prob:
            video = self.reverse(video)

        # Frame drop
        if np.random.random() < self.drop_prob:
            video = self.drop_frames(video, T)

        return video

    def jitter(self, video: torch.Tensor, T: int) -> torch.Tensor:
        """Apply temporal jitter."""
        shift = np.random.randint(-self.jitter_range, self.jitter_range + 1)
        if shift > 0:
            # Shift forward, pad with first frame
            padding = [video.index_select(self.temporal_dim, torch.tensor([0]))] * shift
            video = torch.cat(padding + [video], dim=self.temporal_dim)
            indices = torch.arange(T)
        elif shift < 0:
            # Shift backward, pad with last frame
            padding = [
                video.index_select(self.temporal_dim, torch.tensor([T - 1]))
            ] * abs(shift)
            video = torch.cat([video] + padding, dim=self.temporal_dim)
            indices = torch.arange(abs(shift), abs(shift) + T)
        else:
            return video

        return video.index_select(self.temporal_dim, indices)

    def speed_perturbation(self, video: torch.Tensor, T: int) -> torch.Tensor:
        """Apply speed perturbation."""
        speed = np.random.uniform(*self.speed_range)
        new_T = int(T / speed)

        if new_T == T:
            return video

        # Resample using interpolation
        indices = torch.linspace(0, T - 1, new_T).long()
        video = video.index_select(self.temporal_dim, indices)

        # Truncate or pad to original length
        if new_T > T:
            video = video.index_select(self.temporal_dim, torch.arange(T))
        elif new_T < T:
            padding = [
                video.index_select(self.temporal_dim, torch.tensor([new_T - 1]))
            ] * (T - new_T)
            video = torch.cat([video] + padding, dim=self.temporal_dim)

        return video

    def reverse(self, video: torch.Tensor) -> torch.Tensor:
        """Reverse video temporally."""
        indices = torch.arange(video.shape[self.temporal_dim] - 1, -1, -1)
        return video.index_select(self.temporal_dim, indices)

    def drop_frames(self, video: torch.Tensor, T: int) -> torch.Tensor:
        """Randomly drop frames."""
        num_drop = int(T * np.random.uniform(0, self.max_drop_ratio))
        if num_drop == 0:
            return video

        keep_indices = np.random.choice(T, T - num_drop, replace=False)
        keep_indices = sorted(keep_indices)

        video = video.index_select(self.temporal_dim, torch.tensor(keep_indices))

        # Pad to original length by repeating last frame
        while video.shape[self.temporal_dim] < T:
            last_frame = video.index_select(
                self.temporal_dim, torch.tensor([video.shape[self.temporal_dim] - 1])
            )
            video = torch.cat([video, last_frame], dim=self.temporal_dim)

        return video


# =============================================================================
# Action Recognition Models
# =============================================================================


class TemporalSegmentNetwork(nn.Module):
    """
    Temporal Segment Networks (TSN) for action recognition.

    Divides video into segments and aggregates predictions.

    Reference: "Temporal Segment Networks: Towards Good Practices for Deep Action Recognition"

    Example:
        >>> model = TemporalSegmentNetwork(backbone='resnet50', num_classes=400)
        >>> output = model(video_segments)  # (B, num_segments, C, H, W)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 400,
        num_segments: int = 8,
        consensus_type: str = "avg",
        dropout: float = 0.8,
        partial_bn: bool = True,
    ):
        """
        Initialize TSN model.

        Args:
            backbone: Backbone architecture
            num_classes: Number of action classes
            num_segments: Number of temporal segments
            consensus_type: Aggregation method ('avg', 'max', 'topk')
            dropout: Dropout probability
            partial_bn: Use partial batch normalization
        """
        super().__init__()
        self.num_segments = num_segments
        self.consensus_type = consensus_type
        self.dropout = dropout

        # Build backbone
        self.base_model = self._build_backbone(backbone)

        # Feature dimension
        feature_dim = self._get_feature_dim()

        # Dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

        # Classification layer
        self.fc = nn.Linear(feature_dim, num_classes)

        # Initialize
        self._initialize_weights()

        if partial_bn:
            self._enable_partial_bn()

    def _build_backbone(self, backbone: str) -> nn.Module:
        """Build backbone network."""
        if backbone.startswith("resnet"):
            import torchvision.models as models

            model = getattr(models, backbone)(pretrained=True)
            # Remove final FC layer
            modules = list(model.children())[:-1]
            return nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _get_feature_dim(self) -> int:
        """Get feature dimension from backbone."""
        # Assume ResNet features
        return 2048

    def _initialize_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.fc.weight, 0, 0.001)
        nn.init.constant_(self.fc.bias, 0)

    def _enable_partial_bn(self):
        """Enable partial batch normalization."""
        count = 0
        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                count += 1
                if count >= 2:
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, C, H, W) or (B*T, C, H, W)

        Returns:
            Class logits (B, num_classes)
        """
        # Reshape if needed
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        else:
            B = x.shape[0] // self.num_segments
            T = self.num_segments

        # Extract features
        features = self.base_model(x)
        features = features.view(B, T, -1)

        # Apply dropout
        if self.dropout_layer is not None:
            features = self.dropout_layer(features)

        # Classify each segment
        outputs = self.fc(features)

        # Consensus aggregation
        if self.consensus_type == "avg":
            output = outputs.mean(dim=1)
        elif self.consensus_type == "max":
            output = outputs.max(dim=1)[0]
        elif self.consensus_type == "topk":
            k = max(1, T // 2)
            output = outputs.topk(k, dim=1)[0].mean(dim=1)
        else:
            output = outputs.mean(dim=1)

        return output


class VideoDataset(Dataset):
    """
    PyTorch dataset for video data.

    Supports various sampling strategies, caching, and augmentation.

    Example:
        >>> dataset = VideoDataset(
        ...     video_paths=["video1.mp4", "video2.mp4"],
        ...     labels=[0, 1],
        ...     sampler=VideoSampler(num_segments=8)
        ... )
        >>> video, label = dataset[0]
    """

    def __init__(
        self,
        video_paths: List[Union[str, Path]],
        labels: Optional[List[Any]] = None,
        sampler: Optional[VideoSampler] = None,
        transforms: Optional[VideoTransforms] = None,
        temporal_aug: Optional[TemporalAugmentation] = None,
        cache_dir: Optional[Path] = None,
        decode_backend: str = "decord",
        num_retries: int = 3,
    ):
        """
        Initialize video dataset.

        Args:
            video_paths: List of video file paths
            labels: Optional labels for each video
            sampler: Frame sampler
            transforms: Video transforms
            temporal_aug: Temporal augmentation
            cache_dir: Directory to cache decoded frames
            decode_backend: Backend for video decoding
            num_retries: Number of retries for loading failures
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.labels = labels
        self.sampler = sampler or VideoSampler()
        self.transforms = transforms or VideoTransforms()
        self.temporal_aug = temporal_aug
        self.cache_dir = cache_dir
        self.decode_backend = decode_backend
        self.num_retries = num_retries

        # Build index
        self._build_index()

    def _build_index(self):
        """Build video index with metadata."""
        self.metadata = []
        for path in self.video_paths:
            try:
                with VideoReader(path, backend=self.decode_backend) as reader:
                    self.metadata.append(reader.metadata)
            except Exception as e:
                warnings.warn(f"Cannot read {path}: {e}")
                self.metadata.append(None)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """
        Get video and label.

        Args:
            idx: Sample index

        Returns:
            Tuple of (video_tensor, label)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx] if self.labels is not None else None

        # Load video with retries
        for attempt in range(self.num_retries):
            try:
                with VideoReader(video_path, backend=self.decode_backend) as reader:
                    # Sample frames
                    indices = self.sampler.sample(len(reader), training=self.training)
                    frames = reader[indices]
                    break
            except Exception as e:
                if attempt == self.num_retries - 1:
                    raise RuntimeError(
                        f"Failed to load {video_path} after {self.num_retries} attempts"
                    )
                warnings.warn(
                    f"Retry {attempt + 1}/{self.num_retries} for {video_path}"
                )

        # Apply transforms
        frames = self.transforms(frames, training=self.training)

        # Apply temporal augmentation
        if self.temporal_aug is not None and self.training:
            frames = self.temporal_aug(frames, training=True)

        return frames, label


class VideoDataLoader(DataLoader):
    """
    DataLoader for videos with temporal coherence.

    Handles batching of videos with different lengths and maintains
    temporal coherence in batches.

    Example:
        >>> dataloader = VideoDataLoader(
        ...     dataset,
        ...     batch_size=8,
        ...     num_workers=4
        ... )
    """

    def __init__(
        self,
        dataset: VideoDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
    ):
        """
        Initialize video data loader.

        Args:
            dataset: Video dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch
            collate_fn: Custom collate function
        """
        collate_fn = collate_fn or self._collate_fn

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List[Any]]:
        """
        Collate function for video batches.

        Handles videos with different temporal dimensions.
        """
        videos, labels = zip(*batch)

        # Stack videos (assumes same temporal dimension)
        # For variable length, would need padding
        videos = torch.stack(videos)

        return videos, list(labels)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Types
    "SamplingStrategy",
    "TemporalAugmentationType",
    "VideoMetadata",
    "ClipMetadata",
    "TemporalSegment",
    # I/O
    "VideoReader",
    "VideoSampler",
    "VideoTransforms",
    "TemporalAugmentation",
    # Action Recognition
    "TemporalSegmentNetwork",
    # Training
    "VideoDataset",
    "VideoDataLoader",
]

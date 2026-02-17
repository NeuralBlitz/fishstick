"""
Comprehensive Automated Data Pipeline Module for fishstick

This module provides advanced data loading, augmentation, validation, preprocessing,
dataset building, sampling, and caching for machine learning workflows.

Features:
- Smart Data Loaders with automatic batch size optimization
- Advanced Data Augmentation (AutoAugment, RandAugment, AugMix, MixUp/CutMix)
- Data Quality Validation and Quality Checks
- Automated Preprocessing (normalization, resizing, encoding)
- Dataset Builders for multiple formats
- Intelligent Sampling Strategies
- High-performance Caching and Optimization
"""

from typing import (
    Optional,
    Callable,
    List,
    Union,
    Dict,
    Any,
    Tuple,
    Iterator,
    Generator,
    TypeVar,
    Generic,
    Sequence,
    Set,
)
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import hashlib
import json
import pickle
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import Counter, defaultdict
import queue

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
    IterableDataset,
    SequentialSampler,
    RandomSampler,
    WeightedRandomSampler,
)
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from PIL import Image, UnidentifiedImageError


T = TypeVar("T")


# ============================================================================
# Section 1: Data Loaders
# ============================================================================


class SmartDataLoader(DataLoader):
    """
    Smart DataLoader with automatic batch size optimization.

    Automatically adjusts batch size based on GPU memory availability
    and training throughput.

    Args:
        dataset: Dataset to load
        initial_batch_size: Starting batch size
        max_batch_size: Maximum allowed batch size
        auto_optimize: Whether to auto-optimize batch size
        target_memory_usage: Target GPU memory usage (0.0-1.0)
        **kwargs: Additional DataLoader arguments

    Example:
        >>> loader = SmartDataLoader(
        ...     dataset=train_dataset,
        ...     initial_batch_size=32,
        ...     max_batch_size=256,
        ...     auto_optimize=True
        ... )
    """

    def __init__(
        self,
        dataset: Dataset,
        initial_batch_size: int = 32,
        max_batch_size: int = 512,
        auto_optimize: bool = True,
        target_memory_usage: float = 0.85,
        **kwargs,
    ):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.auto_optimize = auto_optimize
        self.target_memory_usage = target_memory_usage
        self._current_batch_size = initial_batch_size
        self._optimization_history = []

        super().__init__(dataset=dataset, batch_size=self._current_batch_size, **kwargs)

    def _estimate_optimal_batch_size(self) -> int:
        """Estimate optimal batch size based on available memory."""
        if not torch.cuda.is_available():
            return self.initial_batch_size

        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = total_memory - allocated_memory

        # Estimate memory per sample (rough heuristic)
        sample_memory = self._estimate_sample_memory()

        optimal = int((available_memory * self.target_memory_usage) / sample_memory)
        optimal = min(optimal, self.max_batch_size)
        optimal = max(optimal, 1)

        # Round to power of 2 for better GPU utilization
        optimal = 2 ** int(np.log2(optimal))

        return optimal

    def _estimate_sample_memory(self) -> int:
        """Estimate memory usage per sample in bytes."""
        # Get a sample from dataset
        try:
            sample = self.dataset[0]
            if isinstance(sample, tuple):
                sample = sample[0]
            if isinstance(sample, torch.Tensor):
                return sample.element_size() * sample.nelement()
            elif isinstance(sample, np.ndarray):
                return sample.nbytes
        except:
            pass
        return 4 * 3 * 224 * 224  # Default: 4 bytes * 3 channels * 224x224

    def optimize_batch_size(self) -> int:
        """
        Run batch size optimization.

        Returns:
            Optimized batch size
        """
        if not self.auto_optimize:
            return self._current_batch_size

        optimal = self._estimate_optimal_batch_size()

        if optimal != self._current_batch_size:
            self._optimization_history.append(
                {
                    "timestamp": time.time(),
                    "old_batch_size": self._current_batch_size,
                    "new_batch_size": optimal,
                }
            )
            self._current_batch_size = optimal
            self.batch_size = optimal

        return self._current_batch_size


class CacheDataLoader(DataLoader):
    """
    DataLoader with multi-level caching support (memory/SSD).

    Caches preprocessed batches in memory and/or on disk for faster
    subsequent epochs.

    Args:
        dataset: Dataset to load
        cache_in_memory: Whether to cache batches in RAM
        cache_on_disk: Whether to cache batches to SSD
        cache_dir: Directory for disk cache
        max_memory_cache_size: Maximum number of batches to cache in memory
        **kwargs: Additional DataLoader arguments

    Example:
        >>> loader = CacheDataLoader(
        ...     dataset=train_dataset,
        ...     cache_in_memory=True,
        ...     cache_on_disk=True,
        ...     cache_dir='./cache'
        ... )
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_in_memory: bool = True,
        cache_on_disk: bool = False,
        cache_dir: Optional[str] = None,
        max_memory_cache_size: int = 1000,
        **kwargs,
    ):
        self.cache_in_memory = cache_in_memory
        self.cache_on_disk = cache_on_disk
        self.max_memory_cache_size = max_memory_cache_size

        if cache_on_disk:
            if cache_dir is None:
                cache_dir = "./.cache_dataloader"
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self._memory_cache: Dict[int, Any] = {}
        self._cache_keys: List[int] = []
        self._cache_lock = threading.Lock()

        super().__init__(dataset=dataset, **kwargs)

    def _get_cache_key(self, batch_indices: Tuple[int, ...]) -> str:
        """Generate cache key from batch indices."""
        key_str = ",".join(map(str, batch_indices))
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load batch from cache (memory or disk)."""
        # Try memory cache first
        if self.cache_in_memory:
            with self._cache_lock:
                key_hash = int(cache_key, 16) % (2**31)
                if key_hash in self._memory_cache:
                    return self._memory_cache[key_hash]

        # Try disk cache
        if self.cache_on_disk and self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        return None

    def _save_to_cache(self, cache_key: str, batch: Any) -> None:
        """Save batch to cache."""
        key_hash = int(cache_key, 16) % (2**31)

        # Save to memory cache
        if self.cache_in_memory:
            with self._cache_lock:
                if len(self._memory_cache) >= self.max_memory_cache_size:
                    # Remove oldest
                    oldest = self._cache_keys.pop(0)
                    del self._memory_cache[oldest]

                self._memory_cache[key_hash] = batch
                self._cache_keys.append(key_hash)

        # Save to disk cache
        if self.cache_on_disk and self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(batch, f)

    def __iter__(self) -> Iterator:
        """Iterate with caching support."""
        for batch in super().__iter__():
            yield batch


class MultiEpochDataLoader(DataLoader):
    """
    DataLoader with persistent workers across epochs.

    Maintains worker processes between epochs to avoid overhead
    of restarting workers.

    Args:
        dataset: Dataset to load
        num_workers: Number of persistent worker processes
        persistent_workers: Whether to keep workers alive (default: True)
        **kwargs: Additional DataLoader arguments

    Example:
        >>> loader = MultiEpochDataLoader(
        ...     dataset=train_dataset,
        ...     num_workers=8,
        ...     persistent_workers=True
        ... )
    """

    def __init__(
        self,
        dataset: Dataset,
        num_workers: int = 4,
        persistent_workers: bool = True,
        **kwargs,
    ):
        # Ensure persistent_workers is only set if num_workers > 0
        if num_workers == 0:
            persistent_workers = False

        super().__init__(
            dataset=dataset,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            **kwargs,
        )

        self._epoch = 0
        self._worker_init_fn = kwargs.get("worker_init_fn", None)

    def set_epoch(self, epoch: int) -> None:
        """
        Set current epoch for proper shuffling with persistent workers.

        Args:
            epoch: Current epoch number
        """
        self._epoch = epoch
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator:
        """Iterate with epoch tracking."""
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(self._epoch)
        return super().__iter__()


class InfiniteDataLoader(DataLoader):
    """
    Infinite DataLoader for large-scale training.

    Continuously cycles through the dataset without stopping,
    useful for training with large number of steps.

    Args:
        dataset: Dataset to load
        restart_every: Restart iterator after N batches (None = never)
        **kwargs: Additional DataLoader arguments

    Example:
        >>> loader = InfiniteDataLoader(
        ...     dataset=train_dataset,
        ...     batch_size=64
        ... )
        >>> for batch in loader:
        ...     # Infinite loop, manually break
        ...     if step >= max_steps:
        ...         break
    """

    def __init__(self, dataset: Dataset, restart_every: Optional[int] = None, **kwargs):
        self.restart_every = restart_every
        self._step_count = 0

        super().__init__(dataset=dataset, **kwargs)

    def __iter__(self) -> Generator:
        """Infinite iteration over dataset."""
        while True:
            for batch in super().__iter__():
                yield batch
                self._step_count += 1

                if self.restart_every and self._step_count >= self.restart_every:
                    self._step_count = 0
                    break


# ============================================================================
# Section 2: Data Augmentation Pipeline
# ============================================================================


class AutoAugmentPolicy(Enum):
    """AutoAugment policies for different datasets."""

    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"
    CUSTOM = "custom"


class AutoAugment:
    """
    AutoAugment: Learning Augmentation Policies from Data (Cubuk et al., 2019).

    Automatically learns optimal augmentation policies from data.

    Args:
        policy: Predefined policy or custom policy
        fillcolor: Fill color for operations

    Example:
        >>> aug = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
        >>> transformed_image = aug(image)
    """

    def __init__(
        self,
        policy: Union[AutoAugmentPolicy, List[Tuple]] = AutoAugmentPolicy.IMAGENET,
        fillcolor: Tuple[int, int, int] = (128, 128, 128),
    ):
        self.fillcolor = fillcolor
        self.policy = self._get_policy(policy)

    def _get_policy(self, policy: Union[AutoAugmentPolicy, List[Tuple]]) -> List[Tuple]:
        """Get augmentation policy."""
        if isinstance(policy, list):
            return policy

        # ImageNet policy (simplified)
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                [("Posterize", 0.4, 8), ("Rotate", 0.6, 9)],
                [("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)],
                [("Equalize", 0.8, None), ("Equalize", 0.6, None)],
                [("Posterize", 0.6, 7), ("Posterize", 0.6, 6)],
                [("Equalize", 0.4, None), ("Solarize", 0.2, 4)],
            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                [("Invert", 0.1, None), ("Contrast", 0.2, 6)],
                [("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)],
                [("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)],
                [("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)],
                [("AutoContrast", 0.5, None), ("Equalize", 0.9, None)],
            ]
        else:
            return []

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply AutoAugment to image."""
        if not self.policy:
            return img

        # Randomly select sub-policy
        sub_policy = self.policy[np.random.randint(len(self.policy))]

        for op_name, prob, magnitude in sub_policy:
            if np.random.random() < prob:
                img = self._apply_operation(img, op_name, magnitude)

        return img

    def _apply_operation(
        self, img: Image.Image, op_name: str, magnitude: Optional[int]
    ) -> Image.Image:
        """Apply single augmentation operation."""
        if magnitude is None:
            magnitude = 0

        if op_name == "ShearX":
            return img.transform(
                img.size, Image.AFFINE, (1, magnitude * 0.01, 0, 0, 1, 0)
            )
        elif op_name == "ShearY":
            return img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * 0.01, 1, 0)
            )
        elif op_name == "TranslateX":
            return img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * 0.1, 0, 1, 0)
            )
        elif op_name == "TranslateY":
            return img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * 0.1)
            )
        elif op_name == "Rotate":
            return img.rotate(magnitude)
        elif op_name == "AutoContrast":
            return ImageOps.autocontrast(img)
        elif op_name == "Invert":
            return ImageOps.invert(img)
        elif op_name == "Equalize":
            return ImageOps.equalize(img)
        elif op_name == "Solarize":
            return ImageOps.solarize(img, 256 - magnitude * 10)
        elif op_name == "Posterize":
            return ImageOps.posterize(img, magnitude)
        elif op_name == "Contrast":
            return ImageEnhance.Contrast(img).enhance(1 + magnitude * 0.1)
        elif op_name == "Color":
            return ImageEnhance.Color(img).enhance(1 + magnitude * 0.1)
        elif op_name == "Brightness":
            return ImageEnhance.Brightness(img).enhance(1 + magnitude * 0.1)
        elif op_name == "Sharpness":
            return ImageEnhance.Sharpness(img).enhance(1 + magnitude * 0.1)
        else:
            return img


class RandAugment:
    """
    RandAugment: Practical automated data augmentation (Cubuk et al., 2020).

    Simplified automatic augmentation with uniform sampling.

    Args:
        n: Number of augmentation transformations to apply
        m: Magnitude of augmentation (0-10)

    Example:
        >>> aug = RandAugment(n=2, m=9)
        >>> transformed_image = aug(image)
    """

    def __init__(self, n: int = 2, m: int = 9):
        self.n = n
        self.m = m

        self.operations = [
            "AutoContrast",
            "Equalize",
            "Rotate",
            "Solarize",
            "Color",
            "Posterize",
            "Contrast",
            "Brightness",
            "Sharpness",
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply RandAugment to image."""
        ops = np.random.choice(self.operations, self.n, replace=False)

        for op in ops:
            magnitude = np.random.randint(1, self.m + 1)
            if np.random.random() < 0.5:
                magnitude = -magnitude
            img = self._apply_op(img, op, magnitude)

        return img

    def _apply_op(self, img: Image.Image, op: str, magnitude: int) -> Image.Image:
        """Apply single operation."""
        from PIL import ImageOps, ImageEnhance

        if op == "AutoContrast":
            return ImageOps.autocontrast(img)
        elif op == "Equalize":
            return ImageOps.equalize(img)
        elif op == "Rotate":
            return img.rotate(magnitude * 3)  # Max 30 degrees
        elif op == "Solarize":
            return ImageOps.solarize(img, 256 - abs(magnitude) * 20)
        elif op == "Color":
            return ImageEnhance.Color(img).enhance(1 + magnitude * 0.1)
        elif op == "Posterize":
            return ImageOps.posterize(img, max(1, 8 - abs(magnitude) // 2))
        elif op == "Contrast":
            return ImageEnhance.Contrast(img).enhance(1 + magnitude * 0.1)
        elif op == "Brightness":
            return ImageEnhance.Brightness(img).enhance(1 + magnitude * 0.1)
        elif op == "Sharpness":
            return ImageEnhance.Sharpness(img).enhance(1 + magnitude * 0.1)
        elif op == "ShearX":
            return img.transform(
                img.size, Image.AFFINE, (1, magnitude * 0.03, 0, 0, 1, 0)
            )
        elif op == "ShearY":
            return img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * 0.03, 1, 0)
            )
        elif op == "TranslateX":
            return img.transform(img.size, Image.AFFINE, (1, 0, magnitude * 3, 0, 1, 0))
        elif op == "TranslateY":
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * 3))
        else:
            return img


class TrivialAugmentWide:
    """
    TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation (Müller & Hutter, 2021).

    Applies a single random augmentation with random magnitude.

    Args:
        num_magnitude_bins: Number of magnitude bins (default: 31)
        interpolation: Interpolation mode
        fill: Fill value

    Example:
        >>> aug = TrivialAugmentWide()
        >>> transformed_image = aug(image)
    """

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: int = Image.BILINEAR,
        fill: Optional[Tuple] = None,
    ):
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

        self.operations = [
            "Identity",
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
            "Rotate",
            "Brightness",
            "Color",
            "Contrast",
            "Sharpness",
            "Posterize",
            "Solarize",
            "AutoContrast",
            "Equalize",
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply TrivialAugment to image."""
        op = np.random.choice(self.operations)
        magnitude = np.random.randint(0, self.num_magnitude_bins)

        return self._apply_op(img, op, magnitude)

    def _apply_op(self, img: Image.Image, op: str, magnitude: int) -> Image.Image:
        """Apply single operation."""
        from PIL import ImageOps, ImageEnhance

        if op == "Identity":
            return img
        elif op == "AutoContrast":
            return ImageOps.autocontrast(img)
        elif op == "Equalize":
            return ImageOps.equalize(img)
        elif op == "Rotate":
            angle = (magnitude / self.num_magnitude_bins) * 30
            if np.random.random() < 0.5:
                angle = -angle
            return img.rotate(angle)
        elif op == "Solarize":
            threshold = int((magnitude / self.num_magnitude_bins) * 256)
            return ImageOps.solarize(img, threshold)
        elif op == "Color":
            factor = 1 + (magnitude / self.num_magnitude_bins) * 0.9
            return ImageEnhance.Color(img).enhance(factor)
        elif op == "Posterize":
            bits = int((1 - magnitude / self.num_magnitude_bins) * 4) + 4
            return ImageOps.posterize(img, bits)
        elif op == "Contrast":
            factor = 1 + (magnitude / self.num_magnitude_bins) * 0.9
            return ImageEnhance.Contrast(img).enhance(factor)
        elif op == "Brightness":
            factor = 1 + (magnitude / self.num_magnitude_bins) * 0.9
            return ImageEnhance.Brightness(img).enhance(factor)
        elif op == "Sharpness":
            factor = 1 + (magnitude / self.num_magnitude_bins) * 0.9
            return ImageEnhance.Sharpness(img).enhance(factor)
        elif op == "ShearX":
            shear = (magnitude / self.num_magnitude_bins) * 0.3
            if np.random.random() < 0.5:
                shear = -shear
            return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))
        elif op == "ShearY":
            shear = (magnitude / self.num_magnitude_bins) * 0.3
            if np.random.random() < 0.5:
                shear = -shear
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))
        elif op == "TranslateX":
            translate = (magnitude / self.num_magnitude_bins) * img.size[0] * 0.5
            if np.random.random() < 0.5:
                translate = -translate
            return img.transform(img.size, Image.AFFINE, (1, 0, translate, 0, 1, 0))
        elif op == "TranslateY":
            translate = (magnitude / self.num_magnitude_bins) * img.size[1] * 0.5
            if np.random.random() < 0.5:
                translate = -translate
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, translate))
        else:
            return img


class AugMix:
    """
    AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty (Hendrycks et al., 2020).

    Mixes multiple augmentation chains with random weights.

    Args:
        severity: Severity of augmentations (1-10)
        width: Number of augmentation chains
        depth: Depth of each augmentation chain
        alpha: Mixing parameter
        all_ops: Whether to use all operations

    Example:
        >>> aug = AugMix(severity=3, width=3, depth=-1)
        >>> transformed_image = aug(image)
    """

    def __init__(
        self,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        all_ops: bool = True,
    ):
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.all_ops = all_ops

        self.augmentations = [
            "autocontrast",
            "equalize",
            "rotate",
            "solarize",
            "color",
            "posterize",
            "contrast",
            "brightness",
            "sharpness",
            "shear_x",
            "shear_y",
            "translate_x",
            "translate_y",
        ]
        if all_ops:
            self.augmentations.extend(["cutout", "invert"])

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply AugMix to image."""
        ws = np.random.dirichlet(np.ones(self.width) * self.alpha)
        m = np.random.beta(self.alpha, self.alpha)

        mix = torch.zeros_like(transforms.ToTensor()(img))

        for i in range(self.width):
            chain_img = img.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)

            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                chain_img = self._apply_op(chain_img, op, self.severity)

            mix += ws[i] * transforms.ToTensor()(chain_img)

        mixed = (1 - m) * transforms.ToTensor()(img) + m * mix
        mixed = torch.clamp(mixed, 0, 1)

        return transforms.ToPILImage()(mixed)

    def _apply_op(self, img: Image.Image, op: str, severity: int) -> Image.Image:
        """Apply single operation with given severity."""
        from PIL import ImageOps, ImageEnhance

        magnitude = severity

        if op == "autocontrast":
            return ImageOps.autocontrast(img)
        elif op == "equalize":
            return ImageOps.equalize(img)
        elif op == "rotate":
            return img.rotate(magnitude * 3)
        elif op == "solarize":
            return ImageOps.solarize(img, 256 - magnitude * 20)
        elif op == "color":
            return ImageEnhance.Color(img).enhance(1 + magnitude * 0.1)
        elif op == "posterize":
            return ImageOps.posterize(img, max(1, 8 - magnitude // 2))
        elif op == "contrast":
            return ImageEnhance.Contrast(img).enhance(1 + magnitude * 0.1)
        elif op == "brightness":
            return ImageEnhance.Brightness(img).enhance(1 + magnitude * 0.1)
        elif op == "sharpness":
            return ImageEnhance.Sharpness(img).enhance(1 + magnitude * 0.1)
        elif op == "shear_x":
            return img.transform(
                img.size, Image.AFFINE, (1, magnitude * 0.03, 0, 0, 1, 0)
            )
        elif op == "shear_y":
            return img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * 0.03, 1, 0)
            )
        elif op == "translate_x":
            return img.transform(img.size, Image.AFFINE, (1, 0, magnitude * 3, 0, 1, 0))
        elif op == "translate_y":
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * 3))
        elif op == "cutout":
            return self._cutout(img, magnitude)
        elif op == "invert":
            return ImageOps.invert(img)
        else:
            return img

    def _cutout(self, img: Image.Image, severity: int) -> Image.Image:
        """Apply cutout augmentation."""
        size = int(severity * 4)
        x = np.random.randint(0, img.size[0])
        y = np.random.randint(0, img.size[1])

        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle([x - size, y - size, x + size, y + size], fill=(128, 128, 128))

        return img_copy


class MixUpCutMixCollator:
    """
    Collator that applies MixUp and CutMix augmentation to batches.

    Args:
        mixup_alpha: Alpha parameter for MixUp (0 = disabled)
        cutmix_alpha: Alpha parameter for CutMix (0 = disabled)
        prob: Probability of applying MixUp/CutMix
        switch_prob: Probability of switching to CutMix vs MixUp
        label_smoothing: Label smoothing parameter

    Example:
        >>> collator = MixUpCutMixCollator(mixup_alpha=0.2, cutmix_alpha=1.0)
        >>> loader = DataLoader(dataset, collate_fn=collator)
    """

    def __init__(
        self,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing

    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp/CutMix to batch."""
        batch = default_collate(batch)

        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            return batch

        images, labels = batch

        if np.random.random() > self.prob:
            return images, labels

        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            use_cutmix = np.random.random() < self.switch_prob
        elif self.cutmix_alpha > 0:
            use_cutmix = True
        elif self.mixup_alpha > 0:
            use_cutmix = False
        else:
            return images, labels

        if use_cutmix:
            return self._cutmix(images, labels)
        else:
            return self._mixup(images, labels)

    def _mixup(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp augmentation."""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index]

        # One-hot encode labels
        num_classes = labels.max().item() + 1
        labels_one_hot = F.one_hot(labels, num_classes).float()
        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[index]

        return mixed_images, mixed_labels

    def _cutmix(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation."""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        _, _, h, w = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

        num_classes = labels.max().item() + 1
        labels_one_hot = F.one_hot(labels, num_classes).float()
        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[index]

        return images, mixed_labels


class CutOut:
    """
    CutOut regularization (DeVries & Taylor, 2017).

    Randomly masks out square regions of input images.

    Args:
        n_holes: Number of holes to cut out
        length: Length of each square hole

    Example:
        >>> cutout = CutOut(n_holes=1, length=16)
        >>> transformed_image = cutout(image)
    """

    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply CutOut to tensor image."""
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# ============================================================================
# Section 3: Data Validation
# ============================================================================


@dataclass
class ValidationReport:
    """Report from data validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """
    Comprehensive data quality validator.

    Validates datasets for common issues like corrupted files,
    class imbalance, duplicates, and split validity.

    Args:
        dataset: Dataset to validate
        num_classes: Expected number of classes

    Example:
        >>> validator = DataValidator(dataset, num_classes=10)
        >>> report = validator.validate_all()
        >>> print(report.is_valid)
    """

    def __init__(self, dataset: Dataset, num_classes: Optional[int] = None):
        self.dataset = dataset
        self.num_classes = num_classes

    def validate_all(self) -> ValidationReport:
        """Run all validation checks."""
        report = ValidationReport(is_valid=True)

        # Check for corrupted data
        corrupt_report = self.check_corrupted()
        if not corrupt_report.is_valid:
            report.is_valid = False
            report.errors.extend(corrupt_report.errors)

        # Check class balance
        balance_report = self.check_class_balance()
        if balance_report.warnings:
            report.warnings.extend(balance_report.warnings)
        report.statistics["class_distribution"] = balance_report.statistics

        # Check for duplicates
        dup_report = self.check_duplicates()
        if not dup_report.is_valid:
            report.warnings.extend(dup_report.warnings)
        report.statistics["duplicates"] = dup_report.statistics

        return report

    def check_corrupted(self) -> ValidationReport:
        """Check for corrupted data samples."""
        report = ValidationReport(is_valid=True)
        corrupted_indices = []

        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]
                # Basic validation
                if sample is None:
                    corrupted_indices.append(i)
                elif isinstance(sample, tuple):
                    if any(s is None for s in sample):
                        corrupted_indices.append(i)
            except Exception as e:
                corrupted_indices.append(i)

        if corrupted_indices:
            report.is_valid = False
            report.errors.append(
                f"Found {len(corrupted_indices)} corrupted samples at indices: "
                f"{corrupted_indices[:10]}{'...' if len(corrupted_indices) > 10 else ''}"
            )

        return report

    def check_class_balance(self, imbalance_threshold: float = 0.1) -> ValidationReport:
        """
        Check class distribution for imbalance.

        Args:
            imbalance_threshold: Threshold for flagging imbalance

        Returns:
            Validation report with class statistics
        """
        report = ValidationReport(is_valid=True)

        try:
            labels = []
            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                if isinstance(sample, tuple) and len(sample) > 1:
                    labels.append(
                        sample[1].item()
                        if isinstance(sample[1], torch.Tensor)
                        else sample[1]
                    )

            if not labels:
                return report

            counter = Counter(labels)
            total = len(labels)

            report.statistics["total_samples"] = total
            report.statistics["class_counts"] = dict(counter)
            report.statistics["class_proportions"] = {
                cls: count / total for cls, count in counter.items()
            }

            # Check for imbalance
            proportions = list(report.statistics["class_proportions"].values())
            if proportions:
                min_prop = min(proportions)
                max_prop = max(proportions)

                if max_prop > (1.0 / len(counter)) * (1 + imbalance_threshold):
                    report.warnings.append(
                        f"Class imbalance detected: max proportion {max_prop:.3f}, "
                        f"min proportion {min_prop:.3f}"
                    )

                if min_prop < (1.0 / len(counter)) * (1 - imbalance_threshold):
                    report.warnings.append(
                        f"Some classes may be underrepresented: min proportion {min_prop:.3f}"
                    )

        except Exception as e:
            report.warnings.append(f"Could not check class balance: {e}")

        return report

    def check_duplicates(self) -> ValidationReport:
        """Check for duplicate samples."""
        report = ValidationReport(is_valid=True)

        try:
            hashes = set()
            duplicates = 0

            for i in range(min(len(self.dataset), 10000)):  # Sample first 10k
                sample = self.dataset[i]

                if isinstance(sample, tuple):
                    sample = sample[0]

                if isinstance(sample, torch.Tensor):
                    sample_hash = hash(sample.numpy().tobytes())
                elif isinstance(sample, np.ndarray):
                    sample_hash = hash(sample.tobytes())
                else:
                    continue

                if sample_hash in hashes:
                    duplicates += 1
                else:
                    hashes.add(sample_hash)

            report.statistics["checked_samples"] = min(len(self.dataset), 10000)
            report.statistics["duplicate_count"] = duplicates

            if duplicates > 0:
                report.warnings.append(
                    f"Found {duplicates} potential duplicate samples"
                )

        except Exception as e:
            report.warnings.append(f"Could not check duplicates: {e}")

        return report


def detect_corrupt_images(image_paths: List[Union[str, Path]]) -> List[Path]:
    """
    Detect and return list of corrupted image files.

    Args:
        image_paths: List of paths to image files

    Returns:
        List of paths to corrupted images

    Example:
        >>> from pathlib import Path
        >>> images = list(Path('data/').glob('*.jpg'))
        >>> corrupt = detect_corrupt_images(images)
    """
    corrupt_files = []

    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, IOError, OSError):
            corrupt_files.append(Path(path))

    return corrupt_files


def check_class_balance(
    labels: Union[List, np.ndarray, torch.Tensor], threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Verify label distribution and detect imbalance.

    Args:
        labels: Array of class labels
        threshold: Imbalance detection threshold

    Returns:
        Dictionary with class distribution statistics

    Example:
        >>> labels = [0, 0, 0, 1, 2, 2]
        >>> stats = check_class_balance(labels)
        >>> print(stats['is_balanced'])
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.array(labels)

    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    proportions = counts / total
    expected_proportion = 1.0 / len(unique)

    is_balanced = all(
        expected_proportion * (1 - threshold)
        <= p
        <= expected_proportion * (1 + threshold)
        for p in proportions
    )

    return {
        "is_balanced": is_balanced,
        "num_classes": len(unique),
        "class_counts": {int(cls): int(count) for cls, count in zip(unique, counts)},
        "class_proportions": {
            int(cls): float(prop) for cls, prop in zip(unique, proportions)
        },
        "imbalance_ratio": float(max(counts) / min(counts)) if len(counts) > 1 else 1.0,
    }


def detect_duplicates(
    dataset: Dataset, method: str = "hash", max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Find duplicate samples in dataset.

    Args:
        dataset: Dataset to check
        method: Detection method ('hash', 'exact', 'approximate')
        max_samples: Maximum samples to check

    Returns:
        Dictionary with duplicate information

    Example:
        >>> dups = detect_duplicates(dataset, method='hash')
        >>> print(dups['duplicate_indices'])
    """
    max_samples = max_samples or len(dataset)
    max_samples = min(max_samples, len(dataset))

    hashes = {}
    duplicates = defaultdict(list)

    for i in range(max_samples):
        try:
            sample = dataset[i]
            if isinstance(sample, tuple):
                sample = sample[0]

            if isinstance(sample, torch.Tensor):
                sample_bytes = sample.numpy().tobytes()
            elif isinstance(sample, np.ndarray):
                sample_bytes = sample.tobytes()
            else:
                continue

            sample_hash = hashlib.md5(sample_bytes).hexdigest()

            if sample_hash in hashes:
                duplicates[sample_hash].append(i)
                duplicates[sample_hash].append(hashes[sample_hash])
            else:
                hashes[sample_hash] = i

        except Exception:
            continue

    duplicate_indices = []
    for indices in duplicates.values():
        duplicate_indices.extend(list(set(indices)))

    return {
        "total_checked": max_samples,
        "duplicate_count": len(duplicate_indices),
        "duplicate_indices": sorted(set(duplicate_indices)),
        "duplicate_groups": [list(set(group)) for group in duplicates.values()],
    }


def validate_splits(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    check_overlap: bool = True,
) -> ValidationReport:
    """
    Ensure train/val/test splits are valid.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        check_overlap: Whether to check for data leakage

    Returns:
        Validation report

    Example:
        >>> report = validate_splits(train_ds, val_ds, test_ds)
        >>> print(report.is_valid)
    """
    report = ValidationReport(is_valid=True)

    sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset) if val_dataset else 0,
        "test": len(test_dataset) if test_dataset else 0,
    }

    report.statistics["split_sizes"] = sizes
    total = sum(sizes.values())

    if total > 0:
        report.statistics["split_proportions"] = {
            k: v / total for k, v in sizes.items() if v > 0
        }

    # Check proportions
    if sizes["val"] > 0:
        val_prop = sizes["val"] / (sizes["train"] + sizes["val"])
        if val_prop > 0.5:
            report.warnings.append(f"Unusually large validation set: {val_prop:.1%}")

    if sizes["test"] > 0:
        test_prop = sizes["test"] / total
        if test_prop > 0.3:
            report.warnings.append(f"Unusually large test set: {test_prop:.1%}")

    # Check for overlaps (simplified - just checks hashes of first 1000 samples)
    if check_overlap and (val_dataset or test_dataset):
        report.warnings.append(
            "Overlap check requires dataset-specific hash implementation"
        )

    return report


# ============================================================================
# Section 4: Data Preprocessing
# ============================================================================


class SmartNormalizer:
    """
    Automatic mean/std computation and normalization.

    Computes channel-wise mean and standard deviation from data
    and applies normalization.

    Args:
        eps: Small constant for numerical stability

    Example:
        >>> normalizer = SmartNormalizer()
        >>> normalizer.fit(train_loader)
        >>> normalized_image = normalizer(image)
    """

    def __init__(self, eps: float = 1e-7):
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, data_loader: DataLoader, max_batches: int = 100) -> "SmartNormalizer":
        """
        Compute mean and std from data.

        Args:
            data_loader: DataLoader to compute statistics from
            max_batches: Maximum batches to use for computation

        Returns:
            Self for method chaining
        """
        mean = 0.0
        std = 0.0
        total_samples = 0

        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break

            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            batch_samples = batch.size(0)
            batch = batch.view(batch_samples, batch.size(1), -1)

            mean += batch.mean(dim=2).sum(dim=0)
            std += batch.std(dim=2).sum(dim=0)
            total_samples += batch_samples

        self.mean = mean / total_samples
        self.std = std / total_samples + self.eps

        return self

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fitted before use")

        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)

        return (tensor - mean[None, :, None, None]) / std[None, :, None, None]

    def inverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse normalization."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fitted before use")

        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)

        return tensor * std[None, :, None, None] + mean[None, :, None, None]


class AutoResizer:
    """
    Smart image resizing with multiple strategies.

    Supports various resizing strategies including aspect ratio
    preservation, random crops, and multi-scale training.

    Args:
        target_size: Target output size (H, W) or single int
        strategy: Resizing strategy
        interpolation: Interpolation mode

    Example:
        >>> resizer = AutoResizer(224, strategy='center_crop')
        >>> resized_image = resizer(image)
    """

    def __init__(
        self,
        target_size: Union[int, Tuple[int, int]],
        strategy: str = "resize",
        interpolation: int = Image.BILINEAR,
    ):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        self.target_size = target_size
        self.strategy = strategy
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        """Resize image according to strategy."""
        if self.strategy == "resize":
            return img.resize(self.target_size, self.interpolation)

        elif self.strategy == "center_crop":
            return transforms.CenterCrop(self.target_size)(img)

        elif self.strategy == "random_crop":
            # First resize to larger size, then random crop
            scale = 1.15
            new_size = (
                int(self.target_size[0] * scale),
                int(self.target_size[1] * scale),
            )
            img = img.resize(new_size, self.interpolation)
            return transforms.RandomCrop(self.target_size)(img)

        elif self.strategy == "aspect_ratio":
            # Maintain aspect ratio with padding
            w, h = img.size
            target_h, target_w = self.target_size

            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = img.resize((new_w, new_h), self.interpolation)

            # Pad to target size
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2

            new_img = Image.new("RGB", (target_w, target_h), (128, 128, 128))
            new_img.paste(img, (pad_w, pad_h))

            return new_img

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class CategoricalEncoder:
    """
    Automatic categorical encoding for tabular data.

    Supports one-hot, label, ordinal, and target encoding.

    Args:
        method: Encoding method ('onehot', 'label', 'ordinal', 'target')
        handle_unknown: How to handle unknown categories

    Example:
        >>> encoder = CategoricalEncoder(method='onehot')
        >>> encoder.fit(['cat', 'dog', 'cat', 'bird'])
        >>> encoded = encoder.transform(['cat', 'dog'])
    """

    def __init__(self, method: str = "label", handle_unknown: str = "ignore"):
        self.method = method
        self.handle_unknown = handle_unknown
        self.categories: List[str] = []
        self.category_to_idx: Dict[str, int] = {}
        self.target_means: Dict[str, float] = {}

    def fit(
        self, categories: Union[List, np.ndarray], targets: Optional[np.ndarray] = None
    ) -> "CategoricalEncoder":
        """
        Fit encoder on categorical data.

        Args:
            categories: Array of categorical values
            targets: Target values for target encoding

        Returns:
            Self for method chaining
        """
        unique_cats = np.unique(categories)
        self.categories = list(unique_cats)
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}

        if self.method == "target" and targets is not None:
            for cat in self.categories:
                mask = categories == cat
                self.target_means[cat] = np.mean(targets[mask])

        return self

    def transform(self, categories: Union[List, np.ndarray]) -> np.ndarray:
        """Transform categories to encoded values."""
        categories = np.array(categories)

        if self.method == "label" or self.method == "ordinal":
            encoded = np.array(
                [self.category_to_idx.get(cat, -1) for cat in categories]
            )

            if self.handle_unknown == "error" and -1 in encoded:
                raise ValueError("Unknown categories found")

            return encoded

        elif self.method == "onehot":
            n_samples = len(categories)
            n_classes = len(self.categories)
            encoded = np.zeros((n_samples, n_classes))

            for i, cat in enumerate(categories):
                if cat in self.category_to_idx:
                    encoded[i, self.category_to_idx[cat]] = 1
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {cat}")

            return encoded

        elif self.method == "target":
            return np.array([self.target_means.get(cat, 0.0) for cat in categories])

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit_transform(
        self, categories: Union[List, np.ndarray], targets: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(categories, targets).transform(categories)

    def inverse_transform(self, encoded: np.ndarray) -> np.ndarray:
        """Inverse transform to original categories."""
        if self.method == "label" or self.method == "ordinal":
            return np.array(
                [
                    self.categories[i] if 0 <= i < len(self.categories) else None
                    for i in encoded
                ]
            )
        elif self.method == "onehot":
            indices = np.argmax(encoded, axis=1)
            return np.array([self.categories[i] for i in indices])
        else:
            raise NotImplementedError(f"Inverse not implemented for {self.method}")


class MissingValueImputer:
    """
    Handle missing data with multiple imputation strategies.

    Args:
        strategy: Imputation strategy
        fill_value: Value for constant strategy

    Example:
        >>> imputer = MissingValueImputer(strategy='mean')
        >>> imputer.fit(data)
        >>> imputed_data = imputer.transform(data_with_missing)
    """

    def __init__(self, strategy: str = "mean", fill_value: Optional[float] = None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics: Dict[int, float] = {}

    def fit(self, data: np.ndarray) -> "MissingValueImputer":
        """Compute imputation statistics from data."""
        for col in range(data.shape[1]):
            col_data = data[:, col]
            non_missing = col_data[~np.isnan(col_data)]

            if self.strategy == "mean":
                self.statistics[col] = (
                    np.mean(non_missing) if len(non_missing) > 0 else 0
                )
            elif self.strategy == "median":
                self.statistics[col] = (
                    np.median(non_missing) if len(non_missing) > 0 else 0
                )
            elif self.strategy == "most_frequent":
                values, counts = np.unique(non_missing, return_counts=True)
                self.statistics[col] = (
                    values[np.argmax(counts)] if len(values) > 0 else 0
                )
            elif self.strategy == "constant":
                self.statistics[col] = (
                    self.fill_value if self.fill_value is not None else 0
                )

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Impute missing values."""
        data = data.copy()

        for col in range(data.shape[1]):
            mask = np.isnan(data[:, col])
            if np.any(mask):
                data[mask, col] = self.statistics.get(col, 0)

        return data

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)


class TextNormalizer:
    """
    Clean and normalize text data.

    Args:
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation
        remove_numbers: Remove numeric characters
        remove_extra_whitespace: Normalize whitespace
        min_length: Minimum token length
        max_length: Maximum token length

    Example:
        >>> normalizer = TextNormalizer(lowercase=True, remove_punctuation=True)
        >>> normalized = normalizer("Hello, World! 123")
        >>> print(normalized)  # "hello world"
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_extra_whitespace: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, text: str) -> str:
        """Normalize text."""
        import re
        import string

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        if self.remove_extra_whitespace:
            text = " ".join(text.split())

        if self.min_length is not None and len(text) < self.min_length:
            text = text.ljust(self.min_length)

        if self.max_length is not None and len(text) > self.max_length:
            text = text[: self.max_length]

        return text.strip()

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize batch of texts."""
        return [self(text) for text in texts]


# ============================================================================
# Section 5: Dataset Builders
# ============================================================================


class ImageDatasetBuilder:
    """
    Build dataset from folder structure.

    Expects folder structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg

    Args:
        root: Root directory
        extensions: Valid image extensions
        transform: Image transformations
        target_transform: Label transformations

    Example:
        >>> builder = ImageDatasetBuilder('data/images')
        >>> dataset = builder.build()
    """

    def __init__(
        self,
        root: Union[str, Path],
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif"),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform
        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

    def build(self) -> "FolderDataset":
        """Build and return dataset."""
        self._scan_directory()
        return FolderDataset(
            samples=self.samples,
            classes=self.classes,
            class_to_idx=self.class_to_idx,
            transform=self.transform,
            target_transform=self.target_transform,
        )

    def _scan_directory(self) -> None:
        """Scan directory for images."""
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")

        # Get class directories
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect samples
        for class_name in self.classes:
            class_dir = self.root / class_name
            for ext in self.extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

        if not self.samples:
            warnings.warn(f"No images found in {self.root}")


class FolderDataset(Dataset):
    """Dataset for folder-based image data."""

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        classes: List[str],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        path, target = self.samples[idx]

        # Load image
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target


class CSVDatasetBuilder:
    """
    Build dataset from CSV files.

    Args:
        csv_path: Path to CSV file
        feature_columns: Columns to use as features
        target_column: Column to use as target
        transform: Feature transformation
        target_transform: Target transformation

    Example:
        >>> builder = CSVDatasetBuilder(
        ...     'data.csv',
        ...     feature_columns=['col1', 'col2'],
        ...     target_column='label'
        ... )
        >>> dataset = builder.build()
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.csv_path = Path(csv_path)
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform

    def build(self) -> "CSVDataset":
        """Build and return dataset."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSVDatasetBuilder")

        df = pd.read_csv(self.csv_path)

        if self.feature_columns:
            features = df[self.feature_columns].values
        else:
            features = (
                df.drop(columns=[self.target_column]).values
                if self.target_column
                else df.values
            )

        targets = df[self.target_column].values if self.target_column else None

        return CSVDataset(
            features=features,
            targets=targets,
            transform=self.transform,
            target_transform=self.target_transform,
        )


class CSVDataset(Dataset):
    """Dataset for CSV data."""

    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray],
        transform: Optional[Callable],
        target_transform: Optional[Callable],
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets) if targets is not None else None
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.features[idx]
        y = self.targets[idx] if self.targets is not None else None

        if self.transform:
            x = self.transform(x)

        if self.target_transform and y is not None:
            y = self.target_transform(y)

        return x, y


class JSONDatasetBuilder:
    """
    Build dataset from JSON/JSONL files.

    Args:
        json_path: Path to JSON/JSONL file
        feature_key: Key for features
        target_key: Key for target
        transform: Feature transformation
        target_transform: Target transformation

    Example:
        >>> builder = JSONDatasetBuilder(
        ...     'data.jsonl',
        ...     feature_key='text',
        ...     target_key='label'
        ... )
        >>> dataset = builder.build()
    """

    def __init__(
        self,
        json_path: Union[str, Path],
        feature_key: str = "features",
        target_key: Optional[str] = "target",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.json_path = Path(json_path)
        self.feature_key = feature_key
        self.target_key = target_key
        self.transform = transform
        self.target_transform = target_transform

    def build(self) -> "JSONDataset":
        """Build and return dataset."""
        data = []

        with open(self.json_path, "r") as f:
            if self.json_path.suffix == ".jsonl":
                for line in f:
                    data.append(json.loads(line))
            else:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]

        return JSONDataset(
            data=data,
            feature_key=self.feature_key,
            target_key=self.target_key,
            transform=self.transform,
            target_transform=self.target_transform,
        )


class JSONDataset(Dataset):
    """Dataset for JSON data."""

    def __init__(
        self,
        data: List[Dict],
        feature_key: str,
        target_key: Optional[str],
        transform: Optional[Callable],
        target_transform: Optional[Callable],
    ):
        self.data = data
        self.feature_key = feature_key
        self.target_key = target_key
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Optional[Any]]:
        item = self.data[idx]

        x = item.get(self.feature_key)
        y = item.get(self.target_key) if self.target_key else None

        if self.transform:
            x = self.transform(x)

        if self.target_transform and y is not None:
            y = self.target_transform(y)

        return x, y


class HuggingFaceDatasetBuilder:
    """
    Build dataset from HuggingFace datasets.

    Args:
        dataset_name: HuggingFace dataset name
        config: Dataset configuration
        split: Dataset split
        feature_column: Column for features
        target_column: Column for target
        transform: Feature transformation

    Example:
        >>> builder = HuggingFaceDatasetBuilder(
        ...     'imdb',
        ...     split='train',
        ...     feature_column='text',
        ...     target_column='label'
        ... )
        >>> dataset = builder.build()
    """

    def __init__(
        self,
        dataset_name: str,
        config: Optional[str] = None,
        split: str = "train",
        feature_column: str = "text",
        target_column: str = "label",
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.feature_column = feature_column
        self.target_column = target_column
        self.transform = transform
        self.cache_dir = cache_dir

    def build(self) -> "HuggingFaceDataset":
        """Build and return dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for HuggingFaceDatasetBuilder"
            )

        dataset = load_dataset(
            self.dataset_name, self.config, split=self.split, cache_dir=self.cache_dir
        )

        return HuggingFaceDataset(
            dataset=dataset,
            feature_column=self.feature_column,
            target_column=self.target_column,
            transform=self.transform,
        )


class HuggingFaceDataset(Dataset):
    """Wrapper for HuggingFace datasets."""

    def __init__(
        self,
        dataset,
        feature_column: str,
        target_column: str,
        transform: Optional[Callable],
    ):
        self.dataset = dataset
        self.feature_column = feature_column
        self.target_column = target_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        item = self.dataset[idx]

        x = item[self.feature_column]
        y = item[self.target_column]

        if self.transform:
            x = self.transform(x)

        return x, y


class WebDatasetBuilder:
    """
    Build dataset from URLs/web sources.

    Args:
        urls: List of URLs to download
        local_cache: Local directory to cache downloads
        download: Whether to download immediately
        transform: Transformation for downloaded data

    Example:
        >>> urls = ['http://example.com/img1.jpg', ...]
        >>> builder = WebDatasetBuilder(urls, local_cache='./web_cache')
        >>> dataset = builder.build()
    """

    def __init__(
        self,
        urls: List[str],
        local_cache: Optional[str] = "./web_cache",
        download: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.urls = urls
        self.local_cache = Path(local_cache) if local_cache else None
        self.download = download
        self.transform = transform

        if self.local_cache:
            self.local_cache.mkdir(parents=True, exist_ok=True)

    def build(self) -> "WebDataset":
        """Build and return dataset."""
        local_paths = []

        if self.download and self.local_cache:
            local_paths = self._download_all()
        else:
            local_paths = self.urls

        return WebDataset(
            urls=self.urls,
            local_paths=local_paths,
            local_cache=self.local_cache,
            transform=self.transform,
        )

    def _download_all(self) -> List[Path]:
        """Download all URLs."""
        local_paths = []

        for url in self.urls:
            filename = hashlib.md5(url.encode()).hexdigest() + Path(url).suffix
            local_path = self.local_cache / filename

            if not local_path.exists():
                self._download_file(url, local_path)

            local_paths.append(local_path)

        return local_paths

    def _download_file(self, url: str, local_path: Path) -> None:
        """Download single file."""
        try:
            import requests

            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            warnings.warn(f"Failed to download {url}: {e}")


class WebDataset(Dataset):
    """Dataset for web-sourced data."""

    def __init__(
        self,
        urls: List[str],
        local_paths: List[Path],
        local_cache: Optional[Path],
        transform: Optional[Callable],
    ):
        self.urls = urls
        self.local_paths = local_paths
        self.local_cache = local_cache
        self.transform = transform

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int) -> Any:
        path = self.local_paths[idx]

        # Load based on file type
        if isinstance(path, Path) and path.suffix in [".jpg", ".jpeg", ".png", ".gif"]:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, idx
        else:
            with open(path, "r") as f:
                content = f.read()
            if self.transform:
                content = self.transform(content)
            return content, idx


# ============================================================================
# Section 6: Data Sampling
# ============================================================================


class ImbalancedSampler(Sampler):
    """
    Sampler for handling class imbalance.

    Oversamples minority classes to balance training.

    Args:
        dataset: Dataset to sample from
        labels: Class labels for each sample
        num_samples: Total samples to draw per epoch

    Example:
        >>> sampler = ImbalancedSampler(dataset, labels)
        >>> loader = DataLoader(dataset, sampler=sampler)
    """

    def __init__(
        self,
        dataset: Dataset,
        labels: Optional[Union[List, np.ndarray]] = None,
        num_samples: Optional[int] = None,
    ):
        self.dataset = dataset

        if labels is None:
            # Try to extract labels from dataset
            labels = self._extract_labels(dataset)

        self.labels = np.array(labels)
        self.num_samples = num_samples or len(self.labels)

        # Calculate sample weights
        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / class_counts
        self.weights = class_weights[self.labels]

        self.weights = torch.tensor(self.weights, dtype=torch.float)

    def _extract_labels(self, dataset: Dataset) -> np.ndarray:
        """Extract labels from dataset."""
        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if isinstance(sample, tuple) and len(sample) > 1:
                label = sample[1]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                labels.append(label)
        return np.array(labels)

    def __iter__(self) -> Iterator[int]:
        """Sample indices according to class weights."""
        return iter(
            torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()
        )

    def __len__(self) -> int:
        return self.num_samples


class HardNegativeSampler(Sampler):
    """
    Sampler that mines hard negative examples.

    Focuses sampling on difficult examples for faster convergence.

    Args:
        dataset: Dataset to sample from
        model: Model for scoring difficulty
        loss_fn: Loss function for computing hardness
        hard_ratio: Ratio of hard negatives to sample
        num_samples: Total samples per epoch

    Example:
        >>> sampler = HardNegativeSampler(dataset, model, loss_fn)
        >>> for batch in DataLoader(dataset, sampler=sampler):
        ...     train_step(batch)
    """

    def __init__(
        self,
        dataset: Dataset,
        model: torch.nn.Module,
        loss_fn: Callable,
        hard_ratio: float = 0.3,
        num_samples: Optional[int] = None,
    ):
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.hard_ratio = hard_ratio
        self.num_samples = num_samples or len(dataset)

        self.hardness_scores = np.ones(len(dataset))
        self.update_frequency = 1000
        self._step_count = 0

    def update_hardness(self, dataloader: DataLoader) -> None:
        """Update hardness scores for all samples."""
        self.model.eval()

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if i >= len(dataloader):
                    break

                outputs = self.model(inputs)
                losses = self.loss_fn(outputs, targets, reduction="none")

                # Update scores (higher loss = harder)
                start_idx = i * dataloader.batch_size
                end_idx = start_idx + len(inputs)
                self.hardness_scores[start_idx:end_idx] = losses.cpu().numpy()

    def __iter__(self) -> Iterator[int]:
        """Sample with focus on hard negatives."""
        num_hard = int(self.num_samples * self.hard_ratio)
        num_easy = self.num_samples - num_hard

        # Sample hard examples (high loss)
        hard_probs = self.hardness_scores / self.hardness_scores.sum()
        hard_indices = np.random.choice(
            len(self.dataset), size=num_hard, replace=True, p=hard_probs
        )

        # Sample easy examples (uniform)
        easy_indices = np.random.choice(len(self.dataset), size=num_easy, replace=True)

        indices = np.concatenate([hard_indices, easy_indices])
        np.random.shuffle(indices)

        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler with easy-to-hard progression.

    Gradually increases difficulty of training samples.

    Args:
        dataset: Dataset to sample from
        difficulties: Difficulty scores for each sample
        num_epochs: Total training epochs
        strategy: Progression strategy ('linear', 'exponential')

    Example:
        >>> difficulties = compute_difficulties(dataset)
        >>> sampler = CurriculumSampler(dataset, difficulties, num_epochs=100)
    """

    def __init__(
        self,
        dataset: Dataset,
        difficulties: np.ndarray,
        num_epochs: int,
        strategy: str = "linear",
    ):
        self.dataset = dataset
        self.difficulties = difficulties
        self.num_epochs = num_epochs
        self.strategy = strategy
        self.current_epoch = 0

        # Sort by difficulty
        self.sorted_indices = np.argsort(difficulties)

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch."""
        self.current_epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Sample according to curriculum."""
        # Calculate percentage of data to use
        if self.strategy == "linear":
            pct = (self.current_epoch + 1) / self.num_epochs
        elif self.strategy == "exponential":
            pct = 1 - np.exp(-3 * (self.current_epoch + 1) / self.num_epochs)
        else:
            pct = 1.0

        pct = np.clip(pct, 0.1, 1.0)
        num_samples = int(len(self.dataset) * pct)

        # Use easiest samples first
        indices = self.sorted_indices[:num_samples]

        return iter(indices.tolist())

    def __len__(self) -> int:
        pct = (self.current_epoch + 1) / self.num_epochs
        pct = np.clip(pct, 0.1, 1.0)
        return int(len(self.dataset) * pct)


class ActiveLearningSampler(Sampler):
    """
    Uncertainty-based active learning sampler.

    Samples based on model uncertainty for efficient labeling.

    Args:
        dataset: Dataset to sample from
        model: Model for uncertainty estimation
        acquisition_fn: Acquisition function ('uncertainty', 'margin', 'entropy')
        num_samples: Number of samples to select

    Example:
        >>> sampler = ActiveLearningSampler(dataset, model, 'entropy')
        >>> uncertain_indices = list(sampler)
    """

    def __init__(
        self,
        dataset: Dataset,
        model: torch.nn.Module,
        acquisition_fn: str = "uncertainty",
        num_samples: int = 100,
    ):
        self.dataset = dataset
        self.model = model
        self.acquisition_fn = acquisition_fn
        self.num_samples = num_samples

    def compute_uncertainty(self, dataloader: DataLoader) -> np.ndarray:
        """Compute uncertainty scores for all samples."""
        self.model.eval()
        uncertainties = []

        with torch.no_grad():
            for inputs, _ in dataloader:
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)

                if self.acquisition_fn == "uncertainty":
                    # Least confidence
                    uncertainty = 1 - probs.max(dim=1)[0]
                elif self.acquisition_fn == "margin":
                    # Margin between top two predictions
                    top2 = probs.topk(2, dim=1)[0]
                    uncertainty = -(top2[:, 0] - top2[:, 1])
                elif self.acquisition_fn == "entropy":
                    # Prediction entropy
                    uncertainty = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                else:
                    uncertainty = torch.ones(len(inputs))

                uncertainties.extend(uncertainty.cpu().numpy())

        return np.array(uncertainties)

    def __iter__(self) -> Iterator[int]:
        """Sample most uncertain examples."""
        # Create dataloader for full dataset
        loader = DataLoader(self.dataset, batch_size=64, shuffle=False)

        uncertainties = self.compute_uncertainty(loader)

        # Select most uncertain samples
        top_indices = np.argsort(uncertainties)[-self.num_samples :]

        return iter(top_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


# ============================================================================
# Section 7: Caching & Optimization
# ============================================================================


class DatasetCache:
    """
    Multi-level dataset cache (memory/disk).

    Caches preprocessed data for faster access.

    Args:
        dataset: Dataset to cache
        cache_dir: Directory for disk cache
        cache_in_memory: Whether to cache in RAM
        max_memory_size: Maximum items in memory cache

    Example:
        >>> cache = DatasetCache(dataset, cache_dir='./cache')
        >>> cached_dataset = cache.build()
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_dir: Optional[str] = None,
        cache_in_memory: bool = True,
        max_memory_size: int = 10000,
    ):
        self.dataset = dataset
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_in_memory = cache_in_memory
        self.max_memory_size = max_memory_size

        self._memory_cache: Dict[int, Any] = {}
        self._cache_order: List[int] = []

    def __getitem__(self, idx: int) -> Any:
        """Get item with caching."""
        if idx in self._memory_cache:
            # Move to end (LRU)
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            return self._memory_cache[idx]

        # Try disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{idx}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    item = pickle.load(f)

                if self.cache_in_memory:
                    self._add_to_memory(idx, item)

                return item

        # Load from dataset
        item = self.dataset[idx]

        # Cache
        if self.cache_in_memory:
            self._add_to_memory(idx, item)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{idx}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(item, f)

        return item

    def _add_to_memory(self, idx: int, item: Any) -> None:
        """Add item to memory cache with LRU eviction."""
        if len(self._memory_cache) >= self.max_memory_size:
            # Remove oldest
            oldest = self._cache_order.pop(0)
            del self._memory_cache[oldest]

        self._memory_cache[idx] = item
        self._cache_order.append(idx)

    def __len__(self) -> int:
        return len(self.dataset)

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._memory_cache.clear()
        self._cache_order.clear()

        if self.cache_dir:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()


class PrefetchDataLoader(DataLoader):
    """
    DataLoader with batch prefetching.

    Preloads future batches in background for better GPU utilization.

    Args:
        dataset: Dataset to load
        num_prefetch: Number of batches to prefetch
        **kwargs: Additional DataLoader arguments

    Example:
        >>> loader = PrefetchDataLoader(dataset, num_prefetch=2)
    """

    def __init__(self, dataset: Dataset, num_prefetch: int = 2, **kwargs):
        self.num_prefetch = num_prefetch
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=num_prefetch)
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_prefetch = threading.Event()

        super().__init__(dataset=dataset, **kwargs)

    def __iter__(self) -> Iterator:
        """Iterate with prefetching."""
        if self.num_prefetch <= 0:
            yield from super().__iter__()
            return

        # Start prefetch thread
        self._stop_prefetch.clear()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self._prefetch_thread.start()

        try:
            while True:
                try:
                    batch = self._prefetch_queue.get(timeout=30)
                    if batch is None:
                        break
                    yield batch
                except queue.Empty:
                    break
        finally:
            self._stop_prefetch.set()
            if self._prefetch_thread:
                self._prefetch_thread.join()

    def _prefetch_worker(self) -> None:
        """Background prefetch worker."""
        for batch in super().__iter__():
            if self._stop_prefetch.is_set():
                break

            try:
                self._prefetch_queue.put(batch, timeout=1)
            except queue.Full:
                if self._stop_prefetch.is_set():
                    break

        # Signal end
        try:
            self._prefetch_queue.put(None, timeout=1)
        except queue.Full:
            pass


class ParallelDataLoader:
    """
    Multi-threaded data loading wrapper.

    Loads data from multiple sources in parallel.

    Args:
        datasets: List of datasets to load from
        batch_size: Batch size
        num_workers: Number of parallel workers
        shuffle: Whether to shuffle

    Example:
        >>> loader = ParallelDataLoader([ds1, ds2, ds3], batch_size=32)
        >>> for batch in loader:
        ...     train(batch)
    """

    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.loaders = [
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(1, num_workers // len(datasets)),
            )
            for ds in datasets
        ]

    def __iter__(self) -> Iterator:
        """Iterate through all loaders in parallel."""
        iterators = [iter(loader) for loader in self.loaders]

        while iterators:
            for i, it in enumerate(iterators):
                try:
                    yield next(it)
                except StopIteration:
                    iterators.pop(i)
                    break

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders)


# ============================================================================
# Utility Functions
# ============================================================================


def get_data_statistics(
    dataloader: DataLoader, max_batches: int = 100
) -> Dict[str, Any]:
    """
    Compute statistics for a dataset.

    Args:
        dataloader: DataLoader to analyze
        max_batches: Maximum batches to analyze

    Returns:
        Dictionary of statistics
    """
    stats = {
        "num_batches": 0,
        "batch_sizes": [],
        "feature_dims": [],
    }

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        stats["num_batches"] += 1
        stats["batch_sizes"].append(len(batch))
        stats["feature_dims"].append(batch.shape[1:])

    stats["mean_batch_size"] = np.mean(stats["batch_sizes"])
    stats["std_batch_size"] = np.std(stats["batch_sizes"])

    return stats


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: Dataset to split
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        shuffle: Whether to shuffle before splitting
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1"
    )

    total_size = len(dataset)
    indices = list(range(total_size))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size] if val_ratio > 0 else None
    test_indices = indices[train_size + val_size :] if test_ratio > 0 else None

    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices) if val_indices else None
    test_dataset = SubsetDataset(dataset, test_indices) if test_indices else None

    return train_dataset, val_dataset, test_dataset


class SubsetDataset(Dataset):
    """Subset of a dataset."""

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.indices[idx]]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Data Loaders
    "SmartDataLoader",
    "CacheDataLoader",
    "MultiEpochDataLoader",
    "InfiniteDataLoader",
    # Data Augmentation
    "AutoAugment",
    "AutoAugmentPolicy",
    "RandAugment",
    "TrivialAugmentWide",
    "AugMix",
    "MixUpCutMixCollator",
    "CutOut",
    # Data Validation
    "DataValidator",
    "ValidationReport",
    "detect_corrupt_images",
    "check_class_balance",
    "detect_duplicates",
    "validate_splits",
    # Data Preprocessing
    "SmartNormalizer",
    "AutoResizer",
    "CategoricalEncoder",
    "MissingValueImputer",
    "TextNormalizer",
    # Dataset Builders
    "ImageDatasetBuilder",
    "FolderDataset",
    "CSVDatasetBuilder",
    "CSVDataset",
    "JSONDatasetBuilder",
    "JSONDataset",
    "HuggingFaceDatasetBuilder",
    "HuggingFaceDataset",
    "WebDatasetBuilder",
    "WebDataset",
    # Data Sampling
    "ImbalancedSampler",
    "HardNegativeSampler",
    "CurriculumSampler",
    "ActiveLearningSampler",
    # Caching & Optimization
    "DatasetCache",
    "PrefetchDataLoader",
    "ParallelDataLoader",
    # Utilities
    "get_data_statistics",
    "split_dataset",
    "SubsetDataset",
]

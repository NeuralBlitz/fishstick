"""
Base Augmentation Classes

Provides foundational abstractions for the augmentation framework
following the fishstick design patterns.
"""

from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
import numpy as np
from torch import Tensor
import random


class AugmentationBase(ABC):
    """Abstract base class for all augmentation operations."""

    def __init__(self, probability: float = 0.5, seed: Optional[int] = None):
        """
        Initialize the augmentation.

        Args:
            probability: Probability of applying the augmentation
            seed: Random seed for reproducibility
        """
        self.probability = probability
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()

    @abstractmethod
    def __call__(self, data: Any, **kwargs: Any) -> Any:
        """Apply the augmentation to the data."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(probability={self.probability})"

    def _should_apply(self) -> bool:
        """Determine if augmentation should be applied."""
        return random.random() < self.probability


@dataclass
class AugmentationConfig:
    """Configuration for augmentation operations."""

    name: str
    probability: float = 1.0
    intensity: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "probability": self.probability,
            "intensity": self.intensity,
            "parameters": self.parameters,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }


class AugmentationScheduler:
    """Dynamic scheduler for augmentation intensity over training."""

    def __init__(
        self,
        initial_intensity: float = 0.0,
        final_intensity: float = 1.0,
        total_steps: int = 10000,
        schedule_type: str = "linear",
    ):
        """
        Initialize the scheduler.

        Args:
            initial_intensity: Starting augmentation intensity
            final_intensity: Final augmentation intensity
            total_steps: Total training steps
            schedule_type: Type of schedule (linear, cosine, exponential)
        """
        self.initial_intensity = initial_intensity
        self.final_intensity = final_intensity
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.current_step = 0

    def step(self) -> float:
        """Step the scheduler and return current intensity."""
        self.current_step += 1
        return self.get_intensity()

    def get_intensity(self) -> float:
        """Get current intensity based on schedule."""
        progress = min(self.current_step / self.total_steps, 1.0)

        if self.schedule_type == "linear":
            return self.initial_intensity + progress * (
                self.final_intensity - self.initial_intensity
            )
        elif self.schedule_type == "cosine":
            return self.initial_intensity + (1 - np.cos(progress * np.pi / 2)) * (
                self.final_intensity - self.initial_intensity
            )
        elif self.schedule_type == "exponential":
            return (
                self.initial_intensity
                * (self.final_intensity / self.initial_intensity) ** progress
            )
        else:
            return self.final_intensity

    def reset(self) -> None:
        """Reset the scheduler."""
        self.current_step = 0


class AdaptiveAugmentation:
    """Adaptive augmentation that adjusts based on training metrics."""

    def __init__(
        self,
        augmentations: List[AugmentationBase],
        metric_fn: Optional[Callable[[], float]] = None,
        target_metric: float = 0.5,
        adjustment_factor: float = 0.1,
    ):
        """
        Initialize adaptive augmentation.

        Args:
            augmentations: List of augmentation operations
            metric_fn: Function to compute current metric
            target_metric: Target metric value
            adjustment_factor: Factor to adjust intensity
        """
        self.augmentations = augmentations
        self.metric_fn = metric_fn
        self.target_metric = target_metric
        self.adjustment_factor = adjustment_factor
        self.intensity_scales = {aug: 1.0 for aug in augmentations}
        self.best_metric = float("-inf")

    def update(self, current_metric: Optional[float] = None) -> None:
        """Update augmentation intensities based on metric."""
        if current_metric is None and self.metric_fn is not None:
            current_metric = self.metric_fn()

        if current_metric is not None:
            if current_metric > self.best_metric:
                self.best_metric = current_metric

            error = self.target_metric - current_metric
            for aug in self.augmentations:
                self.intensity_scales[aug] = max(
                    0.1,
                    min(
                        2.0, self.intensity_scales[aug] + error * self.adjustment_factor
                    ),
                )

    def apply(
        self,
        data: Any,
        labels: Optional[Any] = None,
    ) -> Tuple[Any, Optional[Any]]:
        """Apply augmentations with adaptive intensity."""
        result = data
        mixed_labels = labels

        for aug in self.augmentations:
            if hasattr(aug, "intensity"):
                original_intensity = aug.intensity
                aug.intensity *= self.intensity_scales[aug]

            result = aug(result)

            if hasattr(aug, "intensity"):
                aug.intensity = original_intensity

        return result, mixed_labels


class AugmentationPipeline:
    """Composable pipeline for chaining augmentations."""

    def __init__(
        self,
        augmentations: List[Union[AugmentationBase, Callable]],
        probabilities: Optional[List[float]] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            augmentations: List of augmentations to apply
            probabilities: Probability for each augmentation
        """
        self.augmentations = augmentations
        self.probabilities = probabilities or [1.0] * len(augmentations)

    def __call__(
        self,
        data: Any,
        return_applied: bool = False,
    ) -> Any:
        """Apply the pipeline to data."""
        applied = []
        result = data

        for aug, prob in zip(self.augmentations, self.probabilities):
            if random.random() < prob:
                result = aug(result)
                applied.append(
                    aug.__class__.__name__ if hasattr(aug, "__class__") else str(aug)
                )

        if return_applied:
            return result, applied
        return result

    def add(self, augmentation: AugmentationBase, probability: float = 1.0) -> None:
        """Add an augmentation to the pipeline."""
        self.augmentations.append(augmentation)
        self.probabilities.append(probability)


class MixupCutmixCollator:
    """Collator for MixUp and CutMix batch operations."""

    def __init__(
        self,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize the collator.

        Args:
            mixup_alpha: Alpha parameter for MixUp
            cutmix_alpha: Alpha parameter for CutMix
            switch_prob: Probability of switching between MixUp and CutMix
            label_smoothing: Label smoothing factor
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.rng = np.random.RandomState()

    def __call__(
        self, batch: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
        """
        Collate batch with MixUp or CutMix.

        Args:
            batch: Tuple of (images, labels)

        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lam)
        """
        images, labels = batch

        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            use_cutmix = self.rng.random() < self.switch_prob
            alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        elif self.mixup_alpha > 0:
            alpha = self.mixup_alpha
        elif self.cutmix_alpha > 0:
            alpha = self.cutmix_alpha
        else:
            return images, labels, labels, 1.0

        lam = self._sample_lambda(alpha)

        if lam == 1.0:
            return images, labels, labels, 1.0

        batch_size = images.size(0)
        index = self.rng.permutation(batch_size)

        if self.cutmix_alpha > 0 and (
            self.mixup_alpha == 0 or self.rng.random() < self.switch_prob
        ):
            images = self._cutmix(images, index, lam)
        else:
            images = self._mixup(images, index, lam)

        labels_a, labels_b = labels, labels[index]

        if self.label_smoothing > 0:
            labels_a = self._smooth_labels(labels_a)
            labels_b = self._smooth_labels(labels_b)

        return images, labels_a, labels_b, lam

    def _mixup(self, x: Tensor, index: Tensor, lam: float) -> Tensor:
        """Apply MixUp."""
        return lam * x + (1 - lam) * x[index]

    def _cutmix(self, x: Tensor, index: Tensor, lam: float) -> Tensor:
        """Apply CutMix."""
        lam = 1 - lam
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        return x

    def _rand_bbox(
        self, size: Tuple[int, ...], lam: float
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box for CutMix."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = self.rng.randint(W)
        cy = self.rng.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def _sample_lambda(self, alpha: float) -> float:
        """Sample lambda from Beta distribution."""
        if alpha > 0:
            lam = self.rng.beta(alpha, alpha)
        else:
            lam = 1.0
        return lam

    def _smooth_labels(self, labels: Tensor) -> Tensor:
        """Apply label smoothing."""
        return labels * (1 - self.label_smoothing) + self.label_smoothing / labels.size(
            -1
        )


def get_augmentation_pipeline(
    task_type: str = "image_classification",
    intensity: float = 1.0,
    **kwargs: Dict[str, Any],
) -> AugmentationPipeline:
    """
    Get a pre-configured augmentation pipeline.

    Args:
        task_type: Type of task (image_classification, object_detection, etc.)
        intensity: Overall intensity of augmentations
        **kwargs: Additional configuration

    Returns:
        Configured AugmentationPipeline
    """
    from fishstick.augmentation_ext.image_augmentation import (
        RandomHorizontalFlip,
        RandomRotation,
        ColorJitter,
        RandomAffine,
    )

    augmentations = [
        RandomHorizontalFlip(p=0.5 * intensity),
        RandomRotation(degrees=15 * intensity),
        ColorJitter(
            brightness=0.2 * intensity,
            contrast=0.2 * intensity,
            saturation=0.2 * intensity,
            hue=0.1 * intensity,
        ),
        RandomAffine(
            degrees=0,
            translate=(0.1 * intensity, 0.1 * intensity),
            scale=(0.9, 1.1),
        ),
    ]

    return AugmentationPipeline(augmentations)

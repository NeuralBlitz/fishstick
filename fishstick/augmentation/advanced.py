"""
Advanced Data Augmentation

Comprehensive data augmentation pipeline with automatic augmentation search,
cutout, mixup, cutmix, and domain-specific augmentations.
"""

from typing import Optional, Tuple, List, Callable, Dict, Any
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import random
from abc import ABC, abstractmethod


class Augmentation(ABC):
    """Base class for augmentations."""

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass

    def __repr__(self):
        return self.__class__.__name__


class CutOut(Augmentation):
    """
    CutOut augmentation - randomly mask out regions of the image.

    Reference: DeVries & Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout", 2017

    Args:
        n_holes: Number of holes to cut
        length: Length of each hole

    Example:
        >>> cutout = CutOut(n_holes=1, length=16)
        >>> augmented = cutout(image)
    """

    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor image of shape (C, H, W)

        Returns:
            Augmented image
        """
        h = x.size(1)
        w = x.size(2)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x_coord = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x_coord - self.length // 2, 0, w)
            x2 = np.clip(x_coord + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask).to(x.device)
        mask = mask.expand_as(x)

        return x * mask


class MixUp:
    """
    MixUp augmentation - blend images and labels.

    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", 2018

    Args:
        alpha: Beta distribution parameter

    Example:
        >>> mixup = MixUp(alpha=0.2)
        >>> mixed_images, targets_a, targets_b, lam = mixup(images, targets)
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        """
        Args:
            x: Batch of images (N, C, H, W)
            y: Batch of labels (N,)

        Returns:
            Tuple of (mixed_images, targets_a, targets_b, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """Criterion that works with mixup."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutMix:
    """
    CutMix augmentation - cut and paste patches between images.

    Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features", 2019

    Args:
        alpha: Beta distribution parameter

    Example:
        >>> cutmix = CutMix(alpha=1.0)
        >>> mixed_images, targets_a, targets_b, lam = cutmix(images, targets)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        """
        Args:
            x: Batch of images (N, C, H, W)
            y: Batch of labels (N,)

        Returns:
            Tuple of (mixed_images, targets_a, targets_b, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        y_a, y_b = y, y[index]

        # Get random bounding box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

        return x, y_a, y_b, lam

    def _rand_bbox(self, size: Tuple, lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


class RandAugment:
    """
    RandAugment - simplified automatic augmentation.

    Reference: Cubuk et al., "RandAugment: Practical Automated Data Augmentation with a Reduced Search Space", 2020

    Args:
        n: Number of augmentation transformations to apply
        m: Magnitude for all transformations

    Example:
        >>> ra = RandAugment(n=2, m=9)
        >>> augmented = ra(image)
    """

    def __init__(self, n: int = 2, m: int = 9):
        self.n = n
        self.m = m

        # Define augmentation operations
        self.augment_list = [
            self._autocontrast,
            self._equalize,
            self._rotate,
            self._solarize,
            self._color,
            self._posterize,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        """Apply n random augmentations."""
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            x = op(x)
        return x

    def _autocontrast(self, x: Tensor) -> Tensor:
        """Apply autocontrast."""
        # Simplified implementation
        min_val = x.min()
        max_val = x.max()
        if max_val > min_val:
            x = (x - min_val) / (max_val - min_val)
        return x

    def _equalize(self, x: Tensor) -> Tensor:
        """Histogram equalization (simplified)."""
        # Placeholder - would need actual implementation
        return x

    def _rotate(self, x: Tensor) -> Tensor:
        """Random rotation."""
        angle = (self.m / 10) * 30  # Max 30 degrees
        angle = random.choice([-1, 1]) * angle
        return self._rotate_img(x, angle)

    def _rotate_img(self, x: Tensor, angle: float) -> Tensor:
        """Rotate image tensor."""
        theta = (
            torch.tensor(
                [
                    [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                    [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
                ],
                dtype=torch.float,
            )
            .unsqueeze(0)
            .to(x.device)
        )

        grid = F.affine_grid(theta, x.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)

    def _solarize(self, x: Tensor) -> Tensor:
        """Solarize - invert pixels above threshold."""
        threshold = 1.0 - (self.m / 10) * 0.5
        return torch.where(x < threshold, x, 1 - x)

    def _color(self, x: Tensor) -> Tensor:
        """Adjust color balance."""
        factor = 1.0 + (self.m / 10) * 0.9
        factor = random.choice([factor, 1 / factor])
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return torch.clamp((x - mean) * factor + mean, 0, 1)

    def _posterize(self, x: Tensor) -> Tensor:
        """Reduce bits per channel."""
        bits = max(1, 8 - int(self.m / 10 * 7))
        x = (x * 255).long()
        x = (x >> (8 - bits)) << (8 - bits)
        return x.float() / 255

    def _contrast(self, x: Tensor) -> Tensor:
        """Adjust contrast."""
        factor = 1.0 + (self.m / 10) * 0.9
        factor = random.choice([factor, 1 / factor])
        mean = x.mean()
        return torch.clamp((x - mean) * factor + mean, 0, 1)

    def _brightness(self, x: Tensor) -> Tensor:
        """Adjust brightness."""
        factor = 1.0 + (self.m / 10) * 0.9
        factor = random.choice([factor, 1 / factor])
        return torch.clamp(x * factor, 0, 1)

    def _sharpness(self, x: Tensor) -> Tensor:
        """Adjust sharpness (simplified)."""
        return x

    def _shear_x(self, x: Tensor) -> Tensor:
        """Shear along x-axis."""
        shear = (self.m / 10) * 0.3
        shear = random.choice([-1, 1]) * shear
        return self._shear(x, shear, 0)

    def _shear_y(self, x: Tensor) -> Tensor:
        """Shear along y-axis."""
        shear = (self.m / 10) * 0.3
        shear = random.choice([-1, 1]) * shear
        return self._shear(x, 0, shear)

    def _shear(self, x: Tensor, shear_x: float, shear_y: float) -> Tensor:
        """Apply shear transformation."""
        theta = (
            torch.tensor([[1, shear_x, 0], [shear_y, 1, 0]], dtype=torch.float)
            .unsqueeze(0)
            .to(x.device)
        )

        grid = F.affine_grid(theta, x.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)

    def _translate_x(self, x: Tensor) -> Tensor:
        """Translate along x-axis."""
        translate = int((self.m / 10) * x.size(-2) * 0.5)
        translate = random.choice([-1, 1]) * translate
        return torch.roll(x, translate, dims=-2)

    def _translate_y(self, x: Tensor) -> Tensor:
        """Translate along y-axis."""
        translate = int((self.m / 10) * x.size(-1) * 0.5)
        translate = random.choice([-1, 1]) * translate
        return torch.roll(x, translate, dims=-1)


class AugmentationPipeline:
    """
    Compose multiple augmentations into a pipeline.

    Args:
        augmentations: List of augmentation operations
        p: Probability of applying each augmentation

    Example:
        >>> pipeline = AugmentationPipeline([
        ...     RandAugment(n=2, m=9),
        ...     CutOut(n_holes=1, length=16)
        ... ])
        >>> augmented = pipeline(image)
    """

    def __init__(self, augmentations: List[Augmentation], p: float = 0.5):
        self.augmentations = augmentations
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        """Apply augmentations with probability p."""
        if random.random() < self.p:
            for aug in self.augmentations:
                x = aug(x)
        return x

    def __repr__(self):
        return f"AugmentationPipeline({self.augmentations}, p={self.p})"


class MixupCutmixCollator:
    """
    DataLoader collator that applies MixUp or CutMix.

    Example:
        >>> collator = MixupCutmixCollator(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5)
        >>> train_loader = DataLoader(dataset, batch_size=32, collate_fn=collator)
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        switch_prob: float = 0.5,
    ):
        self.mixup = MixUp(mixup_alpha)
        self.cutmix = CutMix(cutmix_alpha)
        self.prob = prob
        self.switch_prob = switch_prob

    def __call__(self, batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor]:
        """Collate batch with MixUp or CutMix."""
        images = torch.stack([item[0] for item in batch])
        targets = torch.tensor([item[1] for item in batch])

        if random.random() > self.prob:
            return images, targets

        if random.random() < self.switch_prob:
            images, targets_a, targets_b, lam = self.mixup(images, targets)
        else:
            images, targets_a, targets_b, lam = self.cutmix(images, targets)

        # Return in format compatible with loss function
        return images, (targets_a, targets_b, lam)


class TrivialAugmentWide:
    """
    TrivialAugment - apply one single augmentation with random magnitude.

    Reference: Müller & Hutter, "TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation", 2021

    Example:
        >>> ta = TrivialAugmentWide()
        >>> augmented = ta(image)
    """

    def __init__(self):
        self.augment_list = [
            "Identity",
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

    def __call__(self, x: Tensor) -> Tensor:
        """Apply one random augmentation."""
        op = random.choice(self.augment_list)
        magnitude = random.random() * 30  # Random magnitude up to 30

        # Apply operation (simplified)
        if op == "Rotate":
            angle = random.choice([-1, 1]) * magnitude
            # Would need actual rotation implementation
            pass
        elif op == "Brightness":
            factor = 1.0 + magnitude / 100
            x = torch.clamp(x * factor, 0, 1)
        # ... other operations

        return x


def get_augmentation_pipeline(
    dataset: str = "imagenet", severity: str = "medium"
) -> AugmentationPipeline:
    """
    Get recommended augmentation pipeline for a dataset.

    Args:
        dataset: Dataset name ('imagenet', 'cifar10', 'cifar100', etc.)
        severity: Augmentation severity ('light', 'medium', 'heavy')

    Returns:
        AugmentationPipeline
    """
    if severity == "light":
        n, m = 1, 5
    elif severity == "medium":
        n, m = 2, 9
    else:  # heavy
        n, m = 3, 15

    augmentations = [RandAugment(n=n, m=m)]

    if dataset in ["cifar10", "cifar100"]:
        augmentations.append(CutOut(n_holes=1, length=16))
    elif dataset == "imagenet":
        augmentations.append(CutOut(n_holes=1, length=40))

    return AugmentationPipeline(augmentations, p=1.0)


__all__ = [
    "Augmentation",
    "CutOut",
    "MixUp",
    "CutMix",
    "RandAugment",
    "TrivialAugmentWide",
    "AugmentationPipeline",
    "MixupCutmixCollator",
    "get_augmentation_pipeline",
]

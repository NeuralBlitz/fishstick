"""
Image Augmentations

Comprehensive image augmentation transforms including geometric,
color, and advanced augmentation techniques.
"""

from typing import Optional, Tuple, List, Union
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import random
from dataclasses import dataclass


class RandomCrop:
    """Random crop with optional resize back to original size."""

    def __init__(
        self, size: Union[int, Tuple[int, int]], padding: Optional[int] = None
    ):
        self.size = (size, size) if isinstance(size, int) else size
        self.padding = padding

    def __call__(self, x: Tensor) -> Tensor:
        if self.padding is not None:
            x = F.pad(x, [self.padding] * 4, mode="constant", value=0)

        h, w = x.size(-2), x.size(-1)
        th, tw = self.size

        if h < th or w < tw:
            return F.interpolate(
                x.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False
            ).squeeze(0)

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return x[..., i : i + th, j : j + tw]


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() < self.p:
            return x.flip(-1)
        return x


class RandomVerticalFlip:
    """Random vertical flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() < self.p:
            return x.flip(-2)
        return x


class ColorJitter:
    """Randomly change brightness, contrast, saturation and hue."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        p: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = x * factor + mean * (1 - factor)

        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = x.mean()
            x = (x - mean) * factor + mean

        if self.saturation > 0 and x.size(0) > 1:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            gray = x[:1] * 0.299 + x[1:2] * 0.587 + x[2:3] * 0.114
            x = x * factor + gray * (1 - factor)

        if self.hue > 0 and x.size(0) > 1:
            shift = random.uniform(-self.hue, self.hue)
            x = torch.roll(x, shifts=int(shift * x.size(0)), dims=0)

        return torch.clamp(x, 0, 1)


class RandomErasing:
    """Random erasing augmentation."""

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        c, h, w = x.shape
        area = h * w

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            cut_h = int(round(np.sqrt(target_area * aspect_ratio)))
            cut_w = int(round(np.sqrt(target_area / aspect_ratio)))

            if cut_h < h and cut_w < w:
                i = random.randint(0, h - cut_h)
                j = random.randint(0, w - cut_w)

                x[:, i : i + cut_h, j : j + cut_w] = self.value
                return x

        return x


class Cutout(RandomErasing):
    """Cutout augmentation - alias for RandomErasing."""

    def __init__(self, n_holes: int = 1, length: int = 16):
        super().__init__(p=1.0, value=0.0)
        self.n_holes = n_holes
        self.length = length

    def __call__(self, x: Tensor) -> Tensor:
        c, h, w = x.shape

        for _ in range(self.n_holes):
            y = random.randint(0, h)
            x_coord = random.randint(0, w)

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x_coord - self.length // 2)
            x2 = min(w, x_coord + self.length // 2)

            x[:, y1:y2, x1:x2] = 0.0

        return x


class Mixup:
    """Mixup augmentation for batches."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self, x: Tensor, y: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]

        if y is not None:
            return mixed_x, y, y[index], lam
        return mixed_x, None, None, lam

    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Cutmix:
    """Cutmix augmentation for batches."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self, x: Tensor, y: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

        if y is not None:
            return x, y, y[index], lam
        return x, None, None, lam

    def _rand_bbox(self, size: Tuple, lam: float) -> Tuple[int, int, int, int]:
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = random.randint(0, W)
        cy = random.randint(0, H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


class AutoAugment:
    """AutoAugment policy application."""

    def __init__(self, policy: str = "imagenet"):
        self.policy = policy
        self.policies = self._get_policies(policy)

    def _get_policies(self, policy: str) -> List[List[Tuple]]:
        policies = {
            "imagenet": [
                [("Posterize", 0.4, 8), ("Rotate", 0.6, 9)],
                [("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)],
                [("Equalize", 0.8, None), ("Equalize", 0.6, None)],
                [("Posterize", 0.6, 7), ("Posterize", 0.6, 6)],
                [("Equalize", 0.4, None), ("Solarize", 0.6, 4)],
            ],
            "cifar10": [
                [("Invert", 0.2, None), ("Contrast", 0.2, 0.6)],
                [("Rotate", 0.6, 2), ("TranslateX", 0.6, 2)],
                [("Sharpness", 0.6, 1), ("Sharpness", 0.6, 3)],
                [("ShearY", 0.6, 0.3), ("TranslateY", 0.6, 3)],
                [("Autocontrast", 0.8, None), ("Equalize", 0.8, None)],
            ],
        }
        return policies.get(policy, policies["imagenet"])

    def __call__(self, x: Tensor) -> Tensor:
        policy = random.choice(self.policies)

        for op_name, prob, mag in policy:
            if random.random() < prob:
                x = self._apply_op(x, op_name, mag)

        return x

    def _apply_op(self, x: Tensor, op_name: str, mag: Optional[int]) -> Tensor:
        if op_name == "Rotate":
            angle = (mag / 10) * 30 * random.choice([-1, 1])
            return self._rotate(x, angle)
        elif op_name == "Posterize":
            bits = max(1, 8 - mag)
            x = (x * 255).long()
            x = (x >> (8 - bits)) << (8 - bits)
            return x.float() / 255
        elif op_name == "Solarize":
            threshold = 1.0 - (mag / 10) * 0.5
            return torch.where(x < threshold, x, 1 - x)
        elif op_name == "AutoContrast":
            min_val, max_val = x.min(), x.max()
            if max_val > min_val:
                x = (x - min_val) / (max_val - min_val)
            return x
        elif op_name == "Equalize":
            return x
        elif op_name == "TranslateX":
            shift = int((mag / 10) * x.size(-1) * 0.3) * random.choice([-1, 1])
            return torch.roll(x, shift, dims=-1)
        elif op_name == "TranslateY":
            shift = int((mag / 10) * x.size(-2) * 0.3) * random.choice([-1, 1])
            return torch.roll(x, shift, dims=-2)
        elif op_name == "Contrast":
            factor = 1.0 + (mag / 10) * 0.9
            mean = x.mean()
            return torch.clamp((x - mean) * factor + mean, 0, 1)
        elif op_name == "Sharpness":
            return x
        elif op_name == "ShearY":
            shear = (mag / 10) * 0.3 * random.choice([-1, 1])
            return self._shear(x, shear, 0)

        return x

    def _rotate(self, x: Tensor, angle: float) -> Tensor:
        theta = (
            torch.tensor(
                [
                    [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                    [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
                ],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(x.device)
        )

        grid = F.affine_grid(theta, x.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)

    def _shear(self, x: Tensor, shear_x: float, shear_y: float) -> Tensor:
        theta = (
            torch.tensor([[1, shear_x, 0], [shear_y, 1, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .to(x.device)
        )

        grid = F.affine_grid(theta, x.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)


class GaussianBlur:
    """Gaussian blur augmentation."""

    def __init__(
        self,
        kernel_size: int = 3,
        sigma: Tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        sigma = random.uniform(*self.sigma)
        kernel = self._get_gaussian_kernel(self.kernel_size, sigma)

        if x.dim() == 3:
            kernel = kernel.unsqueeze(0)

        return F.conv2d(
            x.unsqueeze(0), kernel, padding=self.kernel_size // 2, groups=x.size(0)
        ).squeeze(0)

    def _get_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:
        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")

        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()

        return kernel


class RandomRotation:
    """Random rotation augmentation."""

    def __init__(self, degrees: Union[float, Tuple[float, float]] = 15, p: float = 0.5):
        self.degrees = degrees if isinstance(degrees, tuple) else (-degrees, degrees)
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        angle = random.uniform(*self.degrees)

        theta = (
            torch.tensor(
                [
                    [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                    [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
                ],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(x.device)
        )

        grid = F.affine_grid(theta, x.unsqueeze(0).size(), align_corners=False)
        return F.grid_sample(
            x.unsqueeze(0),
            grid,
            align_corners=False,
            mode="bilinear",
            padding_mode="zeros",
        ).squeeze(0)


@dataclass
class ImageAugmentConfig:
    """Configuration for image augmentations."""

    crop_size: Tuple[int, int] = (224, 224)
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    rotation_degrees: float = 15
    gaussian_blur: float = 0.0
    random_erasing: float = 0.0


def get_image_augmentations(config: Optional[ImageAugmentConfig] = None) -> List:
    """Get list of image augmentations from config."""
    if config is None:
        config = ImageAugmentConfig()

    augmentations = [
        RandomCrop(config.crop_size),
        RandomHorizontalFlip(p=config.horizontal_flip),
        RandomVerticalFlip(p=config.vertical_flip),
        ColorJitter(
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue=config.hue,
        ),
        RandomRotation(degrees=config.rotation_degrees),
    ]

    if config.gaussian_blur > 0:
        augmentations.append(GaussianBlur(p=config.gaussian_blur))

    if config.random_erasing > 0:
        augmentations.append(RandomErasing(p=config.random_erasing))

    return augmentations


__all__ = [
    "RandomCrop",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "ColorJitter",
    "RandomErasing",
    "Cutout",
    "Mixup",
    "Cutmix",
    "AutoAugment",
    "GaussianBlur",
    "RandomRotation",
    "ImageAugmentConfig",
    "get_image_augmentations",
]

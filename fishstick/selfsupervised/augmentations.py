"""
Self-Supervised Learning Augmentations

Augmentation pipelines for self-supervised learning:
- BYOL augmentations
- SimCLR augmentations
- MAE augmentations
- Individual augmentation modules
"""

from typing import Tuple, List, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RandomResizedCrop(nn.Module):
    """Random resized crop with specific parameters for SSL."""

    def __init__(
        self,
        size: int = 224,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.333),
    ):
        super().__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def forward(self, x: Tensor) -> Tensor:
        i, j, h, w = T.RandomResizedCrop.get_params(x, self.scale, self.ratio)
        return TF.resized_crop(x, i, j, h, w, [self.size, self.size])


class ColorJitter(nn.Module):
    """Color jitter augmentation."""

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, x: Tensor) -> Tensor:
        transforms = []

        if self.brightness > 0:
            brightness_factor = (
                torch.empty(1).uniform_(1 - self.brightness, 1 + self.brightness).item()
            )
            transforms.append(lambda img: TF.adjust_brightness(img, brightness_factor))

        if self.contrast > 0:
            contrast_factor = (
                torch.empty(1).uniform_(1 - self.contrast, 1 + self.contrast).item()
            )
            transforms.append(lambda img: TF.adjust_contrast(img, contrast_factor))

        if self.saturation > 0:
            saturation_factor = (
                torch.empty(1).uniform_(1 - self.saturation, 1 + self.saturation).item()
            )
            transforms.append(lambda img: TF.adjust_saturation(img, saturation_factor))

        if self.hue > 0:
            hue_factor = torch.empty(1).uniform_(-self.hue, self.hue).item()
            transforms.append(lambda img: TF.adjust_hue(img, hue_factor))

        for transform in transforms:
            x = transform(x)

        return x


class GaussianBlur(nn.Module):
    """Gaussian blur augmentation."""

    def __init__(
        self,
        kernel_size: int = 23,
        sigma: Tuple[float, float] = (0.1, 2.0),
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        sigma = torch.empty(1).uniform_(*self.sigma).item()
        return TF.gaussian_blur(x, self.kernel_size, [sigma, sigma])


class Solarization(nn.Module):
    """Solarization augmentation."""

    def __init__(self, threshold: float = 128.0, p: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if torch.rand(1) < self.p:
            return torch.where(x > self.threshold, 1 - x, x)
        return x


class RandomSolarization(nn.Module):
    """Random solarization with configurable probability."""

    def __init__(
        self,
        threshold: float = 128.0,
        p: float = 0.5,
        min_p: float = 0.0,
        max_p: float = 0.5,
    ):
        super().__init__()
        self.threshold = threshold
        self.p = p
        self.min_p = min_p
        self.max_p = max_p

    def forward(self, x: Tensor) -> Tensor:
        if torch.rand(1) < self.p:
            p = torch.empty(1).uniform_(self.min_p, self.max_p).item()
            if torch.rand(1) < p:
                return torch.where(x > self.threshold, 1 - x, x)
        return x


class RandomGrayscale(nn.Module):
    """Random grayscale conversion."""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if torch.rand(1) < self.p:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device)
            gray = (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
            return gray.repeat(1, 3, 1, 1)
        return x


class SolarizeAndBlur(nn.Module):
    """Combined solarization and blur for MAE."""

    def __init__(
        self,
        blur_kernel: int = 23,
        blur_sigma: Tuple[float, float] = (0.1, 2.0),
        solar_thresh: float = 128.0,
        solar_p: float = 0.0,
    ):
        super().__init__()
        self.blur = GaussianBlur(blur_kernel, blur_sigma)
        self.solar = Solarization(solar_thresh, solar_p)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blur(x)
        x = self.solar(x)
        return x


class BYOLAugmentations(nn.Module):
    """BYOL-style augmentations.

    Strong augmentations suitable for BYOL training.
    """

    def __init__(
        self,
        size: int = 224,
        crop_scale: Tuple[float, float] = (0.08, 1.0),
        color_jitter: float = 0.4,
        grayscale: float = 0.2,
        gaussian_blur: float = 0.5,
    ):
        super().__init__()
        self.size = size

        self.transforms = nn.Sequential(
            RandomResizedCrop(size, scale=crop_scale),
            TF.RandomHorizontalFlip(),
            ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=0.1,
            ),
            RandomGrayscale(p=grayscale),
            GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            if gaussian_blur > 0
            else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transforms(x)


class SimCLRAugmentations(nn.Module):
    """SimCLR-style augmentations.

    Standard contrastive learning augmentations.
    """

    def __init__(
        self,
        size: int = 224,
        crop_scale: Tuple[float, float] = (0.08, 1.0),
        color_jitter: float = 0.4,
        gaussian_blur: float = 0.5,
        severity: float = 1.0,
    ):
        super().__init__()
        self.size = size
        self.severity = severity

        self.transforms = nn.Sequential(
            RandomResizedCrop(size, scale=crop_scale),
            TF.RandomHorizontalFlip(),
            T.RandomApply(
                [
                    ColorJitter(
                        brightness=color_jitter * severity,
                        contrast=color_jitter * severity,
                        saturation=color_jitter * severity,
                        hue=0.1 * severity,
                    )
                ],
                p=0.8,
            ),
            T.RandomApply(
                [GaussianBlur(kernel_size=23)],
                p=gaussian_blur,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transforms(x)


class MAEAugmentations(nn.Module):
    """MAE-style augmentations.

    Minimal augmentations suitable for masked autoencoder training.
    """

    def __init__(
        self,
        size: int = 224,
        crop_scale: Tuple[float, float] = (0.2, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.333),
    ):
        super().__init__()

        self.transforms = nn.Sequential(
            RandomResizedCrop(size, scale=crop_scale, ratio=ratio),
            TF.RandomHorizontalFlip(p=0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transforms(x)


class TwoCropsTransform(nn.Module):
    """Create two different crops of the same image for contrastive learning."""

    def __init__(self, base_transform: nn.Module):
        super().__init__()
        self.base_transform = base_transform

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.base_transform(x), self.base_transform(x)


class MultiCropsTransform(nn.Module):
    """Create multiple crops (global + local) for SwAV-style training."""

    def __init__(
        self,
        global_transform: nn.Module,
        local_transform: nn.Module,
        n_local_crops: int = 8,
    ):
        super().__init__()
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_local_crops = n_local_crops

    def forward(self, x: Tensor) -> List[Tensor]:
        crops = [self.global_transform(x)]
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(x))
        return crops


class GaussianNoise(nn.Module):
    """Add Gaussian noise to input."""

    def __init__(self, mean: float = 0.0, std: float = 0.01):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


class Cutout(nn.Module):
    """Cutout augmentation: randomly mask out square patches."""

    def __init__(self, n_holes: int = 1, length: int = 16):
        super().__init__()
        self.n_holes = n_holes
        self.length = length

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2], x.shape[3]
        mask = torch.ones(h, w, dtype=x.dtype, device=x.device)

        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x_start = torch.randint(0, w - self.length, (1,)).item()
            mask[y, x_start : x_start + self.length] = 0

        return x * mask.unsqueeze(0)

"""
Image Augmentation Module

Advanced image augmentation techniques including MixUp, CutMix, RandAugment,
AutoAugment, GridMask, RandomErasing, Mosaic, and more.
"""

from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

from fishstick.augmentation_ext.base import AugmentationBase


class RandomHorizontalFlip(AugmentationBase):
    """Random horizontal flip augmentation."""

    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        super().__init__(probability=p, seed=seed)

    def __call__(self, data: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        if self._should_apply():
            if isinstance(data, Tensor):
                return torch.flip(data, dims=[-1])
            elif isinstance(data, Image.Image):
                return data.transpose(Image.FLIP_LEFT_RIGHT)
        return data


class RandomVerticalFlip(AugmentationBase):
    """Random vertical flip augmentation."""

    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        super().__init__(probability=p, seed=seed)

    def __call__(self, data: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        if self._should_apply():
            if isinstance(data, Tensor):
                return torch.flip(data, dims=[-2])
            elif isinstance(data, Image.Image):
                return data.transpose(Image.FLIP_TOP_BOTTOM)
        return data


class RandomRotation(AugmentationBase):
    """Random rotation augmentation."""

    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]] = 10,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees

    def __call__(self, data: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        if not self._should_apply():
            return data

        if isinstance(data, Tensor):
            return self._rotate_tensor(data)
        elif isinstance(data, Image.Image):
            return self._rotate_pil(data)
        return data

    def _rotate_tensor(self, x: Tensor) -> Tensor:
        angle = random.uniform(*self.degrees)
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=x.dtype, device=x.device
        )
        theta = theta.unsqueeze(0).repeat(x.size(0), 1, 1)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(
            x, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )

    def _rotate_pil(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(*self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)


class ColorJitter(AugmentationBase):
    """Color jitter augmentation."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, data: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        if not self._should_apply():
            return data

        if isinstance(data, Tensor):
            return self._jitter_tensor(data)
        elif isinstance(data, Image.Image):
            return self._jitter_pil(data)
        return data

    def _jitter_tensor(self, x: Tensor) -> Tensor:
        if self.brightness > 0:
            brightness_factor = 1 + random.uniform(-self.brightness, self.brightness)
            x = x * brightness_factor

        if self.contrast > 0:
            contrast_factor = 1 + random.uniform(-self.contrast, self.contrast)
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            x = (x - mean) * contrast_factor + mean

        return torch.clamp(x, 0, 1)

    def _jitter_pil(self, img: Image.Image) -> Image.Image:
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)

        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)

        if self.saturation > 0:
            factor = 1 + random.uniform(-self.saturation, self.saturation)
            img = ImageEnhance.Color(img).enhance(factor)

        if self.hue > 0:
            hsv = np.array(img.convert("HSV"))
            hsv[:, :, 0] = (
                hsv[:, :, 0].astype(float) + random.uniform(-self.hue, self.hue) * 255
            ) % 256
            img = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")

        return img


class RandomAffine(AugmentationBase):
    """Random affine transformation."""

    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]] = 0,
        translate: Tuple[float, float] = (0, 0),
        scale: Tuple[float, float] = (1, 1),
        shear: Union[float, Tuple[float, float]] = 0,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, data: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        if not self._should_apply():
            return data

        if isinstance(data, Tensor):
            return self._affine_tensor(data)
        return data

    def _affine_tensor(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        theta_list = []

        for _ in range(batch_size):
            angle = (
                random.uniform(*self.degrees)
                if isinstance(self.degrees, tuple)
                else random.uniform(-self.degrees, self.degrees)
            )
            tx = random.uniform(-self.translate[0], self.translate[0])
            ty = random.uniform(-self.translate[1], self.translate[1])
            scale_factor = random.uniform(self.scale[0], self.scale[1])

            angle_rad = np.deg2rad(angle)
            cos_a = np.cos(angle_rad) * scale_factor
            sin_a = np.sin(angle_rad) * scale_factor

            theta = torch.tensor(
                [[cos_a, -sin_a, tx], [sin_a, cos_a, ty]],
                dtype=x.dtype,
                device=x.device,
            )
            theta_list.append(theta)

        theta = torch.stack(theta_list)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(
            x, grid, align_corners=False, mode="bilinear", padding_mode="zeros"
        )


class MixUp(AugmentationBase):
    """
    MixUp augmentation - blend images and labels.

    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", 2018
    """

    def __init__(
        self,
        alpha: float = 0.2,
        p: float = 1.0,
        label_smoothing: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.rng = np.random.RandomState(seed)

    def __call__(self, x: Tensor, y: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        """
        Apply MixUp to a batch.

        Args:
            x: Batch of images (N, C, H, W)
            y: Optional batch of labels (N,)

        Returns:
            Mixed images and optionally (labels_a, labels_b, lambda)
        """
        if self.alpha > 0:
            lam = self.rng.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = self.rng.permutation(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index]

        if y is not None:
            y_a, y_b = y, y[index]
            if self.label_smoothing > 0:
                y_a = self._smooth_labels(y_a)
                y_b = self._smooth_labels(y_b)
            return mixed_x, y_a, y_b, lam

        return (mixed_x,)

    def _smooth_labels(self, labels: Tensor) -> Tensor:
        n_classes = labels.max().item() + 1
        return labels * (1 - self.label_smoothing) + self.label_smoothing / n_classes


class CutMix(AugmentationBase):
    """
    CutMix augmentation - cut and paste regions between images.

    Reference: Yun et al., "CutMix: Regularization Strategy", 2019
    """

    def __init__(
        self,
        alpha: float = 1.0,
        p: float = 1.0,
        label_smoothing: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.rng = np.random.RandomState(seed)

    def __call__(self, x: Tensor, y: Optional[Tensor] = None) -> Tuple[Tensor, ...]:
        """
        Apply CutMix to a batch.

        Args:
            x: Batch of images (N, C, H, W)
            y: Optional batch of labels (N,)

        Returns:
            Mixed images and optionally (labels_a, labels_b, lambda)
        """
        if self.alpha > 0:
            lam = self.rng.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = self.rng.permutation(batch_size)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

        if y is not None:
            y_a, y_b = y, y[index]
            if self.label_smoothing > 0:
                y_a = self._smooth_labels(y_a)
                y_b = self._smooth_labels(y_b)
            return x, y_a, y_b, lam

        return (x,)

    def _rand_bbox(
        self, size: Tuple[int, ...], lam: float
    ) -> Tuple[int, int, int, int]:
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

    def _smooth_labels(self, labels: Tensor) -> Tensor:
        n_classes = labels.max().item() + 1
        return labels * (1 - self.label_smoothing) + self.label_smoothing / n_classes


class GridMask(AugmentationBase):
    """
    GridMask augmentation.

    Reference: Chen et al., "GridMask Data Augmentation", 2020
    """

    def __init__(
        self,
        ratio: float = 0.6,
        rotate: int = 1,
        grid_size: int = 96,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.ratio = ratio
        self.rotate = rotate
        self.grid_size = grid_size

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply GridMask to an image batch.

        Args:
            x: Image tensor (N, C, H, W)

        Returns:
            Masked image tensor
        """
        if not self._should_apply():
            return x

        batch_size, channels, height, width = x.size()
        mask = torch.ones(height, width, dtype=x.dtype, device=x.device)

        d = self.grid_size
        l = int(d * self.ratio)

        for i in range(0, height, d):
            start = i
            end = min(i + l, height)
            mask[start:end, :] = 0

        for j in range(0, width, d):
            start = j
            end = min(j + l, width)
            mask[:, start:end] = 0

        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(x)
        return x * mask


class RandomErasing(AugmentationBase):
    """
    Random Erasing augmentation.

    Reference: Zhong et al., "Random Erasing Data Augmentation", 2017
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.rng = np.random.RandomState(seed)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply random erasing to an image batch.

        Args:
            x: Image tensor (N, C, H, W)

        Returns:
            Augmented image tensor
        """
        if not self._should_apply():
            return x

        batch_size, channels, height, width = x.size()
        area = height * width

        for i in range(batch_size):
            target_area = self.rng.uniform(*self.scale) * area
            aspect_ratio = self.rng.uniform(*self.ratio)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = self.rng.randint(0, width - w)
                y1 = self.rng.randint(0, height - h)
                x[i, :, y1 : y1 + h, x1 : x1 + w] = self.value

        return x


class Mosaic(AugmentationBase):
    """
    Mosaic augmentation - combine 4 images into one.

    Reference: Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy", 2020
    """

    def __init__(
        self,
        p: float = 0.5,
        img_size: Optional[Tuple[int, int]] = None,
        border: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.img_size = img_size
        self.border = border or [-320, -320]
        self.rng = np.random.RandomState(seed)

    def __call__(self, images: Tensor) -> Tensor:
        """
        Apply mosaic augmentation to a batch of 4+ images.

        Args:
            images: Batch of images (N, C, H, W) where N >= 4

        Returns:
            Single mosaic image
        """
        if not self._should_apply() or images.size(0) < 4:
            return images

        batch_size, channels, height, width = images.size()
        mosaic_img = torch.zeros(channels, height, width, device=images.device)

        yc = int(self.rng.uniform(*self.border) + height / 2)
        xc = int(self.rng.uniform(*self.border) + width / 2)

        indices = self.rng.choice(batch_size, size=4, replace=False)

        for idx, (img, (i, j)) in enumerate(
            zip(images[indices], [(0, 0), (0, width), (height, 0), (height, width)])
        ):
            h, w = height // 2, width // 2

            img_resized = F.interpolate(
                img.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            y1 = max(0, yc - h + j)
            y2 = min(yc + j, height)
            x1 = max(0, xc - w + i)
            x2 = min(xc + i, width)

            y1_res = h - (yc - y1) if y1 < yc else 0
            y2_res = h + (height - y2) if y2 > yc else h
            x1_res = w - (xc - x1) if x1 < xc else 0
            x2_res = w + (width - x2) if x2 > xc else w

            mosaic_img[:, y1:y2, x1:x2] = img_resized[:, y1_res:y2_res, x1_res:x2_res]

        return mosaic_img.unsqueeze(0)


class Blend(AugmentationBase):
    """Blend two images together."""

    def __init__(
        self,
        p: float = 0.5,
        alpha: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def __call__(self, x: Tensor) -> Tensor:
        """Blend images in a batch."""
        if not self._should_apply() or x.size(0) < 2:
            return x

        indices = self.rng.choice(x.size(0), size=2, replace=False)
        lam = self.rng.beta(self.alpha, self.alpha)
        return lam * x[indices[0]] + (1 - lam) * x[indices[1]]


class RandAugment(AugmentationBase):
    """
    RandAugment: Practical automated data augmentation.

    Reference: Cubuk et al., "RandAugment", 2019
    """

    def __init__(
        self,
        n: int = 2,
        m: int = 10,
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.n = n
        self.m = m
        self.rng = np.random.RandomState(seed)

        self.operations = [
            self._auto_contrast,
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

    def __call__(self, img: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        ops = self.rng.choice(self.operations, size=self.n, replace=False)

        for op in ops:
            img = op(img, self.m)

        return img

    def _auto_contrast(self, img: Image.Image, m: int) -> Image.Image:
        return ImageOps.autocontrast(img)

    def _equalize(self, img: Image.Image, m: int) -> Image.Image:
        return ImageOps.equalize(img)

    def _rotate(self, img: Image.Image, m: int) -> Image.Image:
        degrees = int((m / 30) * 30)
        if self.rng.random() > 0.5:
            degrees = -degrees
        return img.rotate(degrees)

    def _solarize(self, img: Image.Image, m: int) -> Image.Image:
        thresh = int((m / 30) * 256)
        return ImageOps.solarize(img, thresh)

    def _color(self, img: Image.Image, m: int) -> Image.Image:
        factor = 1.0 + (m / 30) * 0.9
        if self.rng.random() > 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Color(img).enhance(factor)

    def _posterize(self, img: Image.Image, m: int) -> Image.Image:
        bits = 4 - int((m / 30) * 4)
        bits = max(1, bits)
        return ImageOps.posterize(img, bits)

    def _contrast(self, img: Image.Image, m: int) -> Image.Image:
        factor = 1.0 + (m / 30) * 0.9
        if self.rng.random() > 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Contrast(img).enhance(factor)

    def _brightness(self, img: Image.Image, m: int) -> Image.Image:
        factor = 1.0 + (m / 30) * 0.9
        if self.rng.random() > 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Brightness(img).enhance(factor)

    def _sharpness(self, img: Image.Image, m: int) -> Image.Image:
        factor = 1.0 + (m / 30) * 0.9
        if self.rng.random() > 0.5:
            factor = 1.0 / factor
        return ImageEnhance.Sharpness(img).enhance(factor)

    def _shear_x(self, img: Image.Image, m: int) -> Image.Image:
        shear = (m / 30) * 0.3
        if self.rng.random() > 0.5:
            shear = -shear
        return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))

    def _shear_y(self, img: Image.Image, m: int) -> Image.Image:
        shear = (m / 30) * 0.3
        if self.rng.random() > 0.5:
            shear = -shear
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))

    def _translate_x(self, img: Image.Image, m: int) -> Image.Image:
        pixels = (m / 30) * img.size[0] / 3
        if self.rng.random() > 0.5:
            pixels = -pixels
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0))

    def _translate_y(self, img: Image.Image, m: int) -> Image.Image:
        pixels = (m / 30) * img.size[1] / 3
        if self.rng.random() > 0.5:
            pixels = -pixels
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels))


class AutoAugment(AugmentationBase):
    """
    AutoAugment: Learning Augmentation Policies.

    Reference: Cubuk et al., "AutoAugment", 2019
    """

    def __init__(
        self,
        policy: str = "imagenet",
        p: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.policy = policy
        self.rng = np.random.RandomState(seed)

        self.policies = {
            "imagenet": [
                [("Posterize", 0.4, 4), ("Rotate", 0.6, 9)],
                [("Solarize", 0.6, 5), ("AutoContrast", 0.8, None)],
                [("Equalize", 0.6, None), ("Equalize", 0.8, None)],
                [("Posterize", 0.6, 7), ("Posterize", 0.6, 6)],
                [("Equalize", 0.4, None), ("Solarize", 0.6, 4)],
                [("Equalize", 0.4, None), ("Rotate", 0.6, 8)],
                [("Solarize", 0.6, 3), ("Equalize", 0.6, None)],
                [("Posterize", 0.8, 5), ("Equalize", 1.0, None)],
                [("Rotate", 0.8, 3), ("Posterize", 0.8, 5)],
                [("Rotate", 0.8, 1), ("Equalize", 0.8, None)],
            ],
            "cifar10": [
                [("Invert", 0.1, None), ("Contrast", 0.2, 6)],
                [("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)],
                [("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)],
                [("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)],
                [("Autocontrast", 0.5, None), ("Equalize", 0.9, None)],
                [("ShearY", 0.2, 7), ("Posterize", 0.3, 7)],
                [("Color", 0.7, 7), ("Brightness", 0.5, 7)],
                [("Sharpness", 0.8, 1), ("Invert", 0.9, None)],
                [("ShearX", 0.9, 4), ("Equalize", 0.6, None)],
                [("Equalize", 0.6, None), ("Rotate", 0.5, 3)],
            ],
        }

    def __call__(self, img: Union[Tensor, Image.Image]) -> Union[Tensor, Image.Image]:
        if not isinstance(img, Image.Image) and isinstance(img, Tensor):
            img = self._tensor_to_pil(img)

        policy = self.rng.choice(
            self.policies.get(self.policy, self.policies["imagenet"])
        )

        for op_name, prob, mag in policy:
            if self.rng.random() < prob:
                img = self._apply_op(img, op_name, mag)

        if isinstance(img, Image.Image):
            return self._pil_to_tensor(img)
        return img

    def _apply_op(
        self, img: Image.Image, op_name: str, magnitude: Optional[int]
    ) -> Image.Image:
        ops = {
            "Invert": lambda: ImageOps.invert(img),
            "AutoContrast": lambda: ImageOps.autocontrast(img),
            "Equalize": lambda: ImageOps.equalize(img),
            "Rotate": lambda: img.rotate(magnitude * 3 if magnitude else 0),
            "Posterize": lambda: ImageOps.posterize(
                img, 8 - magnitude if magnitude else 4
            ),
            "Solarize": lambda: ImageOps.solarize(
                img, 256 - magnitude * 16 if magnitude else 128
            ),
            "Contrast": lambda: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * 0.1 if magnitude else 1
            ),
            "Brightness": lambda: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * 0.1 if magnitude else 1
            ),
            "Sharpness": lambda: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * 0.1 if magnitude else 1
            ),
            "Color": lambda: ImageEnhance.Color(img).enhance(
                1 + magnitude * 0.1 if magnitude else 1
            ),
            "TranslateX": lambda: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * 10, 0, 1, 0)
            ),
            "TranslateY": lambda: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * 10)
            ),
            "ShearX": lambda: img.transform(
                img.size, Image.AFFINE, (1, magnitude * 0.1, 0, 0, 1, 0)
            ),
            "ShearY": lambda: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * 0.1, 1, 0)
            ),
        }
        return ops.get(op_name, lambda: img)()

    def _tensor_to_pil(self, tensor: Tensor) -> Image.Image:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)

    def _pil_to_tensor(self, img: Image.Image) -> Tensor:
        array = np.array(img).astype(np.float32) / 255.0
        if array.ndim == 2:
            array = array[:, :, None]
        return torch.from_numpy(array).permute(2, 0, 1)


class GaussianBlur(AugmentationBase):
    """Gaussian blur augmentation."""

    def __init__(
        self,
        kernel_size: int = 3,
        sigma: Tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)

    def __call__(self, x: Tensor) -> Tensor:
        if not self._should_apply():
            return x

        sigma = self.rng.uniform(*self.sigma)
        kernel = self._get_gaussian_kernel(self.kernel_size, sigma)
        kernel = kernel.to(x.device)

        padding = self.kernel_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.size(1))

    def _get_gaussian_kernel(self, kernel_size: int, sigma: float) -> Tensor:
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)


class Cutout(AugmentationBase):
    """
    Cutout augmentation - randomly mask out square regions.

    Reference: DeVries & Taylor, "Improved Regularization of CNNs", 2017
    """

    def __init__(
        self,
        n_holes: int = 1,
        length: int = 16,
        p: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(probability=p, seed=seed)
        self.n_holes = n_holes
        self.length = length

    def __call__(self, x: Tensor) -> Tensor:
        h = x.size(2)
        w = x.size(3)

        mask = torch.ones(h, w, dtype=x.dtype, device=x.device)

        for _ in range(self.n_holes):
            y = random.randint(0, h)
            x_pos = random.randint(0, w)

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x_pos - self.length // 2)
            x2 = min(w, x_pos + self.length // 2)

            mask[y1:y2, x1:x2] = 0.0

        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(x)
        return x * mask


def get_image_augmentation_pipeline(
    task: str = "classification",
    intensity: float = 1.0,
) -> List[Union[AugmentationBase, Callable]]:
    """
    Get a pre-configured image augmentation pipeline.

    Args:
        task: Task type (classification, detection, segmentation)
        intensity: Overall augmentation intensity (0-1)

    Returns:
        List of augmentation operations
    """
    return [
        RandomHorizontalFlip(p=0.5),
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
        GaussianBlur(p=0.2 * intensity),
        RandomErasing(p=0.3 * intensity),
    ]

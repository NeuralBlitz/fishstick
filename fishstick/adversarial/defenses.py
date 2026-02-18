import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple, List
import numpy as np
from collections import OrderedDict


class AdversarialTraining:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        steps: int = 10,
        beta: float = 6.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.beta = beta

    def step(
        self, x: torch.Tensor, y: torch.Tensor, loss_fn: Optional[Callable] = None
    ) -> float:
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.model.train()

        delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        for _ in range(self.steps):
            x_adv = torch.clamp(x + delta, 0, 1)
            outputs = self.model(x_adv)
            loss = loss_fn(outputs, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                delta = delta + self.alpha * delta.grad.sign()
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                delta.requires_grad = True

        return loss.item()


class TRADES:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epsilon: float = 0.03,
        beta: float = 6.0,
        num_steps: int = 10,
        step_size: float = 0.01,
    ):
        self.model = model
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.beta = beta
        self.num_steps = num_steps
        self.step_size = step_size

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()

        batch_size = x.shape[0]
        delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        for _ in range(self.num_steps):
            x_adv = torch.clamp(x + delta, 0, 1)

            outputs_adv = self.model(x_adv)
            outputs_nat = self.model(x).detach()

            loss_kl = F.kl_div(
                F.log_softmax(outputs_adv, dim=1),
                F.softmax(outputs_nat, dim=1),
                reduction="batchmean",
            )

            loss_natural = F.cross_entropy(outputs_nat, y)
            loss = loss_natural + self.beta * loss_kl

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                delta = delta + self.step_size * delta.grad.sign()
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                delta.requires_grad = True

        return loss.item()


class ADP:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epsilon: float = 0.03,
        gamma: float = 0.5,
        num_steps: int = 10,
        step_size: float = 0.01,
    ):
        self.model = model
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_steps = num_steps
        self.step_size = step_size

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()

        delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        for _ in range(self.num_steps):
            x_adv = torch.clamp(x + delta, 0, 1)

            outputs = self.model(x_adv)
            loss = F.cross_entropy(outputs, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                delta = delta + self.step_size * delta.grad.sign()
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                delta.requires_grad = True

        return loss.item()


class InputTransformation:
    def __init__(
        self,
        model: nn.Module,
        transforms: Optional[List[Callable]] = None,
        num_samples: int = 5,
    ):
        self.model = model
        self.transforms = transforms or [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
        ]
        self.num_samples = num_samples

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        batch_size = x.shape[0]
        all_probs = []

        for _ in range(self.num_samples):
            x_transformed = x
            for transform in self.transforms:
                x_transformed = transform(x_transformed)

            with torch.no_grad():
                outputs = self.model(x_transformed)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)

        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs


class RandomCrop:
    def __init__(self, size: int, padding: int = 0):
        self.size = size
        self.padding = padding

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding > 0:
            x = F.pad(x, [self.padding] * 4, mode="reflect")

        _, h, w = x.shape[-3:]
        new_h, new_w = self.size, self.size

        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()

        return x[..., top : top + new_h, left : left + new_w]


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return x.flip(-1)
        return x


class ColorJitter:
    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.brightness > 0:
            factor = 1 + torch.rand(1).item() * self.brightness * 2 - self.brightness
            x = x * factor

        if self.contrast > 0:
            factor = 1 + torch.rand(1).item() * self.contrast * 2 - self.contrast
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - mean) * factor + mean

        return torch.clamp(x, 0, 1)


class RandomizedSmoothing:
    def __init__(
        self,
        model: nn.Module,
        sigma: float = 0.25,
        num_samples: int = 100,
        num_classes: int = 10,
    ):
        self.model = model
        self.sigma = sigma
        self.num_samples = num_samples
        self.num_classes = num_classes

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()

        batch_size = x.shape[0]
        noise = (
            torch.randn(batch_size, self.num_samples, *x.shape[1:], device=x.device)
            * self.sigma
        )

        x_noisy = x.unsqueeze(1) + noise
        x_noisy = torch.clamp(x_noisy, 0, 1)

        with torch.no_grad():
            outputs = self.model(x_noisy.view(-1, *x.shape[1:]))
            probs = F.softmax(outputs, dim=1)
            probs = probs.view(batch_size, self.num_samples, -1)

        avg_probs = probs.mean(dim=1)
        predictions = avg_probs.argmax(dim=1)

        return predictions, avg_probs

    def certify(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions, probs = self.predict(x)

        top2, _ = probs.topk(2, dim=1)
        radius = self.sigma * (top2[:, 0] - top2[:, 1]).sqrt()

        return predictions, radius


class FeatureDecompositionDefense:
    def __init__(self, model: nn.Module, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        features = self._extract_features(x)
        noise_features = features * (features.abs() > self.threshold).float()
        cleaned_features = features - noise_features

        return self._classify_from_features(cleaned_features)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = []

        def hook(module, input, output):
            features.append(output)

        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                handle = layer.register_forward_hook(hook)

        with torch.no_grad():
            self.model(x)

        for handle in self.model._forward_hooks.values():
            handle.remove()

        return features[-1] if features else x

    def _classify_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.classifier(features.flatten(1))


class HighLevelRepresentationDefense:
    def __init__(self, model: nn.Module, k: int = 5):
        self.model = model
        self.k = k

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        with torch.no_grad():
            features = self.model.forward_before_classifier(x)

        features_sorted, _ = torch.sort(features, dim=-1)
        top_k_features = features_sorted[..., -self.k :]

        reconstructed = self.model.classifier(top_k_features.flatten(1))

        return reconstructed.argmax(dim=1)


class Smoothing:
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

        kernel = self._gaussian_kernel(kernel_size, sigma)
        self.register_buffer("kernel", kernel)

    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            kernel = self.kernel.unsqueeze(0).unsqueeze(0)
            return F.conv2d(x, kernel, padding=self.kernel_size // 2)
        return x


class BitDepthReduction:
    def __init__(self, bits: int = 4):
        self.bits = bits
        self.levels = 2**bits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x * self.levels) / self.levels


class JpegCompression:
    def __init__(self, quality: int = 75):
        self.quality = quality

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


class InputDiversity:
    def __init__(
        self,
        model: nn.Module,
        prob: float = 0.5,
        diversity_range: Tuple[float, float] = (0.5, 1.0),
    ):
        self.model = model
        self.prob = prob
        self.diversity_range = diversity_range

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        outputs_list = []

        for _ in range(10):
            if torch.rand(1).item() < self.prob:
                scale = (
                    torch.rand(1).item()
                    * (self.diversity_range[1] - self.diversity_range[0])
                    + self.diversity_range[0]
                )
                h, w = x.shape[-2:]
                new_h, new_w = int(h * scale), int(w * scale)

                x_resized = F.interpolate(
                    x, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                x_aug = F.interpolate(
                    x_resized, size=(h, w), mode="bilinear", align_corners=False
                )
            else:
                x_aug = x

            with torch.no_grad():
                outputs = self.model(x_aug)
                outputs_list.append(outputs)

        avg_outputs = torch.stack(outputs_list).mean(dim=0)
        return avg_outputs.argmax(dim=1)

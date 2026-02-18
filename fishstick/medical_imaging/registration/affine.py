"""
Affine and Rigid Registration

Affine and rigid transformation registration for medical images.
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


@dataclass
class RegistrationResult:
    """Result of a registration operation."""

    transformed_image: Tensor
    transform_params: Dict[str, Tensor]
    similarity_metric: float
    converged: bool


class AffineRegistration(nn.Module):
    """Affine image registration.

    Aligns two images using affine transformation with optimization
    for similarity metrics (NCC, MI, SSD).

    Example:
        >>> reg = AffineRegistration(num_iterations=100)
        >>> result = reg(moving_image, fixed_image)
    """

    def __init__(
        self,
        num_iterations: int = 100,
        learning_rate: float = 1.0,
        similarity: str = "ncc",
        init_params: Optional[Tensor] = None,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.similarity = similarity

        self.params = nn.Parameter(
            init_params if init_params is not None else torch.eye(4)
        )

    def forward(
        self,
        moving: Tensor,
        fixed: Tensor,
    ) -> RegistrationResult:
        """Perform affine registration.

        Args:
            moving: Moving image tensor
            fixed: Fixed (reference) image tensor

        Returns:
            RegistrationResult with transformed image and parameters
        """
        device = moving.device

        self.params.data = torch.eye(4, device=device)

        for i in range(self.num_iterations):
            transformed = self._apply_transform(moving, self.params)

            loss = self._compute_similarity(transformed, fixed)

            loss.backward()

            with torch.no_grad():
                self.params.data -= self.learning_rate * self.params.grad
                self.params.grad.zero_()

        final_transformed = self._apply_transform(moving, self.params)
        final_similarity = self._compute_similarity(final_transformed, fixed)

        return RegistrationResult(
            transformed_image=final_transformed,
            transform_params={"affine": self.params.data.clone()},
            similarity_metric=float(final_similarity),
            converged=True,
        )

    def _apply_transform(self, image: Tensor, transform: Tensor) -> Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 4:
            image = image.unsqueeze(0)

        grid = self._affine_grid(image.shape, transform)

        transformed = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return transformed.squeeze(0) if image.shape[0] == 1 else transformed

    def _affine_grid(self, size: Tuple[int, ...], transform: Tensor) -> Tensor:
        if len(size) == 4:
            b, c, h, w = size
            d = 1
        else:
            b, c, d, h, w = size

        theta = transform[:3, :].unsqueeze(0).expand(b, -1, -1)

        if len(size) == 4:
            grid = F.affine_grid(theta, [b, c, h, w], align_corners=False)
        else:
            grid = F.affine_grid(theta, [b, c, d, h, w], align_corners=False)

        return grid

    def _compute_similarity(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.similarity == "ncc":
            return self._normalized_cross_correlation(pred, target)
        elif self.similarity == "ssd":
            return F.mse_loss(pred, target)
        elif self.similarity == "mi":
            return self._mutual_information(pred, target)
        else:
            return F.mse_loss(pred, target)

    def _normalized_cross_correlation(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_mean = pred.mean()
        target_mean = target.mean()

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        numerator = (pred_centered * target_centered).sum()
        denominator = (
            torch.sqrt((pred_centered**2).sum() * (target_centered**2).sum()) + 1e-8
        )

        return -numerator / denominator

    def _mutual_information(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        hist_2d = torch.histc(
            pred_flat * target_flat,
            bins=256,
            min=0,
            max=1,
        )

        pxy = hist_2d / hist_2d.sum()

        px = pxy.sum(dim=0)
        py = pxy.sum(dim=1)

        px_py = px.unsqueeze(0) * py.unsqueeze(1)

        mi = (pxy * torch.log(pxy / (px_py + 1e-8) + 1e-8)).sum()

        return -mi


class RigidRegistration(AffineRegistration):
    """Rigid registration (translation + rotation only).

    Constrains affine transformation to rigid body transformations
    (rotation and translation, no scaling or shearing).
    """

    def __init__(
        self,
        num_iterations: int = 100,
        learning_rate: float = 1.0,
        similarity: str = "ncc",
    ):
        super().__init__(
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            similarity=similarity,
        )

    def _apply_transform(self, image: Tensor, transform: Tensor) -> Tensor:
        rigid_transform = self._extract_rigid_transform(transform)
        return super()._apply_transform(image, rigid_transform)

    def _extract_rigid_transform(self, affine: Tensor) -> Tensor:
        rigid = affine.clone()

        rigid.data[0, 0] = 1.0
        rigid.data[0, 1] = 0.0
        rigid.data[1, 0] = 0.0
        rigid.data[1, 1] = 1.0

        rigid.data[0, 2] = torch.clamp(rigid.data[0, 2], -0.5, 0.5)
        rigid.data[1, 2] = torch.clamp(rigid.data[1, 2], -0.5, 0.5)

        if affine.shape[0] == 4:
            rigid.data[2, 2] = 1.0
            rigid.data[2, :3] = 0.0
            rigid.data[:3, 2] = 0.0

        return rigid


class SimilarityRegistration(nn.Module):
    """Similarity registration (rigid + uniform scaling)."""

    def __init__(
        self,
        num_iterations: int = 100,
        learning_rate: float = 1.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        self.scale = nn.Parameter(torch.tensor(1.0))
        self.rotation = nn.Parameter(torch.tensor(0.0))
        self.translation = nn.Parameter(torch.zeros(2))

    def forward(self, moving: Tensor, fixed: Tensor) -> Tensor:
        device = moving.device

        for _ in range(self.num_iterations):
            transformed = self._apply_transform(moving)

            loss = F.mse_loss(transformed, fixed)

            loss.backward()

            with torch.no_grad():
                self.scale.data -= self.learning_rate * self.scale.grad
                self.rotation.data -= self.learning_rate * self.rotation.grad
                self.translation.data -= self.learning_rate * self.translation.grad

                self.scale.grad.zero_()
                self.rotation.grad.zero_()
                self.translation.grad.zero_()

        return self._apply_transform(moving)

    def _apply_transform(self, image: Tensor) -> Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 4:
            image = image.unsqueeze(0)

        b, c, h, w = image.shape

        theta = torch.zeros(b, 2, 3, device=image.device)

        theta[:, 0, 0] = self.scale * torch.cos(self.rotation)
        theta[:, 0, 1] = -self.scale * torch.sin(self.rotation)
        theta[:, 1, 0] = self.scale * torch.sin(self.rotation)
        theta[:, 1, 1] = self.scale * torch.cos(self.rotation)

        theta[:, 0, 2] = self.translation[0]
        theta[:, 1, 2] = self.translation[1]

        grid = F.affine_grid(theta, [b, c, h, w], align_corners=False)

        transformed = F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return transformed.squeeze(0) if c == 1 else transformed

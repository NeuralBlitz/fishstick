"""
Deformable Registration

Deformable image registration using displacement fields and DEMons.
"""

from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DeformableRegistration(nn.Module):
    """Deformable image registration using learnable displacement field.
    
    Uses a CNN to predict a dense displacement field for non-linear alignment.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        num_iterations: int = 100,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        
        self.field_net = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, 3, 3, padding=1),
        )

    def forward(
        self,
        moving: Tensor,
        fixed: Tensor,
    ) -> "RegistrationResult":
        device = moving.device
        
        if moving.ndim == 3:
            moving = moving.unsqueeze(0).unsqueeze(0)
            fixed = fixed.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        b, c, d, h, w = moving.shape
        
        displacement = torch.zeros(b, 3, d, h, w, device=device)
        
        for _ in range(self.num_iterations):
            moving_warped = self._apply_displacement(moving, displacement)
            input_feat = torch.cat([moving_warped, fixed], dim=1)
            displacement_update = self.field_net(input_feat)
            displacement = displacement + displacement_update * 0.1
        
        final_warped = self._apply_displacement(moving, displacement)
        
        from fishstick.medical_imaging.registration.transforms import RegistrationResult
        return RegistrationResult(
            transformed_image=final_warped.squeeze(0) if squeeze_output else final_warped,
            transform_field=displacement.squeeze(0) if squeeze_output else displacement,
            similarity_metric=float(F.mse_loss(final_warped, fixed)),
            converged=True,
        )

    def _apply_displacement(self, image: Tensor, displacement: Tensor) -> Tensor:
        b, c, d, h, w = image.shape
        
        z = torch.linspace(-1, 1, d, device=image.device, dtype=image.dtype)
        y = torch.linspace(-1, 1, h, device=image.device, dtype=image.dtype)
        x = torch.linspace(-1, 1, w, device=image.device, dtype=image.dtype)
        
        grid = torch.stack(torch.meshgrid(x, y, z, indexing="xy"), dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1)
        
        new_grid = grid + displacement.permute(0, 1, 3, 4, 2)
        new_grid = new_grid.permute(0, 1, 4, 3, 2)
        
        return F.grid_sample(image, new_grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class DemonsRegistration(nn.Module):
    """Demons-based deformable registration."""

    def __init__(
        self,
        num_iterations: int = 50,
        sigma: float = 1.0,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.sigma = sigma
        self.alpha = alpha

    def forward(self, moving: Tensor, fixed: Tensor) -> "RegistrationResult":
        device = moving.device
        
        if moving.ndim == 3:
            moving = moving.unsqueeze(0).unsqueeze(0)
            fixed = fixed.unsqueeze(0).unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        b, c, d, h, w = moving.shape
        velocity = torch.zeros(b, 3, d, h, w, device=device)
        
        for _ in range(self.num_iterations):
            warped = self._apply_velocity(moving, velocity)
            grad_fixed = self._compute_gradient(fixed)
            diff = fixed - warped
            numerator = diff * grad_fixed
            denominator = grad_fixed.square().sum(dim=1, keepdim=True) + (diff.square() / (self.alpha ** 2))
            update = numerator / (denominator + 1e-8)
            velocity = velocity + update
        
        final_warped = self._apply_velocity(moving, velocity)
        
        from fishstick.medical_imaging.registration.transforms import RegistrationResult
        return RegistrationResult(
            transformed_image=final_warped.squeeze(0) if squeeze_output else final_warped,
            transform_field=velocity.squeeze(0) if squeeze_output else velocity,
            similarity_metric=float(F.mse_loss(final_warped, fixed)),
            converged=True,
        )

    def _apply_velocity(self, image: Tensor, velocity: Tensor) -> Tensor:
        b, c, d, h, w = image.shape
        grid = self._make_grid(d, h, w, image.device, image.dtype).unsqueeze(0).expand(b, -1, -1, -1, -1)
        new_grid = grid + velocity.permute(0, 1, 3, 4, 2).permute(0, 1, 4, 3, 2)
        return F.grid_sample(image, new_grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _compute_gradient(self, image: Tensor) -> Tensor:
        grad = torch.zeros_like(image)
        grad[:, :, 1:-1, :, :] = (image[:, :, 2:, :, :] - image[:, :, :-2, :, :]) / 2
        grad[:, :, :, 1:-1, :] = (image[:, :, :, 2:, :] - image[:, :, :, :-2, :]) / 2
        grad[:, :, :, :, 1:-1] = (image[:, :, :, :, 2:] - image[:, :, :, :, :-2]) / 2
        return grad

    def _make_grid(self, d: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        z = torch.linspace(-1, 1, d, device=device, dtype=dtype)
        y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        return torch.stack(torch.meshgrid(x, y, z, indexing="xy"), dim=0)

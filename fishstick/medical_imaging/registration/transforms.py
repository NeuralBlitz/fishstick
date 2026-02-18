"""
Registration Transformation Utilities

Transformation field operations and utilities for image registration.
"""

from typing import Optional, Tuple, Union, Dict, Any
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
    transform_field: Optional[Tensor] = None
    transform_params: Optional[Dict[str, Tensor]] = None
    similarity_metric: float = 0.0
    converged: bool = False


class TransformationField(nn.Module):
    """Dense transformation field for deformable registration."""

    def __init__(self, shape: Tuple[int, int, int]):
        super().__init__()
        self.shape = shape
        self.field = nn.Parameter(torch.zeros(1, 3, *shape))

    def forward(self) -> Tensor:
        return self.field


def apply_transform(
    image: Tensor,
    transform: Tensor,
    mode: str = "bilinear",
) -> Tensor:
    """Apply transformation to image.
    
    Args:
        image: Input image (C, D, H, W) or (D, H, W)
        transform: Transformation field (3, D, H, W) or (D, H, W)
        mode: Interpolation mode
        
    Returns:
        Transformed image
    """
    if image.ndim == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 4:
        image = image.unsqueeze(0)
    
    if transform.ndim == 3:
        transform = transform.unsqueeze(0)
    
    b, c, d, h, w = image.shape
    
    grid = _make_identity_grid(d, h, w, image.device, image.dtype)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1, -1)
    
    new_grid = grid + transform.permute(0, 1, 3, 4, 2).permute(0, 1, 4, 3, 2)
    new_grid = new_grid.permute(0, 1, 4, 3, 2)
    
    transformed = F.grid_sample(
        image,
        new_grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )
    
    return transformed.squeeze(0) if image.shape[0] == 1 else transformed


def compose_transforms(
    transform1: Tensor,
    transform2: Tensor,
) -> Tensor:
    """Compose two transformation fields.
    
    Args:
        transform1: First transform field
        transform2: Second transform field
        
    Returns:
        Composed transform
    """
    warped = apply_transform(transform2, transform1)
    return transform1 + warped


def compute_jacobian_determinant(field: Tensor) -> Tensor:
    """Compute Jacobian determinant of transformation field.
    
    Args:
        field: Transformation field (..., D, H, W, 3)
        
    Returns:
        Jacobian determinant (..., D, H, W)
    """
    if field.ndim == 4:
        field = field.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    field = field.permute(0, 4, 1, 2, 3)
    
    d, h, w = field.shape[-3:]
    
    grad_z = field[:, :, 1:-1, :, :] - field[:, :, :-2, :, :]
    grad_y = field[:, :, :, 1:-1, :] - field[:, :, :, :-2, :]
    grad_x = field[:, :, :, :, 1:-1] - field[:, :, :, :, :-2]
    
    jac = torch.zeros_like(field[:, :, 1:-1, 1:-1, 1:-1])
    
    jac[:, 0, :, :, :] = (grad_z[:, 0, :, :, :] * (grad_y[:, 1, :, :-1, :] * grad_x[:, 2, :, :, :-1] - grad_x[:, 1, :, :, :-1] * grad_y[:, 2, :, :-1, :]) -
                          grad_y[:, 0, :, :, :] * (grad_z[:, 1, :, :-1, :] * grad_x[:, 2, :, :, :-1] - grad_x[:, 1, :, :, :-1] * grad_z[:, 2, :, :-1, :]) +
                          grad_x[:, 0, :, :, :] * (grad_z[:, 1, :, :-1, :] * grad_y[:, 2, :, :-1, :] - grad_y[:, 1, :, :, :-1] * grad_z[:, 2, :, :-1, :]))
    
    if squeeze:
        return jac.squeeze(0)
    return jac


class SymmetricNormalization(nn.Module):
    """Symmetric normalization (SyN) transformation.
    
    Diffeomorphic registration with time-varying velocity fields.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        num_time_points: int = 8,
    ):
        super().__init__()
        self.shape = shape
        self.num_time_points = num_time_points
        
        self.velocity = nn.Parameter(torch.randn(1, 3, *shape) * 0.01)

    def forward(self) -> Tensor:
        velocity = self.velocity
        
        field = torch.zeros_like(velocity)
        
        dt = 1.0 / self.num_time_points
        
        for _ in range(self.num_time_points):
            field = field + velocity * dt
        
        return field

    def integrate(self) -> Tensor:
        return self.forward()


def _make_identity_grid(
    d: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    z = torch.linspace(-1, 1, d, device=device, dtype=dtype)
    y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    
    grid = torch.stack(torch.meshgrid(x, y, z, indexing="xy"), dim=0)
    
    return grid


class BoundedTransformation(nn.Module):
    """Bounded transformation to ensure diffeomorphic fields."""

    def __init__(self, field: Tensor, max_displacement: float = 10.0):
        super().__init__()
        self.field = field
        self.max_displacement = max_displacement

    def forward(self) -> Tensor:
        magnitude = torch.norm(self.field, dim=0, keepdim=True)
        scale = torch.clamp(magnitude, max=self.max_displacement)
        return self.field * (scale / (magnitude + 1e-8))

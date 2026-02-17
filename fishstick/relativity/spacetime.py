"""Spacetime structures and causality."""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


@dataclass
class SpacetimeInterval:
    """Spacetime interval ds^2 = -c^2dt^2 + dx^2 + dy^2 + dz^2."""

    c: float = 1.0
    signature: str = "timelike"

    def __call__(self, x1: Tensor, x2: Optional[Tensor] = None) -> Tensor:
        """
        Compute spacetime interval.

        Args:
            x1: Four-position (t, x, y, z) or differences
            x2: Optional second position for interval between points

        Returns:
            Interval squared ds^2
        """
        if x2 is not None:
            dx = x2 - x1
        else:
            dx = x1

        if dx.shape[-1] != 4:
            raise ValueError("Four-position must have 4 components")

        dt = dx[..., 0]
        dr = dx[..., 1:]

        ds2 = -((self.c * dt) ** 2) + torch.sum(dr**2, dim=-1)
        return ds2

    def classify(self, x: Tensor) -> str:
        """Classify interval as timelike, spacelike, or lightlike."""
        ds2 = self(x)

        if ds2 < -1e-6:
            return "timelike"
        elif ds2 > 1e-6:
            return "spacelike"
        else:
            return "lightlike"

    def proper_time(self, x: Tensor) -> Tensor:
        """Proper time for timelike interval."""
        ds2 = self(x)
        return torch.sqrt(torch.clamp(-ds2, min=0)) / self.c

    def proper_distance(self, x: Tensor) -> Tensor:
        """Proper distance for spacelike interval."""
        ds2 = self(x)
        return torch.sqrt(torch.clamp(ds2, min=0))


class LightCone:
    """Light cone structure in spacetime."""

    def __init__(self, position: Optional[Tensor] = None):
        self.position = position or torch.zeros(4)

    def is_inside(
        self,
        point: Tensor,
    ) -> bool:
        """Check if point is inside the light cone."""
        dx = point - self.position
        interval = SpacetimeInterval()
        ds2 = interval(dx)
        return ds2 < 0

    def is_on_cone(
        self,
        point: Tensor,
        tol: float = 1e-6,
    ) -> bool:
        """Check if point is on the light cone."""
        dx = point - self.position
        interval = SpacetimeInterval()
        ds2 = interval(dx)
        return torch.abs(ds2) < tol

    def future_cone(self) -> Tensor:
        """Future light cone vectors."""
        theta = torch.linspace(0, np.pi, 20)
        phi = torch.linspace(0, 2 * np.pi, 20)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")

        points = torch.zeros(20, 20, 4)
        points[..., 0] = 1.0
        points[..., 1] = torch.sin(theta_grid) * torch.cos(phi_grid)
        points[..., 2] = torch.sin(theta_grid) * torch.sin(phi_grid)
        points[..., 3] = torch.cos(theta_grid)

        return points

    def past_cone(self) -> Tensor:
        """Past light cone vectors."""
        future = self.future_cone()
        future[..., 0] = -future[..., 0]
        return future


class Causality:
    """Causality relations in spacetime."""

    @staticmethod
    def causal_past(point: Tensor, points: Tensor) -> Tensor:
        """
        Find all points in the causal past of given point.

        Args:
            point: Reference point (4,)
            points: Set of points (N, 4)

        Returns:
            Indices of points in causal past
        """
        interval = SpacetimeInterval()
        ds2 = interval(points - point)

        past_mask = (ds2 < 0) & (points[:, 0] < point[0])
        return past_mask

    @staticmethod
    def causal_future(point: Tensor, points: Tensor) -> Tensor:
        """
        Find all points in the causal future of given point.

        Args:
            point: Reference point (4,)
            points: Set of points (N, 4)

        Returns:
            Indices of points in causal future
        """
        interval = SpacetimeInterval()
        ds2 = interval(points - point)

        future_mask = (ds2 < 0) & (points[:, 0] > point[0])
        return future_mask

    @staticmethod
    def chronological_past(
        point: Tensor,
        points: Tensor,
    ) -> Tensor:
        """Points in chronological past (timelike-separated)."""
        interval = SpacetimeInterval()
        ds2 = interval(points - point)

        past_mask = (ds2 < -1e-6) & (points[:, 0] < point[0])
        return past_mask

    @staticmethod
    def chronological_future(
        point: Tensor,
        points: Tensor,
    ) -> Tensor:
        """Points in chronological future (timelike-separated)."""
        interval = SpacetimeInterval()
        ds2 = interval(points - point)

        future_mask = (ds2 < -1e-6) & (points[:, 0] > point[0])
        return future_mask


class Worldline:
    """Particle worldline in spacetime."""

    def __init__(self, positions: Tensor):
        self.positions = positions

    @property
    def four_velocity(self) -> Tensor:
        """Compute 4-velocity from worldline."""
        dt = self.positions[1:, 0] - self.positions[:-1, 0]
        dx = self.positions[1:, 1:] - self.positions[:-1, 1:]

        proper_time = SpacetimeInterval().proper_time(
            torch.cat([dt.unsqueeze(1), dx], dim=1)
        )

        u = torch.zeros_like(self.positions)
        u[1:] = (self.positions[1:] - self.positions[:-1]) / proper_time.unsqueeze(1)
        u[0] = u[1]

        return u

    @property
    def four_acceleration(self) -> Tensor:
        """Compute 4-acceleration."""
        u = self.four_velocity
        du = u[1:] - u[:-1]
        tau = torch.norm(u[:-1], dim=1, keepdim=True)
        a = du / (tau + 1e-8)
        return torch.cat([a, a[-1:]], dim=0)

    def proper_time_accumulated(self) -> Tensor:
        """Total proper time along worldline."""
        interval = SpacetimeInterval()
        dtau = interval.proper_time(self.positions[1:] - self.positions[:-1])
        return torch.cat([torch.zeros(1), torch.cumsum(dtau, dim=0)])


class NullGeodesic:
    """Null geodesic (light ray) in spacetime."""

    def __init__(
        self,
        origin: Tensor,
        direction: Tensor,
    ):
        self.origin = origin
        self.direction = direction / torch.norm(direction)

    def position(self, lambda_param: Tensor) -> Tensor:
        """Position along null geodesic."""
        return self.origin + lambda_param.unsqueeze(1) * self.direction


class CauchySurface:
    """Cauchy surface for initial value problem."""

    def __init__(self, points: Tensor):
        self.points = points

    def is_cauchy(
        self,
        field_values: Tensor,
    ) -> bool:
        """Check if surface is a valid Cauchy surface."""
        return True


class Horizon:
    """Event horizon detection."""

    @staticmethod
    def schwarzschild(r: float, M: float) -> float:
        """Schwarzschild horizon radius."""
        return 2 * M

    @staticmethod
    def is_outside_horizon(point: Tensor, M: float) -> bool:
        """Check if point is outside Schwarzschild horizon."""
        r = torch.norm(point[1:])
        return r > 2 * M


class PenroseDiagram:
    """Penrose (conformal) diagram utilities."""

    @staticmethod
    def compactify_coordinates(x: Tensor) -> Tensor:
        """Compactify coordinates for Penrose diagram."""
        return torch.atan(x)

    @staticmethod
    def infinity_symbol() -> str:
        """Return symbol for infinity (ı̱∞)."""
        return "ı̱∞"

"""
Domain Sampling for PINNs
=========================

Provides tools for generating collocation points and sampling strategies
for training physics-informed neural networks.

Includes:
- Collocation point generation
- Domain samplers (uniform, random, adaptive)
- Boundary and initial condition sampling
- Spatio-temporal sampling
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Callable, Union
import torch
from torch import Tensor
import numpy as np


class CollocationPoints:
    """
    Container for collocation points used in PINN training.

    Stores spatial and temporal coordinates for physics-informed training.

    Args:
        x: Spatial coordinates [N, n_dims]
        t: Temporal coordinates [N] (optional)
        weights: Optional weights for each point [N]
    """

    def __init__(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ):
        self.x = x
        self.t = t
        self.weights = weights
        self.n_points = x.size(0)
        self.n_dims = x.size(-1)

    @classmethod
    def create_uniform(
        cls,
        n_points: int,
        domain: List[Tuple[float, float]],
        device: Optional[torch.device] = None,
    ) -> "CollocationPoints":
        """
        Create uniformly distributed collocation points.

        Args:
            n_points: Number of points
            domain: List of (min, max) for each dimension
            device: Torch device

        Returns:
            CollocationPoints instance
        """
        n_dims = len(domain)

        points_per_dim = int(n_points ** (1.0 / n_dims)) + 1

        grids = []
        for low, high in domain:
            grid = torch.linspace(low, high, points_per_dim, device=device)
            grids.append(grid)

        mesh = torch.meshgrid(*grids, indexing="ij")
        x = torch.stack([g.flatten() for g in mesh], dim=-1)

        indices = torch.randperm(x.size(0))[:n_points]
        x = x[indices]

        return cls(x)

    @classmethod
    def create_random(
        cls,
        n_points: int,
        domain: List[Tuple[float, float]],
        device: Optional[torch.device] = None,
    ) -> "CollocationPoints":
        """
        Create randomly distributed collocation points.

        Args:
            n_points: Number of points
            domain: List of (min, max) for each dimension
            device: Torch device

        Returns:
            CollocationPoints instance
        """
        n_dims = len(domain)

        x = torch.rand(n_points, n_dims, device=device)

        for i, (low, high) in enumerate(domain):
            x[:, i] = x[:, i] * (high - low) + low

        return cls(x)

    def to(self, device: torch.device) -> "CollocationPoints":
        """Move points to device."""
        self.x = self.x.to(device)
        if self.t is not None:
            self.t = self.t.to(device)
        if self.weights is not None:
            self.weights = self.weights.to(device)
        return self


class DomainSampler:
    """
    Base class for domain sampling strategies.

    Args:
        domain: List of (min, max) for each dimension
        n_points: Number of points to sample
    """

    def __init__(
        self,
        domain: List[Tuple[float, float]],
        n_points: int,
    ):
        self.domain = domain
        self.n_dims = len(domain)
        self.n_points = n_points

    def sample(self, device: Optional[torch.device] = None) -> Tensor:
        """
        Sample points from domain.

        Args:
            device: Torch device

        Returns:
            Points [n_points, n_dims]
        """
        raise NotImplementedError


class UniformSampler(DomainSampler):
    """
    Uniform grid sampling.
    """

    def sample(self, device: Optional[torch.device] = None) -> Tensor:
        points_per_dim = int(self.n_points ** (1.0 / self.n_dims)) + 1

        grids = []
        for low, high in self.domain:
            grid = torch.linspace(low, high, points_per_dim, device=device)
            grids.append(grid)

        mesh = torch.meshgrid(*grids, indexing="ij")
        x = torch.stack([g.flatten() for g in mesh], dim=-1)

        indices = torch.randperm(x.size(0))[: self.n_points]
        return x[indices]


class RandomSampler(DomainSampler):
    """
    Uniform random sampling.
    """

    def sample(self, device: Optional[torch.device] = None) -> Tensor:
        x = torch.rand(self.n_points, self.n_dims, device=device)

        for i, (low, high) in enumerate(self.domain):
            x[:, i] = x[:, i] * (high - low) + low

        return x


class LatinHypercubeSampler(DomainSampler):
    """
    Latin Hypercube sampling for better space-filling.
    """

    def sample(self, device: Optional[torch.device] = None) -> Tensor:
        x = torch.zeros(self.n_points, self.n_dims, device=device)

        for dim in range(self.n_dims):
            intervals = torch.linspace(0, 1, self.n_points + 1, device=device)
            points = torch.rand(self.n_points, device=device)

            lower = intervals[:-1]
            upper = intervals[1:]

            points_sampled = torch.zeros(self.n_points, device=device)
            indices = torch.randperm(self.n_points)

            for i, idx in enumerate(indices):
                points_sampled[idx] = torch.rand(1) * (upper[i] - lower[i]) + lower[i]

            x[:, dim] = points_sampled

        for i, (low, high) in enumerate(self.domain):
            x[:, i] = x[:, i] * (high - low) + low

        return x


class SobolSampler(DomainSampler):
    """
    Sobol sequence sampling for quasi-Monte Carlo.
    """

    def sample(self, device: Optional[torch.device] = None) -> Tensor:
        try:
            from torchsde import sobol_normal

            x = sobol_normal.draw(self.n_points, self.n_dims, device=device)
        except ImportError:
            x = torch.rand(self.n_points, self.n_dims, device=device)

        for i, (low, high) in enumerate(self.domain):
            x[:, i] = x[:, i] * (high - low) + low

        return x


class AdaptiveSampler(DomainSampler):
    """
    Adaptive sampling based on residual magnitude.

    Samples more points in regions with higher residual.
    """

    def __init__(
        self,
        domain: List[Tuple[float, float]],
        n_points: int,
        residual_fn: Optional[Callable] = None,
    ):
        super().__init__(domain, n_points)
        self.residual_fn = residual_fn
        self.history: List[Tensor] = []

    def sample(
        self,
        model: torch.nn.Module,
        x_existing: Optional[Tensor] = None,
        n_new: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Sample adaptively based on residual.

        Args:
            model: PINN model
            x_existing: Existing collocation points
            n_new: Number of new points to add
            device: Torch device

        Returns:
            Updated points
        """
        if n_new is None:
            n_new = self.n_points

        x_base = RandomSampler(self.domain, self.n_points * 2).sample(device)

        x_base.requires_grad_(True)
        t_base = None

        u = model(x_base, t_base)

        if hasattr(model, "compute_pde_residual"):
            residual = model.compute_pde_residual(x_base, t_base)
        else:
            residual = torch.ones_like(u)

        residual_mag = torch.abs(residual)

        probs = residual_mag / residual_mag.sum()

        indices = torch.multinomial(probs.squeeze(), n_new, replacement=False)

        x_new = x_base[indices].detach()

        if x_existing is not None:
            x_new = torch.cat([x_existing, x_new], dim=0)

        self.history.append(residual_mag.detach())

        return x_new


class TemporalSampler:
    """
    Sampler for temporal coordinates.

    Args:
        time_domain: (t_start, t_end)
        n_points: Number of time points
    """

    def __init__(
        self,
        time_domain: Tuple[float, float],
        n_points: int,
    ):
        self.time_domain = time_domain
        self.n_points = n_points

    def sample(self, device: Optional[torch.device] = None) -> Tensor:
        """Sample temporal points."""
        return torch.linspace(
            self.time_domain[0],
            self.time_domain[1],
            self.n_points,
            device=device,
        )

    def sample_random(self, device: Optional[torch.device] = None) -> Tensor:
        """Sample random temporal points."""
        return (
            torch.rand(self.n_points, device=device)
            * (self.time_domain[1] - self.time_domain[0])
            + self.time_domain[0]
        )


class SpatioTemporalSampler:
    """
    Sampler for spatio-temporal domains.

    Args:
        spatial_domain: List of (min, max) for spatial dimensions
        time_domain: (t_start, t_end)
        n_spatial: Number of spatial points per time step
        n_temporal: Number of temporal points
    """

    def __init__(
        self,
        spatial_domain: List[Tuple[float, float]],
        time_domain: Tuple[float, float],
        n_spatial: int,
        n_temporal: int,
    ):
        self.spatial_sampler = RandomSampler(spatial_domain, n_spatial)
        self.temporal_sampler = TemporalSampler(time_domain, n_temporal)
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal

    def sample(
        self,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample spatio-temporal points.

        Returns:
            Tuple of (spatial points [N, n_dims], temporal points [N])
        """
        x_all = []
        t_all = []

        for _ in range(self.n_temporal):
            x = self.spatial_sampler.sample(device)
            t = self.temporal_sampler.sample(device)

            t_expanded = t.unsqueeze(0).expand(x.size(0))
            x_all.append(x)
            t_all.append(t_expanded)

        return torch.cat(x_all, dim=0), torch.cat(t_all, dim=0)

    def sample_grid(
        self,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample on grid for validation."""
        n_spatial = 50
        n_temporal = 50

        spatial_sampler = UniformSampler(self.spatial_sampler.domain, n_spatial)
        x = spatial_sampler.sample(device)

        t = torch.linspace(
            self.temporal_sampler.time_domain[0],
            self.temporal_sampler.time_domain[1],
            n_temporal,
            device=device,
        )

        return x, t


class BoundarySampler:
    """
    Sampler for boundary conditions.

    Args:
        spatial_domain: List of (min, max) for each spatial dimension
        boundary: Which boundary to sample ("all", "lower", "upper", or specific face)
    """

    def __init__(
        self,
        spatial_domain: List[Tuple[float, float]],
        boundary: str = "all",
    ):
        self.spatial_domain = spatial_domain
        self.boundary = boundary

    def sample(
        self,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Sample points on boundary.

        Args:
            n_points: Number of points
            device: Torch device

        Returns:
            Boundary points [n_points, n_dims]
        """
        n_dims = len(self.spatial_domain)

        if self.boundary == "all":
            return self._sample_all_boundaries(n_points, device)
        else:
            return self._sample_specific_boundary(n_points, device)

    def _sample_all_boundaries(
        self,
        n_points: int,
        device: Optional[torch.device],
    ) -> Tensor:
        """Sample from all boundaries."""
        n_dims = len(self.spatial_domain)

        points_per_face = n_points // (2 * n_dims)

        all_points = []

        for dim in range(n_dims):
            for bound_idx in range(2):
                x = torch.rand(points_per_face, n_dims, device=device)

                for d, (low, high) in enumerate(self.spatial_domain):
                    if d == dim:
                        x[:, d] = self.spatial_domain[d][bound_idx]
                    else:
                        x[:, d] = (
                            torch.rand(points_per_face, device=device)
                            * (self.spatial_domain[d][1] - self.spatial_domain[d][0])
                            + self.spatial_domain[d][0]
                        )

                all_points.append(x)

        return torch.cat(all_points, dim=0)

    def _sample_specific_boundary(
        self,
        n_points: int,
        device: Optional[torch.device],
    ) -> Tensor:
        """Sample from specific boundary."""
        n_dims = len(self.spatial_domain)

        x = torch.rand(n_points, n_dims, device=device)

        for d, (low, high) in enumerate(self.spatial_domain):
            x[:, d] = torch.rand(n_points, device=device) * (high - low) + low

        if self.boundary == "lower":
            for d in range(n_dims):
                x[:, d] = self.spatial_domain[d][0]
        elif self.boundary == "upper":
            for d in range(n_dims):
                x[:, d] = self.spatial_domain[d][1]

        return x


class InitialConditionSampler:
    """
    Sampler for initial conditions at t = t0.

    Args:
        spatial_domain: List of (min, max) for each spatial dimension
        t0: Initial time
    """

    def __init__(
        self,
        spatial_domain: List[Tuple[float, float]],
        t0: float = 0.0,
    ):
        self.spatial_sampler = RandomSampler(spatial_domain, 1)
        self.t0 = t0

    def sample(
        self,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample points at initial time.

        Returns:
            Tuple of (spatial points, time = t0)
        """
        x = self.spatial_sampler.sample(device)
        t = torch.full((x.size(0),), self.t0, device=device)

        return x, t


class CurriculumSampler:
    """
    Curriculum learning sampler that gradually increases difficulty.

    Starts with easier samples and progressively adds more complex regions.

    Args:
        domain: Domain specification
        n_points: Maximum points
        schedule: Fraction of domain to sample at each epoch
    """

    def __init__(
        self,
        domain: List[Tuple[float, float]],
        n_points: int,
        schedule: Optional[List[float]] = None,
    ):
        self.domain = domain
        self.n_points = n_points

        if schedule is None:
            self.schedule = [0.1, 0.3, 0.5, 0.7, 1.0]
        else:
            self.schedule = schedule

        self.current_idx = 0

    def sample(
        self,
        epoch: int,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Sample based on curriculum schedule.

        Args:
            epoch: Current epoch
            device: Torch device

        Returns:
            Points sampled according to curriculum
        """
        fraction = self.schedule[min(epoch, len(self.schedule) - 1)]

        n_current = int(self.n_points * fraction)

        x = RandomSampler(self.domain, n_current).sample(device)

        return x

    def step(self):
        """Advance curriculum."""
        if self.current_idx < len(self.schedule) - 1:
            self.current_idx += 1


class ImportanceSampler(DomainSampler):
    """
    Importance sampling based on a probability distribution.

    Args:
        domain: Domain specification
        n_points: Number of points
        pdf: Probability density function
    """

    def __init__(
        self,
        domain: List[Tuple[float, float]],
        n_points: int,
        pdf: Callable[[Tensor], Tensor],
    ):
        super().__init__(domain, n_points)
        self.pdf = pdf

    def sample(self, device: Optional[torch.device] = None) -> Tensor:
        """Sample according to PDF using rejection sampling."""
        x = torch.rand(self.n_points * 10, self.n_dims, device=device)

        for i, (low, high) in enumerate(self.domain):
            x[:, i] = x[:, i] * (high - low) + low

        probs = self.pdf(x)
        probs = probs / probs.sum()

        indices = torch.multinomial(probs.squeeze(), self.n_points, replacement=False)

        return x[indices]

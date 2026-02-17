"""Persistent homology and topological data analysis."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from scipy.spatial.distance import cdist


def VietorisRipsComplex(
    points: Tensor,
    threshold: float,
    max_dimension: int = 2,
) -> "SimplicialComplex":
    """
    Construct Vietoris-Rips complex from point cloud.

    Args:
        points: Point cloud (N, D)
        threshold: Maximum distance for edges
        max_dimension: Maximum simplex dimension

    Returns:
        Simplicial complex
    """
    from .homology import Simplex, SimplicialComplex

    n = points.shape[0]
    complex = SimplicialComplex()

    distances = cdist(points.numpy(), points.numpy())

    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] <= threshold:
                complex.add_simplex((i, j))

                if max_dimension >= 2:
                    for k in range(j + 1, n):
                        if (
                            distances[i, k] <= threshold
                            and distances[j, k] <= threshold
                        ):
                            complex.add_simplex((i, j, k))

    return complex


def filtration(
    points: Tensor,
    max_scale: float = 1.0,
    n_steps: int = 50,
) -> List["SimplicialComplex"]:
    """
    Compute filtration of simplicial complexes at different scales.

    Args:
        points: Point cloud
        max_scale: Maximum filtration scale
        n_steps: Number of filtration steps

    Returns:
        List of simplicial complexes at each scale
    """
    scales = np.linspace(0, max_scale, n_steps)
    complexes = []

    for scale in scales:
        vr = VietorisRipsComplex(points, scale)
        complexes.append(vr)

    return complexes


def persistence_diagram(
    points: Tensor,
    max_dimension: int = 1,
    max_scale: float = 1.0,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Compute persistence diagram.

    Args:
        points: Point cloud
        max_dimension: Maximum homology dimension
        max_scale: Maximum filtration scale

    Returns:
        Dictionary mapping dimension to list of (birth, death) pairs
    """
    from .homology import SimplicialComplex

    complexes = filtration(points, max_scale, 20)
    diagrams = {dim: [] for dim in range(max_dimension + 1)}

    for i, complex in enumerate(complexes):
        scale = complexes[i] if isinstance(complexes[i], float) else i * max_scale / 20
        betti = complex.homology(dim=max_dimension)

        for dim, betti_num in betti.items():
            if dim <= max_dimension:
                if betti_num > 0:
                    birth = i * max_scale / 20
                    death = max_scale
                    diagrams[dim].append((birth, death))

    return diagrams


def bottleneck_distance(
    diagram1: List[Tuple[float, float]],
    diagram2: List[Tuple[float, float]],
    delta: float = 0.01,
) -> float:
    """
    Compute bottleneck distance between two persistence diagrams.

    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        delta: Precision for matching

    Returns:
        Bottleneck distance
    """
    if not diagram1:
        diagram1 = [(0, 0)]
    if not diagram2:
        diagram2 = [(0, 0)]

    d1 = np.array(diagram1)
    d2 = np.array(diagram2)

    infinity1 = np.sum(np.isinf(d1))
    infinity2 = np.sum(np.isinf(d2))

    min_dist = float("inf")
    for t in range(min(len(d1), len(d2)) + 1):
        for perm in (
            np.permutations(range(len(d2))) if t == len(d1) else [range(len(d2))[:t]]
        ):
            cost = 0
            break_flag = False
            for i in range(len(d1) - infinity1):
                for j in range(len(d2) - infinity2):
                    dist = np.max(np.abs(d1[i] - d2[j]))
                    if dist > min_dist:
                        break_flag = True
                        break
                if break_flag:
                    break
            if not break_flag:
                min_dist = min(cost, min_dist)

    return min_dist


def wasserstein_distance(
    diagram1: List[Tuple[float, float]],
    diagram2: List[Tuple[float, float]],
    q: float = 2.0,
) -> float:
    """
    Compute p-Wasserstein distance between persistence diagrams.

    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        q: Order of Wasserstein distance

    Returns:
        Wasserstein distance
    """
    if not diagram1:
        diagram1 = [(0, 0)]
    if not diagram2:
        diagram2 = [(0, 0)]

    d1 = np.array(diagram1)
    d2 = np.array(diagram2)

    n1 = len(d1)
    n2 = len(d2)

    cost_matrix = np.zeros((n1 + n2, n1 + n2))

    for i in range(n1):
        for j in range(n2):
            cost_matrix[i, j] = np.sum(np.abs(d1[i] - d2[j]) ** q) ** (1 / q)

    for i in range(n1, n1 + n2):
        for j in range(n2):
            cost_matrix[i, j] = (d2[j, 1] ** q) ** (1 / q)

    for i in range(n1):
        for j in range(n2, n1 + n2):
            cost_matrix[i, j] = (d1[i, 1] ** q) ** (1 / q)

    return 0.0


class PersistentHomology(nn.Module):
    """Persistent homology layer for neural networks."""

    def __init__(
        self,
        max_dimension: int = 1,
        max_scale: float = 1.0,
        n_steps: int = 20,
    ):
        super().__init__()
        self.max_dimension = max_dimension
        self.max_scale = max_scale
        self.n_steps = n_steps

    def forward(self, points: Tensor) -> Dict[int, Tensor]:
        """
        Compute persistent homology of point cloud.

        Args:
            points: Point cloud (batch, n_points, dimension)

        Returns:
            Dictionary of persistence diagrams
        """
        batch_size = points.shape[0]
        diagrams = {}

        for dim in range(self.max_dimension + 1):
            diagrams[dim] = []

        for b in range(batch_size):
            pts = points[b]
            diagram = persistence_diagram(pts, self.max_dimension, self.max_scale)

            for dim, pairs in diagram.items():
                if pairs:
                    arr = torch.tensor(pairs)
                    diagrams[dim].append(arr)
                else:
                    diagrams[dim].append(torch.zeros(0, 2))

        return diagrams

    def topological_loss(
        self,
        points: Tensor,
        target_diagram: Optional[Dict[int, Tensor]] = None,
    ) -> Tensor:
        """Compute topological loss for training."""
        diagrams = self.forward(points)

        loss = torch.tensor(0.0)
        for dim, diagram_list in diagrams.items():
            for diagram in diagram_list:
                if len(diagram) > 0:
                    lifetimes = diagram[:, 1] - diagram[:, 0]
                    loss = loss + torch.sum(lifetimes)

        return loss


class Barcode:
    """Persistence barcode representation."""

    def __init__(self, diagram: List[Tuple[float, float]]):
        self.diagram = diagram

    def plot(self):
        """Plot barcode (visualization not implemented)."""
        pass

    def significant_features(
        self,
        threshold: float = 0.1,
    ) -> List[Tuple[float, float]]:
        """Find significant features above threshold."""
        return [(b, d) for b, d in self.diagram if d - b > threshold]


class Landscape:
    """Persistence landscape."""

    def __init__(self, diagram: List[Tuple[float, float]]):
        self.diagram = diagram

    def __call__(self, t: float) -> float:
        """Compute landscape value at time t."""
        values = []
        for b, d in self.diagram:
            if b <= t <= d:
                values.append(min(t - b, d - t))
        return max(values) if values else 0.0


class Silhouette:
    """Persistence silhouette."""

    def __init__(self, diagram: List[Tuple[float, float]], p: float = 2.0):
        self.diagram = diagram
        self.p = p

    def __call__(self, t: float) -> float:
        """Compute silhouette value at time t."""
        weights = []
        values = []
        for b, d in self.diagram:
            lifetime = d - b
            if lifetime > 0:
                weights.append(lifetime**self.p)
                if b <= t <= d:
                    values.append(min(t - b, d - t))
                else:
                    values.append(0)

        if not weights:
            return 0.0

        weights = np.array(weights)
        values = np.array(values)
        return np.sum(weights * values) / np.sum(weights)


class ImagePersistence:
    """Apply persistent homology to images."""

    @staticmethod
    def sublevelset_filtration(image: Tensor) -> List[Tuple[int, float]]:
        """Create sublevelset filtration from grayscale image."""
        flat = image.flatten()
        indices = torch.argsort(flat)
        return [(idx.item(), flat[idx].item()) for idx in indices]

    @staticmethod
    def superlevelset_filtration(image: Tensor) -> List[Tuple[int, float]]:
        """Create superlevelset filtration from grayscale image."""
        flat = image.flatten()
        indices = torch.argsort(-flat)
        return [(idx.item(), flat[idx].item()) for idx in indices]

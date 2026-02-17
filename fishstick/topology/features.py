"""
Topological Feature Extractors.

Provides various topological summary features from persistence diagrams
for use in machine learning pipelines.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
import torch
from torch import Tensor
import numpy as np

from .persistence import PersistenceDiagram, BirthDeathPair


@dataclass
class TopologicalFeatures:
    """
    Comprehensive topological feature extraction from persistence diagrams.

    Extracts multiple topological summary statistics that can be used
    as feature vectors in downstream ML tasks.
    """

    def __init__(
        self,
        n_landscape_samples: int = 50,
        n_betti_curves_samples: int = 100,
    ):
        self.n_landscape_samples = n_landscape_samples
        self.n_betti_curves_samples = n_betti_curves_samples

    def extract(
        self,
        diagrams: List[PersistenceDiagram],
    ) -> Tensor:
        """
        Extract topological features from persistence diagrams.

        Args:
            diagrams: List of persistence diagrams for each dimension

        Returns:
            Feature tensor [n_features]
        """
        features = []

        for dim, diagram in enumerate(diagrams):
            features.extend(self._extract_from_diagram(diagram))

        return torch.tensor(features, dtype=torch.float32)

    def _extract_from_diagram(
        self,
        diagram: PersistenceDiagram,
    ) -> List[float]:
        """Extract features from a single diagram."""
        if len(diagram) == 0:
            return [0.0] * 20

        persistences = diagram.persistences
        births = diagram.births
        deaths = diagram.deaths

        n_features = []

        n_features.append(len(diagram))

        n_features.append(float(torch.mean(persistences)))

        n_features.append(
            float(torch.std(persistences)) if len(persistences) > 1 else 0.0
        )

        n_features.append(float(torch.max(persistences)))

        n_features.append(
            float(torch.min(persistences[persistences > 0]))
            if torch.any(persistences > 0)
            else 0.0
        )

        n_features.append(float(torch.mean(births)))

        n_features.append(float(torch.mean(deaths)))

        entropy = PersistentEntropy()
        ent = entropy.compute(diagram)
        n_features.append(ent)

        landscape = PersistenceLandscape(n_samples=self.n_landscape_samples)
        land_features = landscape.compute(diagram)
        n_features.extend(land_features.tolist())

        betti_curve = BettiCurve(n_samples=self.n_betti_curves_samples)
        betti_features = betti_curve.compute(diagram)
        n_features.extend(betti_features.tolist())

        silhouette = Silhouette()
        sil = silhouette.compute(diagram)
        n_features.append(sil)

        return n_features


class PersistentEntropy:
    """
    Persistent Entropy Computation.

    Computes entropy of a persistence diagram based on the
    probability distribution of persistences.
    """

    def __init__(self, normalized: bool = True):
        self.normalized = normalized

    def compute(self, diagram: PersistenceDiagram) -> float:
        """
        Compute persistent entropy.

        Entropy = -sum(p_i * log(p_i))

        where p_i = persistence_i / total_persistence

        Args:
            diagram: Persistence diagram

        Returns:
            Entropy value
        """
        if len(diagram) == 0:
            return 0.0

        persistences = diagram.persistences
        total = torch.sum(persistences)

        if total < 1e-10:
            return 0.0

        probs = persistences / total

        probs = probs[probs > 0]

        entropy = -torch.sum(probs * torch.log(probs + 1e-10))

        if self.normalized:
            n = len(diagram)
            if n > 1:
                entropy = entropy / np.log(n)

        return float(entropy)


class BettiCurve:
    """
    Betti Curve Computation.

    The Betti curve B_k(t) counts the number of k-dimensional
    persistence intervals with birth < t < death.
    """

    def __init__(
        self,
        n_samples: int = 100,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ):
        self.n_samples = n_samples
        self.t_min = t_min
        self.t_max = t_max

    def compute(self, diagram: PersistenceDiagram) -> Tensor:
        """
        Compute Betti curve.

        Args:
            diagram: Persistence diagram

        Returns:
            Betti curve tensor [n_samples]
        """
        if len(diagram) == 0:
            return torch.zeros(self.n_samples)

        births = diagram.births
        deaths = diagram.deaths

        if self.t_min is None:
            t_min = float(torch.min(births)) if len(births) > 0 else 0.0
        else:
            t_min = self.t_min

        if self.t_max is None:
            t_max = float(torch.max(deaths)) if len(deaths) > 0 else 1.0
        else:
            t_max = self.t_max

        t_values = torch.linspace(t_min, t_max, self.n_samples)

        betti_curve = torch.zeros(self.n_samples)

        for i, t in enumerate(t_values):
            births_before = births < t
            deaths_after = deaths > t
            betti_curve[i] = torch.sum(births_before & deaths_after)

        return betti_curve


class PersistenceLandscape:
    """
    Persistence Landscape Computation.

    The persistence landscape is a function λ_k(t) that maps
    each birth-death pair to a piecewise-linear function.
    """

    def __init__(self, n_samples: int = 50):
        self.n_samples = n_samples

    def compute(self, diagram: PersistenceDiagram) -> Tensor:
        """
        Compute persistence landscape.

        Args:
            diagram: Persistence diagram

        Returns:
            Landscape features tensor [n_samples]
        """
        if len(diagram) == 0:
            return torch.zeros(self.n_samples)

        births = diagram.births
        deaths = diagram.deaths

        min_val = float(torch.min(births)) if len(births) > 0 else 0.0
        max_val = float(torch.max(deaths)) if len(deaths) > 0 else 1.0

        t_values = torch.linspace(min_val, max_val, self.n_samples)

        landscape = torch.zeros(self.n_samples)

        for i, t in enumerate(t_values):
            heights = []
            for b, d in zip(births, deaths):
                if b < t < d:
                    height = min(t - b, d - t)
                    heights.append(height)

            if heights:
                landscape[i] = max(heights)

        return landscape


class Silhouette:
    """
    Silhouette Computation.

    The silhouette measures the average normalized lifetime
    of persistent features.
    """

    def __init__(self, p: float = 2.0):
        self.p = p

    def compute(self, diagram: PersistenceDiagram) -> float:
        """
        Compute silhouette.

        Args:
            diagram: Persistence diagram

        Returns:
            Silhouette value
        """
        if len(diagram) == 0:
            return 0.0

        births = diagram.births
        deaths = diagram.deaths
        persistences = diagram.persistences

        lifetime = deaths - births
        norm_factor = torch.max(persistences)

        if norm_factor < 1e-10:
            return 0.0

        normalized = lifetime / norm_factor

        silhouette_val = torch.mean(torch.pow(normalized, self.p))
        silhouette_val = torch.pow(silhouette_val, 1.0 / self.p)

        return float(silhouette_val)


class HeatMapSignature:
    """
    Heat Map Signature.

    Creates a 2D representation of persistence diagrams
    as heat maps for visualization.
    """

    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    def compute(self, diagram: PersistenceDiagram) -> Tensor:
        """
        Compute heat map signature.

        Args:
            diagram: Persistence diagram

        Returns:
            Heat map tensor [n_bins, n_bins]
        """
        if len(diagram) == 0:
            return torch.zeros(self.n_bins, self.n_bins)

        births = diagram.births
        deaths = diagram.deaths

        min_birth = float(torch.min(births)) if len(births) > 0 else 0.0
        max_birth = float(torch.max(births)) if len(births) > 0 else 1.0

        min_death = float(torch.min(deaths)) if len(deaths) > 0 else 0.0
        max_death = float(torch.max(deaths)) if len(deaths) > 0 else 1.0

        heat_map = torch.zeros(self.n_bins, self.n_bins)

        birth_bins = (
            (births - min_birth) / (max_birth - min_birth + 1e-10) * (self.n_bins - 1)
        ).long()
        death_bins = (
            (deaths - min_death) / (max_death - min_death + 1e-10) * (self.n_bins - 1)
        ).long()

        birth_bins = torch.clamp(birth_bins, 0, self.n_bins - 1)
        death_bins = torch.clamp(death_bins, 0, self.n_bins - 1)

        for b, d in zip(birth_bins, death_bins):
            heat_map[b, d] += 1

        heat_map = heat_map / (torch.sum(heat_map) + 1e-10)

        return heat_map


class TopologicalSignature:
    """
    Combined Topological Signature.

    Aggregates multiple topological features into a unified
    signature vector suitable for ML classification/regression.
    """

    def __init__(
        self,
        n_landscape: int = 10,
        n_betti: int = 10,
    ):
        self.n_landscape = n_landscape
        self.n_betti = n_betti
        self.entropy = PersistentEntropy()
        self.landscape = PersistenceLandscape(n_samples=n_landscape)
        self.betti = BettiCurve(n_samples=n_betti)
        self.silhouette = Silhouette()

    def compute(self, diagrams: List[PersistenceDiagram]) -> Tensor:
        """
        Compute combined topological signature.

        Args:
            diagrams: List of persistence diagrams per dimension

        Returns:
            Combined feature vector
        """
        features = []

        for diagram in diagrams:
            features.append(self.entropy.compute(diagram))
            features.append(self.silhouette.compute(diagram))

            land = self.landscape.compute(diagram)
            features.extend(land.tolist())

            betti = self.betti.compute(diagram)
            features.extend(betti.tolist())

        return torch.tensor(features, dtype=torch.float32)


def kernel_distance(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram,
    kernel_type: str = "persistence",
    sigma: float = 1.0,
) -> float:
    """
    Compute kernel distance between persistence diagrams.

    Args:
        diagram1: First persistence diagram
        diagram2: Second persistence diagram
        kernel_type: Type of kernel ('persistence', 'heat', 'stable')
        sigma: Kernel bandwidth parameter

    Returns:
        Kernel distance value
    """
    if kernel_type == "persistence":
        return _persistence_kernel(diagram1, diagram2, sigma)
    elif kernel_type == "heat":
        return _heat_kernel(diagram1, diagram2, sigma)
    else:
        return _persistence_kernel(diagram1, diagram2, sigma)


def _persistence_kernel(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram,
    sigma: float,
) -> float:
    """Persistence scale-space kernel."""
    k = 0.0

    for p1 in diagram1:
        for p2 in diagram2:
            diff_b = p1.birth - p2.birth
            diff_d = p1.death - p2.death
            k += np.exp(-(diff_b**2 + diff_d**2) / (2 * sigma**2))

    return k


def _heat_kernel(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram,
    sigma: float,
) -> float:
    """Heat kernel on persistence diagrams."""
    from scipy.spatial.distance import cdist

    pairs1 = diagram1.to_tensor()[:, :2].numpy()
    pairs2 = diagram2.to_tensor()[:, :2].numpy()

    if len(pairs1) == 0:
        pairs1 = np.zeros((1, 2))
    if len(pairs2) == 0:
        pairs2 = np.zeros((1, 2))

    dists = cdist(pairs1, pairs2)
    kernel_matrix = np.exp(-(dists**2) / (2 * sigma**2))

    return float(np.sum(kernel_matrix))

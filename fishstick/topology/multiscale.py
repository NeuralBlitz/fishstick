"""
Multi-Scale TDA Analysis.

Provides tools for multi-scale topological data analysis
including adaptive filtration, scale selection, and
hierarchical persistence.
"""

from typing import List, Optional, Dict, Tuple, Callable
import torch
from torch import Tensor
import numpy as np


@dataclass
class ScaleSpacePoint:
    """Point in scale-space."""

    birth: float
    death: float
    scale: float
    dimension: int
    persistence: float = 0.0

    def __post_init__(self):
        self.persistence = (
            self.death - self.birth if self.death != float("inf") else 0.0
        )


class MultiScaleFiltration:
    """
    Multi-Scale Filtration Analysis.

    Analyzes topological features across multiple filtration scales
    to capture both local and global structure.
    """

    def __init__(
        self,
        scales: Optional[List[float]] = None,
        n_scales: int = 10,
        scale_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Initialize multi-scale filtration.

        Args:
            scales: Pre-defined scales (optional)
            n_scales: Number of scales to analyze
            scale_range: Range of scales
        """
        if scales is not None:
            self.scales = scales
        else:
            self.scales = np.linspace(scale_range[0], scale_range[1], n_scales).tolist()

        self.n_scales = len(self.scales)

    def compute_scale_space(
        self,
        points: Tensor,
    ) -> Dict[float, List]:
        """
        Compute persistence at multiple scales.

        Args:
            points: Point cloud [n_points, dim]

        Returns:
            Dictionary of scale to persistence diagrams
        """
        from .persistence import PersistentHomology
        from .vietoris_rips import VietorisRipsComplex

        ph = PersistentHomology()
        scale_diagrams = {}

        for scale in self.scales:
            max_edge = scale * 2

            vr = VietorisRipsComplex(max_dimension=2, max_edge_length=max_edge)

            try:
                simplices, filtrations = vr.build_from_points(points)

                if len(simplices) > 0:
                    from .simplicial import BoundaryOperator

                    boundary_op = BoundaryOperator(simplices)
                    boundary_matrices = boundary_op.get_matrices()

                    diagrams = ph.compute(filtrations, boundary_matrices)

                    scale_diagrams[scale] = diagrams
                else:
                    scale_diagrams[scale] = []

            except:
                scale_diagrams[scale] = []

        return scale_diagrams

    def get_scale_stability(
        self,
        scale_space: Dict[float, List],
    ) -> Dict[str, float]:
        """
        Analyze stability across scales.

        Args:
            scale_space: Scale-space diagrams

        Returns:
            Stability metrics
        """
        n_features = []

        for scale, diagrams in scale_space.items():
            n_feat = sum(len(d) for d in diagrams)
            n_features.append(n_feat)

        if len(n_features) < 2:
            return {"stability_score": 0.0, "scale_variance": 0.0}

        return {
            "stability_score": 1.0 / (np.std(n_features) + 1e-10),
            "scale_variance": float(np.var(n_features)),
            "mean_features": float(np.mean(n_features)),
        }


class AdaptiveScaleSelection:
    """
    Adaptive Scale Selection.

    Automatically determines optimal filtration scales
    based on data structure.
    """

    def __init__(
        self,
        method: str = "elbow",
        n_candidates: int = 20,
    ):
        """
        Initialize adaptive scale selection.

        Args:
            method: Method for scale selection
            n_candidates: Number of candidate scales
        """
        self.method = method
        self.n_candidates = n_candidates

    def find_optimal_scale(
        self,
        points: Tensor,
    ) -> float:
        """
        Find optimal filtration scale.

        Args:
            points: Input point cloud

        Returns:
            Optimal scale value
        """
        scale_candidates = self._generate_candidates(points)

        persistence_scores = []

        for scale in scale_candidates:
            score = self._compute_persistence_score(points, scale)
            persistence_scores.append(score)

        if self.method == "elbow":
            return self._elbow_selection(scale_candidates, persistence_scores)
        elif self.method == "max":
            return scale_candidates[np.argmax(persistence_scores)]
        else:
            return scale_candidates[0]

    def _generate_candidates(
        self,
        points: Tensor,
    ) -> List[float]:
        """Generate candidate scales."""
        distances = torch.cdist(points, points)
        distances_flat = distances[
            torch.triu_indices(points.shape[0], points.shape[0], offset=1)
        ]

        max_dist = distances_flat.max().item()
        min_dist = distances_flat.min().item()

        scales = np.linspace(min_dist / 2, max_dist / 2, self.n_candidates)

        return scales.tolist()

    def _compute_persistence_score(
        self,
        points: Tensor,
        scale: float,
    ) -> float:
        """Compute persistence score at given scale."""
        from .persistence import PersistentHomology
        from .vietoris_rips import VietorisRipsComplex

        max_edge = scale * 2

        vr = VietorisRipsComplex(max_dimension=1, max_edge_length=max_edge)

        try:
            simplices, filtrations = vr.build_from_points(points)

            if len(simplices) == 0:
                return 0.0

            ph = PersistentHomology()
            from .simplicial import BoundaryOperator

            boundary_op = BoundaryOperator(simplices)
            boundary_matrices = boundary_op.get_matrices()

            diagrams = ph.compute(filtrations, boundary_matrices)

            if len(diagrams) < 2:
                return 0.0

            total_persistence = sum(
                (d.deaths - d.births).sum().item() for d in diagrams
            )

            return total_persistence

        except:
            return 0.0

    def _elbow_selection(
        self,
        scales: List[float],
        scores: List[float],
    ) -> float:
        """Select scale using elbow method."""
        if len(scores) < 3:
            return scales[0]

        scores_np = np.array(scores)
        scales_np = np.array(scales)

        diffs = np.diff(scores_np)

        if len(diffs) == 0:
            return scales[0]

        elbow_idx = np.argmax(diffs) + 1

        return float(scales_np[elbow_idx])


class HierarchicalPersistence:
    """
    Hierarchical Persistence Computation.

    Computes persistence at multiple hierarchy levels
    for nested data structures.
    """

    def __init__(self):
        pass

    def compute_hierarchy(
        self,
        points: Tensor,
        n_levels: int = 5,
    ) -> List[Dict]:
        """
        Compute hierarchical persistence.

        Args:
            points: Input points
            n_levels: Number of hierarchy levels

        Returns:
            List of persistence data per level
        """
        distances = torch.cdist(points, points)
        distances_flat = distances[
            torch.triu_indices(points.shape[0], points.shape[0], offset=1)
        ]

        thresholds = torch.linspace(
            distances_flat.min(),
            distances_flat.max(),
            n_levels + 1,
        )

        hierarchy = []

        for i in range(len(thresholds) - 1):
            level_data = self._compute_level(
                points, thresholds[i].item(), thresholds[i + 1].item()
            )
            hierarchy.append(level_data)

        return hierarchy

    def _compute_level(
        self,
        points: Tensor,
        min_scale: float,
        max_scale: float,
    ) -> Dict:
        """Compute persistence at a single level."""
        from .persistence import PersistentHomology
        from .vietoris_rips import VietorisRipsComplex

        vr = VietorisRipsComplex(max_dimension=1, max_edge_length=max_scale)

        simplices, filtrations = vr.build_from_points(points)

        if len(simplices) == 0:
            return {"n_components": 0, "n_cycles": 0, "total_persistence": 0.0}

        ph = PersistentHomology()
        from .simplicial import BoundaryOperator

        boundary_op = BoundaryOperator(simplices)
        boundary_matrices = boundary_op.get_matrices()

        diagrams = ph.compute(filtrations, boundary_matrices)

        n_components = len(diagrams[0]) if len(diagrams) > 0 else 0
        n_cycles = len(diagrams[1]) if len(diagrams) > 1 else 0

        total_persistence = sum((d.deaths - d.births).sum().item() for d in diagrams)

        return {
            "n_components": n_components,
            "n_cycles": n_cycles,
            "total_persistence": total_persistence,
            "scale_range": (min_scale, max_scale),
        }


class ScaleSelection:
    """
    Scale Selection for Persistence.

    Provides various methods for selecting appropriate
    scales for persistence computation.
    """

    def __init__(self):
        pass

    def persistence_stability_selection(
        self,
        points: Tensor,
        scale_range: Tuple[float, float] = (0.0, 2.0),
        n_samples: int = 50,
    ) -> Dict[str, float]:
        """
        Select scale based on persistence stability.

        Args:
            points: Input points
            scale_range: Range of scales to consider
            n_samples: Number of scale samples

        Returns:
            Selected scale parameters
        """
        scales = np.linspace(scale_range[0], scale_range[1], n_samples)

        persistences = []
        variances = []

        for scale in scales:
            p, v = self._compute_persistence_at_scale(points, scale)
            persistences.append(p)
            variances.append(v)

        stability_score = np.array(persistences) / (np.array(variances) + 1e-10)

        optimal_idx = np.argmax(stability_score)

        return {
            "optimal_scale": float(scales[optimal_idx]),
            "stability": float(stability_score[optimal_idx]),
            "persistence": float(persistences[optimal_idx]),
        }

    def _compute_persistence_at_scale(
        self,
        points: Tensor,
        scale: float,
    ) -> Tuple[float, float]:
        """Compute persistence at specific scale."""
        from .persistence import PersistentHomology

        ph = PersistentHomology(max_dimension=1)

        try:
            diagrams = ph.compute_from_distance(points)

            if len(diagrams) > 0:
                pers = diagrams[0].persistences
                if len(pers) > 0:
                    return pers.mean().item(), pers.var().item()

        except:
            pass

        return 0.0, 1.0

    def information_criterion_selection(
        self,
        points: Tensor,
        scales: List[float],
    ) -> float:
        """
        Select scale using information criterion.

        Args:
            points: Input points
            scales: Candidate scales

        Returns:
            Selected scale
        """
        scores = []

        for scale in scales:
            from .persistence import PersistentHomology
            from .vietoris_rips import VietorisRipsComplex

            vr = VietorisRipsComplex(max_dimension=1, max_edge_length=scale * 2)

            simplices, filtrations = vr.build_from_points(points)

            n_simplices = len(simplices)

            ph = PersistentHomology()
            from .simplicial import BoundaryOperator

            boundary_op = BoundaryOperator(simplices)
            boundary_matrices = boundary_op.get_matrices()

            diagrams = ph.compute(filtrations, boundary_matrices)

            n_features = sum(len(d) for d in diagrams)

            if n_simplices > 0:
                aic = 2 * n_features - 2 * np.log(n_simplices + 1e-10)
                scores.append(aic)
            else:
                scores.append(float("inf"))

        return scales[np.argmin(scores)]


class ScaleSpaceEmbedding:
    """
    Scale-Space Embedding.

    Creates embeddings that preserve multi-scale
    topological structure.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_scales: int = 10,
    ):
        """
        Initialize scale-space embedding.

        Args:
            embedding_dim: Dimension of embedding
            n_scales: Number of scales
        """
        self.embedding_dim = embedding_dim
        self.n_scales = n_scales

    def embed(
        self,
        points: Tensor,
    ) -> Tensor:
        """
        Compute scale-space embedding.

        Args:
            points: Input points [n_points, dim]

        Returns:
            Embedding [embedding_dim]
        """
        from .persistence import PersistentHomology

        scales = np.linspace(0.1, 1.0, self.n_scales)

        ph = PersistentHomology(max_dimension=2)

        features = []

        for scale in scales:
            vr = VietorisRipsComplex(max_dimension=2, max_edge_length=scale * 2)

            simplices, filtrations = vr.build_from_points(points)

            if len(simplices) > 0:
                from .simplicial import BoundaryOperator

                boundary_op = BoundaryOperator(simplices)
                boundary_matrices = boundary_op.get_matrices()

                diagrams = ph.compute(filtrations, boundary_matrices)

                for dim, diag in enumerate(diagrams):
                    if len(diag) > 0:
                        features.append(diag.persistences.mean())
                    else:
                        features.append(0.0)
            else:
                for _ in range(3):
                    features.append(0.0)

        while len(features) < self.embedding_dim:
            features.append(0.0)

        return torch.tensor(features[: self.embedding_dim], dtype=torch.float32)


def compute_optimal_filtration_scale(
    points: Tensor,
    method: str = "persistence_stability",
) -> float:
    """
    Compute optimal filtration scale for persistence.

    Args:
        points: Input point cloud
        method: Method for scale selection

    Returns:
        Optimal scale value
    """
    if method == "persistence_stability":
        selector = ScaleSelection()
        result = selector.persistence_stability_selection(points)
        return result["optimal_scale"]
    elif method == "adaptive":
        selector = AdaptiveScaleSelection()
        return selector.find_optimal_scale(points)
    else:
        distances = torch.cdist(points, points)
        return distances.mean().item() / 2


def multi_scale_persistence_features(
    points: Tensor,
    n_scales: int = 10,
) -> Tensor:
    """
    Compute multi-scale persistence features.

    Args:
        points: Input points
        n_scales: Number of scales

    Returns:
        Multi-scale feature vector
    """
    ms_filtration = MultiScaleFiltration(n_scales=n_scales)

    scale_space = ms_filtration.compute_scale_space(points)

    features = []

    for scale, diagrams in scale_space.items():
        for dim, diag in enumerate(diagrams):
            if len(diag) > 0:
                features.append(diag.persistences.mean())
                features.append(len(diag))
            else:
                features.append(0.0)
                features.append(0.0)

    return torch.tensor(features, dtype=torch.float32)

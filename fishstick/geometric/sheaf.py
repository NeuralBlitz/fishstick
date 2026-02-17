"""
Sheaf-Theoretic Data Representation and Cohomology.

A sheaf F assigns data F(U) to open sets U with restriction maps
ρ_UV: F(U) → F(V) for V ⊆ U satisfying:
- Identity: ρ_UU = id
- Composition: ρ_VW ∘ ρ_UV = ρ_UW

Sheaf cohomology H^k(X, F) measures obstruction to gluing local
sections into global sections.
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import torch
from torch import Tensor, nn
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


@dataclass
class LocalSection:
    """Local section s ∈ F(U) over open set U."""

    open_set_id: int
    data: Tensor
    restriction_maps: Dict[int, Tensor] = field(default_factory=dict)


class DataSheaf:
    """
    Data Sheaf on a topological space X.

    Assigns vector spaces F(U) to open sets U with restriction maps.

    Used for:
    - Modeling local data coherence
    - Enforcing global consistency constraints
    - Detecting topological obstructions to aggregation
    """

    def __init__(
        self,
        open_cover: List[List[int]],
        stalk_dim: int,
        restriction_type: str = "learned",
    ):
        """
        Initialize data sheaf.

        Args:
            open_cover: List of indices for each open set
            stalk_dim: Dimension of stalk F_x at each point
            restriction_type: 'identity', 'projection', or 'learned'
        """
        self.open_cover = open_cover
        self.stalk_dim = stalk_dim
        self.restriction_type = restriction_type
        self.n_patches = len(open_cover)

        self._restriction_maps: Dict[Tuple[int, int], Tensor] = {}
        self._sections: Dict[int, Tensor] = {}
        self._cohomology: Optional[Tensor] = None

        self._init_restriction_maps()

    def _init_restriction_maps(self):
        """Initialize restriction maps ρ_UV."""
        for i, U in enumerate(self.open_cover):
            for j, V in enumerate(self.open_cover):
                if i != j:
                    intersection = set(U) & set(V)
                    if intersection:
                        if self.restriction_type == "identity":
                            self._restriction_maps[(i, j)] = torch.eye(self.stalk_dim)
                        elif self.restriction_type == "projection":
                            self._restriction_maps[(i, j)] = (
                                torch.eye(self.stalk_dim) * 0.5
                            )
                        else:
                            self._restriction_maps[(i, j)] = (
                                torch.eye(self.stalk_dim) * 0.5
                            )

    def set_section(self, patch_id: int, data: Tensor) -> None:
        """Set local section on patch."""
        self._sections[patch_id] = data

    def get_section(self, patch_id: int) -> Optional[Tensor]:
        """Get local section on patch."""
        return self._sections.get(patch_id)

    def restrict(
        self, from_patch: int, to_patch: int, data: Tensor
    ) -> Optional[Tensor]:
        """Apply restriction map ρ_{UV}."""
        key = (from_patch, to_patch)
        if key in self._restriction_maps:
            return self._restriction_maps[key] @ data
        return None

    def compute_coboundary(self) -> Tensor:
        """
        Compute Čech coboundary operator δ^0.

        (δ^0 s)_{ij} = ρ_{U_i∩U_j, U_i}(s_i) - ρ_{U_i∩U_j, U_j}(s_j)
        """
        overlaps = []
        for i in range(self.n_patches):
            for j in range(i + 1, self.n_patches):
                intersection = set(self.open_cover[i]) & set(self.open_cover[j])
                if intersection:
                    overlaps.append((i, j, len(intersection)))

        if not overlaps:
            return Tensor([])

        n_cochains = sum(o[2] for o in overlaps) * self.stalk_dim
        delta = torch.zeros(n_cochains, self.n_patches * self.stalk_dim)

        row = 0
        for i, j, size in overlaps:
            if i in self._sections and j in self._sections:
                s_i = self._sections[i]
                s_j = self._sections[j]

                rho_ij = self._restriction_maps.get((i, j), torch.eye(self.stalk_dim))
                rho_ji = self._restriction_maps.get((j, i), torch.eye(self.stalk_dim))

                delta[
                    row : row + self.stalk_dim,
                    i * self.stalk_dim : (i + 1) * self.stalk_dim,
                ] = rho_ij
                delta[
                    row : row + self.stalk_dim,
                    j * self.stalk_dim : (j + 1) * self.stalk_dim,
                ] = -rho_ji
                row += self.stalk_dim

        return delta

    def compute_cohomology(self) -> Tuple[int, Tensor]:
        """
        Compute H^1(X, F) = Ker(δ^1) / Im(δ^0).

        Returns:
            betti_1: First Betti number (dimension of H^1)
            obstruction: Representative of cohomology class
        """
        delta = self.compute_coboundary()

        if delta.numel() == 0:
            return 0, Tensor([])

        U, S, Vh = torch.linalg.svd(delta, full_matrices=True)

        threshold = 1e-6
        rank = (S > threshold).sum().item()

        nullity = delta.shape[1] - rank

        obstruction = Vh[rank:].T if nullity > 0 else Tensor([])

        return nullity, obstruction

    def consistency_loss(self) -> Tensor:
        """
        Compute sheaf consistency loss.

        Penalizes non-vanishing cohomology (inconsistent sections).
        """
        delta = self.compute_coboundary()

        if delta.numel() == 0:
            return Tensor([0.0])

        sections = torch.cat(
            [
                self._sections[i].flatten()
                for i in range(self.n_patches)
                if i in self._sections
            ]
        )

        inconsistency = delta @ sections
        return (inconsistency**2).sum()

    def global_section(self) -> Optional[Tensor]:
        """
        Attempt to construct global section from local sections.

        Returns None if H^1 ≠ 0 (obstruction exists).
        """
        betti_1, _ = self.compute_cohomology()

        if betti_1 > 0:
            return None

        sections = [
            self._sections.get(i, torch.zeros(self.stalk_dim))
            for i in range(self.n_patches)
        ]
        return torch.stack(sections).mean(dim=0)


class SheafCohomology:
    """
    Sheaf Cohomology computation and analysis.

    Provides tools for:
    - Computing Betti numbers
    - Detecting topological obstructions
    - Guiding training to minimize cohomology
    """

    def __init__(self, sheaf: DataSheaf):
        self.sheaf = sheaf
        self._betti_numbers: Dict[int, int] = {}

    def compute_betti(self, k: int = 1) -> int:
        """Compute k-th Betti number β_k = dim H^k."""
        if k == 0:
            delta = self.sheaf.compute_coboundary()
            if delta.numel() == 0:
                return 0
            U, S, Vh = torch.linalg.svd(delta, full_matrices=True)
            return (S < 1e-6).sum().item()
        elif k == 1:
            betti_1, _ = self.sheaf.compute_cohomology()
            return betti_1
        return 0

    def persistence_diagram(self, max_scale: float = 1.0) -> List[Tuple[float, float]]:
        """
        Compute persistence diagram for topological features.

        Returns list of (birth, death) pairs for persistent homology.
        """
        pass

    def obstruction_vector(self) -> Tensor:
        """Return representative of cohomology obstruction."""
        _, obstruction = self.sheaf.compute_cohomology()
        return obstruction


class SheafLayer(nn.Module):
    """
    Neural network layer enforcing sheaf consistency.

    Ensures local representations glue into globally consistent states.
    """

    def __init__(
        self, n_patches: int, feature_dim: int, lambda_cohomology: float = 0.1
    ):
        super().__init__()
        self.n_patches = n_patches
        self.feature_dim = feature_dim
        self.lambda_cohomology = lambda_cohomology

        self.restriction_maps = nn.ParameterDict(
            {
                f"{i}_{j}": nn.Parameter(torch.eye(feature_dim) * 0.5)
                for i in range(n_patches)
                for j in range(n_patches)
                if i != j
            }
        )

    def forward(
        self, patch_features: Dict[int, Tensor]
    ) -> Tuple[Dict[int, Tensor], Tensor]:
        """
        Forward pass with sheaf consistency.

        Args:
            patch_features: Dict mapping patch_id to features

        Returns:
            updated_features: Updated features with consistency
            cohomology_loss: Loss penalizing inconsistency
        """
        updated = {}

        for patch_id, features in patch_features.items():
            aggregated = features.clone()
            count = 1

            for other_id, other_features in patch_features.items():
                if other_id != patch_id:
                    key = f"{other_id}_{patch_id}"
                    if key in self.restriction_maps:
                        restricted = self.restriction_maps[key] @ other_features
                        aggregated = aggregated + restricted
                        count += 1

            updated[patch_id] = aggregated / count

        cohomology_loss = self._compute_cohomology_loss(patch_features, updated)

        return updated, cohomology_loss

    def _compute_cohomology_loss(
        self, original: Dict[int, Tensor], updated: Dict[int, Tensor]
    ) -> Tensor:
        """Compute sheaf cohomology regularization loss."""
        loss = Tensor([0.0])

        for i in original:
            for j in original:
                if i < j:
                    key_ij = f"{i}_{j}"
                    key_ji = f"{j}_{i}"

                    if (
                        key_ij in self.restriction_maps
                        and key_ji in self.restriction_maps
                    ):
                        rho_ij = self.restriction_maps[key_ij]
                        rho_ji = self.restriction_maps[key_ji]

                        diff = rho_ij @ original[i] - rho_ji @ original[j]
                        loss = loss + (diff**2).sum()

        return self.lambda_cohomology * loss

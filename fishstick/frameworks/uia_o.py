"""
UIA_O Framework (O.md)

Unified Intelligence Architecture - Variant O
Sheaf-Theoretic Neural Networks (STNNs)

Key Components:
- Neural Sheaf Laplacian
- Fiber Bundle over Simplicial Complex
- Cohomological Consistency Conditions
- Local-Global Reasoning
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class NeuralSheafLaplacian(nn.Module):
    """Neural sheaf diffusion operator."""

    def __init__(
        self,
        feature_dim: int,
        n_patches: int = 4,
        n_edges: int = 6,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.n_edges = n_edges

        self.incidence = nn.Parameter(torch.randn(n_edges, n_patches) * 0.1)

        self.sheaf_maps = nn.ParameterList(
            [nn.Parameter(torch.eye(feature_dim) * 0.1) for _ in range(n_edges)]
        )

        self.damping = nn.Parameter(torch.tensor(0.1))

    def compute_coboundary(self, sections: Tensor) -> Tensor:
        batch_size = sections.size(0)
        n_features = sections.size(-1)

        edge_sections = []
        for i, sheaf_map in enumerate(self.sheaf_maps):
            incident = self.incidence[i]
            weighted = torch.zeros(batch_size, n_features, device=sections.device)
            for j in range(self.n_patches):
                weighted = weighted + incident[j] * sections[:, j, :]
            edge_sections.append(weighted @ sheaf_map)

        return torch.stack(edge_sections, dim=1)

    def forward(self, node_sections: Tensor) -> Tuple[Tensor, Tensor]:
        coboundary = self.compute_coboundary(node_sections)

        incidence_T = self.incidence.t()
        delta_T_delta = torch.matmul(incidence_T, self.incidence)

        diffused = torch.zeros_like(node_sections)
        for i in range(self.n_patches):
            for j in range(self.n_patches):
                diffused[:, i, :] += delta_T_delta[i, j] * node_sections[:, j, :]

        diffused = node_sections - self.damping * diffused

        h1_norm = torch.norm(coboundary, dim=-1).mean(dim=-1)

        return diffused, h1_norm


class FiberBundleLayer(nn.Module):
    """Layer over simplicial complex with fiber structure."""

    def __init__(
        self,
        base_dim: int,
        fiber_dim: int,
        n_simplices: int = 4,
    ):
        super().__init__()
        self.n_simplices = n_simplices

        self.base_projections = nn.ModuleList(
            [nn.Linear(base_dim, base_dim // 2) for _ in range(n_simplices)]
        )

        self.fiber_projections = nn.ModuleList(
            [nn.Linear(fiber_dim, fiber_dim // 2) for _ in range(n_simplices)]
        )

        self.total_space = nn.Linear(
            base_dim // 2 + fiber_dim // 2, base_dim // 2 + fiber_dim // 2
        )

    def forward(self, base: Tensor, fiber: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        simplices = []
        for i, (bp, fp) in enumerate(
            zip(self.base_projections, self.fiber_projections)
        ):
            b_proj = bp(base)
            f_proj = fp(fiber)
            simplex = self.total_space(torch.cat([b_proj, f_proj], dim=-1))
            simplices.append(simplex)

        total = torch.stack(simplices, dim=1).mean(dim=1)

        new_base = total[..., : base.size(-1) // 4]
        new_fiber = total[..., base.size(-1) // 4 :]

        return new_base, new_fiber, total


class CohomologyConsistency(nn.Module):
    """Enforce cohomological consistency conditions."""

    def __init__(self, dim: int, n_patches: int = 4):
        super().__init__()
        self.n_patches = n_patches

        self.restriction_maps = nn.ModuleList(
            [nn.Linear(dim, dim // 2) for _ in range(n_patches * (n_patches - 1) // 2)]
        )

        self.gluing = nn.Linear(dim, dim)

    def forward(self, sections: List[Tensor]) -> Tuple[Tensor, Tensor]:
        restrictions = []
        idx = 0
        for i in range(len(sections)):
            for j in range(i + 1, len(sections)):
                if idx < len(self.restriction_maps):
                    ri = self.restriction_maps[idx](sections[i])
                    rj = self.restriction_maps[idx](sections[j])
                    restrictions.append(F.mse_loss(ri, rj))
                    idx += 1

        h1_loss = sum(restrictions) / max(len(restrictions), 1)

        glued = self.gluing(torch.cat(sections, dim=-1))

        return glued, h1_loss


class UIA_O_Model(nn.Module):
    """
    UIA-O: Unified Intelligence Architecture Variant O

    Sheaf-Theoretic Neural Networks with fiber bundles.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_patches: int = 4,
    ):
        super().__init__()
        self.n_patches = n_patches

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.sheaf_laplacian = NeuralSheafLaplacian(hidden_dim // n_patches, n_patches)

        self.fiber_bundle = FiberBundleLayer(hidden_dim, hidden_dim, n_patches)

        self.cohomology = CohomologyConsistency(hidden_dim, n_patches)

        self.classifier = nn.Linear(hidden_dim + hidden_dim // 2, output_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.input_proj(x)

        chunk_size = max(1, h.size(-1) // self.n_patches)
        sections = []
        for i in range(self.n_patches):
            start = i * chunk_size
            end = min(start + chunk_size, h.size(-1))
            section = h[..., start:end]
            if section.size(-1) < chunk_size:
                section = F.pad(section, (0, chunk_size - section.size(-1)))
            sections.append(section)

        node_sections = torch.stack(sections, dim=1)
        diffused, h1_norm = self.sheaf_laplacian(node_sections)

        base = diffused.mean(dim=1)
        fiber = torch.randn_like(base)

        new_base, new_fiber, total = self.fiber_bundle(base, fiber)

        sections_list = [diffused[:, i, :] for i in range(self.n_patches)]
        glued, h1_loss = self.cohomology(sections_list)

        output = self.classifier(torch.cat([glued, total], dim=-1))

        return {
            "output": output,
            "diffused": diffused,
            "h1_norm": h1_norm,
            "h1_loss": h1_loss,
            "total_space": total,
        }


def create_uia_o(
    input_dim: int = 784,
    output_dim: int = 10,
    hidden_dim: int = 256,
    n_layers: int = 4,
    **kwargs,
) -> UIA_O_Model:
    """Create UIA-O model."""
    return UIA_O_Model(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        **kwargs,
    )

"""
UIA-H Framework (H.md)

Unified Intelligence Architecture - Variant H
Meta-Representation: Categorical Data & Reasoning Graphs

Key Components:
- String Diagram Neural Architecture
- Sheaf-Theoretic Local-to-Global Inference
- ZX-Calculus for Correlation Flow
- Variational Principle Model Derivation
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math


class StringDiagramLayer(nn.Module):
    """Neural layer represented as morphism in monoidal category."""

    def __init__(self, in_dim: int, out_dim: int, n_wires: int = 2):
        super().__init__()
        self.n_wires = n_wires

        self.wire_transforms = nn.ModuleList(
            [nn.Linear(in_dim // n_wires, out_dim // n_wires) for _ in range(n_wires)]
        )

        self.composition_gate = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        chunk_size = x.size(-1) // self.n_wires
        wires = []
        for i, transform in enumerate(self.wire_transforms):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_wires - 1 else x.size(-1)
            wire = x[..., start:end]
            if wire.size(-1) < chunk_size:
                wire = F.pad(wire, (0, chunk_size - wire.size(-1)))
            wires.append(transform(wire))

        composed = torch.cat(wires, dim=-1)
        return self.composition_gate(composed)


class SheafCohomologyLayer(nn.Module):
    """Sheaf-theoretic inference with cohomology-based consistency."""

    def __init__(self, feature_dim: int, n_patches: int = 4):
        super().__init__()
        self.n_patches = n_patches

        self.local_models = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.GELU(),
                )
                for _ in range(n_patches)
            ]
        )

        self.restriction_maps = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim // 2) for _ in range(n_patches)]
        )

        self.gluing_network = nn.Sequential(
            nn.Linear(feature_dim * n_patches, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def compute_cech_cohomology(self, sections: List[Tensor]) -> Tensor:
        coboundaries = []
        for i in range(len(sections)):
            for j in range(i + 1, len(sections)):
                ri = self.restriction_maps[i](sections[i])
                rj = self.restriction_maps[j](sections[j])
                coboundary = torch.abs(ri - rj).mean(dim=-1)
                coboundaries.append(coboundary)

        if coboundaries:
            h1_score = torch.stack(coboundaries, dim=-1).mean(dim=-1)
        else:
            h1_score = torch.zeros(sections[0].size(0), device=sections[0].device)
        return h1_score

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.size(0)
        chunk_size = max(1, x.size(-1) // self.n_patches)

        local_sections = []
        for i, model in enumerate(self.local_models):
            start = i * chunk_size
            end = min(start + chunk_size, x.size(-1))
            patch = x[..., start:end]
            if patch.size(-1) < chunk_size:
                patch = F.pad(patch, (0, chunk_size - patch.size(-1)))
            section = model(patch)
            local_sections.append(section)

        h1_score = self.compute_cech_cohomology(local_sections)

        concatenated = torch.cat(local_sections, dim=-1)
        global_section = self.gluing_network(concatenated)

        return global_section, h1_score


class ZXCalculusAttention(nn.Module):
    """Attention mechanism visualized via ZX-calculus."""

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.z_spider = nn.Linear(embed_dim, embed_dim)
        self.x_spider = nn.Linear(embed_dim, embed_dim)

        self.phase_modulation = nn.Parameter(torch.randn(n_heads, self.head_dim))

        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        z_state = self.z_spider(x)
        x_state = self.x_spider(x)

        z_heads = z_state.view(batch_size, seq_len, self.n_heads, self.head_dim)
        x_heads = x_state.view(batch_size, seq_len, self.n_heads, self.head_dim)

        phase = torch.sin(self.phase_modulation)
        modulated = z_heads * phase.unsqueeze(0).unsqueeze(0)

        scores = torch.matmul(modulated, x_heads.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        attn_weights = F.softmax(scores, dim=-1)

        attended = torch.matmul(attn_weights, x_heads)
        attended = attended.contiguous().view(batch_size, seq_len, -1)

        return self.output_proj(attended)


class VariationalDerivationLayer(nn.Module):
    """Derive models from variational principles."""

    def __init__(self, dim: int, n_symmetries: int = 3):
        super().__init__()
        self.n_symmetries = n_symmetries

        self.lagrangian_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.symmetry_projections = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(n_symmetries)]
        )

        self.conservation_checker = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, q: Tensor, p: Tensor) -> Tuple[Tensor, Tensor]:
        qp = torch.cat([q, p], dim=-1)
        L = self.lagrangian_net(qp)

        conserved = []
        for proj in self.symmetry_projections:
            projected = proj(L)
            conservation = self.conservation_checker(projected)
            conserved.append(conservation)

        conservation_score = torch.stack(conserved, dim=-1).mean(dim=-1)
        return L, conservation_score


class UIAHModel(nn.Module):
    """
    UIA-H: Unified Intelligence Architecture Variant H

    Implements categorical data representation with sheaf-theoretic
    inference and variational model derivation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        n_patches: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.string_layers = nn.ModuleList(
            [
                StringDiagramLayer(hidden_dim, hidden_dim, n_wires=n_heads)
                for _ in range(n_layers // 2)
            ]
        )

        self.sheaf_layer = SheafCohomologyLayer(hidden_dim, n_patches)

        self.zx_attention = ZXCalculusAttention(hidden_dim, n_heads)

        self.variational = VariationalDerivationLayer(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.input_proj(x)

        for layer in self.string_layers:
            h = layer(h)

        h_sheaf, h1_score = self.sheaf_layer(h)

        h_seq = h_sheaf.unsqueeze(1)
        h_attn = self.zx_attention(h_seq).squeeze(1)

        p = torch.randn_like(h_attn)
        h_var, conservation = self.variational(h_attn, p)

        output = self.classifier(h_var)

        return {
            "output": output,
            "hidden": h_var,
            "h1_cohomology": h1_score,
            "conservation_score": conservation,
        }


def create_uia_h(
    input_dim: int = 784,
    output_dim: int = 10,
    hidden_dim: int = 256,
    n_layers: int = 4,
    **kwargs,
) -> UIAHModel:
    """Create UIA-H model."""
    return UIAHModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        **kwargs,
    )

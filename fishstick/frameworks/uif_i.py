"""
UIF-I Framework (I.md)

Unified Intelligence Framework - Variant I
Core: Renormalized Attention Module (RAM)

Key Components:
- Scale-Parameterized Attention Kernel
- Multi-Scale Beta Distribution Weights
- Categorical Learner Optics
- Sheaf-Consistent Transformer (ToposFormer)
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from scipy.stats import beta as beta_dist
import numpy as np


class ScaleParameterizedAttention(nn.Module):
    """RAM: Attention kernel parameterized by momentum scale."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        n_scales: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_scales = n_scales
        self.alpha = alpha
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.scale_heads = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(n_scales)]
        )

        coeffs = self._compute_beta_coeffs(n_scales, alpha)
        self.register_buffer("scale_coeffs", torch.tensor(coeffs, dtype=torch.float32))

        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def _compute_beta_coeffs(self, n_scales: int, alpha: float) -> List[float]:
        coeffs = []
        for k in range(n_scales):
            x = k / max(n_scales - 1, 1)
            c = beta_dist.pdf(x, k + 1, alpha)
            coeffs.append(c)
        total = sum(coeffs)
        return [c / total for c in coeffs] if total > 0 else [1.0 / n_scales] * n_scales

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        scale_outputs = []
        for i, head in enumerate(self.scale_heads):
            scale_factor = 2**i
            if scale_factor > 1 and seq_len >= scale_factor:
                pooled = F.avg_pool1d(
                    x.transpose(1, 2), kernel_size=scale_factor, stride=scale_factor
                ).transpose(1, 2)
                scaled = head(pooled)
            else:
                scaled = head(x)
            scale_outputs.append(scaled)

        weighted = torch.zeros_like(scale_outputs[0])
        for i, out in enumerate(scale_outputs):
            coeff = self.scale_coeffs[i]
            weighted = weighted + coeff * out

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, v)

        attended = attended.contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.output_proj(attended + weighted)

        return output, attn_weights


class CategoricalLearnerOptics(nn.Module):
    """Probabilistic lens for compositional Bayesian updating."""

    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.forward_lens = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim if i == 0 else latent_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                    nn.GELU(),
                )
                for i in range(3)
            ]
        )

        self.backward_lens = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                    nn.GELU(),
                )
                for _ in range(3)
            ]
        )

        self.belief_update = nn.GRUCell(latent_dim, latent_dim)
        self.predictor = nn.Linear(latent_dim, output_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = x.size(0)

        h = torch.zeros(batch_size, self.latent_dim, device=x.device)
        forward_states = []
        backward_states = []

        for fl, bl in zip(self.forward_lens, self.backward_lens):
            forward_out = fl(h if len(forward_states) > 0 else x)
            forward_states.append(forward_out)

            backward_out = bl(forward_out)
            backward_states.append(backward_out)

            h = self.belief_update(forward_out, h)

        output = self.predictor(h)

        kl_div = F.kl_div(
            F.log_softmax(h, dim=-1),
            torch.softmax(h.detach(), dim=-1),
            reduction="batchmean",
        )

        return output, h, kl_div


class ToposFormerBlock(nn.Module):
    """Sheaf-consistent transformer block."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        ff_dim: int = None,
        n_patches: int = 4,
    ):
        super().__init__()
        ff_dim = ff_dim or embed_dim * 4

        self.attention = ScaleParameterizedAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.sheaf_restrictions = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim // 2) for _ in range(n_patches)]
        )

        self.cohomology_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        restrictions = [r(x) for r in self.sheaf_restrictions]
        h1_loss = sum(
            F.mse_loss(restrictions[i], restrictions[j])
            for i in range(len(restrictions))
            for j in range(i + 1, len(restrictions))
        ) / max(len(restrictions), 1)

        return x, h1_loss * self.cohomology_weight


class UIF_I_Model(nn.Module):
    """
    UIF-I: Unified Intelligence Framework Variant I

    Renormalized Attention Module with categorical optics and
    sheaf-consistent transformer blocks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        n_scales: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.toposformer_blocks = nn.ModuleList(
            [
                ToposFormerBlock(hidden_dim, n_heads, hidden_dim * 4)
                for _ in range(n_layers)
            ]
        )

        self.optics = CategoricalLearnerOptics(hidden_dim, hidden_dim, hidden_dim // 2)

        self.classifier = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.input_proj(x)

        cohomology_losses = []
        for block in self.toposformer_blocks:
            h, h1_loss = block(h)
            cohomology_losses.append(h1_loss)

        optics_out, latent, kl_div = self.optics(h)

        output = self.classifier(optics_out)

        return {
            "output": output,
            "latent": latent,
            "cohomology_losses": torch.stack(cohomology_losses),
            "kl_divergence": kl_div,
        }


def create_uif_i(
    input_dim: int = 784,
    output_dim: int = 10,
    hidden_dim: int = 256,
    n_layers: int = 4,
    **kwargs,
) -> UIF_I_Model:
    """Create UIF-I model."""
    return UIF_I_Model(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        **kwargs,
    )

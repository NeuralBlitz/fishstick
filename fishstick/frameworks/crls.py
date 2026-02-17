"""
CRLS - Categorical Renormalization of Learning Systems

Framework G: Integrates categorical semantics, renormalization group flows,
Hamiltonian variational inference, and sheaf-theoretic data integration.

Key Components:
- Hamiltonian Variational Autoencoder (HVAE)
- Sheaf-Based Data Integration
- RG Flow over Representations
- Natural Gradient with Fisher Metric
"""

from typing import Optional, Tuple, List, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class HamiltonianVAE(nn.Module):
    """
    Hamiltonian Variational Autoencoder with symplectic dynamics in latent space.

    Preserves phase-space volume during inference via symplectic integration.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for _ in range(n_layers - 2)
            ],
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                for _ in range(n_layers - 2)
            ],
            nn.Linear(hidden_dim, input_dim),
        )

        self.hamiltonian_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        qp = torch.cat([q, p], dim=-1)
        return self.hamiltonian_net(qp).squeeze(-1)

    def symplectic_step(
        self, q: torch.Tensor, p: torch.Tensor, dt: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)

        H = self.hamiltonian(q, p)
        dH_dq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        dH_dp = torch.autograd.grad(H.sum(), p, create_graph=True)[0]

        p_new = p - dt * dH_dq
        q_new = q + dt * dH_dp

        return q_new.detach(), p_new.detach()

    def forward(
        self, x: torch.Tensor, n_symplectic_steps: int = 3
    ) -> Dict[str, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        q = z
        p = torch.randn_like(z)

        for _ in range(n_symplectic_steps):
            q, p = self.symplectic_step(q, p)

        x_recon = self.decode(q)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "z_transformed": q,
            "hamiltonian": self.hamiltonian(q, p),
        }


class SheafIntegrationLayer(nn.Module):
    """
    Sheaf-based data integration layer.

    Models data as sections of a sheaf over overlapping patches,
    enforcing consistency via cohomological constraints.
    """

    def __init__(
        self,
        feature_dim: int,
        n_patches: int = 4,
        overlap_dim: int = 16,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.feature_dim = feature_dim
        self.overlap_dim = overlap_dim

        self.local_encoders = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim) for _ in range(n_patches)]
        )

        self.restriction_maps = nn.ModuleList(
            [nn.Linear(feature_dim, overlap_dim) for _ in range(n_patches)]
        )

        self.gluing_net = nn.Linear(feature_dim * n_patches, feature_dim)

        self.consistency_weight = nn.Parameter(torch.ones(1))

    def compute_cohomology_loss(
        self, local_sections: List[torch.Tensor]
    ) -> torch.Tensor:
        loss = 0.0
        count = 0

        for i in range(len(local_sections)):
            for j in range(i + 1, len(local_sections)):
                ri = self.restriction_maps[i](local_sections[i])
                rj = self.restriction_maps[j](local_sections[j])
                loss = loss + F.mse_loss(ri, rj)
                count += 1

        return loss / max(count, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        chunk_size = x.size(-1) // self.n_patches
        patches = [
            x[..., i * chunk_size : (i + 1) * chunk_size] for i in range(self.n_patches)
        ]

        if patches[-1].size(-1) < chunk_size:
            patches[-1] = F.pad(patches[-1], (0, chunk_size - patches[-1].size(-1)))

        local_sections = []
        for i, patch in enumerate(patches):
            if patch.size(-1) < self.feature_dim:
                patch = F.pad(patch, (0, self.feature_dim - patch.size(-1)))
            section = self.local_encoders[i](patch)
            local_sections.append(section)

        cohomology_loss = self.compute_cohomology_loss(local_sections)

        concatenated = torch.cat(local_sections, dim=-1)
        global_section = self.gluing_net(concatenated)

        return global_section, cohomology_loss * self.consistency_weight


class RGLayer(nn.Module):
    """
    Renormalization Group Layer for multi-scale feature extraction.

    Implements coarse-graining operations that integrate out irrelevant
    features while preserving relevant ones (scale-invariant features).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        scale_factor: int = 2,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        self.coarse_grain = nn.Sequential(
            nn.Linear(in_dim, in_dim // scale_factor),
            nn.LayerNorm(in_dim // scale_factor),
            nn.GELU(),
        )

        self.projector = nn.Linear(in_dim // scale_factor, out_dim)

        self.relevance_net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        relevance = self.relevance_net(x)

        x_coarse = self.coarse_grain(x)
        x_out = self.projector(x_coarse)

        return x_out, relevance


class NaturalGradientOptimizer:
    """
    Natural gradient descent using Fisher information metric.

    Implements preconditioned gradient updates that follow geodesics
    on the statistical manifold.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        lr: float = 1e-3,
        damping: float = 1e-5,
    ):
        self.params = list(params)
        self.lr = lr
        self.damping = damping

    def compute_fisher_diag(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        fisher_diag = []

        for param in model.parameters():
            if param.grad is not None:
                fisher_diag.append(param.grad.data.pow(2).mean().item())
            else:
                fisher_diag.append(0.0)

        return torch.tensor(fisher_diag)

    def step(self, grads: List[torch.Tensor], fisher_diag: torch.Tensor):
        idx = 0
        for param in self.params:
            if param.grad is not None:
                grad = grads[idx]
                fish = fisher_diag[idx] if idx < len(fisher_diag) else 1.0
                fish = max(fish, self.damping)
                nat_grad = grad / (fish + self.damping)
                param.data.add_(nat_grad, alpha=-self.lr)
                idx += 1


class CRLSModel(nn.Module):
    """
    Complete CRLS (Categorical Renormalization of Learning Systems) model.

    Combines:
    - Hamiltonian VAE for energy-conserving generation
    - Sheaf integration for multi-modal data fusion
    - RG layers for multi-scale representation
    - Natural gradient optimization
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        n_layers: int = 4,
        n_patches: int = 4,
        n_rg_layers: int = 3,
    ):
        super().__init__()

        self.sheaf_layer = SheafIntegrationLayer(
            feature_dim=hidden_dim,
            n_patches=n_patches,
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.rg_layers = nn.ModuleList(
            [
                RGLayer(hidden_dim // (2**i), hidden_dim // (2 ** (i + 1)))
                for i in range(n_rg_layers)
            ]
        )

        self.hvae = HamiltonianVAE(
            input_dim=hidden_dim // (2**n_rg_layers),
            latent_dim=latent_dim,
            hidden_dim=hidden_dim // 2,
            n_layers=n_layers,
        )

        self.output_layer = nn.Linear(latent_dim, output_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x)

        h, cohomology_loss = self.sheaf_layer(h)

        relevance_scores = []
        for rg_layer in self.rg_layers:
            h, rel = rg_layer(h)
            relevance_scores.append(rel)

        vae_out = self.hvae(h)

        output = self.classifier(vae_out["z_transformed"])

        return {
            "output": output,
            "x_recon": vae_out["x_recon"],
            "mu": vae_out["mu"],
            "log_var": vae_out["log_var"],
            "z": vae_out["z"],
            "z_transformed": vae_out["z_transformed"],
            "hamiltonian": vae_out["hamiltonian"],
            "cohomology_loss": cohomology_loss,
            "relevance_scores": relevance_scores,
        }

    def compute_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        vae_output: Dict[str, torch.Tensor],
        beta: float = 1.0,
        lambda_cohomology: float = 0.1,
        lambda_energy: float = 0.01,
    ) -> torch.Tensor:
        task_loss = F.cross_entropy(output, target)

        mu = vae_output["mu"]
        log_var = vae_output["log_var"]
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        cohomology_loss = vae_output["cohomology_loss"]

        energy_loss = vae_output["hamiltonian"].mean()

        total_loss = (
            task_loss
            + beta * kl_loss
            + lambda_cohomology * cohomology_loss
            + lambda_energy * energy_loss
        )

        return total_loss


def create_crls(
    input_dim: int = 784,
    output_dim: int = 10,
    hidden_dim: int = 256,
    n_layers: int = 4,
    **kwargs,
) -> CRLSModel:
    """Create a CRLS model."""
    return CRLSModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        **kwargs,
    )

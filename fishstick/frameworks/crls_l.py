"""
CRLS-L Framework (L.md)

Categorical Renormalization Learning Systems - Variant L
Mathematical Intelligence Physics

Key Components:
- Hamiltonian VAE with Symplectic Integration
- Sheaf-Based Data Integration
- RG Flow over Representations
- Thermodynamic Training Bounds
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal


class SymplecticIntegrator(nn.Module):
    """Symplectic Euler integrator for Hamiltonian dynamics."""

    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.dim = dim

        self.hamiltonian = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def compute_gradients(self, q: Tensor, p: Tensor) -> Tuple[Tensor, Tensor]:
        qp = torch.cat([q, p], dim=-1)
        qp.requires_grad_(True)

        H = self.hamiltonian(qp).sum()
        grad = torch.autograd.grad(H, qp, create_graph=True)[0]

        dH_dq = grad[:, : self.dim]
        dH_dp = grad[:, self.dim :]

        return dH_dq, dH_dp

    def step(self, q: Tensor, p: Tensor, dt: float = 0.1) -> Tuple[Tensor, Tensor]:
        dH_dq, dH_dp = self.compute_gradients(q, p)

        p_half = p - dt / 2 * dH_dq
        q_new = q + dt * dH_dp

        dH_dq_new, _ = self.compute_gradients(q_new, p_half)
        p_new = p_half - dt / 2 * dH_dq_new

        return q_new.detach(), p_new.detach()


class HamiltonianVAE_L(nn.Module):
    """VAE with Hamiltonian dynamics in latent space."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_integrate_steps: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_integrate_steps = n_integrate_steps

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.symplectic = SymplecticIntegrator(latent_dim, hidden_dim // 2)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        p = torch.randn_like(z)

        q, p_final = z, p
        for _ in range(self.n_integrate_steps):
            q, p_final = self.symplectic.step(q, p_final)

        x_recon = self.decoder(q)

        H = self.symplectic.hamiltonian(torch.cat([q, p_final], dim=-1))

        return {
            "x_recon": x_recon,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "z_evolved": q,
            "hamiltonian": H,
        }


class RGLayer_L(nn.Module):
    """Renormalization Group layer for multi-scale features."""

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

        self.beta_function = nn.Parameter(torch.ones(out_dim))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        relevance = self.relevance_net(x)

        x_coarse = self.coarse_grain(x)
        x_out = self.projector(x_coarse)

        rg_flow = x_out * self.beta_function

        return x_out, relevance, rg_flow


class ThermodynamicRegularizer(nn.Module):
    """Enforce Landauer bounds on information processing."""

    def __init__(self, dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.kb = 1.0

        self.entropy_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        self.kl_estimator = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, prior: Tensor, posterior: Tensor) -> Tuple[Tensor, Tensor]:
        entropy = self.entropy_estimator(posterior)

        kl_approx = self.kl_estimator(torch.cat([prior, posterior], dim=-1))

        energy_cost = (
            self.kb * self.temperature * (F.softplus(kl_approx) - F.softplus(entropy))
        )

        return energy_cost, entropy


class CRLS_L_Model(nn.Module):
    """
    CRLS-L: Categorical Renormalization Learning Systems Variant L

    Hamiltonian VAE with RG flow and thermodynamic regularization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        n_layers: int = 4,
        n_rg_layers: int = 3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.rg_layers = nn.ModuleList(
            [
                RGLayer_L(hidden_dim // (2**i), hidden_dim // (2 ** (i + 1)))
                for i in range(n_rg_layers)
            ]
        )

        self.hvae = HamiltonianVAE_L(
            input_dim=hidden_dim // (2**n_rg_layers),
            latent_dim=latent_dim,
            hidden_dim=hidden_dim // 2,
        )

        self.thermo = ThermodynamicRegularizer(latent_dim)

        self.classifier = nn.Linear(latent_dim, output_dim)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        h = self.input_proj(x)

        relevance_scores = []
        rg_flows = []
        for rg_layer in self.rg_layers:
            h, rel, flow = rg_layer(h)
            relevance_scores.append(rel)
            rg_flows.append(flow)

        vae_out = self.hvae(h)

        prior = torch.randn_like(vae_out["z_evolved"])
        energy_cost, entropy = self.thermo(prior, vae_out["z_evolved"])

        output = self.classifier(vae_out["z_evolved"])

        return {
            "output": output,
            "x_recon": vae_out["x_recon"],
            "mu": vae_out["mu"],
            "log_var": vae_out["log_var"],
            "z": vae_out["z"],
            "z_evolved": vae_out["z_evolved"],
            "hamiltonian": vae_out["hamiltonian"],
            "relevance_scores": relevance_scores,
            "energy_cost": energy_cost,
            "entropy": entropy,
        }


def create_crls_l(
    input_dim: int = 784,
    output_dim: int = 10,
    hidden_dim: int = 256,
    n_layers: int = 4,
    **kwargs,
) -> CRLS_L_Model:
    """Create CRLS-L model."""
    return CRLS_L_Model(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        **kwargs,
    )

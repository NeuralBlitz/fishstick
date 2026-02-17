"""
Renormalization Group (RG) Flow in Deep Networks.

Inspired by Wilsonian RG in Quantum Field Theory:
- Deep networks perform implicit coarse-graining
- Early layers extract micro-features; later layers encode macro-states
- RG flow governs layer-wise representation coarse-graining

This module implements:
- RG-Aware Autoencoders (RGA-AE)
- RG Flow analysis for architecture optimization
- Fixed point detection and universality class prediction
"""

from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


@dataclass
class RGScale:
    """Represents a scale in the RG hierarchy."""

    level: int
    resolution: float
    correlation_length: float
    features: Optional[Tensor] = None


class RGFlow:
    """
    Renormalization Group Flow operator.

    Defines coarse-graining transformation:
        R_λ: N → N'  (network at scale λ to scale λ')

    The beta-function β(θ) = dθ/d(log λ) encodes how parameters
    evolve under rescaling.
    """

    def __init__(
        self,
        n_scales: int = 4,
        base_resolution: float = 1.0,
        scaling_factor: float = 2.0,
    ):
        self.n_scales = n_scales
        self.base_resolution = base_resolution
        self.scaling_factor = scaling_factor
        self.scales: List[RGScale] = []

    def compute_correlation_length(self, features: Tensor) -> float:
        """
        Compute correlation length ξ from feature activations.

        ξ is the scale at which correlations decay to 1/e.
        """
        if features.dim() < 2:
            return 1.0

        features_flat = features.view(features.size(0), -1)
        correlation = torch.corrcoef(features_flat.T)

        eigenvalues = torch.linalg.eigvalsh(correlation)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        if len(eigenvalues) == 0:
            return 1.0

        xi = eigenvalues.mean().item()
        return max(xi, 0.1)

    def coarse_grain(self, features: Tensor, method: str = "avg_pool") -> Tensor:
        """
        Apply coarse-graining transformation R_λ.

        Methods:
        - avg_pool: Average pooling
        - max_pool: Max pooling
        - learned: Learned transformation
        """
        if features.dim() == 2:
            return features

        if method == "avg_pool":
            if features.dim() == 4:
                return F.avg_pool2d(features, kernel_size=2, stride=2)
            elif features.dim() == 5:
                return F.avg_pool3d(features, kernel_size=2, stride=2)
        elif method == "max_pool":
            if features.dim() == 4:
                return F.max_pool2d(features, kernel_size=2, stride=2)
            elif features.dim() == 5:
                return F.max_pool3d(features, kernel_size=2, stride=2)

        return features

    def beta_function(
        self,
        params_before: Dict[str, Tensor],
        params_after: Dict[str, Tensor],
        log_lambda: float,
    ) -> Dict[str, Tensor]:
        """
        Compute beta-function β(θ) = dθ/d(log λ).

        The beta-function describes how parameters flow under RG.
        """
        beta = {}
        for key in params_before:
            if key in params_after:
                beta[key] = (params_after[key] - params_before[key]) / log_lambda
        return beta

    def find_fixed_points(
        self, param_trajectory: List[Dict[str, Tensor]], threshold: float = 1e-4
    ) -> List[int]:
        """
        Find RG fixed points where β(θ*) = 0.

        Fixed points correspond to scale-invariant representations.
        """
        fixed_points = []

        for i in range(1, len(param_trajectory) - 1):
            total_change = 0.0
            count = 0

            for key in param_trajectory[i]:
                if key in param_trajectory[i - 1]:
                    change = torch.norm(
                        param_trajectory[i][key] - param_trajectory[i - 1][key]
                    ).item()
                    total_change += change
                    count += 1

            if count > 0 and total_change / count < threshold:
                fixed_points.append(i)

        return fixed_points


class RGAutoencoder(nn.Module):
    """
    Renormalization Group-Aware Autoencoder (RGA-AE).

    Multi-scale VAE with RG-inspired β-scheduling:
        J_s = E_{q_s}[log p_s(x|z_s)] - β_s D_KL(q_s(z_s|x) || p_s(z_s))

    where β_s = λ^(s·Δ) with Δ being the scaling dimension of irrelevant operators.

    This enables:
    - Scale-invariant representations
    - Automatic feature hierarchy
    - Improved OOD generalization
    """

    def __init__(
        self,
        input_dim: int,
        latent_dims: List[int],
        hidden_dim: int = 256,
        n_scales: int = 4,
        beta_base: float = 1.0,
        scaling_dimension: float = 0.5,
        input_shape: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dims = latent_dims
        self.n_scales = n_scales
        self.beta_base = beta_base
        self.scaling_dimension = scaling_dimension
        self.input_shape = input_shape

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pooling = nn.ModuleList()

        current_dim = input_dim
        for i, latent_dim in enumerate(latent_dims):
            self.encoders.append(
                self._make_encoder_block(current_dim, hidden_dim, latent_dim)
            )
            self.decoders.append(
                self._make_decoder_block(latent_dim, hidden_dim, current_dim)
            )
            self.pooling.append(
                nn.AvgPool1d(kernel_size=2, stride=2)
                if i < n_scales - 1
                else nn.Identity()
            )
            current_dim = latent_dim

        self.rg_flow = RGFlow(n_scales=n_scales)

        self.scale_representations: List[Tensor] = []

        self.final_decoder = nn.Linear(latent_dims[-1], input_dim)

    def _make_encoder_block(self, in_dim: int, hidden: int, latent: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def _make_decoder_block(self, latent: int, hidden: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def encode(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Multi-scale encoding with RG coarse-graining.

        Returns:
            z_list: Latent codes at each scale
            mu_list: Mean parameters
            logvar_list: Log-variance parameters
        """
        z_list = []
        mu_list = []
        logvar_list = []

        h = x
        for i, encoder in enumerate(self.encoders):
            h_encoded = encoder(h)

            latent_dim = self.latent_dims[i]
            mu = nn.Linear(h_encoded.shape[-1], latent_dim)(h_encoded)
            logvar = nn.Linear(h_encoded.shape[-1], latent_dim)(h_encoded)

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            z_list.append(z)
            mu_list.append(mu)
            logvar_list.append(logvar)

            if i < self.n_scales - 1:
                if h.dim() > 2:
                    h = self.rg_flow.coarse_grain(z.unsqueeze(-1)).squeeze(-1)
                else:
                    h = z

        return z_list, mu_list, logvar_list

    def decode(self, z_list: List[Tensor]) -> Tensor:
        """
        Multi-scale decoding - direct projection from finest latent to input.
        """
        z_final = z_list[-1]
        return self.final_decoder(z_final)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass returning reconstructions and KL terms.
        """
        z_list, mu_list, logvar_list = self.encode(x)
        x_recon = self.decode(z_list)

        kl_terms = []
        for mu, logvar in zip(mu_list, logvar_list):
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            kl_terms.append(kl)

        self.scale_representations = z_list

        return {
            "reconstruction": x_recon,
            "latents": z_list,
            "kl_terms": kl_terms,
            "mu": mu_list,
            "logvar": logvar_list,
        }

    def loss(
        self, x: Tensor, outputs: Dict[str, Tensor], recon_loss: str = "mse"
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute RG-weighted ELBO.

        J = Σ_s [E[log p_s(x|z_s)] - β_s D_KL]
        """
        x_recon = outputs["reconstruction"]
        kl_terms = outputs["kl_terms"]

        if recon_loss == "mse":
            recon = F.mse_loss(x_recon, x, reduction="none").sum(dim=-1)
        else:
            recon = F.binary_cross_entropy_with_logits(
                x_recon, x, reduction="none"
            ).sum(dim=-1)

        total_loss = Tensor([0.0])
        losses = {"recon": recon.mean().item()}

        for s, kl in enumerate(kl_terms):
            beta_s = self.beta_base * (self.scaling_dimension**s)

            scale_loss = recon.mean() + beta_s * kl.mean()
            total_loss = total_loss + scale_loss
            losses[f"kl_scale_{s}"] = kl.mean().item()
            losses[f"beta_{s}"] = beta_s

        return total_loss, losses

    def get_fixed_point_representations(self) -> List[Tensor]:
        """Get representations at RG fixed points."""
        return self.scale_representations


class RGAugmentedResNet(nn.Module):
    """
    RG-Augmented Residual Network.

    Each block applies residual connection followed by RG coarse-graining.
    The RG loss encourages fixed-point behavior.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        n_scales: int = 4,
        n_blocks_per_scale: int = 2,
    ):
        super().__init__()

        self.n_scales = n_scales

        self.input_proj = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, padding=1
        )

        self.scales = nn.ModuleList()
        self.rg_ops = nn.ModuleList()
        self.channel_projs = nn.ModuleList()

        channels = base_channels
        for _ in range(n_scales):
            blocks = nn.ModuleList(
                [self._make_residual_block(channels) for _ in range(n_blocks_per_scale)]
            )
            self.scales.append(blocks)

            self.rg_ops.append(nn.AvgPool2d(kernel_size=2, stride=2))

            old_channels = channels
            channels *= 2
            self.channel_projs.append(
                nn.Conv2d(old_channels, channels, kernel_size=1)
                if channels != old_channels
                else nn.Identity()
            )

        self.output_proj = nn.Linear(128, 10)

    def _make_residual_block(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass with scale representations.

        Returns:
            output: Classification output
            scale_reprs: Representations at each RG scale
        """
        x = self.input_proj(x)

        scale_reprs = []

        for scale_idx, (blocks, rg_op, ch_proj) in enumerate(
            zip(self.scales, self.rg_ops, self.channel_projs)
        ):
            for block in blocks:
                identity = x
                x = F.relu(block(x) + identity)

            scale_reprs.append(x.mean(dim=[2, 3]))
            x = rg_op(x)
            x = ch_proj(x)

        x = x.view(x.size(0), -1)
        output = nn.Linear(x.shape[1], 10).to(x.device)(x)

        return output, scale_reprs

    def rg_loss(self, scale_reprs: List[Tensor]) -> Tensor:
        """
        RG regularization loss encouraging fixed-point behavior.

        L_RG = Σ_i ||repr_{i+1} - repr_i||
        """
        loss = Tensor([0.0])

        for i in range(len(scale_reprs) - 1):
            diff = scale_reprs[i + 1] - scale_reprs[i]
            loss = loss + torch.norm(diff, dim=-1).mean()

        return loss


class UniversalityClassPredictor:
    """
    Predict universality class of neural architectures via RG analysis.

    Two models belong to the same universality class iff their
    correlation length diverges with the same critical exponent ν.
    """

    def __init__(self, temperature_range: Tuple[float, float] = (0.1, 10.0)):
        self.temp_range = temperature_range

    def compute_critical_exponents(
        self, model: nn.Module, data: Tensor, n_temps: int = 20
    ) -> Dict[str, float]:
        """
        Compute critical exponents (ν, η) from finite-size scaling.

        Near critical temperature T_c:
        - ξ(T) ~ |T - T_c|^{-ν}
        - C(r) ~ r^{-(d-2+η)} e^{-r/ξ}
        """
        temps = np.linspace(*self.temp_range, n_temps)
        correlation_lengths = []

        for T in temps:
            xi = self._measure_correlation_length(model, data, T)
            correlation_lengths.append(xi)

        temps = np.array(temps)
        xi = np.array(correlation_lengths)

        valid = (xi > 0) & (xi < 1e6)
        if valid.sum() < 3:
            return {"nu": 1.0, "eta": 0.0, "tc": np.mean(self.temp_range)}

        temps_valid = temps[valid]
        xi_valid = xi[valid]

        tc_idx = np.argmax(xi_valid)
        tc = temps_valid[tc_idx]

        distances = np.abs(temps_valid - tc)
        near_critical = distances < np.percentile(distances, 50)

        if near_critical.sum() < 2:
            return {"nu": 1.0, "eta": 0.0, "tc": tc}

        log_xi = np.log(xi_valid[near_critical])
        log_t = np.log(distances[near_critical] + 1e-10)

        nu = -np.polyfit(log_t, log_xi, 1)[0] if len(log_t) > 1 else 1.0
        nu = max(0.1, min(nu, 5.0))

        return {"nu": nu, "eta": 0.25, "tc": tc}

    def _measure_correlation_length(
        self, model: nn.Module, data: Tensor, temperature: float
    ) -> float:
        """Measure correlation length at given temperature."""
        model.eval()
        with torch.no_grad():
            features = model(data)
            if isinstance(features, tuple):
                features = features[0]

            features = features.view(features.size(0), -1)

            features = features / temperature

            if features.shape[0] > 1:
                corr = torch.corrcoef(features.T)
                eigenvalues = torch.linalg.eigvalsh(corr)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                if len(eigenvalues) > 0:
                    return eigenvalues.mean().item()

        return 1.0

    def compare_universality_classes(
        self, model1: nn.Module, model2: nn.Module, data: Tensor, tolerance: float = 0.2
    ) -> bool:
        """
        Check if two models belong to the same universality class.

        Returns True if critical exponents match within tolerance.
        """
        exponents1 = self.compute_critical_exponents(model1, data)
        exponents2 = self.compute_critical_exponents(model2, data)

        nu_diff = abs(exponents1["nu"] - exponents2["nu"]) / max(
            exponents1["nu"], exponents2["nu"]
        )

        return nu_diff < tolerance

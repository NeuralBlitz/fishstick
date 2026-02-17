"""
Information Bottleneck Module

Implements the Information Bottleneck (IB) method for representation learning:
- Classic IB with beta parameter
- Variational IB
- Conditional IB
- Lagrangian IB optimization
"""

from typing import Optional, Tuple, Callable, Dict, Any
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
from dataclasses import dataclass


@dataclass
class IBParameters:
    """Information Bottleneck parameters."""

    beta: float = 1.0
    alpha: float = 1.0
    latent_dim: int = 32
    hidden_dim: int = 128


class InformationBottleneck(nn.Module):
    """
    Classic Information Bottleneck layer.

    Optimizes: min I(X;Z) - beta * I(Z;Y)

    Where Z is the bottleneck representation.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        beta: float = 1.0,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize IB layer.

        Args:
            input_dim: Input dimension
            latent_dim: Bottleneck dimension
            beta: IB trade-off parameter
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        hidden_dim = hidden_dim or max(64, latent_dim * 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode to latent distribution parameters (mean, logvar)."""
        params = self.encoder(x)
        mean, logvar = params.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mean, logvar

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode from latent to reconstruction."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with reconstruction.

        Returns:
            Tuple of (reconstruction, mean, logvar)
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar

    def compute_ib_loss(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        target_mi: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute IB objective.

        Args:
            x: Input data
            y: Target labels (if available)
            target_mi: Target I(Z;Y) value

        Returns:
            Dictionary with loss components
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)

        recon_loss = nn.functional.mse_loss(reconstruction, x)

        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        if target_mi is not None and y is not None:
            from .mutual_info import info_nce

            mi_loss = torch.abs(info_nce(z, y) - target_mi)
        else:
            mi_loss = torch.tensor(0.0)

        total_loss = kl_div - self.beta * mi_loss + recon_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_div": kl_div,
            "mi_loss": mi_loss,
            "latent": z,
        }


class VariationalInformationBottleneck(nn.Module):
    """
    Variational Information Bottleneck (VIB).

    Uses variational approximation for the bottleneck.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        beta: float = 1e-3,
        output_dim: Optional[int] = None,
    ):
        """
        Initialize VIB.

        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            beta: Regularization strength
            output_dim: Output dimension (for classification)
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_dim * 2),
        )

        if output_dim is not None:
            self.classifier = nn.Linear(latent_dim, output_dim)
        else:
            self.classifier = nn.Linear(latent_dim, input_dim)

    def forward(
        self,
        x: Tensor,
        return_latent: bool = False,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            return_latent: Whether to return latent representation

        Returns:
            Output logits or tuple of (logits, latent)
        """
        params = self.encoder(x)
        mean, logvar = params.chunk(2, dim=-1)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std

        logits = self.classifier(z)

        if return_latent:
            return logits, z, mean, logvar
        return logits

    def compute_loss(
        self,
        x: Tensor,
        labels: Tensor,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute VIB loss.

        Args:
            x: Input
            labels: Target labels
            criterion: Loss criterion

        Returns:
            Loss dictionary
        """
        criterion = criterion or nn.CrossEntropyLoss()

        logits, z, mean, logvar = self.forward(x, return_latent=True)

        ce_loss = criterion(logits, labels)

        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        total_loss = ce_loss + self.beta * kl_div

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "kl_div": kl_div,
            "accuracy": (logits.argmax(dim=-1) == labels).float().mean(),
        }


class LagrangianIB:
    """
    Lagrangian Information Bottleneck with adaptive beta.

    Optimizes using dual gradient descent.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        target_ib: float = 1.0,
        eta: float = 0.01,
        beta_min: float = 1e-6,
        beta_max: float = 1e6,
    ):
        """
        Initialize Lagrangian IB.

        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            target_ib: Target IB value
            eta: Learning rate for beta
            beta_min: Minimum beta
            beta_max: Maximum beta
        """
        self.model = InformationBottleneck(input_dim, latent_dim)
        self.target_ib = target_ib
        self.eta = eta
        self.beta = 1.0
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def step(self, x: Tensor, y: Tensor) -> Dict[str, Tensor]:
        """
        Single optimization step.

        Args:
            x: Input data
            y: Labels

        Returns:
            Loss dictionary
        """
        self.optimizer.zero_grad()

        mean, logvar = self.model.encode(x)
        z = self.model.reparameterize(mean, logvar)

        from .mutual_info import info_nce

        mi_zy = info_nce(z, y)

        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        recon = self.model.decode(z)
        recon_loss = nn.functional.mse_loss(recon, x)

        ib = kl_div - self.beta * mi_zy

        loss = recon_loss + ib
        loss.backward()

        self.optimizer.step()

        constraint = kl_div.item() - self.target_ib
        self.beta = max(
            self.beta_min, min(self.beta_max, self.beta + self.eta * constraint)
        )

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_div": kl_div,
            "mi_zy": mi_zy,
            "beta": torch.tensor(self.beta),
        }


class ConditionalInformationBottleneck(nn.Module):
    """
    Conditional IB for class-conditional representations.

    Optimizes: min I(X;Z) - alpha * I(Z;Y) - gamma * I(Z;C)

    Where C is class information.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        alpha: float = 1.0,
        gamma: float = 0.1,
        hidden_dim: int = 256,
    ):
        """
        Initialize Conditional IB.

        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            num_classes: Number of classes
            alpha: I(Z;Y) weight
            gamma: I(Z;C) weight
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.gamma = gamma

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass."""
        params = self.encoder(x)
        mean, logvar = params.chunk(2, dim=-1)

        z = self.reparameterize(mean, logvar)
        logits = self.classifier(z)

        return logits, mean, logvar

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization."""
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def compute_loss(
        self,
        x: Tensor,
        y: Tensor,
        c: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute conditional IB loss.

        Args:
            x: Input
            y: Labels
            c: Class embeddings
        """
        logits, mean, logvar = self.forward(x)

        ce_loss = nn.functional.cross_entropy(logits, y)

        kl_div = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        z = self.reparameterize(mean, logvar)

        if c is not None:
            from .mutual_info import info_nce

            mi_zy = info_nce(z, y)
            mi_zc = info_nce(z, c)
            total_loss = ce_loss + kl_div - self.alpha * mi_zy + self.gamma * mi_zc
        else:
            mi_zy = torch.tensor(0.0)
            mi_zc = torch.tensor(0.0)
            total_loss = ce_loss + kl_div - self.alpha * mi_zy

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "kl_div": kl_div,
            "mi_zy": mi_zy,
            "mi_zc": mi_zc,
        }


class DeepInfoMax(nn.Module):
    """
    Deep InfoMax: Maximize mutual information between input and latent.

    I(X;Z) - beta * I(Z;Y)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        beta: float = 1.0,
    ):
        """
        Initialize Deep InfoMax.

        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            hidden_dim: Hidden dimension
            beta: IB trade-off
        """
        super().__init__()
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode input."""
        return self.encoder(x)

    def compute_loss(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute Deep InfoMax loss.

        Args:
            x: Input data
            y: Labels (optional)
        """
        batch_size = x.shape[0]

        z = self.encoder(x)

        pos_pairs = torch.cat([x, z], dim=-1)
        neg_indices = torch.randperm(batch_size)
        neg_pairs = torch.cat([x, z[neg_indices]], dim=-1)

        pos_score = self.discriminator(pos_pairs)
        neg_score = self.discriminator(neg_pairs)

        mi_loss = (
            -torch.log(torch.sigmoid(pos_score) + 1e-10).mean()
            - torch.log(1 - torch.sigmoid(neg_score) + 1e-10).mean()
        )

        if y is not None:
            from .mutual_info import info_nce

            mi_zy = info_nce(z, y)
            total_loss = mi_loss - self.beta * mi_zy
        else:
            mi_zy = torch.tensor(0.0)
            total_loss = mi_loss

        return {
            "total_loss": total_loss,
            "mi_loss": mi_loss,
            "mi_zy": mi_zy,
        }


def soft_ib_loss(
    z_mean: Tensor,
    z_logvar: Tensor,
    x: Tensor,
    reconstruction: Tensor,
    beta: float = 1.0,
) -> Dict[str, Tensor]:
    """
    Soft Information Bottleneck loss.

    L = Reconstruction + beta * KL(Z|X) || N(0,I)

    Args:
        z_mean: Latent mean
        z_logvar: Latent log variance
        x: Original input
        reconstruction: Reconstructed input
        beta: Beta parameter

    Returns:
        Loss dictionary
    """
    recon_loss = nn.functional.mse_loss(reconstruction, x)

    kl_div = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

    total_loss = recon_loss + beta * kl_div

    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_div": kl_div,
    }

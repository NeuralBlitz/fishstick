"""
Barlow Twins: Self-Supervised Learning via Redundancy Reduction

Implementation based on "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
(Zbontar et al., 2021).

Key ideas:
- Learn representations that are invariant to augmentations
- Reduce redundancy between dimensions of the embedding vector
- Use cross-correlation matrix between two augmented views
"""

from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss function.

    Computes the redundancy reduction loss between two augmented views.

    Args:
        embedding_dim: Dimension of the embedding space
        lambd: Trade-off parameter for off-diagonal terms (default: 0.005)
    """

    def __init__(self, embedding_dim: int, lambd: float = 0.005):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lambd = lambd

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        batch_size = z1.shape[0]

        c = torch.mm(z1.T, z2) / batch_size

        diag = torch.eye(self.embedding_dim, device=c.device)
        off_diag = 1 - diag

        loss_diag = (1 - c.diagonal()).pow(2).sum()
        loss_off_diag = (c * off_diag).pow(2).sum() / self.embedding_dim

        loss = loss_diag + self.lambd * loss_off_diag

        return loss


class BarlowTwins(nn.Module):
    """Barlow Twins model for self-supervised learning.

    Barlow Twins learns representations by:
    1. Encoding two augmented views of the same image
    2. Computing cross-correlation matrix between embeddings
    3. Minimizing diagonal terms (invariance) and off-diagonal terms (redundancy reduction)

    Args:
        encoder: Backbone neural network
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers in projection head
        lambd: Trade-off parameter for off-diagonal terms
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 8192,
        hidden_dim: int = 8192,
        lambd: float = 0.005,
        encoder_output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder

        # Get encoder output dimension
        if encoder_output_dim is not None:
            encoder_out_dim = encoder_output_dim
        elif hasattr(encoder, "output_dim"):
            encoder_out_dim = encoder.output_dim
        elif hasattr(encoder, "embed_dim"):
            encoder_out_dim = encoder.embed_dim
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                encoder_out_dim = encoder(dummy).flatten(1).shape[1]

        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.loss_fn = BarlowTwinsLoss(projection_dim, lambd)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        loss = self.loss_fn(z1, z2)
        return loss

    def get_embeddings(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=-1)


class VicRegLoss(nn.Module):
    """VICReg: Variance-Invariance-Covariance Regularization Loss.

    Based on "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
    (Bardes et al., 2022).

    Args:
        sim_coef: Similarity loss coefficient
        var_coef: Variance loss coefficient
        cov_coef: Covariance loss coefficient
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        sim_coef: float = 25.0,
        var_coef: float = 25.0,
        cov_coef: float = 1.0,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef
        self.epsilon = epsilon

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        sim_loss = F.mse_loss(z1, z2)

        z = torch.cat([z1, z2], dim=0)
        batch_size = z.shape[0]
        dim = z.shape[-1]

        std_z = torch.sqrt(z.var(dim=0) + self.epsilon)
        std_loss = torch.mean(F.relu(1 - std_z))

        z = z - z.mean(dim=0)
        cov_z = (z.T @ z) / batch_size
        cov_loss = cov_z.pow(2).sum() / dim - cov_z.diagonal().pow(2).sum() / dim
        cov_loss = cov_loss

        loss = (
            self.sim_coef * sim_loss
            + self.var_coef * std_loss
            + self.cov_coef * cov_loss
        )

        return loss


class VicReg(nn.Module):
    """VICReg: Variance-Invariance-Covariance Regularization.

    Self-supervised method that combines:
    - Invariance: Same image should have similar representations
    - Variance: Representations should have high variance
    - Covariance: Dimensions should be uncorrelated

    Args:
        encoder: Backbone neural network
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of hidden layers
        sim_coef: Similarity loss coefficient
        var_coef: Variance loss coefficient
        cov_coef: Covariance loss coefficient
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 8192,
        hidden_dim: int = 8192,
        sim_coef: float = 25.0,
        var_coef: float = 25.0,
        cov_coef: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder

        encoder_out_dim = (
            encoder.output_dim if hasattr(encoder, "output_dim") else encoder.embed_dim
        )

        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

        self.loss_fn = VicRegLoss(sim_coef, var_coef, cov_coef)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        loss = self.loss_fn(z1, z2)
        return loss

    def get_embeddings(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=-1)

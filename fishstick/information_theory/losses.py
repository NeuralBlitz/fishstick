"""
Entropy-Based Loss Functions Module

Provides information-theoretic loss functions for training:
- Information gain losses
- Entropy-regularized losses
- Contrastive losses based on mutual information
- Redundancy-penalized losses
- Compression-aware losses
"""

from typing import Optional, Tuple, Callable, Dict, Any
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """
    InfoNCE contrastive loss (Noise Contrastive Estimation).

    Lower bound on mutual information between positive pairs.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = "mean",
    ):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature for softmax
            reduction: Reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        z_i: Tensor,
        z_j: Tensor,
        neg_pairs: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute InfoNCE loss.

        Args:
            z_i: First view embeddings (batch_size, dim)
            z_j: Second view embeddings (batch_size, dim)
            neg_pairs: Negative pairs (batch_size, num_neg, dim)

        Returns:
            InfoNCE loss
        """
        batch_size = z_i.shape[0]

        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        pos_sim = (z_i * z_j).sum(dim=-1) / self.temperature

        if neg_pairs is not None:
            neg_pairs = neg_pairs.view(-1, neg_pairs.shape[-1])
            neg_sim = torch.mm(z_i, neg_pairs.T) / self.temperature
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)
            loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        else:
            z_all = torch.cat([z_i, z_j], dim=0)
            sim_matrix = torch.mm(z_all, z_all.T) / self.temperature

            mask = torch.eye(batch_size * 2, device=z_i.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

            pos_sim = (
                torch.cat([(z_i * z_j).sum(-1), (z_j * z_i).sum(-1)]) / self.temperature
            )

            loss = -pos_sim + torch.logsumexp(sim_matrix, dim=-1)

            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()

        return loss


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins loss for self-supervised learning.

    Invariance + redundancy reduction.
    """

    def __init__(
        self,
        lambda_: float = 0.005,
        scale_factor: float = 0.024,
    ):
        """
        Initialize Barlow Twins loss.

        Args:
            lambda_: Off-diagonal penalty weight
            scale_factor: Scaling factor for similarity matrix
        """
        super().__init__()
        self.lambda_ = lambda_
        self.scale_factor = scale_factor

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """
        Compute Barlow Twins loss.

        Args:
            z_i: First view embeddings
            z_j: Second view embeddings

        Returns:
            Barlow Twins loss
        """
        z_i = (z_i - z_i.mean(dim=0)) / (z_i.std(dim=0) + 1e-10)
        z_j = (z_j - z_j.mean(dim=0)) / (z_j.std(dim=0) + 1e-10)

        c = torch.mm(z_i.T, z_j) / z_i.shape[0]
        c = c * self.scale_factor

        diagonal = torch.eye(c.shape[0], device=c.device)
        off_diagonal = c * (1 - diagonal)

        loss = (1 - diagonal.sum() / c.shape[0]) + self.lambda_ * off_diagonal.pow(
            2
        ).sum()

        return loss


class EntropyRegularizedCrossEntropy(nn.Module):
    """
    Cross-entropy with entropy regularization.

    L = CE(p, y) - alpha * H(p)

    Encourages confident predictions while preventing overconfidence.
    """

    def __init__(self, alpha: float = 0.1, reduction: str = "mean"):
        """
        Initialize entropy-regularized CE.

        Args:
            alpha: Entropy penalty weight
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute loss.

        Args:
            logits: Model outputs
            targets: Ground truth labels

        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction=self.reduction)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        if self.reduction == "mean":
            entropy = entropy.mean()

        return ce_loss - self.alpha * entropy


class InformationGainLoss(nn.Module):
    """
    Information gain loss for active learning / uncertainty.

    Maximizes information gain from predictions.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize information gain loss.

        Args:
            temperature: Temperature for softmax
        """
        self.temperature = temperature

    def forward(
        self,
        logits: Tensor,
        prior: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute information gain loss.

        Args:
            logits: Model predictions
            prior: Prior distribution

        Returns:
            Negative information gain
        """
        probs = F.softmax(logits / self.temperature, dim=-1)

        if prior is None:
            prior = torch.ones_like(probs) / probs.shape[-1]

        prior = prior.to(probs.device)

        kl_div = (probs * torch.log(probs / (prior + 1e-10) + 1e-10)).sum(dim=-1)

        return -kl_div.mean()


class RedundancyPenalty(nn.Module):
    """
    Penalizes redundant features in representations.

    Based on total correlation / multi-information.
    """

    def __init__(self, penalty: float = 0.1):
        """
        Initialize redundancy penalty.

        Args:
            penalty: Weight for redundancy penalty
        """
        super().__init__()
        self.penalty = penalty

    def forward(self, features: Tensor) -> Tensor:
        """
        Compute redundancy penalty.

        Args:
            features: Feature tensor (batch, dim)

        Returns:
            Redundancy penalty
        """
        features = features - features.mean(dim=0, keepdim=True)

        cov = torch.mm(features.T, features) / features.shape[0]

        sign, logdet = torch.slogdet(
            cov + 1e-8 * torch.eye(features.shape[1], device=features.device)
        )

        log_cov_diag = torch.log(torch.diag(cov) + 1e-10).sum()

        redundancy = logdet - log_cov_diag

        return self.penalty * redundancy


class CompressionAwareLoss(nn.Module):
    """
    Compression-aware loss for rate-distortion optimization.

    L = Distortion + lambda * Rate
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        distortion_type: str = "mse",
    ):
        """
        Initialize compression-aware loss.

        Args:
            lambda_: Rate-distortion trade-off
            distortion_type: Type of distortion measure
        """
        super().__init__()
        self.lambda_ = lambda_
        self.distortion_type = distortion_type

    def forward(
        self,
        x: Tensor,
        reconstructed: Tensor,
        latents: Tensor,
    ) -> Tensor:
        """
        Compute compression-aware loss.

        Args:
            x: Original input
            reconstructed: Reconstructed input
            latents: Latent representation

        Returns:
            Rate-distortion loss
        """
        if self.distortion_type == "mse":
            distortion = F.mse_loss(reconstructed, x)
        elif self.distortion_type == "l1":
            distortion = F.l1_loss(reconstructed, x)
        else:
            raise ValueError(f"Unknown distortion type: {self.distortion_type}")

        std = latents.std(dim=0).mean()
        rate = torch.log(std + 1e-10)

        return distortion + self.lambda_ * rate


class ConditionalEntropyLoss(nn.Module):
    """
    Conditional entropy loss for cluster assignment.

    Minimizes H(Y|X) to make confident predictions.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize conditional entropy loss.

        Args:
            temperature: Temperature for softening
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: Tensor) -> Tensor:
        """
        Compute conditional entropy.

        Args:
            logits: Model predictions

        Returns:
            Negative conditional entropy (to minimize)
        """
        probs = F.softmax(logits / self.temperature, dim=-1)

        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        return entropy.mean()


class MutualInformationMaximizationLoss(nn.Module):
    """
    Maximizes mutual information between views.

    Uses contrastive approach.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        temperature: float = 0.07,
    ):
        """
        Initialize MI maximization loss.

        Args:
            hidden_dim: Projection hidden dim
            temperature: Temperature for similarity
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(
        self,
        view1: Tensor,
        view2: Tensor,
    ) -> Tensor:
        """
        Compute MI maximization loss.

        Args:
            view1: First view
            view2: Second view

        Returns:
            Negative MI (to minimize)
        """
        batch_size = view1.shape[0]

        proj1 = nn.Linear(view1.shape[1], self.hidden_dim).to(view1.device)(view1)
        proj2 = nn.Linear(view2.shape[1], self.hidden_dim).to(view2.device)(view2)

        proj1 = F.normalize(proj1, dim=-1)
        proj2 = F.normalize(proj2, dim=-1)

        sim_matrix = torch.mm(proj1, proj2.T) / self.temperature

        labels = torch.arange(batch_size, device=view1.device)

        loss_i = F.cross_entropy(sim_matrix, labels)
        loss_j = F.cross_entropy(sim_matrix.T, labels)

        return (loss_i + loss_j) / 2


class TripletInfoNCE(nn.Module):
    """
    Triplet InfoNCE loss with anchor, positive, negative.
    """

    def __init__(self, temperature: float = 0.1, margin: float = 0.5):
        """
        Initialize triplet InfoNCE.

        Args:
            temperature: Temperature for softmax
            margin: Margin for negative pairs
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """
        Compute triplet InfoNCE loss.

        Args:
            anchor: Anchor embeddings
            positive: Positive pair embeddings
            negative: Negative pair embeddings

        Returns:
            Loss value
        """
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature
        neg_sim = (anchor * negative).sum(dim=-1) / self.temperature

        loss = F.relu(self.margin - pos_sim + neg_sim).mean()

        return loss


class ClusterSeparationLoss(nn.Module):
    """
    Loss for cluster separation using information theory.

    Maximizes inter-cluster MI, minimizes intra-cluster MI.
    """

    def __init__(
        self,
        inter_weight: float = 1.0,
        intra_weight: float = 0.5,
    ):
        """
        Initialize cluster separation loss.

        Args:
            inter_weight: Weight for inter-cluster loss
            intra_weight: Weight for intra-cluster loss
        """
        super().__init__()
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight

    def forward(
        self,
        features: Tensor,
        assignments: Tensor,
    ) -> Tensor:
        """
        Compute cluster separation loss.

        Args:
            features: Feature embeddings
            assignments: Cluster assignments (one-hot or indices)

        Returns:
            Cluster separation loss
        """
        unique_clusters = torch.unique(assignments)

        cluster_means = []
        for c in unique_clusters:
            mask = (assignments == c).float().unsqueeze(-1)
            mean = (features * mask).sum(dim=0) / (mask.sum() + 1e-10)
            cluster_means.append(mean)

        cluster_means = torch.stack(cluster_means)

        inter_cluster = torch.cdist(cluster_means, cluster_means).mean()

        intra_cluster = 0.0
        for c in unique_clusters:
            mask = (assignments == c).float().unsqueeze(-1)
            cluster_features = features * mask
            if mask.sum() > 1:
                cov = torch.mm(cluster_features.T, cluster_features) / (mask.sum() - 1)
                intra_cluster += torch.logdet(
                    cov + 1e-8 * torch.eye(cov.shape[0], device=cov.device)
                )

        return (
            self.inter_weight * (1 / (inter_cluster + 1e-10))
            + self.intra_weight * intra_cluster
        )


def spectral_entropy(
    logits: Tensor,
    dim: int = -1,
) -> Tensor:
    """
    Compute spectral entropy from logits.

    Args:
        logits: Model output logits
        dim: Dimension for softmax

    Returns:
        Spectral entropy
    """
    probs = F.softmax(logits, dim=dim)
    return -(probs * torch.log(probs + 1e-10)).sum(dim=dim)


def gini_entropy(
    features: Tensor,
    dim: int = -1,
) -> Tensor:
    """
    Compute Gini entropy (impurity).

    Args:
        features: Feature tensor
        dim: Dimension for computing Gini

    Returns:
        Gini impurity
    """
    probs = F.softmax(features, dim=dim)
    return 1 - (probs**2).sum(dim=dim)


class EntropyPenalizedBCE(nn.Module):
    """
    Binary cross-entropy with entropy penalty.

    Useful for preventing overconfident predictions.
    """

    def __init__(self, penalty: float = 0.01):
        """
        Initialize entropy-penalized BCE.

        Args:
            penalty: Entropy penalty weight
        """
        super().__init__()
        self.penalty = penalty

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute loss.

        Args:
            logits: Model outputs (before sigmoid)
            targets: Binary targets

        Returns:
            Loss value
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")

        probs = torch.sigmoid(logits)
        entropy = -(
            probs * torch.log(probs + 1e-10)
            + (1 - probs) * torch.log(1 - probs + 1e-10)
        )

        return bce + self.penalty * entropy.mean()

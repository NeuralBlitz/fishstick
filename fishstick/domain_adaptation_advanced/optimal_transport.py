import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass


@dataclass
class OTConfig:
    """Configuration for Optimal Transport based domain adaptation."""

    epsilon: float = 0.1
    max_iter: int = 1000
    num_iiter: int = 10
    reduction: str = "mean"


@dataclass
class SinkhornConfig:
    """Configuration for Sinkhorn algorithm."""

    epsilon: float = 0.1
    max_iter: int = 100
    num_inner_iter: int = 10
    reduction: str = "mean"
    clamp_between: Optional[Tuple[float, float]] = None


class SinkhornDistance(nn.Module):
    """Sinkhorn distance for optimal transport."""

    def __init__(self, config: Optional[SinkhornConfig] = None):
        super().__init__()
        self.config = config or SinkhornConfig()

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_weights: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Sinkhorn distance between source and target features.

        Args:
            source_features: Source domain features (batch_size, feature_dim)
            target_features: Target domain features (batch_size, feature_dim)
            source_weights: Optional weights for source (batch_size,)
            target_weights: Optional weights for target (batch_size,)

        Returns:
            Tuple of (sinkhorn_distance, coupling_matrix, log_dict)
        """
        device = source_features.device
        batch_size_s = source_features.size(0)
        batch_size_t = target_features.size(0)

        if source_weights is None:
            source_weights = torch.ones(batch_size_s, device=device)
        if target_weights is None:
            target_weights = torch.ones(batch_size_t, device=device)

        source_weights = source_weights / source_weights.sum()
        target_weights = target_weights / target_weights.sum()

        cost_matrix = self._compute_cost_matrix(source_features, target_features)

        kernel_matrix = self._compute_kernel_matrix(cost_matrix)

        u = torch.ones_like(source_weights).unsqueeze(1)
        v = torch.ones_like(target_weights).unsqueeze(1)

        for _ in range(self.config.num_inner_iter):
            u = source_weights.unsqueeze(1) / (kernel_matrix @ v + 1e-8)
            v = target_weights.unsqueeze(1) / (kernel_matrix.t() @ u + 1e-8)

        coupling_matrix = u * kernel_matrix * v.t()

        sinkhorn_dist = torch.sum(coupling_matrix * cost_matrix)

        if self.config.reduction == "mean":
            sinkhorn_dist = sinkhorn_dist / (batch_size_s * batch_size_t)

        return sinkhorn_dist, coupling_matrix, {"u": u, "v": v}

    def _compute_cost_matrix(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise cost matrix using Euclidean distance."""
        source_expanded = source_features.unsqueeze(1)
        target_expanded = target_features.unsqueeze(0)
        cost_matrix = torch.norm(source_expanded - target_expanded, p=2, dim=2)
        return cost_matrix

    def _compute_kernel_matrix(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """Compute Gibbs kernel from cost matrix."""
        kernel = torch.exp(-cost_matrix / self.config.epsilon)
        return kernel


@dataclass
class JCPOTConfig:
    """Configuration for Joint Coupling Optimal Transport (JCPOT)."""

    epsilon: float = 0.1
    max_iter: int = 100
    num_inner_iter: int = 10
    margin: float = 1.0


class JCPOTModel(nn.Module):
    """Joint Partial Optimal Transport for domain adaptation."""

    def __init__(self, config: Optional[JCPOTConfig] = None):
        super().__init__()
        self.config = config or JCPOTConfig()
        self.sinkhorn = SinkhornDistance(
            SinkhornConfig(
                epsilon=self.config.epsilon,
                max_iter=self.config.max_iter,
                num_inner_iter=self.config.num_inner_iter,
            )
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute JCPOT loss for domain adaptation.

        Args:
            source_features: Source features (n_s, d)
            target_features: Target features (n_t, d)
            source_labels: Source labels (n_s,) for class-aware OT
            class_weights: Weights for each class

        Returns:
            Dictionary containing OT loss and coupling matrix
        """
        device = source_features.device
        n_s, n_t = source_features.size(0), target_features.size(0)

        cost_matrix = self._compute_cost_matrix(source_features, target_features)

        source_weights = torch.ones(n_s, device=device) / n_s
        target_weights = torch.ones(n_t, device=device) / n_t

        if source_labels is not None and class_weights is not None:
            class_matrix = self._compute_class_matrix(
                source_labels, n_s, n_t, class_weights, device
            )
            cost_matrix = cost_matrix * class_matrix

        ot_dist, coupling, _ = self.sinkhorn(
            source_features, target_features, source_weights, target_weights
        )

        return {"ot_loss": ot_dist, "coupling": coupling, "cost_matrix": cost_matrix}

    def _compute_cost_matrix(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute cost matrix with optional class-aware weighting."""
        source_expanded = source_features.unsqueeze(1)
        target_expanded = target_features.unsqueeze(0)
        cost_matrix = torch.norm(source_expanded - target_expanded, p=2, dim=2)
        return cost_matrix

    def _compute_class_matrix(
        self,
        source_labels: torch.Tensor,
        n_s: int,
        n_t: int,
        class_weights: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute class indicator matrix for class-aware OT."""
        unique_labels = torch.unique(source_labels)
        class_matrix = torch.zeros(n_s, n_t, device=device)

        for label in unique_labels:
            source_mask = (source_labels == label).float().unsqueeze(1)
            class_indicator = source_mask @ source_mask.t()
            weight = class_weights[label.item()].item()
            class_matrix += class_indicator * weight

        return class_matrix


@dataclass
class WDGRLConfig:
    """Configuration for Wasserstein Distance Guided Representation Learning (WDGRL)."""

    hidden_dim: int = 256
    critic_iterations: int = 5
    weight_clip: float = 0.01
    learning_rate: float = 1e-4


class WDGRLModel(nn.Module):
    """Wasserstein Distance Guided Representation Learning (WDGRL)."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        critic: nn.Module,
        config: Optional[WDGRLConfig] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.critic = critic
        self.config = config or WDGRLConfig()

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        if return_features:
            return {"features": features, "logits": logits}
        return {"logits": logits}

    def compute_wasserstein_distance(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute Wasserstein distance using the critic network."""
        source_scores = self.critic(source_features).mean()
        target_scores = self.critic(target_features).mean()

        wasserstein_dist = source_scores - target_scores
        return wasserstein_dist

    def compute_adversarial_loss(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute adversarial loss for domain alignment."""
        source_scores = self.critic(source_features)
        target_scores = self.critic(target_features)

        source_labels = torch.ones_like(source_scores)
        target_labels = torch.zeros_like(target_scores)

        source_loss = F.mse_loss(source_scores, source_labels)
        target_loss = F.mse_loss(target_scores, target_labels)

        return source_loss + target_loss

    def clip_critic_weights(self):
        """Clip critic weights for Lipschitz constraint."""
        for param in self.critic.parameters():
            param.data.clamp_(-self.config.weight_clip, self.config.weight_clip)


@dataclass
class OTBasedClassifierConfig:
    """Configuration for OT-based domain classifier."""

    epsilon: float = 0.1
    max_iter: int = 100


class OTBasedDomainAdaptation(nn.Module):
    """General optimal transport based domain adaptation framework."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        config: Optional[OTConfig] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.config = config or OTConfig()
        self.sinkhorn = SinkhornDistance(
            SinkhornConfig(
                epsilon=self.config.epsilon,
                max_iter=self.config.max_iter,
                num_inner_iter=self.config.num_iiter,
                reduction=self.config.reduction,
            )
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        if return_features:
            return {"features": features, "logits": logits}
        return {"logits": logits}

    def compute_ot_loss(
        self, source_features: torch.Tensor, target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute optimal transport loss for domain alignment."""
        source_weights = torch.ones(
            source_features.size(0), device=source_features.device
        )
        source_weights = source_weights / source_weights.sum()

        target_weights = torch.ones(
            target_features.size(0), device=target_features.device
        )
        target_weights = target_weights / target_weights.sum()

        ot_dist, _, _ = self.sinkhorn(
            source_features, target_features, source_weights, target_weights
        )

        return ot_dist

    def compute_class_aware_ot_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class-aware optimal transport loss."""
        unique_labels = torch.unique(source_labels)
        total_loss = 0.0
        count = 0

        for label in unique_labels:
            mask = source_labels == label
            if mask.sum() > 0:
                source_subset = source_features[mask]

                source_weights = torch.ones(
                    source_subset.size(0), device=source_features.device
                )
                source_weights = source_weights / source_weights.sum()

                target_weights = torch.ones(
                    target_features.size(0), device=target_features.device
                )
                target_weights = target_weights / target_weights.sum()

                ot_dist, _, _ = self.sinkhorn(
                    source_subset, target_features, source_weights, target_weights
                )
                total_loss += ot_dist
                count += 1

        return total_loss / max(count, 1)


def create_sinkhorn(
    epsilon: float = 0.1, max_iter: int = 100, num_inner_iter: int = 10
) -> SinkhornDistance:
    """Factory function to create Sinkhorn distance layer."""
    return SinkhornDistance(
        SinkhornConfig(
            epsilon=epsilon, max_iter=max_iter, num_inner_iter=num_inner_iter
        )
    )


def create_wdgrl(
    feature_extractor: nn.Module,
    classifier: nn.Module,
    feature_dim: int,
    config: Optional[WDGRLConfig] = None,
) -> WDGRLModel:
    """Factory function to create WDGRL model."""
    config = config or WDGRLConfig()

    critic = nn.Sequential(
        nn.Linear(feature_dim, config.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(config.hidden_dim, config.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(config.hidden_dim, 1),
    )

    return WDGRLModel(
        feature_extractor=feature_extractor,
        classifier=classifier,
        critic=critic,
        config=config,
    )


def create_jcpot(config: Optional[JCPOTConfig] = None) -> JCPOTModel:
    """Factory function to create JCPOT model."""
    return JCPOTModel(config=config)


def create_ot_adaptation(
    feature_extractor: nn.Module,
    classifier: nn.Module,
    config: Optional[OTConfig] = None,
) -> OTBasedDomainAdaptation:
    """Factory function to create OT-based domain adaptation model."""
    return OTBasedDomainAdaptation(
        feature_extractor=feature_extractor, classifier=classifier, config=config
    )

"""
Advanced Contrastive Loss Functions

Extended contrastive loss implementations beyond standard InfoNCE,
including supervised, proto-typical, and multi-view variants.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Supervised version of contrastive learning that uses label information
    to pull together samples of the same class and push apart different classes.

    Args:
        temperature: Softmax temperature for scaling similarities.
        base_temperature: Base temperature for normalization.
        reduction: Specifies the reduction to apply: 'none', 'mean', 'sum'.

    Example:
        >>> loss_fn = SupConLoss(temperature=0.07)
        >>> features = torch.randn(16, 128)
        >>> labels = torch.randint(0, 4, (16,))
        >>> loss = loss_fn(features, labels)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        batch_size = features.size(0)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        logits = features @ features.T / self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * (
            1 - torch.eye(batch_size, device=features.device)
        )

        log_probs = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos = mask.sum(1)
        mask_pos = torch.where(mask_pos > 0, mask_pos, torch.ones_like(mask_pos))

        mean_log_prob_pos = -(mask * log_probs).sum(1) / mask_pos

        loss = mean_log_prob_pos / torch.tensor(
            self.temperature / self.base_temperature, device=features.device
        )
        loss = loss * self.base_temperature

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ProtoNCELoss(nn.Module):
    """
    Prototypical NCE Loss.

    Contrastive loss using prototype representations as anchors.

    Args:
        temperature: Softmax temperature.
        num_prototypes: Number of prototypes per class.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        num_prototypes: int = 5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.num_prototypes = num_prototypes
        self.reduction = reduction

    def forward(
        self,
        features: Tensor,
        prototypes: Tensor,
        labels: Tensor,
    ) -> Tensor:
        batch_size = features.size(0)
        num_classes = prototypes.size(0)

        features_normalized = F.normalize(features, p=2, dim=1)
        prototypes_normalized = F.normalize(prototypes, p=2, dim=1)

        similarities = features_normalized @ prototypes_normalized.T / self.temperature

        labels_expanded = labels.unsqueeze(1).expand(-1, self.num_prototypes)
        class_indices = torch.arange(num_classes, device=features.device).unsqueeze(0)
        class_indices = class_indices.expand(batch_size, -1)
        mask = torch.eq(labels_expanded, class_indices)

        logits = similarities
        exp_logits = torch.exp(logits - logits.max(1, keepdim=True)[0])

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos = mask.float()
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / mask_pos.sum(1).clamp(min=1)

        loss = -mean_log_prob_pos

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiViewContrastiveLoss(nn.Module):
    """
    Multi-View Contrastive Loss.

    Contrastive loss for multi-view data where different views of the
    same sample should be close in representation space.

    Args:
        temperature: Softmax temperature.
        view_weights: Optional weights for different views.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        view_weights: Optional[List[float]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.view_weights = view_weights
        self.reduction = reduction

    def forward(
        self,
        views: List[Tensor],
    ) -> Tensor:
        num_views = len(views)
        if self.view_weights is None:
            view_weights = [1.0 / num_views] * num_views
        else:
            view_weights = self.view_weights

        all_features = torch.cat(views, dim=0)
        batch_size = views[0].size(0)

        features_normalized = F.normalize(all_features, p=2, dim=1)
        similarities = features_normalized @ features_normalized.T / self.temperature

        logits = similarities
        exp_logits = torch.exp(logits - logits.max(1, keepdim=True)[0])

        mask = torch.zeros_like(logits, dtype=torch.bool)
        for i in range(num_views):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            for j in range(num_views):
                if i != j:
                    other_start = j * batch_size
                    other_end = (j + 1) * batch_size
                    mask[start_idx:end_idx, other_start:other_end] = True

        exp_logits = exp_logits.masked_fill(~mask, 0)
        log_probs = logits - torch.log(exp_logits.sum(1, keepdim=True))

        diagonal_mask = torch.zeros_like(logits, dtype=torch.bool)
        for i in range(num_views):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            diagonal_mask[start_idx:end_idx, start_idx:end_idx] = True

        positive_logits = logits.masked_fill(~diagonal_mask, float("-inf"))
        positive_log_probs = positive_logits - torch.log(
            exp_logits.sum(1, keepdim=True).masked_fill(~mask, 1)
        )

        loss = -positive_log_probs.masked_fill(~diagonal_mask, 0).sum() / (
            num_views * batch_size
        )

        if self.reduction == "mean":
            return loss
        elif self.reduction == "sum":
            return loss * num_views * batch_size
        return loss


class InstanceContrastiveLoss(nn.Module):
    """
    Instance Contrastive Loss with Hard Negative Mining.

    Contrastive loss that focuses on hard negatives for more
    effective learning.

    Args:
        temperature: Softmax temperature.
        num_hard_negatives: Number of hard negatives to use per sample.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        num_hard_negatives: int = 5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives
        self.reduction = reduction

    def forward(self, features: Tensor, indices: Optional[Tensor] = None) -> Tensor:
        batch_size = features.size(0)

        features_normalized = F.normalize(features, p=2, dim=1)
        similarities = features_normalized @ features_normalized.T

        mask = torch.ones_like(similarities, dtype=torch.bool)
        mask.fill_diagonal_(False)

        hard_negatives_topk = torch.topk(
            similarities.masked_fill(~mask, float("-inf")),
            k=min(self.num_hard_negatives, batch_size - 2),
            dim=1,
        )[0]

        hard_negative_scores = hard_negatives_topk / self.temperature
        exp_hard_negatives = torch.exp(
            hard_negative_scores - hard_negative_scores.max()
        )

        positive_mask = torch.zeros_like(similarities, dtype=torch.bool)
        for i in range(batch_size):
            if indices is not None:
                matches = indices == indices[i]
                matches[i] = False
                positive_mask[i] = matches
            else:
                positive_mask[i, i] = True
                positive_mask[i] = ~positive_mask[i]

        positive_similarities = similarities.masked_fill(~positive_mask, float("-inf"))
        positive_scores = positive_similarities / self.temperature

        exp_positive = torch.exp(
            positive_scores - positive_scores.max(dim=1, keepdim=True)[0]
        )

        denominator = exp_positive.sum(dim=1, keepdim=True) + exp_hard_negatives.sum(
            dim=1, keepdim=True
        )
        log_probs = positive_scores - torch.log(denominator)

        loss = -log_probs.masked_fill(positive_mask, 0).sum(dim=1) / positive_mask.sum(
            dim=1
        ).clamp(min=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ClusterContrastiveLoss(nn.Module):
    """
    Cluster-Instance Contrastive Loss.

    Combines cluster-level and instance-level contrastive learning.

    Args:
        temperature: Softmax temperature.
        cluster_weight: Weight for cluster-level contrastive term.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        cluster_weight: float = 0.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.cluster_weight = cluster_weight
        self.reduction = reduction

    def forward(
        self,
        features: Tensor,
        cluster_labels: Tensor,
        instance_labels: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = features.size(0)
        features_normalized = F.normalize(features, p=2, dim=1)

        instance_loss = self._instance_contrastive(features_normalized, instance_labels)
        cluster_loss = self._cluster_contrastive(features_normalized, cluster_labels)

        loss = (
            1 - self.cluster_weight
        ) * instance_loss + self.cluster_weight * cluster_loss

        return loss

    def _instance_contrastive(
        self,
        features: Tensor,
        instance_labels: Optional[Tensor],
    ) -> Tensor:
        batch_size = features.size(0)
        similarities = features @ features.T / self.temperature

        mask = torch.ones_like(similarities, dtype=torch.bool)
        mask.fill_diagonal_(False)

        exp_logits = torch.exp(similarities - similarities.max(1, keepdim=True)[0])
        exp_logits = exp_logits.masked_fill(~mask, 0)

        log_probs = similarities - torch.log(
            exp_logits.sum(1, keepdim=True).clamp(min=1e-8)
        )

        if instance_labels is not None:
            mask_same = instance_labels.unsqueeze(0) == instance_labels.unsqueeze(1)
            mask_same.fill_diagonal_(False)
        else:
            mask_same = torch.zeros_like(mask)

        loss = -(mask_same * log_probs).sum() / mask_same.sum().clamp(min=1)
        return loss

    def _cluster_contrastive(self, features: Tensor, cluster_labels: Tensor) -> Tensor:
        cluster_centers = []
        unique_labels = torch.unique(cluster_labels)
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_centers.append(features[mask].mean(0))
        prototypes = torch.stack(cluster_centers)

        prototypes_normalized = F.normalize(prototypes, p=2, dim=1)
        similarities = features @ prototypes_normalized.T / self.temperature

        cluster_idx = (cluster_labels.unsqueeze(1) == unique_labels.unsqueeze(0)).long()
        target_idx = cluster_idx.argmax(1)

        loss = F.cross_entropy(similarities, target_idx)
        return loss


class NTNContrastiveLoss(nn.Module):
    """
    Neural Tensor Network enhanced Contrastive Loss.

    Uses bilinear scoring for computing similarity between samples.

    Args:
        embedding_dim: Dimension of input embeddings.
        num_filters: Number of tensor slices.
        temperature: Softmax temperature.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_filters: int = 16,
        temperature: float = 0.07,
        reduction: str = "mean",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.temperature = temperature
        self.reduction = reduction

        self.tensor_layer = nn.Bilinear(embedding_dim, embedding_dim, num_filters)
        self.weight_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_filters),
        )

    def forward(
        self,
        features1: Tensor,
        features2: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = features1.size(0)

        tensor_scores = self.tensor_layer(features1, features2).squeeze(-1)

        concat_features = torch.cat([features1, features2], dim=1)
        linear_scores = self.weight_net(concat_features)

        scores = tensor_scores + linear_scores
        scores = scores / self.temperature

        if labels is not None:
            mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        else:
            mask = torch.eye(batch_size, device=features1.device, dtype=torch.bool)

        exp_scores = torch.exp(scores - scores.max(1, keepdim=True)[0])
        exp_scores = exp_scores.masked_fill(
            torch.eye(batch_size, device=features1.device), 0
        )

        positive_scores = scores.masked_fill(~mask, float("-inf"))
        positive_exp = torch.exp(positive_scores - scores.max(1, keepdim=True)[0])

        denominator = positive_exp.sum(1) + exp_scores.sum(1)
        loss = -(positive_scores - torch.log(denominator.clamp(min=1e-8)))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AngularContrastiveLoss(nn.Module):
    """
    Angular Contrastive Loss with Additive Angular Margin.

    Adds angular margin to increase inter-class separation.

    Args:
        embedding_dim: Dimension of embeddings.
        num_classes: Number of classes.
        margin: Angular margin in radians.
        scale: Scaling factor for logits.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 30.0,
    ):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        normalized_features = F.normalize(features, p=2, dim=1)
        normalized_weight = F.normalize(self.weight, p=2, dim=1)

        cosine = normalized_features @ normalized_weight.T

        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_cosine = torch.cos(theta + self.margin)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = torch.where(one_hot.bool(), target_cosine, cosine)
        logits = logits * self.scale

        loss = F.cross_entropy(logits, labels)
        return loss


__all__ = [
    "SupConLoss",
    "ProtoNCELoss",
    "MultiViewContrastiveLoss",
    "InstanceContrastiveLoss",
    "ClusterContrastiveLoss",
    "NTNContrastiveLoss",
    "AngularContrastiveLoss",
]

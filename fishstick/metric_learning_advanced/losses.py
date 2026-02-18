from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContrastiveConfig:
    margin: float = 1.0
    embedding_dim: int = 128


@dataclass
class TripletConfig:
    margin: float = 0.5
    embedding_dim: int = 128


@dataclass
class NPairConfig:
    embedding_dim: int = 128
    lr: float = 0.001


@dataclass
class NTXentConfig:
    temperature: float = 0.07
    embedding_dim: int = 128


@dataclass
class ArcFaceConfig:
    embedding_dim: int = 128
    num_classes: int = 1000
    s: float = 64.0
    m: float = 0.5


@dataclass
class CosFaceConfig:
    embedding_dim: int = 128
    num_classes: int = 1000
    s: float = 64.0
    m: float = 0.35


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(embeddings, embeddings, p=2)
        same_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff_mask = ~same_mask
        same_mask = same_mask.float().fill_diagonal_(0)

        pos_dist = (distances * same_mask).sum() / (same_mask.sum() + 1e-8)
        neg_dist = F.relu(self.margin - distances * diff_mask.float()).sum() / (
            diff_mask.sum() + 1e-8
        )

        return pos_dist + neg_dist


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class NPairLoss(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.t())

        labels = labels.unsqueeze(0)
        mask = (labels == labels.t()).float()
        mask = mask - torch.eye(mask.size(0), device=mask.device)

        pos_sim = (similarity_matrix * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        log_prob = pos_sim - torch.logsumexp(similarity_matrix, dim=1)
        return -log_prob.mean()


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature

        batch_size = embeddings.size(0)
        labels = torch.arange(batch_size, device=embeddings.device)
        labels = torch.cat([labels[batch_size // 2 :], labels[: batch_size // 2]])

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


class ArcFaceLoss(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 1000,
        s: float = 64.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = torch.matmul(embeddings, weight.t())
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-8, 1.0 - 1e-8))

        target_logits = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = cosine * (1 - one_hot) + target_logits * one_hot

        return F.cross_entropy(logits * self.s, labels)


class CosFaceLoss(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 1000,
        s: float = 64.0,
        m: float = 0.35,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        cosine = torch.matmul(embeddings, weight.t())
        target_logits = cosine - self.m

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = cosine * (1 - one_hot) + target_logits * one_hot

        return F.cross_entropy(logits * self.s, labels)


class MetricLossEnsemble(nn.Module):
    def __init__(self, losses: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights else [1.0 / len(losses)] * len(losses)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(embeddings, labels)
        return total_loss

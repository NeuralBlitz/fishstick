from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from sklearn.cluster import KMeans


@dataclass
class DeepClusterConfig:
    n_clusters: int = 10000
    embed_dim: int = 256
    batch_size: int = 256
    lr: float = 0.001
    epochs: int = 100
    assignment_epochs: int = 10


@dataclass
class SeLaConfig:
    n_clusters: int = 10000
    embed_dim: int = 256
    batch_size: int = 256
    lr: float = 0.001
    epochs: int = 100
    assigner_lr: float = 0.05


@dataclass
class SCANConfig:
    n_clusters: int = 100
    embed_dim: int = 256
    batch_size: int = 256
    lr: float = 0.001
    epochs: int = 100
    momentum: float = 0.999
    tau: float = 0.1
    entropy_weight: float = 0.5


@dataclass
class ClusterConsistencyConfig:
    n_clusters: int = 100
    embed_dim: int = 256
    batch_size: int = 256
    lr: float = 0.001
    epochs: int = 100
    consistency_weight: float = 0.5


class ClusterHead(nn.Module):
    def __init__(self, input_dim: int, n_clusters: int):
        super().__init__()
        self.cluster_head = nn.Linear(input_dim, n_clusters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cluster_head(x)


class DeepCluster(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int = 512,
        config: Optional[DeepClusterConfig] = None,
    ):
        super().__init__()
        self.config = config or DeepClusterConfig()
        self.encoder = encoder
        self.n_clusters = self.config.n_clusters
        self.cluster_head = ClusterHead(input_dim, self.n_clusters)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.cluster_head(features)
        return features, logits

    def assign_clusters(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
            labels = kmeans.fit_predict(features.cpu().numpy())
            labels = torch.from_numpy(labels).to(features.device)
        return labels

    def cluster_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class AssignerHead(nn.Module):
    def __init__(self, input_dim: int, n_clusters: int):
        super().__init__()
        self.assigner = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, n_clusters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.assigner(x)


class SeLa(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int = 512,
        config: Optional[SeLaConfig] = None,
    ):
        super().__init__()
        self.config = config or SeLaConfig()
        self.encoder = encoder
        self.n_clusters = self.config.n_clusters
        self.assigner_head = AssignerHead(input_dim, self.n_clusters)
        self.assigner_lr = self.config.assigner_lr

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        assigner_logits = self.assigner_head(features)
        return features, assigner_logits

    def assign_labels(self, assigner_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            labels = torch.argmax(assigner_logits, dim=1)
        return labels

    def sela_loss(
        self,
        features: torch.Tensor,
        assigner_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cluster_centers = F.normalize(
            torch.randn(self.n_clusters, features.size(1), device=features.device),
            dim=1,
        )

        assignments = F.softmax(assigner_logits / self.assigner_lr, dim=1)

        cluster_features = torch.matmul(assignments.T, features)

        similarity = torch.matmul(
            F.normalize(features, dim=1), F.normalize(cluster_features, dim=1).T
        )

        loss = F.cross_entropy(similarity, labels)
        return loss


class SCANLoss(nn.Module):
    def __init__(self, tau: float = 0.1, entropy_weight: float = 0.5):
        super().__init__()
        self.tau = tau
        self.entropy_weight = entropy_weight

    def forward(
        self,
        anchors: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        anchors = F.normalize(anchors, dim=1)
        neighbors = F.normalize(neighbors, dim=1)

        similarity = torch.matmul(anchors, neighbors.T) / self.tau

        loss = F.cross_entropy(
            similarity, torch.arange(len(anchors), device=anchors.device)
        )

        probs = F.softmax(similarity, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        return loss - self.entropy_weight * entropy


class SCAN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int = 512,
        config: Optional[SCANConfig] = None,
    ):
        super().__init__()
        self.config = config or SCANConfig()
        self.encoder = encoder
        self.n_clusters = self.config.n_clusters
        self.momentum = self.config.momentum
        self.tau = self.config.tau
        self.entropy_weight = self.config.entropy_weight

        self.cluster_head = ClusterHead(input_dim, self.n_clusters)
        self.momentum_encoder = self._create_momentum_encoder(encoder, input_dim)

        for param in self.momentum_encoder.parameters():
            param.requires_grad = False

        self.scan_loss = SCANLoss(self.tau, self.entropy_weight)

    def _create_momentum_encoder(self, encoder: nn.Module, input_dim: int) -> nn.Module:
        from copy import deepcopy

        return deepcopy(encoder)

    @torch.no_grad()
    def _momentum_update_encoder(self):
        for param_online, param_momentum in zip(
            self.encoder.parameters(), self.momentum_encoder.parameters()
        ):
            param_momentum.data.mul_(self.momentum).add_(
                param_online.data, alpha=1 - self.momentum
            )

    def forward(
        self,
        x_anchor: torch.Tensor,
        x_neighbor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._momentum_update_encoder()

        features_anchor = self.encoder(x_anchor)
        features_neighbor = self.momentum_encoder(x_neighbor)

        logits_anchor = self.cluster_head(features_anchor)
        logits_neighbor = self.cluster_head(features_neighbor)

        return features_anchor, logits_anchor, logits_neighbor

    def compute_loss(
        self,
        features_anchor: torch.Tensor,
        logits_anchor: torch.Tensor,
        features_neighbor: torch.Tensor,
    ) -> torch.Tensor:
        return self.scan_loss(features_anchor, features_neighbor)


class ClusterAssignmentConsistency(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int = 512,
        config: Optional[ClusterConsistencyConfig] = None,
    ):
        super().__init__()
        self.config = config or ClusterConsistencyConfig()
        self.encoder = encoder
        self.n_clusters = self.config.n_clusters
        self.consistency_weight = self.config.consistency_weight

        self.cluster_head = ClusterHead(input_dim, self.n_clusters)
        self.ema_cluster_head = self._create_ema_cluster_head(input_dim)

        for param in self.ema_cluster_head.parameters():
            param.requires_grad = False

    def _create_ema_cluster_head(self, input_dim: int) -> nn.Module:
        from copy import deepcopy

        return deepcopy(self.cluster_head)

    @torch.no_grad()
    def _update_ema_cluster_head(self, momentum: float = 0.999):
        for param_online, param_ema in zip(
            self.cluster_head.parameters(), self.ema_cluster_head.parameters()
        ):
            param_ema.data.mul_(momentum).add_(param_online.data, alpha=1 - momentum)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features1 = self.encoder(x1)
        features2 = self.encoder(x2)

        logits1 = self.cluster_head(features1)
        logits2 = self.ema_cluster_head(features2)

        return logits1, logits2

    def consistency_loss(
        self, logits1: torch.Tensor, logits2: torch.Tensor
    ) -> torch.Tensor:
        prob1 = F.softmax(logits1, dim=1)
        prob2 = F.softmax(logits2, dim=1)

        consistency = F.kl_div(prob1.log(), prob2, reduction="batchmean")
        return consistency

    def cluster_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


def train_deepcluster(
    model: DeepCluster,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    features_buffer: List[torch.Tensor],
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        features, logits = model(images)
        features_buffer.append(features.detach())

        labels = model.assign_clusters(features.detach())
        loss = model.cluster_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if len(features_buffer) > 10000:
        features_buffer = features_buffer[-10000:]

    return total_loss / len(dataloader)


def train_sela(
    model: SeLa,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        features, assigner_logits = model(images)

        labels = model.assign_labels(assigner_logits.detach())

        loss = model.sela_loss(features, assigner_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_scan(
    model: SCAN,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        x_anchor = images[: images.size(0) // 2]
        x_neighbor = images[images.size(0) // 2 :]

        features_anchor, logits_anchor, logits_neighbor = model(x_anchor, x_neighbor)

        loss = model.compute_loss(features_anchor, logits_anchor, logits_neighbor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_cluster_consistency(
    model: ClusterAssignmentConsistency,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        x1 = images[: images.size(0) // 2]
        x2 = images[images.size(0) // 2 :]

        logits1, logits2 = model(x1, x2)

        labels = torch.argmax(logits1, dim=1)

        cluster_loss = model.cluster_loss(logits1, labels)
        consistency_loss = model.consistency_loss(logits1, logits2)

        loss = cluster_loss + model.consistency_weight * consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model._update_ema_cluster_head()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def update_cluster_centers(
    model: DeepCluster,
    features: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    with torch.no_grad():
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        centers = kmeans.fit_transform(features.cpu().numpy())
        centers = torch.from_numpy(kmeans.cluster_centers_).float().to(features.device)
    return centers

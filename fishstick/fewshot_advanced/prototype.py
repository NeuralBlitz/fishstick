import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class PrototypicalNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        distance: str = "euclidean",
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.distance = distance

    @property
    def out_features(self):
        return self.encoder.out_features

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_support = support.size(1)

        support_emb = self.encoder(support.view(-1, *support.shape[2:]))
        query_emb = self.encoder(query.view(-1, *query.shape[2:]))

        support_emb = support_emb.view(-2, n_support, -1)
        query_emb = query_emb.view(-2, query.size(1), -1)

        prototypes = self._compute_prototypes(support_emb, support_labels)

        distances = self._compute_distances(query_emb, prototypes)
        logits = -distances

        if support_labels is not None:
            return self._compute_loss(distances, support_labels, n_support)

        return logits

    def _compute_prototypes(
        self,
        support_emb: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        n_way = support_emb.size(0)
        prototypes = torch.zeros(n_way, support_emb.size(-1), device=support_emb.device)

        for i in range(n_way):
            mask = support_labels == i
            if mask.sum() > 0:
                prototypes[i] = support_emb[mask].mean(0)
            else:
                prototypes[i] = support_emb[i * support_emb.size(1) // n_way]

        return prototypes

    def _compute_distances(
        self, query_emb: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        if self.distance == "euclidean":
            distances = torch.cdist(query_emb, prototypes, p=2)
        elif self.distance == "cosine":
            query_norm = F.normalize(query_emb, p=2, dim=-1)
            proto_norm = F.normalize(prototypes, p=2, dim=-1)
            distances = 1 - torch.einsum("bnd,cd->bnc", query_norm, proto_norm)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        return distances

    def _compute_loss(
        self, distances: torch.Tensor, labels: torch.Tensor, n_support: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(distances.size(0), -1)
        query_labels = labels[:, n_support:]

        return F.cross_entropy(-distances[:, n_support:], query_labels)


class ClusterProtoNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        n_clusters: int = 5,
        distance: str = "euclidean",
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.n_clusters = n_clusters
        self.distance = distance

    @property
    def out_features(self):
        return self.encoder.out_features

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_support = support.size(1)

        support_emb = self.encoder(support.view(-1, *support.shape[2:]))
        query_emb = self.encoder(query.view(-1, *query.shape[2:]))

        support_emb = support_emb.view(-2, n_support, -1)
        query_emb = query_emb.view(-2, query.size(1), -1)

        cluster_prototypes = self._compute_cluster_prototypes(support_emb)

        class_prototypes = self._cluster_to_class(cluster_prototypes, support_labels)

        distances = self._compute_distances(query_emb, class_prototypes)
        logits = -distances

        if support_labels is not None:
            return self._compute_loss(distances, support_labels, n_support)

        return logits

    def _compute_cluster_prototypes(self, support_emb: torch.Tensor) -> torch.Tensor:
        n_way = support_emb.size(0)
        cluster_prototypes = []

        for i in range(n_way):
            class_emb = support_emb[i]
            k = min(self.n_clusters, len(class_emb))

            centroids = self._kmeans(class_emb, k)
            cluster_prototypes.append(centroids)

        return torch.stack(cluster_prototypes)

    def _kmeans(self, x: torch.Tensor, k: int, max_iter: int = 100) -> torch.Tensor:
        indices = torch.randperm(len(x))[:k]
        centroids = x[indices].clone()

        for _ in range(max_iter):
            distances = torch.cdist(x, centroids, p=2)
            assignments = distances.argmin(dim=1)

            new_centroids = torch.zeros_like(centroids)
            for j in range(k):
                mask = assignments == j
                if mask.sum() > 0:
                    new_centroids[j] = x[mask].mean(0)
                else:
                    new_centroids[j] = centroids[j]

            if torch.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return centroids

    def _cluster_to_class(
        self,
        cluster_prototypes: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        return cluster_prototypes.mean(dim=1)

    def _compute_distances(
        self, query_emb: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        if self.distance == "euclidean":
            distances = torch.cdist(query_emb, prototypes, p=2)
        elif self.distance == "cosine":
            query_norm = F.normalize(query_emb, p=2, dim=-1)
            proto_norm = F.normalize(prototypes, p=2, dim=-1)
            distances = 1 - torch.einsum("bnd,cd->bnc", query_norm, proto_norm)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
        return distances

    def _compute_loss(
        self, distances: torch.Tensor, labels: torch.Tensor, n_support: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(distances.size(0), -1)
        query_labels = labels[:, n_support:]
        return F.cross_entropy(-distances[:, n_support:], query_labels)


class CentroidNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        use_softmax: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.use_softmax = use_softmax

    @property
    def out_features(self):
        return self.encoder.out_features

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_support = support.size(1)

        support_emb = self.encoder(support.view(-1, *support.shape[2:]))
        query_emb = self.encoder(query.view(-1, *query.shape[2:]))

        support_emb = support_emb.view(-2, n_support, -1)
        query_emb = query_emb.view(-2, query.size(1), -1)

        centroids = self._compute_centroids(support_emb, support_labels)

        logits = self._compute_logits(query_emb, centroids)

        if support_labels is not None:
            return self._compute_loss(logits, support_labels, n_support)

        return logits

    def _compute_centroids(
        self,
        support_emb: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        n_way = support_emb.size(0)
        centroids = torch.zeros(n_way, support_emb.size(-1), device=support_emb.device)

        for i in range(n_way):
            mask = support_labels == i
            if mask.sum() > 0:
                centroids[i] = support_emb[mask].mean(0)

        return centroids

    def _compute_logits(
        self, query_emb: torch.Tensor, centroids: torch.Tensor
    ) -> torch.Tensor:
        similarities = torch.einsum("bnd,cd->bnc", query_emb, centroids)

        if self.use_softmax:
            logits = F.softmax(similarities, dim=-1)
        else:
            logits = similarities

        return logits

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, n_support: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(logits.size(0), -1)
        query_labels = labels[:, n_support:]

        return F.cross_entropy(logits[:, n_support:], query_labels)


class MaskedPrototypicalNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        temperature: float = 10.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.temperature = temperature

    @property
    def out_features(self):
        return self.encoder.out_features

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_support = support.size(1)

        support_emb = self.encoder(support.view(-1, *support.shape[2:]))
        query_emb = self.encoder(query.view(-1, *query.shape[2:]))

        support_emb = support_emb.view(-2, n_support, -1)
        query_emb = query_emb.view(-2, query.size(1), -1)

        prototypes = self._compute_prototypes(support_emb, support_labels)

        masked_prototypes = self._apply_attention_mask(
            support_emb, prototypes, support_labels
        )

        distances = torch.cdist(query_emb, masked_prototypes, p=2)
        logits = -distances / self.temperature

        if support_labels is not None:
            return self._compute_loss(logits, support_labels, n_support)

        return logits

    def _compute_prototypes(
        self,
        support_emb: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        n_way = support_emb.size(0)
        prototypes = torch.zeros(n_way, support_emb.size(-1), device=support_emb.device)

        for i in range(n_way):
            mask = support_labels == i
            if mask.sum() > 0:
                prototypes[i] = support_emb[mask].mean(0)

        return prototypes

    def _apply_attention_mask(
        self,
        support_emb: torch.Tensor,
        prototypes: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        n_way = support_emb.size(0)
        distances_to_proto = torch.cdist(support_emb, prototypes, p=2)

        attention_weights = F.softmax(-distances_to_proto, dim=-1)

        attention_weights = attention_weights.unsqueeze(-1)
        masked_emb = support_emb * attention_weights

        masked_prototypes = torch.zeros_like(prototypes)
        for i in range(n_way):
            mask = support_labels == i
            if mask.sum() > 0:
                masked_prototypes[i] = masked_emb[mask].mean(0)
            else:
                masked_prototypes[i] = prototypes[i]

        return masked_prototypes

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, n_support: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(logits.size(0), -1)
        query_labels = labels[:, n_support:]
        return F.cross_entropy(logits[:, n_support:], query_labels)


class VariationalPrototypicalNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        self.mu_head = nn.Linear(encoder.out_features, latent_dim)
        self.logvar_head = nn.Linear(encoder.out_features, latent_dim)

    @property
    def out_features(self):
        return self.latent_dim

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_support = support.size(1)

        support_emb = self.encoder(support.view(-1, *support.shape[2:]))
        query_emb = self.encoder(query.view(-1, *query.shape[2:]))

        support_emb = support_emb.view(-2, n_support, -1)
        query_emb = query_emb.view(-2, query.size(1), -1)

        support_z = self._reparameterize(support_emb)
        query_z = self._reparameterize(query_emb)

        prototypes = self._compute_prototypes(support_z, support_labels)

        distances = torch.cdist(query_z, prototypes, p=2)
        logits = -distances

        if support_labels is not None:
            return self._compute_loss(logits, support_labels, n_support)

        return logits

    def _reparameterize(self, emb: torch.Tensor) -> torch.Tensor:
        mu = self.mu_head(emb)
        logvar = self.logvar_head(emb)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def _compute_prototypes(
        self,
        support_emb: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        n_way = support_emb.size(0)
        prototypes = torch.zeros(n_way, self.latent_dim, device=support_emb.device)

        for i in range(n_way):
            mask = support_labels == i
            if mask.sum() > 0:
                prototypes[i] = support_emb[mask].mean(0)

        return prototypes

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, n_support: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(logits.size(0), -1)
        query_labels = labels[:, n_support:]
        return F.cross_entropy(logits[:, n_support:], query_labels)

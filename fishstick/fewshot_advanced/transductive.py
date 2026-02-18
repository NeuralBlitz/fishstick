import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class TransductivePrototypicalNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        transductive_steps: int = 5,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.transductive_steps = transductive_steps

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

        for _ in range(self.transductive_steps):
            query_labels = self._assign_labels(query_emb, prototypes)
            prototypes = self._update_prototypes(
                support_emb, query_emb, support_labels, query_labels
            )

        distances = torch.cdist(query_emb, prototypes, p=2)
        logits = -distances

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

    def _assign_labels(
        self, query_emb: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        distances = torch.cdist(query_emb, prototypes, p=2)
        labels = distances.argmin(dim=-1)
        return labels

    def _update_prototypes(
        self,
        support_emb: torch.Tensor,
        query_emb: torch.Tensor,
        support_labels: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> torch.Tensor:
        n_way = support_emb.size(0)
        all_emb = torch.cat([support_emb, query_emb], dim=1)
        all_labels = torch.cat([support_labels, query_labels], dim=1)

        prototypes = torch.zeros(n_way, all_emb.size(-1), device=all_emb.device)

        for i in range(n_way):
            mask = all_labels == i
            if mask.sum() > 0:
                prototypes[i] = all_emb[mask].mean(0)

        return prototypes

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, n_support: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(logits.size(0), -1)
        query_labels = labels[:, n_support:]
        return F.cross_entropy(logits[:, n_support:], query_labels)


class LabelPropagation(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        alpha: float = 0.5,
        iterations: int = 10,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.alpha = alpha
        self.iterations = iterations

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

        all_emb = torch.cat([support_emb, query_emb], dim=1)

        labels = self._propagate_labels(all_emb, support_labels)

        query_labels = labels[:, n_support:]

        if support_labels is not None:
            return self._compute_loss(query_labels, support_labels, n_support)

        return query_labels

    def _propagate_labels(
        self,
        embeddings: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        n_way = support_labels.max().item() + 1
        n_samples = embeddings.size(1)

        similarity = torch.mm(embeddings.squeeze(0), embeddings.squeeze(0).t())
        adjacency = (similarity > 0.5).float()
        degree = adjacency.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1
        normalized = adjacency / degree

        label_matrix = torch.zeros(n_samples, n_way, device=embeddings.device)
        for i in range(n_way):
            label_matrix[support_labels == i, i] = 1.0

        labels = label_matrix.clone()

        for _ in range(self.iterations):
            labels = (1 - self.alpha) * labels + self.alpha * torch.mm(
                normalized, labels
            )

        return labels

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, n_support: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(logits.size(0), -1)
        query_labels = labels[:, n_support:]
        return F.cross_entropy(logits, query_labels)


class TransductiveAttentionProtocol(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        num_propagations: int = 3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.num_propagations = num_propagations
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

        all_emb = torch.cat([support_emb, query_emb], dim=1)

        prototypes = self._compute_prototypes(support_emb, support_labels)

        attention_weights = self._compute_attention(all_emb, prototypes)

        for _ in range(self.num_propagations):
            prototypes = self._propagate_attention(
                all_emb, prototypes, attention_weights
            )
            attention_weights = self._update_attention(all_emb, prototypes)

        distances = torch.cdist(query_emb, prototypes, p=2)
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

    def _compute_attention(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        distances = torch.cdist(embeddings.squeeze(0), prototypes, p=2)
        attention = F.softmax(-distances / self.temperature, dim=-1)
        return attention

    def _update_attention(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        return self._compute_attention(embeddings, prototypes)

    def _propagate_attention(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        n_way = prototypes.size(0)
        n_samples = embeddings.size(1)

        weighted_emb = embeddings.squeeze(0).unsqueeze(-1) * attention.unsqueeze(-1)

        new_prototypes = torch.zeros_like(prototypes)
        for i in range(n_way):
            new_prototypes[i] = weighted_emb[:, i, :].mean(0)

        return new_prototypes

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, n_support: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(logits.size(0), -1)
        query_labels = labels[:, n_support:]
        return F.cross_entropy(logits[:, n_support:], query_labels)


class GraphPropagationNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        k_neighbors: int = 5,
        iterations: int = 3,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.k_neighbors = k_neighbors
        self.iterations = iterations

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

        all_emb = torch.cat([support_emb, query_emb], dim=1)

        prototypes = self._compute_prototypes(support_emb, support_labels)

        query_labels = self._graph_propagate(all_emb, prototypes, support_labels)

        logits = self._compute_logits(query_emb, query_labels)

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

    def _build_knn_graph(self, embeddings: torch.Tensor) -> torch.Tensor:
        n_samples = embeddings.size(1)

        distances = torch.cdist(embeddings.squeeze(0), embeddings.squeeze(0), p=2)

        topk = min(self.k_neighbors, n_samples - 1)
        _, indices = distances.topk(topk, largest=False)

        adjacency = torch.zeros(n_samples, n_samples, device=embeddings.device)
        adjacency.scatter_(1, indices, 1)
        adjacency = (adjacency + adjacency.t()) > 0

        return adjacency.float()

    def _graph_propagate(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        n_way = self.n_classes
        n_samples = embeddings.size(1)

        adjacency = self._build_knn_graph(embeddings)

        label_matrix = torch.zeros(n_samples, n_way, device=embeddings.device)
        for i in range(n_way):
            label_matrix[support_labels == i, i] = 1.0

        labels = label_matrix.clone()

        degree = adjacency.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1
        transition = adjacency / degree

        for _ in range(self.iterations):
            labels = torch.mm(transition, labels)

        return labels.argmax(dim=-1)

    def _compute_logits(
        self, query_emb: torch.Tensor, query_labels: torch.Tensor
    ) -> torch.Tensor:
        n_way = self.n_classes
        n_query = query_emb.size(1)

        logits = torch.zeros(query_emb.size(0), n_query, n_way, device=query_emb.device)

        for i in range(n_way):
            mask = query_labels == i
            logits[:, mask, i] = 1.0

        return logits

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, n_support: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(logits.size(0), -1)
        query_labels = labels[:, n_support:]
        return F.cross_entropy(logits[:, n_support:], query_labels)


class SemiSupervisedFewShot(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        unlabeled_weight: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.unlabeled_weight = unlabeled_weight

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

        distances = torch.cdist(query_emb, prototypes, p=2)
        logits = -distances

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

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, n_support: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(logits.size(0), -1)
        query_labels = labels[:, n_support:]

        sup_loss = F.cross_entropy(logits[:, n_support:], query_labels)

        probs = F.softmax(logits[:, n_support:], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        return sup_loss - self.unlabeled_weight * entropy
